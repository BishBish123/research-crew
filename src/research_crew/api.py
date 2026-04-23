"""FastAPI service: POST /research, GET /runs/{id}, health probe."""

from __future__ import annotations

import asyncio
import contextlib
import os
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import redis.asyncio as aioredis
import structlog
import typer
import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from redis.exceptions import RedisError

from research_crew.agents import default_agents
from research_crew.errors import StoreUnavailableError
from research_crew.models import (
    ResearchReport,
    ResearchRequest,
    RunStatus,
    StepStatus,
)
from research_crew.store import RedisRunStore, RunStore
from research_crew.synthesizer import StitchSynthesizer
from research_crew.workflow import WorkflowEngine

_log = structlog.get_logger(__name__)


def _redis_url() -> str:
    return os.environ.get("REDIS_URL", "redis://localhost:6379/0")


@contextlib.asynccontextmanager
async def _lifespan(app_: FastAPI) -> AsyncIterator[None]:
    """Open the Redis client on startup; close it cleanly on shutdown.

    Tests can override `app_.state.redis` / `app_.state.store` *before*
    issuing requests because the lifespan sees ``getattr(..., None)`` and
    only initialises what's missing.

    Also initialises the in-process terminal-state shadow cache: when the
    store rejects a terminal-state write from the background task (the
    whole point of that path is "store is down"), the bg task stashes the
    final RunStatus here so a subsequent `GET /runs/{run_id}` can still
    surface a terminal state instead of "running forever".
    """
    if getattr(app_.state, "redis", None) is None:
        client = aioredis.from_url(_redis_url(), decode_responses=True)
        app_.state.redis = client
        app_.state.store = RedisRunStore(client)
    if getattr(app_.state, "terminal_shadow", None) is None:
        app_.state.terminal_shadow = {}
    try:
        yield
    finally:
        if getattr(app_.state, "redis", None) is not None:
            with contextlib.suppress(Exception):
                await app_.state.redis.aclose()


app = FastAPI(title="research-crew", version="0.1.0", lifespan=_lifespan)


@app.exception_handler(StoreUnavailableError)
async def _store_unavailable_handler(request: Request, exc: StoreUnavailableError) -> JSONResponse:
    """Map any leaked StoreUnavailableError to a typed 503."""
    _log.warning("api.store_unavailable", path=request.url.path, error=str(exc))
    return JSONResponse(status_code=503, content={"detail": f"run store unavailable: {exc}"})


@app.get("/health")
async def health(request: Request) -> dict[str, str]:
    redis_client = getattr(request.app.state, "redis", None)
    if redis_client is None:
        raise HTTPException(status_code=503, detail="redis client not initialised")
    try:
        pong = await redis_client.ping()
    except Exception as exc:  # surface any connection error as 503
        _log.warning("api.health_redis_down", error=str(exc))
        raise HTTPException(status_code=503, detail=f"redis unavailable: {exc}") from exc
    return {"status": "ok", "redis": "up" if pong else "down"}


def _store(request: Request) -> RunStore:
    store = getattr(request.app.state, "store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="run store not initialised")
    return store  # type: ignore[no-any-return]


@app.post("/research", status_code=202)
async def submit_research(
    payload: ResearchRequest, background: BackgroundTasks, request: Request
) -> dict[str, str]:
    """Enqueue a research run. Returns immediately with the run_id."""
    run_id = uuid.uuid4().hex
    store = _store(request)
    run = RunStatus(run_id=run_id, question=payload.question, state=StepStatus.RUNNING)
    try:
        await store.put_run(run)
    except (StoreUnavailableError, RedisError) as exc:
        _log.warning("api.store_down_on_submit", run_id=run_id, error=str(exc))
        raise HTTPException(status_code=503, detail=f"run store unavailable: {exc}") from exc
    _log.info("api.run_submitted", run_id=run_id, question_len=len(payload.question))
    background.add_task(_execute_run, store, _terminal_shadow(request), run_id, payload)
    return {"run_id": run_id, "status_url": f"/runs/{run_id}"}


_TERMINAL_STATES = (StepStatus.SUCCEEDED, StepStatus.FAILED)


def _terminal_shadow(request: Request) -> dict[str, RunStatus]:
    shadow: dict[str, RunStatus] | None = getattr(
        request.app.state, "terminal_shadow", None
    )
    if shadow is None:
        # Lifespan didn't run (some tests poke app.state directly); make
        # the shadow lookup a no-op rather than crash on missing attr.
        shadow = {}
        request.app.state.terminal_shadow = shadow
    return shadow


@app.get("/runs/{run_id}")
async def get_run(run_id: str, request: Request) -> RunStatus:
    """Return the latest known state for `run_id`.

    Lookup precedence (the shadow exists for the bg-task store-outage path
    where the run *did* finish but the store rejected our terminal write):

    1. Store has a terminal record (SUCCEEDED/FAILED) → return it.
    2. Store has a still-RUNNING record but the shadow has a terminal one
       → return the shadow (the bg task hit a store outage and recovered
       to in-process state, while a stale RUNNING blob lingers in Redis).
    3. Store says 404 but the shadow has it → return the shadow.
    4. Store outage → return the shadow if present, else 503.
    """
    store = _store(request)
    shadow = _terminal_shadow(request)
    try:
        run = await store.get_run(run_id)
    except (StoreUnavailableError, RedisError) as exc:
        shadow_run = shadow.get(run_id)
        if shadow_run is not None:
            _log.info("api.get_run_served_from_shadow_on_outage", run_id=run_id)
            return shadow_run
        _log.warning("api.store_down_on_get", run_id=run_id, error=str(exc))
        raise HTTPException(status_code=503, detail=f"run store unavailable: {exc}") from exc

    if run is None:
        shadow_run = shadow.get(run_id)
        if shadow_run is not None:
            _log.info("api.get_run_served_from_shadow_on_404", run_id=run_id)
            return shadow_run
        raise HTTPException(status_code=404, detail=f"run {run_id} not found")

    if run.state not in _TERMINAL_STATES:
        shadow_run = shadow.get(run_id)
        if shadow_run is not None:
            _log.info("api.get_run_shadow_overrides_running", run_id=run_id)
            return shadow_run

    try:
        run.steps = await store.list_steps(run_id)
    except (StoreUnavailableError, RedisError) as exc:
        _log.warning("api.store_down_on_list_steps", run_id=run_id, error=str(exc))
        raise HTTPException(status_code=503, detail=f"run store unavailable: {exc}") from exc
    return run


# ---------------------------------------------------------------------------
# Background execution
# ---------------------------------------------------------------------------


async def _persist_terminal(
    store: RunStore,
    shadow: dict[str, RunStatus],
    run: RunStatus,
    *,
    agent_label: str,
) -> None:
    """Write a terminal RunStatus to the store; on failure stash in the
    in-process shadow so a subsequent GET still surfaces a terminal state.

    The shadow is only written on the failure path. A successful put
    leaves `shadow[run_id]` untouched so happy-path runs never pollute
    the in-process cache.
    """
    try:
        await store.put_run(run)
    except Exception as exc:
        _log.error(
            "api.background_terminal_write_failed",
            run_id=run.run_id,
            agent=agent_label,
            exc_type=type(exc).__name__,
            error=str(exc),
        )
        shadow[run.run_id] = run


async def _execute_run(
    store: RunStore,
    shadow: dict[str, RunStatus],
    run_id: str,
    payload: ResearchRequest,
) -> None:
    """Run the workflow and persist the terminal state.

    Wrapped so any uncaught failure (workflow bug, store outage,
    synthesizer crash) leaves the run in a terminal FAILED state with
    an error message instead of stuck at `running` forever. When the
    store itself is the cause of the outage, the terminal RunStatus is
    written to the in-process `shadow` cache so `GET /runs/{id}` can
    still report a terminal state — the next-best thing to a durable
    write.
    """
    try:
        agents = default_agents()
        if payload.agents:
            wanted = set(payload.agents)
            agents = [a for a in agents if a.name in wanted]
        engine = WorkflowEngine(
            run_id=run_id,
            record_step=store.append_step,
            cache_get=store.cache_get,
            cache_put=store.cache_put,
        )
        results = await engine.run_parallel(agents, payload.question)
        report: ResearchReport = await StitchSynthesizer().synthesize(
            run_id, payload.question, results
        )
        finished = await store.get_run(run_id)
        if finished is None:
            return
        finished.state = (
            StepStatus.FAILED
            if all(r.status is StepStatus.FAILED for r in results)
            else StepStatus.SUCCEEDED
        )
        finished.finished_at = datetime.now(UTC)
        finished.report = report
        await _persist_terminal(store, shadow, finished, agent_label="workflow_done")
    except Exception as exc:
        _log.error(
            "api.background_run_failed",
            run_id=run_id,
            exc_type=type(exc).__name__,
            error=str(exc),
        )
        # Best-effort: mark the run FAILED so callers see a terminal state.
        # If the store itself is the cause of the outage, both the read
        # and the write may raise — fall back to the in-process shadow
        # so the GET path can still surface FAILED.
        try:
            existing = await store.get_run(run_id)
        except Exception as recovery_exc:
            _log.error(
                "api.background_failed_state_read_failed",
                run_id=run_id,
                exc_type=type(recovery_exc).__name__,
                error=str(recovery_exc),
            )
            existing = None

        # Build the FAILED RunStatus to record. If we couldn't even read
        # the original record, synthesise a minimal one from what we know
        # so the shadow still has a terminal entry to serve.
        failed = existing if existing is not None else RunStatus(
            run_id=run_id,
            question=payload.question,
            state=StepStatus.RUNNING,
        )
        failed.state = StepStatus.FAILED
        failed.finished_at = datetime.now(UTC)
        await _persist_terminal(store, shadow, failed, agent_label="workflow_failed")


# ---------------------------------------------------------------------------
# CLI entry: `research-api`
# ---------------------------------------------------------------------------


_cli = typer.Typer(name="research-api", add_completion=False)


@_cli.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind host."),
    port: int = typer.Option(8000, help="Bind port."),
    reload: bool = typer.Option(False, "--reload", help="uvicorn auto-reload"),
) -> None:
    """Run the FastAPI service."""
    uvicorn.run(
        "research_crew.api:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


def main() -> None:  # pragma: no cover
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    _cli()


if __name__ == "__main__":  # pragma: no cover
    main()
