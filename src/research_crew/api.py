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
    """
    if getattr(app_.state, "redis", None) is None:
        client = aioredis.from_url(_redis_url(), decode_responses=True)
        app_.state.redis = client
        app_.state.store = RedisRunStore(client)
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
    background.add_task(_execute_run, store, run_id, payload)
    return {"run_id": run_id, "status_url": f"/runs/{run_id}"}


@app.get("/runs/{run_id}")
async def get_run(run_id: str, request: Request) -> RunStatus:
    store = _store(request)
    try:
        run = await store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"run {run_id} not found")
        run.steps = await store.list_steps(run_id)
    except (StoreUnavailableError, RedisError) as exc:
        _log.warning("api.store_down_on_get", run_id=run_id, error=str(exc))
        raise HTTPException(status_code=503, detail=f"run store unavailable: {exc}") from exc
    return run


# ---------------------------------------------------------------------------
# Background execution
# ---------------------------------------------------------------------------


async def _execute_run(store: RunStore, run_id: str, payload: ResearchRequest) -> None:
    """Run the workflow and persist the terminal state.

    Wrapped so any uncaught failure (workflow bug, store outage,
    synthesizer crash) leaves the run in a terminal FAILED state with
    an error message instead of stuck at `running` forever. If the
    store itself is also down the best we can do is log loudly.
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
        await store.put_run(finished)
    except Exception as exc:
        _log.error(
            "api.background_run_failed",
            run_id=run_id,
            exc_type=type(exc).__name__,
            error=str(exc),
        )
        # Best-effort: mark the run FAILED so callers see a terminal state.
        # If the store itself is the cause of the outage, this will also
        # raise and the loud `error` log above is the only signal.
        try:
            existing = await store.get_run(run_id)
            if existing is not None:
                existing.state = StepStatus.FAILED
                existing.finished_at = datetime.now(UTC)
                await store.put_run(existing)
        except Exception as recovery_exc:
            _log.error(
                "api.background_failed_state_unrecorded",
                run_id=run_id,
                error=str(recovery_exc),
            )


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
