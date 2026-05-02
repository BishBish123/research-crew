"""FastAPI service: POST /research, GET /runs/{id}, health probe."""

from __future__ import annotations

import asyncio
import contextlib
import os
import time
import uuid
from collections import OrderedDict, deque
from collections.abc import AsyncIterator, Iterator
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


# Default ceiling on the in-process terminal-state shadow. The shadow only
# accumulates entries while the run store is unavailable, so 10k is many
# orders of magnitude more than a typical degraded-mode window. Operators
# can override via the RESEARCH_SHADOW_MAX env var.
_DEFAULT_SHADOW_MAX = 10_000

# Default per-IP rate-limit budget for POST /research. The cost of a
# research run is asymmetric (one HTTP call → 5 fan-out agents + Redis
# writes), so a conservative default protects the upstream search APIs
# from a single misbehaving client. Operators can override via
# RESEARCH_RATE_LIMIT_PER_MIN.
_DEFAULT_RATE_LIMIT_PER_MIN = 10
_RATE_LIMIT_WINDOW_S = 60.0


class _RateLimiter:
    """Per-IP sliding-window counter for POST /research.

    The window is a `deque` of timestamps newer than `now - 60s`. When
    the window already holds `limit` entries, the next request is
    rejected with a Retry-After hint computed from the oldest pending
    timestamp.

    Concurrency note: all reads/writes happen on the FastAPI event loop,
    so dict/deque mutations are not interleaved at the bytecode level.
    No explicit lock is needed.
    """

    def __init__(self, limit_per_min: int, window_s: float = _RATE_LIMIT_WINDOW_S) -> None:
        if limit_per_min <= 0:
            raise ValueError("limit_per_min must be positive")
        self._limit = limit_per_min
        self._window = window_s
        self._buckets: dict[str, deque[float]] = {}

    @property
    def limit(self) -> int:
        return self._limit

    def check(self, ip: str, *, now: float | None = None) -> tuple[bool, float]:
        """Record a request for `ip`. Returns ``(allowed, retry_after_s)``.

        On allow, ``retry_after_s`` is 0. On reject, it's the number of
        seconds the caller would need to wait before the oldest in-window
        timestamp ages out.
        """
        ts = time.monotonic() if now is None else now
        cutoff = ts - self._window
        bucket = self._buckets.get(ip)
        if bucket is None:
            bucket = deque()
            self._buckets[ip] = bucket
        # Drop expired entries off the front before counting.
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        if len(bucket) >= self._limit:
            retry_after = max(0.0, bucket[0] + self._window - ts)
            return False, retry_after
        bucket.append(ts)
        return True, 0.0

    def reset(self) -> None:
        """Test-only hook: drop all buckets."""
        self._buckets.clear()


# Header name carrying the bearer token. We deliberately don't use
# FastAPI's `HTTPBearer` security scheme — the OpenAPI noise it adds
# isn't worth it for a single-token deployment, and Bearer is the
# de-facto convention so a plain `Authorization` header check is fine.
_AUTH_HEADER = "Authorization"
_BEARER_PREFIX = "Bearer "


class _TerminalShadow:
    """Bounded FIFO shadow of terminal RunStatuses.

    Used only on the bg-task store-outage path. Behaves like a `dict` for
    the read sites but evicts the oldest entry on insert once `max_size`
    is exceeded — without this cap, a long store outage with continuous
    submits would grow the process heap unboundedly.

    Concurrency note: shadow reads (GET handler) and shadow writes (bg
    task) both run in the same event loop, so dict mutations are not
    interleaved at the bytecode level. No explicit lock is needed.
    """

    def __init__(self, max_size: int = _DEFAULT_SHADOW_MAX) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        self._items: OrderedDict[str, RunStatus] = OrderedDict()
        self._max_size = max_size

    def __setitem__(self, run_id: str, run: RunStatus) -> None:
        if run_id in self._items:
            # Refresh insertion order so the newest write is the youngest.
            self._items.move_to_end(run_id)
        self._items[run_id] = run
        while len(self._items) > self._max_size:
            evicted_id, _ = self._items.popitem(last=False)
            _log.warning(
                "api.terminal_shadow_evicted",
                run_id=evicted_id,
                max_size=self._max_size,
                reason="shadow_full",
            )

    def __getitem__(self, run_id: str) -> RunStatus:
        return self._items[run_id]

    def __contains__(self, run_id: object) -> bool:
        return run_id in self._items

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)

    def get(self, run_id: str, default: RunStatus | None = None) -> RunStatus | None:
        return self._items.get(run_id, default)

    def pop(self, run_id: str, default: RunStatus | None = None) -> RunStatus | None:
        return self._items.pop(run_id, default)

    def clear(self) -> None:
        self._items.clear()


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
        app_.state.terminal_shadow = _TerminalShadow(
            max_size=int(os.environ.get("RESEARCH_SHADOW_MAX", _DEFAULT_SHADOW_MAX))
        )
    # Auth token — when unset, run open and log a loud DEV-ONLY warning
    # so operators don't ship to prod accidentally.
    token = os.environ.get("RESEARCH_API_TOKEN")
    if not token:
        _log.warning(
            "api.auth_disabled",
            message="RESEARCH_API_TOKEN not set; running unauthenticated — DEV ONLY",
        )
    app_.state.api_token = token or None
    # Rate limiter is always wired up; if RESEARCH_RATE_LIMIT_PER_MIN
    # is unset we use the conservative default. Recreate per lifespan
    # so `make api && CTRL+C && make api` doesn't carry counters across
    # the (logical) restart.
    rate_per_min = int(
        os.environ.get("RESEARCH_RATE_LIMIT_PER_MIN", _DEFAULT_RATE_LIMIT_PER_MIN)
    )
    app_.state.rate_limiter = _RateLimiter(limit_per_min=rate_per_min)
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


def _require_auth(request: Request, authorization: str | None) -> None:
    """Enforce bearer-token auth on the protected routes.

    No-ops when ``app.state.api_token`` is unset (the dev path) — the
    lifespan logs a loud warning in that case so unauthenticated mode
    is loud, not silent.
    """
    expected: str | None = getattr(request.app.state, "api_token", None)
    if not expected:
        return
    if authorization is None or not authorization.startswith(_BEARER_PREFIX):
        raise HTTPException(status_code=401, detail="missing or malformed Authorization header")
    presented = authorization[len(_BEARER_PREFIX):].strip()
    if presented != expected:
        raise HTTPException(status_code=401, detail="invalid bearer token")


def _client_ip(request: Request) -> str:
    """Best-effort client IP for the rate-limit key.

    The ASGI transport used in tests doesn't always populate
    ``request.client``; fall back to a sentinel so the test path still
    exercises the limiter (one bucket for everyone) rather than
    short-circuiting on ``None``.
    """
    if request.client is not None and request.client.host:
        return request.client.host
    return "unknown"


def _enforce_rate_limit(request: Request) -> None:
    limiter: _RateLimiter | None = getattr(request.app.state, "rate_limiter", None)
    if limiter is None:
        return
    ip = _client_ip(request)
    allowed, retry_after = limiter.check(ip)
    if not allowed:
        # Round up to a whole second so the Retry-After header is a
        # well-formed integer (the RFC permits a delta-seconds *or* an
        # HTTP-date, integer is the easier contract for clients).
        retry_secs = max(1, int(retry_after) + 1)
        _log.warning(
            "api.rate_limited",
            ip=ip,
            limit_per_min=limiter.limit,
            retry_after_s=retry_secs,
        )
        raise HTTPException(
            status_code=429,
            detail=f"rate limit exceeded: {limiter.limit} requests/min",
            headers={"Retry-After": str(retry_secs)},
        )


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
    _require_auth(request, request.headers.get(_AUTH_HEADER))
    _enforce_rate_limit(request)
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


def _terminal_shadow(request: Request) -> _TerminalShadow:
    shadow: _TerminalShadow | None = getattr(request.app.state, "terminal_shadow", None)
    if shadow is None:
        # Lifespan didn't run (some tests poke app.state directly); make
        # the shadow lookup a no-op rather than crash on missing attr.
        shadow = _TerminalShadow()
        request.app.state.terminal_shadow = shadow
    return shadow


@app.get("/runs/{run_id}")
async def get_run(run_id: str, request: Request) -> RunStatus:
    _require_auth(request, request.headers.get(_AUTH_HEADER))
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
    shadow: _TerminalShadow,
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
    shadow: _TerminalShadow,
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

    On every terminal write we also emit `api.run_completed` with the
    end-to-end latency, agent counts, and terminal state so operators
    have a single log line to grep for run-level SLO tracking.
    """
    run_started_at = time.monotonic()
    succeeded_agents = 0
    failed_agents = 0
    agent_count = 0
    try:
        agents = default_agents()
        if payload.agents:
            wanted = set(payload.agents)
            agents = [a for a in agents if a.name in wanted]
        agent_count = len(agents)
        engine = WorkflowEngine(
            run_id=run_id,
            record_step=store.append_step,
            cache_get=store.cache_get,
            cache_put=store.cache_put,
        )
        results = await engine.run_parallel(agents, payload.question)
        succeeded_agents = sum(1 for r in results if r.status is not StepStatus.FAILED)
        failed_agents = sum(1 for r in results if r.status is StepStatus.FAILED)
        report: ResearchReport = await StitchSynthesizer().synthesize(
            run_id, payload.question, results
        )
        finished = await store.get_run(run_id)
        if finished is None:
            _emit_run_completed(
                run_id=run_id,
                run_started_at=run_started_at,
                terminal_state=StepStatus.FAILED,
                agent_count=agent_count,
                succeeded_agents=succeeded_agents,
                failed_agents=failed_agents,
                note="run_record_missing_at_finalization",
            )
            return
        finished.state = (
            StepStatus.FAILED
            if all(r.status is StepStatus.FAILED for r in results)
            else StepStatus.SUCCEEDED
        )
        finished.finished_at = datetime.now(UTC)
        finished.report = report
        finished.total_latency_ms = (time.monotonic() - run_started_at) * 1000.0
        await _persist_terminal(store, shadow, finished, agent_label="workflow_done")
        _emit_run_completed(
            run_id=run_id,
            run_started_at=run_started_at,
            terminal_state=finished.state,
            agent_count=agent_count,
            succeeded_agents=succeeded_agents,
            failed_agents=failed_agents,
        )
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
        failed.total_latency_ms = (time.monotonic() - run_started_at) * 1000.0
        await _persist_terminal(store, shadow, failed, agent_label="workflow_failed")
        _emit_run_completed(
            run_id=run_id,
            run_started_at=run_started_at,
            terminal_state=StepStatus.FAILED,
            agent_count=agent_count,
            succeeded_agents=succeeded_agents,
            failed_agents=max(failed_agents, agent_count - succeeded_agents),
            note="exception_path",
        )


def _emit_run_completed(
    *,
    run_id: str,
    run_started_at: float,
    terminal_state: StepStatus,
    agent_count: int,
    succeeded_agents: int,
    failed_agents: int,
    note: str | None = None,
) -> None:
    """Single source of truth for the `api.run_completed` event.

    Centralised so the happy path, the missing-record edge, and the
    exception path all emit the same fields with the same names — a
    single grep covers run-level SLO observability.
    """
    payload: dict[str, object] = {
        "run_id": run_id,
        "total_latency_ms": (time.monotonic() - run_started_at) * 1000.0,
        "agent_count": agent_count,
        "succeeded_agents": succeeded_agents,
        "failed_agents": failed_agents,
        "terminal_state": terminal_state.value,
    }
    if note:
        payload["note"] = note
    _log.info("api.run_completed", **payload)


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
