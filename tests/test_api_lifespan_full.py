"""Full-lifespan integration tests for auth, rate-limit, and reconcile-on-startup.

These tests exercise the FastAPI lifespan explicitly: a small async
``_run_lifespan`` context manager sends the ASGI ``lifespan.startup`` /
``lifespan.shutdown`` events so ``_lifespan`` in api.py runs fully.
HTTP requests then go through ``httpx.ASGITransport(app=app)`` against
the already-booted app.state, which is the closest approximation of
``ASGITransport(app=app, lifespan="on")`` available in httpx 0.28.

The lifespan wires up:
* Redis client (via patched ``aioredis.from_url`` → fakeredis)
* Terminal-state shadow
* Bearer token resolved from env
* Rate-limiter
* Orphan-run reconciliation
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta

import fakeredis.aioredis as fake_aioredis
from httpx import ASGITransport, AsyncClient

from research_crew import api as api_module
from research_crew.api import app
from research_crew.models import RunStatus, StepStatus
from research_crew.store import RedisRunStore


@asynccontextmanager
async def _run_lifespan(
    fake: fake_aioredis.FakeRedis,
    *,
    token: str | None = None,
    rate_limit: int = 1000,
) -> AsyncIterator[None]:
    """Explicitly drive the ASGI lifespan protocol on ``app``.

    Patches ``aioredis.from_url`` so the lifespan's connection-open step
    lands on ``fake``.  Clears stale ``app.state.*`` from prior tests.
    Yields after the startup event completes; sends shutdown before exit.
    """
    original_from_url = api_module.aioredis.from_url

    def _fake_from_url(url: str, **kwargs: object) -> fake_aioredis.FakeRedis:
        return fake

    api_module.aioredis.from_url = _fake_from_url  # type: ignore[assignment]

    # Clear stale state so the lifespan re-initialises from scratch.
    for attr in ("redis", "store", "terminal_shadow", "rate_limiter", "api_token"):
        if hasattr(app.state, attr):
            setattr(app.state, attr, None)

    if token:
        os.environ["RESEARCH_API_TOKEN"] = token
    else:
        os.environ.pop("RESEARCH_API_TOKEN", None)
    os.environ["RESEARCH_RATE_LIMIT_PER_MIN"] = str(rate_limit)
    app.openapi_schema = None  # type: ignore[assignment]

    # Build a minimal ASGI lifespan runner.
    receive_queue: asyncio.Queue[dict[str, str]] = asyncio.Queue()
    send_queue: asyncio.Queue[dict[str, str]] = asyncio.Queue()
    await receive_queue.put({"type": "lifespan.startup"})

    async def receive() -> dict[str, str]:
        return await receive_queue.get()

    async def send(msg: dict[str, str]) -> None:
        await send_queue.put(msg)

    lifespan_task = asyncio.create_task(
        app({"type": "lifespan", "asgi": {"version": "3.0"}}, receive, send)
    )
    # Wait for startup.complete
    startup_msg = await send_queue.get()
    assert startup_msg["type"] == "lifespan.startup.complete", startup_msg
    try:
        yield
    finally:
        await receive_queue.put({"type": "lifespan.shutdown"})
        shutdown_msg = await send_queue.get()
        assert shutdown_msg["type"] == "lifespan.shutdown.complete", shutdown_msg
        await lifespan_task
        api_module.aioredis.from_url = original_from_url  # type: ignore[assignment]
        os.environ.pop("RESEARCH_API_TOKEN", None)
        os.environ.pop("RESEARCH_RATE_LIMIT_PER_MIN", None)
        app.openapi_schema = None  # type: ignore[assignment]


class TestLifespanAuth:
    """Auth enforcement exercised through the full lifespan path."""

    async def test_protected_route_requires_token_via_lifespan(self) -> None:
        """When RESEARCH_API_TOKEN is set via the environment (so the
        lifespan resolves it into app.state.api_token), a request without
        a bearer token must be rejected with 401.
        """
        _token = "lifespan-test-token"  # noqa: S105
        fake = fake_aioredis.FakeRedis(decode_responses=True)
        try:
            async with _run_lifespan(fake, token=_token):
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://t") as client:
                    resp = await client.post("/research", json={"question": "what is python"})
                    assert resp.status_code == 401

                    ok = await client.post(
                        "/research",
                        json={"question": "what is python"},
                        headers={"Authorization": "Bearer lifespan-test-token"},
                    )
                    assert ok.status_code == 202
        finally:
            await fake.aclose()

    async def test_health_remains_open_with_token_set(self) -> None:
        _token = "lifespan-test-token"  # noqa: S105
        fake = fake_aioredis.FakeRedis(decode_responses=True)
        try:
            async with _run_lifespan(fake, token=_token):
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://t") as client:
                    resp = await client.get("/health")
                    assert resp.status_code == 200
                    assert resp.json()["status"] == "ok"
        finally:
            await fake.aclose()


class TestLifespanRateLimit:
    """Rate-limit enforcement exercised through the full lifespan path."""

    async def test_rate_limit_fires_after_n_requests(self) -> None:
        """With the limit set to 2 the third POST to /research must 429."""
        fake = fake_aioredis.FakeRedis(decode_responses=True)
        try:
            async with _run_lifespan(fake, rate_limit=2):
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://t") as client:
                    r1 = await client.post("/research", json={"question": "what is python"})
                    assert r1.status_code == 202
                    r2 = await client.post("/research", json={"question": "what is python"})
                    assert r2.status_code == 202
                    r3 = await client.post("/research", json={"question": "what is python"})
                    assert r3.status_code == 429
                    assert r3.headers.get("Retry-After") is not None
        finally:
            await fake.aclose()


class TestLifespanReconcileOnStartup:
    """Orphan reconciliation fires during lifespan startup."""

    async def test_orphan_run_is_reconciled_at_startup(self) -> None:
        """A RUNNING run without a heartbeat that was already in Redis
        before the lifespan starts must be flipped to FAILED by the
        startup reconciliation step.
        """
        fake = fake_aioredis.FakeRedis(decode_responses=True)
        store = RedisRunStore(fake)
        orphan = RunStatus(
            run_id="startup-orphan-1",
            question="what is python",
            state=StepStatus.RUNNING,
        )
        await store.put_run(orphan)

        try:
            async with _run_lifespan(fake):
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://t") as client:
                    resp = await client.get("/runs/startup-orphan-1")
                    assert resp.status_code == 200
                    body = resp.json()
                    assert body["state"] == "failed"
                    assert body["error"] is not None
        finally:
            await fake.aclose()

    async def test_live_peer_run_is_not_reconciled_at_startup(self) -> None:
        """A RUNNING run with a fresh heartbeat must survive startup
        reconciliation unchanged.
        """
        fake = fake_aioredis.FakeRedis(decode_responses=True)
        store = RedisRunStore(fake)
        live = RunStatus(
            run_id="startup-live-1",
            question="what is python",
            state=StepStatus.RUNNING,
            owner_id="peer-worker",
            heartbeat_at=datetime.now(UTC),
        )
        await store.put_run(live)

        try:
            async with _run_lifespan(fake):
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://t") as client:
                    resp = await client.get("/runs/startup-live-1")
                    assert resp.status_code == 200
                    body = resp.json()
                    assert body["state"] == "running"
        finally:
            await fake.aclose()

    async def test_stale_heartbeat_run_is_reconciled_at_startup(self) -> None:
        """A RUNNING run with a heartbeat older than the stale threshold
        must be flipped to FAILED by the startup reconciliation step.
        The reconciled record carries ``abandoned_at`` to distinguish it
        from a run that failed during normal execution.
        """
        fake = fake_aioredis.FakeRedis(decode_responses=True)
        store = RedisRunStore(fake)
        stale_ts = datetime.now(UTC) - timedelta(minutes=10)
        stale = RunStatus(
            run_id="startup-stale-1",
            question="what is python",
            state=StepStatus.RUNNING,
            owner_id="dead-worker",
            heartbeat_at=stale_ts,
        )
        await store.put_run(stale)

        try:
            async with _run_lifespan(fake):
                transport = ASGITransport(app=app)
                async with AsyncClient(transport=transport, base_url="http://t") as client:
                    resp = await client.get("/runs/startup-stale-1")
                    assert resp.status_code == 200
                    body = resp.json()
                    assert body["state"] == "failed"
                    assert body["abandoned_at"] is not None
        finally:
            await fake.aclose()
