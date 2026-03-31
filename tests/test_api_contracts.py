"""Extra API contract tests: bad payloads, store unavailable, /health shape."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import fakeredis.aioredis as fake_aioredis
import pytest
from httpx import ASGITransport, AsyncClient

from research_crew.api import app
from research_crew.errors import StoreUnavailableError
from research_crew.models import AgentResult, RunStatus, StepRecord
from research_crew.store import RedisRunStore, RunStore


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    fake = fake_aioredis.FakeRedis(decode_responses=True)
    app.state.redis = fake
    app.state.store = RedisRunStore(fake)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://t") as c:
        yield c
    await fake.aclose()


class _ToggleFailStore:
    """Wraps a real store and raises StoreUnavailableError once toggled.

    Lets a test simulate "store goes down mid-request" without nilling
    out app.state, which only worked because of an implementation
    detail (the lifespan re-initialised on demand).
    """

    def __init__(self, inner: RunStore) -> None:
        self._inner = inner
        self.fail = False

    def _maybe_fail(self) -> None:
        if self.fail:
            raise StoreUnavailableError("simulated outage")

    async def get_run(self, run_id: str) -> RunStatus | None:
        self._maybe_fail()
        return await self._inner.get_run(run_id)

    async def put_run(self, run: RunStatus) -> None:
        self._maybe_fail()
        await self._inner.put_run(run)

    async def append_step(self, step: StepRecord) -> None:
        self._maybe_fail()
        await self._inner.append_step(step)

    async def list_steps(self, run_id: str) -> list[StepRecord]:
        self._maybe_fail()
        return await self._inner.list_steps(run_id)

    async def cache_get(self, dedup_key: str) -> AgentResult | None:
        self._maybe_fail()
        return await self._inner.cache_get(dedup_key)

    async def cache_put(self, dedup_key: str, result: AgentResult) -> None:
        self._maybe_fail()
        await self._inner.cache_put(dedup_key, result)


class TestHealthShape:
    async def test_health_body_matches_contract(self, client: AsyncClient) -> None:
        resp = await client.get("/health")
        body = resp.json()
        assert resp.status_code == 200
        assert set(body.keys()) == {"status", "redis"}
        assert body["status"] == "ok"
        assert body["redis"] in {"up", "down"}


class TestBadPayloads:
    async def test_question_too_long_422s(self, client: AsyncClient) -> None:
        resp = await client.post("/research", json={"question": "x" * 1024})
        assert resp.status_code == 422

    async def test_question_missing_422s(self, client: AsyncClient) -> None:
        resp = await client.post("/research", json={})
        assert resp.status_code == 422

    async def test_question_wrong_type_422s(self, client: AsyncClient) -> None:
        resp = await client.post("/research", json={"question": 42})
        assert resp.status_code == 422

    async def test_unknown_agent_in_list_422s(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/research",
            json={"question": "what is python", "agents": ["NOT_AN_AGENT"]},
        )
        assert resp.status_code == 422


class TestStoreUnavailable:
    async def test_health_returns_503_when_redis_missing(self) -> None:
        # Brand-new app instance with no .state at all.
        app.state.redis = None
        app.state.store = None
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://t") as c:
            resp = await c.get("/health")
        assert resp.status_code == 503
        assert "redis" in resp.json()["detail"].lower()

    async def test_post_research_returns_503_when_store_missing(self) -> None:
        app.state.redis = None
        app.state.store = None
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://t") as c:
            resp = await c.post("/research", json={"question": "what is python"})
        assert resp.status_code == 503

    async def test_post_research_returns_503_when_store_fails_mid_request(self) -> None:
        """Store is wired up, then toggles to failure: POST must return 503,
        not a raw 500. Exercises the typed exception handler."""
        fake = fake_aioredis.FakeRedis(decode_responses=True)
        toggle = _ToggleFailStore(RedisRunStore(fake))
        app.state.redis = fake
        app.state.store = toggle
        transport = ASGITransport(app=app)
        try:
            async with AsyncClient(transport=transport, base_url="http://t") as c:
                toggle.fail = True
                resp = await c.post("/research", json={"question": "what is python"})
            assert resp.status_code == 503
            assert "store" in resp.json()["detail"].lower()
        finally:
            await fake.aclose()

    async def test_get_run_returns_503_when_store_fails_mid_request(self) -> None:
        """GET /runs/{id} must surface 503 when the backing store is down."""
        fake = fake_aioredis.FakeRedis(decode_responses=True)
        toggle = _ToggleFailStore(RedisRunStore(fake))
        app.state.redis = fake
        app.state.store = toggle
        transport = ASGITransport(app=app)
        try:
            async with AsyncClient(transport=transport, base_url="http://t") as c:
                toggle.fail = True
                resp = await c.get("/runs/anything")
            assert resp.status_code == 503
        finally:
            await fake.aclose()


class TestAgentSubsetting:
    async def test_only_requested_agents_appear_in_run_steps(self, client: AsyncClient) -> None:
        """Subsetting must actually subset: only the requested agents
        should leave per-step audit records for the run."""
        wanted = ["web_search", "scholar"]
        resp = await client.post(
            "/research",
            json={"question": "what is python", "agents": wanted},
        )
        assert resp.status_code == 202
        run_id = resp.json()["run_id"]

        # Poll until the bg task reaches a terminal state.
        body: dict[str, object] = {}
        for _ in range(40):
            r = await client.get(f"/runs/{run_id}")
            body = r.json()
            if body.get("state") in ("succeeded", "failed"):
                break
            await asyncio.sleep(0.05)
        assert body["state"] == "succeeded"

        steps = body["steps"]
        assert isinstance(steps, list)
        agents_seen = {s["agent"] for s in steps}
        assert agents_seen == set(wanted)
