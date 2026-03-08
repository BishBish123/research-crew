"""Extra API contract tests: bad payloads, store unavailable, /health shape."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import fakeredis.aioredis as fake_aioredis
import pytest
from httpx import ASGITransport, AsyncClient

from research_crew.api import app
from research_crew.errors import StoreUnavailableError
from research_crew.models import AgentName, AgentResult, RunStatus, StepRecord, StepStatus
from research_crew.store import RedisRunStore, RunStore


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    fake = fake_aioredis.FakeRedis(decode_responses=True)
    app.state.redis = fake
    app.state.store = RedisRunStore(fake)
    # Reset the terminal-state shadow per-test so cross-module state
    # never leaks between fixtures (prior write from another test could
    # otherwise satisfy a `not in shadow` assertion vacuously).
    if hasattr(app.state, "terminal_shadow") and hasattr(app.state.terminal_shadow, "clear"):
        app.state.terminal_shadow.clear()
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


class TestHealthCounters:
    async def test_active_runs_reflects_running_records(self, client: AsyncClient) -> None:
        """Plant N RUNNING records and assert /health reports them."""
        store = app.state.store
        for i in range(3):
            await store.put_run(
                RunStatus(
                    run_id=f"running-{i}",
                    question="what is python",
                    state=StepStatus.RUNNING,
                )
            )
        # One terminal record that must NOT be counted.
        await store.put_run(
            RunStatus(
                run_id="done-1",
                question="what is python",
                state=StepStatus.SUCCEEDED,
            )
        )
        resp = await client.get("/health")
        body = resp.json()
        assert body["active_runs"] == 3, body

    async def test_active_runs_counts_default_jsondumps_spacing(self, client: AsyncClient) -> None:
        """Regression: a RUNNING blob written with the stdlib
        ``json.dumps`` default spacing serialises ``"state": "running"``
        (note the space after the colon). The previous literal-substring
        pre-filter looked for ``"state":"running"`` and silently skipped
        these records, undercounting active runs.

        Plant such a blob directly via the underlying Redis client so
        the regression is exercised exactly the way an external writer
        — for example a peer instance using a different Pydantic
        version, or a manual recovery script — would produce it.
        """
        store = app.state.store
        redis_client = app.state.redis
        # Compact-spaced blob via Pydantic's model_dump_json — already
        # covered by the existing test, but include one so the count
        # below is unambiguous.
        await store.put_run(
            RunStatus(
                run_id="running-compact",
                question="compact",
                state=StepStatus.RUNNING,
            )
        )
        # Default-spaced blob via json.dumps — this is the regression.
        run = RunStatus(
            run_id="running-spaced",
            question="spaced",
            state=StepStatus.RUNNING,
        )
        spaced_payload = run.model_dump(mode="json")
        spaced_blob = json.dumps(spaced_payload)
        assert '"state": "running"' in spaced_blob, (
            "json.dumps default spacing must produce a space after the colon "
            "for this regression test to be meaningful"
        )
        await redis_client.set(f"{store.prefix}:run:running-spaced", spaced_blob)

        resp = await client.get("/health")
        body = resp.json()
        assert body["active_runs"] == 2, body

    async def test_shadow_size_reports_zero_when_empty(self, client: AsyncClient) -> None:
        if hasattr(app.state, "terminal_shadow") and hasattr(app.state.terminal_shadow, "clear"):
            app.state.terminal_shadow.clear()
        resp = await client.get("/health")
        assert resp.json()["shadow_size"] == 0


class TestHealthShape:
    async def test_health_body_matches_contract(self, client: AsyncClient) -> None:
        resp = await client.get("/health")
        body = resp.json()
        assert resp.status_code == 200
        assert set(body.keys()) == {"status", "redis", "active_runs", "shadow_size"}
        assert body["status"] == "ok"
        assert body["redis"] in {"up", "down"}
        # active_runs is None if SCAN can't run (e.g. memory store) or
        # an int otherwise; under fakeredis it's always an int.
        assert body["active_runs"] is None or isinstance(body["active_runs"], int)
        assert isinstance(body["shadow_size"], int)
        assert body["shadow_size"] >= 0


class TestBadPayloads:
    async def test_question_too_long_422s(self, client: AsyncClient) -> None:
        # Default max is 5000; one character past it must trip 422.
        resp = await client.post("/research", json={"question": "x" * 5001})
        assert resp.status_code == 422

    async def test_extra_field_rejected_422(self, client: AsyncClient) -> None:
        # `extra="forbid"` on the model must reject unknown keys.
        resp = await client.post(
            "/research",
            json={"question": "what is python", "totally_unknown_field": "boom"},
        )
        assert resp.status_code == 422

    async def test_blank_question_rejected_422(self, client: AsyncClient) -> None:
        resp = await client.post("/research", json={"question": "    "})
        assert resp.status_code == 422

    async def test_too_many_agents_rejected_422(self, client: AsyncClient) -> None:
        # The cap is 20; the agent enum only has 5 valid values, but a
        # request that *repeats* valid names well past the cap must
        # still 422 on the length check before duplicates land in the
        # workflow.
        too_many = [AgentName.WEB_SEARCH.value] * 21
        resp = await client.post(
            "/research",
            json={"question": "what is python", "agents": too_many},
        )
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

    async def test_empty_agents_list_422s_with_explicit_message(self, client: AsyncClient) -> None:
        """`agents=[]` is a client bug (used to silently fan-out to
        defaults). Reject it loudly with a helpful message."""
        resp = await client.post(
            "/research",
            json={"question": "what is python", "agents": []},
        )
        assert resp.status_code == 422
        body = resp.json()
        # FastAPI nests Pydantic validator messages under "detail".
        rendered = str(body).lower()
        assert "non-empty" in rendered
        assert "default fan-out" in rendered


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
