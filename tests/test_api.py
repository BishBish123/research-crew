"""End-to-end API tests using fakeredis (no real Redis)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import fakeredis.aioredis as fake_aioredis
import pytest
from httpx import ASGITransport, AsyncClient

from research_crew.api import app
from research_crew.store import RedisRunStore


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    fake = fake_aioredis.FakeRedis(decode_responses=True)
    app.state.redis = fake
    app.state.store = RedisRunStore(fake)
    # Reset shadow between tests; otherwise a prior test that primed it
    # could mask a real regression in shadow-population code paths here.
    if hasattr(app.state, "terminal_shadow") and hasattr(app.state.terminal_shadow, "clear"):
        app.state.terminal_shadow.clear()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://t") as c:
        yield c
    await fake.aclose()


class TestHealth:
    async def test_health_reports_redis_up(self, client: AsyncClient) -> None:
        resp = await client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["redis"] == "up"


class TestResearchEndpoint:
    async def test_submit_returns_run_id_and_status_url(self, client: AsyncClient) -> None:
        resp = await client.post("/research", json={"question": "what is python"})
        assert resp.status_code == 202
        body = resp.json()
        assert "run_id" in body
        # status_url is now absolute (built via request.url_for) so callers
        # behind proxies / on a non-default host can poll without
        # re-deriving the base URL. Should include scheme + host + path.
        status_url = body["status_url"]
        assert status_url.startswith("http://") or status_url.startswith("https://")
        assert status_url.endswith(f"/runs/{body['run_id']}")

    async def test_run_completes_and_renders_report(self, client: AsyncClient) -> None:
        resp = await client.post("/research", json={"question": "what is python"})
        run_id = resp.json()["run_id"]

        # Poll up to a couple seconds for the bg task to finish.
        for _ in range(20):
            r = await client.get(f"/runs/{run_id}")
            body = r.json()
            if body.get("state") in ("succeeded", "failed"):
                break
            await asyncio.sleep(0.1)
        assert body["state"] == "succeeded"
        assert body["report"] is not None
        # Every default agent should appear in the per-step audit.
        agents_seen = {s["agent"] for s in body["steps"]}
        assert "web_search" in agents_seen
        assert "scholar" in agents_seen
        assert "wikipedia" in agents_seen

    async def test_terminal_run_reports_total_latency(self, client: AsyncClient) -> None:
        """`total_latency_ms` is None until the bg task writes the
        terminal state, then a positive monotonic-clock measurement.
        """
        resp = await client.post("/research", json={"question": "what is python"})
        run_id = resp.json()["run_id"]

        body: dict[str, object] = {}
        for _ in range(20):
            r = await client.get(f"/runs/{run_id}")
            body = r.json()
            if body.get("state") in ("succeeded", "failed"):
                break
            await asyncio.sleep(0.1)
        assert body["state"] == "succeeded"
        latency = body.get("total_latency_ms")
        assert isinstance(latency, (int, float))
        assert latency > 0.0

    async def test_get_unknown_run_404s(self, client: AsyncClient) -> None:
        resp = await client.get("/runs/does-not-exist")
        assert resp.status_code == 404

    async def test_blank_question_rejected_422(self, client: AsyncClient) -> None:
        # Whitespace-only input strips to empty and must be rejected.
        resp = await client.post("/research", json={"question": "   "})
        assert resp.status_code == 422

    async def test_422_detail_shape_pinned(self, client: AsyncClient) -> None:
        """Pin the FastAPI 422 envelope so the OpenAPI contract is
        protected against silent shape drift.

        For a `question` whose type is wrong (int instead of str)
        FastAPI emits a `detail[]` array with `loc`, `msg`, `type`
        per error. Generated clients consume this shape — we want a
        red test if it changes.
        """
        resp = await client.post("/research", json={"question": 42})
        assert resp.status_code == 422
        body = resp.json()
        assert "detail" in body and isinstance(body["detail"], list) and body["detail"]
        first = body["detail"][0]
        assert isinstance(first.get("loc"), list)
        # The leading element of `loc` is "body" for body-payload errors.
        assert first["loc"][0] == "body"
        # And the offending field is `question`.
        assert "question" in first["loc"]
        assert isinstance(first.get("msg"), str) and first["msg"]
        # `type` is a stable Pydantic identifier (e.g. "string_type").
        assert isinstance(first.get("type"), str) and first["type"]
        assert "string" in first["type"]
