"""Extra API contract tests: bad payloads, store unavailable, /health shape."""

from __future__ import annotations

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
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://t") as c:
        yield c
    await fake.aclose()


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


class TestAgentSubsetting:
    async def test_can_request_a_specific_agent_subset(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/research",
            json={"question": "what is python", "agents": ["web_search"]},
        )
        assert resp.status_code == 202
        assert "run_id" in resp.json()
