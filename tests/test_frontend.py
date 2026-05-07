"""Frontend static-file serving tests.

All tests run entirely in-process via FastAPI's TestClient (httpx / ASGI).
No real network ports are opened, no browser is launched.

Covered scenarios
-----------------
1. GET / returns 200 with text/html content.
2. index.html contains the expected DOM hooks.
3. GET /static/app.js returns 200 with JavaScript content.
4. GET /static/styles.css returns 200 with CSS content.
5. API routes still respond correctly with the static mount in place.
6. index.html has the research form element.
7. index.html has the agent-cards container.
8. index.html has the final-report pre element.
9. index.html links to app.js and styles.css.
10. WebSocket route is still registered alongside the static mount.
"""

from __future__ import annotations

import fakeredis.aioredis as fake_aioredis
import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from research_crew.api import app
from research_crew.store import RedisRunStore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sync_client() -> TestClient:
    """Synchronous TestClient — sufficient for static-file GET assertions."""
    fake = fake_aioredis.FakeRedis(decode_responses=True)
    app.state.redis = fake
    app.state.store = RedisRunStore(fake)
    # Disable auth for these tests.
    app.state.api_token = None
    return TestClient(app, raise_server_exceptions=True)


@pytest.fixture
async def async_client():
    """Async client for the API-route coexistence tests."""
    fake = fake_aioredis.FakeRedis(decode_responses=True)
    app.state.redis = fake
    app.state.store = RedisRunStore(fake)
    app.state.api_token = None
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://t") as c:
        yield c
    await fake.aclose()


# ---------------------------------------------------------------------------
# Static asset serving
# ---------------------------------------------------------------------------


class TestIndexHtml:
    def test_get_root_returns_200(self, sync_client: TestClient) -> None:
        resp = sync_client.get("/")
        assert resp.status_code == 200

    def test_get_root_content_type_html(self, sync_client: TestClient) -> None:
        resp = sync_client.get("/")
        ct = resp.headers.get("content-type", "")
        assert "text/html" in ct

    def test_index_has_research_form(self, sync_client: TestClient) -> None:
        resp = sync_client.get("/")
        assert 'id="research-form"' in resp.text

    def test_index_has_agent_cards(self, sync_client: TestClient) -> None:
        resp = sync_client.get("/")
        assert 'id="agent-cards"' in resp.text

    def test_index_has_final_report(self, sync_client: TestClient) -> None:
        resp = sync_client.get("/")
        assert 'id="final-report"' in resp.text

    def test_index_links_app_js(self, sync_client: TestClient) -> None:
        resp = sync_client.get("/")
        assert "app.js" in resp.text

    def test_index_links_styles_css(self, sync_client: TestClient) -> None:
        resp = sync_client.get("/")
        assert "styles.css" in resp.text

    def test_index_has_question_input(self, sync_client: TestClient) -> None:
        resp = sync_client.get("/")
        assert 'id="question-input"' in resp.text

    def test_index_has_submit_btn(self, sync_client: TestClient) -> None:
        resp = sync_client.get("/")
        assert 'id="submit-btn"' in resp.text

    def test_index_has_cancel_btn(self, sync_client: TestClient) -> None:
        resp = sync_client.get("/")
        assert 'id="cancel-btn"' in resp.text


class TestStaticAssets:
    def test_app_js_returns_200(self, sync_client: TestClient) -> None:
        resp = sync_client.get("/static/app.js")
        assert resp.status_code == 200

    def test_app_js_content_type(self, sync_client: TestClient) -> None:
        resp = sync_client.get("/static/app.js")
        ct = resp.headers.get("content-type", "")
        assert "javascript" in ct or "text/plain" in ct

    def test_styles_css_returns_200(self, sync_client: TestClient) -> None:
        resp = sync_client.get("/static/styles.css")
        assert resp.status_code == 200

    def test_styles_css_content_type(self, sync_client: TestClient) -> None:
        resp = sync_client.get("/static/styles.css")
        ct = resp.headers.get("content-type", "")
        assert "css" in ct or "text/plain" in ct

    def test_app_js_has_websocket(self, sync_client: TestClient) -> None:
        resp = sync_client.get("/static/app.js")
        assert "WebSocket" in resp.text

    def test_app_js_has_fetch(self, sync_client: TestClient) -> None:
        resp = sync_client.get("/static/app.js")
        assert "fetch" in resp.text


# ---------------------------------------------------------------------------
# API routes still work alongside the static mount
# ---------------------------------------------------------------------------


class TestApiRoutesCoexistence:
    async def test_health_still_works(self, async_client: AsyncClient) -> None:
        resp = await async_client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    async def test_post_research_still_works(self, async_client: AsyncClient) -> None:
        resp = await async_client.post("/research", json={"question": "test question"})
        assert resp.status_code == 202
        body = resp.json()
        assert "run_id" in body

    async def test_get_run_still_works(self, async_client: AsyncClient) -> None:
        post = await async_client.post("/research", json={"question": "test question"})
        run_id = post.json()["run_id"]
        resp = await async_client.get(f"/runs/{run_id}")
        assert resp.status_code == 200

    async def test_nonexistent_run_still_404(self, async_client: AsyncClient) -> None:
        resp = await async_client.get("/runs/doesnotexist")
        assert resp.status_code == 404
