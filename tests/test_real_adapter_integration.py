"""Integration tests: real-mode adapters against local mock servers.

Each test:
- Starts the mock FastAPI app on a random OS-assigned port via uvicorn in a
  background thread.
- Points the adapter at localhost using the ``base_url`` constructor parameter
  together with a fake API key so the adapter enters real-mode (not mock-mode).
- Calls the adapter's ``search`` method (or Langfuse's ingestion path).
- Asserts the adapter returned the expected ``AgentResult`` parsed from the
  fixture, and that the adapter sent the correct auth header.

What this proves vs. unit-mocked tests
---------------------------------------
Unit tests (test_real_agents.py) patch ``httpx.AsyncClient`` at the class
level — no real socket I/O occurs. These integration tests exercise:

  * Real ``httpx.AsyncClient`` with real TCP sockets.
  * Real JSON serialisation over the wire.
  * Real HTTP response parsing (status codes, headers, body).
  * Real retry logic: the ``?force=500`` endpoint returns 500 on the first
    call then 200 on the next, so the adapter's one-retry loop is exercised
    against a real server that actually returns a 5xx response code.

Marked ``integration`` so they are excluded from ``make test`` and run only
via ``make test-integration-mock``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import pathlib
import socket
import threading
import time
from collections.abc import Generator
from typing import Any

import httpx
import pytest
import uvicorn

from research_crew.agents.real import BraveAgent, ExaAgent, TavilyAgent
from research_crew.models import AgentName, StepStatus
from tests.mock_servers import create_mock_app

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Fixture loading helpers
# ---------------------------------------------------------------------------

_FIXTURE_DIR = pathlib.Path(__file__).parent / "fixtures" / "api_responses"


def _load_fixture(filename: str) -> dict[str, Any]:
    return json.loads((_FIXTURE_DIR / filename).read_text())  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Mock server lifecycle context manager
# ---------------------------------------------------------------------------


def _free_port() -> int:
    """Ask the OS for a free TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]  # type: ignore[no-any-return]


@contextlib.contextmanager
def mock_server_ctx() -> Generator[tuple[str, Any], None, None]:
    """Start the mock FastAPI app on a random port, yield (base_url, app).

    The server runs in a daemon thread and is shut down cleanly when the
    context manager exits.
    """
    port = _free_port()
    app = create_mock_app()
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="critical",  # silence uvicorn noise in test output
    )
    server = uvicorn.Server(config)

    # Uvicorn's startup is async; run it in a daemon thread.
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait until the server is ready (polls up to 5 s).
    deadline = 5.0
    interval = 0.05
    elapsed = 0.0
    while not server.started and elapsed < deadline:
        time.sleep(interval)
        elapsed += interval

    if not server.started:
        raise RuntimeError(f"Mock server did not start within {deadline}s")

    base_url = f"http://127.0.0.1:{port}"
    try:
        yield base_url, app
    finally:
        server.should_exit = True
        thread.join(timeout=5.0)


# ---------------------------------------------------------------------------
# TavilyAgent integration tests
# ---------------------------------------------------------------------------


class TestTavilyAdapterIntegration:
    async def test_search_returns_expected_citations(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """TavilyAgent hits real local socket, parses fixture, returns AgentResult."""
        monkeypatch.setenv("TAVILY_API_KEY", "fake-tavily-key")
        fixture = _load_fixture("tavily_search_response.json")

        with mock_server_ctx() as (base_url, _app):
            agent = TavilyAgent(name=AgentName.WEB_SEARCH, base_url=base_url + "/tavily")
            assert agent._mock is False

            result = await agent.search("machine learning transformers")

        assert result.status is StepStatus.SUCCEEDED
        assert result.agent is AgentName.WEB_SEARCH
        assert len(result.citations) == len(fixture["results"])

        # First citation matches fixture[results][0].
        c0 = result.citations[0]
        assert c0.url == fixture["results"][0]["url"]
        assert c0.title == fixture["results"][0]["title"]
        assert c0.snippet == fixture["results"][0]["content"]

    async def test_search_sends_auth_header(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """TavilyAgent must send Authorization header; mock returns 401 if absent."""
        monkeypatch.setenv("TAVILY_API_KEY", "fake-tavily-key")

        with mock_server_ctx() as (base_url, _app):
            agent = TavilyAgent(name=AgentName.WEB_SEARCH, base_url=base_url + "/tavily")
            # The mock validates the header is present; any non-empty value passes.
            result = await agent.search("auth header test")

        assert result.status is StepStatus.SUCCEEDED

    async def test_retry_on_5xx_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """TavilyAgent retries once on 5xx and ultimately returns success.

        The mock server returns 500 on the first call to ?force=500 and 200
        on the next, exercising the real retry loop in _post_with_retry.
        The retry delay is patched to 0 so the test runs fast.
        """
        monkeypatch.setenv("TAVILY_API_KEY", "fake-tavily-key")

        with mock_server_ctx() as (base_url, _app):
            agent = TavilyAgent(name=AgentName.WEB_SEARCH, base_url=base_url + "/tavily")
            # Override the endpoint to include ?force=500.
            agent._endpoint = agent._endpoint + "?force=500"

            # Patch asyncio.sleep so the 1 s retry backoff doesn't slow CI.
            async def _no_sleep(_: float) -> None:
                pass

            monkeypatch.setattr(asyncio, "sleep", _no_sleep)

            result = await agent.search("retry test")

        assert result.status is StepStatus.SUCCEEDED


# ---------------------------------------------------------------------------
# BraveAgent integration tests
# ---------------------------------------------------------------------------


class TestBraveAdapterIntegration:
    async def test_web_search_returns_expected_citations(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """BraveAgent (web) hits real local socket, parses fixture."""
        monkeypatch.setenv("BRAVE_API_KEY", "fake-brave-key")
        fixture = _load_fixture("brave_web_response.json")
        expected_results = fixture["web"]["results"]

        with mock_server_ctx() as (base_url, _app):
            agent = BraveAgent(
                name=AgentName.WEB_SEARCH, endpoint="web", base_url=base_url + "/brave"
            )
            assert agent._mock is False
            assert agent._url == base_url + "/brave/web/search"

            result = await agent.search("machine learning transformers")

        assert result.status is StepStatus.SUCCEEDED
        assert result.agent is AgentName.WEB_SEARCH
        assert len(result.citations) == len(expected_results)
        assert result.citations[0].url == expected_results[0]["url"]
        assert result.citations[0].title == expected_results[0]["title"]
        assert result.citations[0].snippet == expected_results[0]["description"]

    async def test_news_search_returns_expected_citations(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """BraveAgent (news) hits real local socket, parses news fixture."""
        monkeypatch.setenv("BRAVE_API_KEY", "fake-brave-key")
        fixture = _load_fixture("brave_news_response.json")
        expected_results = fixture["news"]["results"]

        with mock_server_ctx() as (base_url, _app):
            agent = BraveAgent(name=AgentName.NEWS, endpoint="news", base_url=base_url + "/brave")
            assert agent._url == base_url + "/brave/news/search"

            result = await agent.search("machine learning transformers 2026")

        assert result.status is StepStatus.SUCCEEDED
        assert result.agent is AgentName.NEWS
        assert len(result.citations) == len(expected_results)
        assert result.citations[0].url == expected_results[0]["url"]

    async def test_sends_subscription_token_header(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """BraveAgent must send X-Subscription-Token; mock returns 401 if absent."""
        monkeypatch.setenv("BRAVE_API_KEY", "fake-brave-key")

        with mock_server_ctx() as (base_url, _app):
            agent = BraveAgent(name=AgentName.NEWS, endpoint="web", base_url=base_url + "/brave")
            result = await agent.search("auth header test")

        assert result.status is StepStatus.SUCCEEDED

    async def test_retry_on_5xx_web_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """BraveAgent retries once on 5xx and returns success."""
        monkeypatch.setenv("BRAVE_API_KEY", "fake-brave-key")

        with mock_server_ctx() as (base_url, _app):
            agent = BraveAgent(
                name=AgentName.WEB_SEARCH, endpoint="web", base_url=base_url + "/brave"
            )
            agent._url = agent._url + "?force=500"

            async def _no_sleep(_: float) -> None:
                pass

            monkeypatch.setattr(asyncio, "sleep", _no_sleep)

            result = await agent.search("retry test")

        assert result.status is StepStatus.SUCCEEDED


# ---------------------------------------------------------------------------
# ExaAgent integration tests
# ---------------------------------------------------------------------------


class TestExaAdapterIntegration:
    async def test_search_returns_expected_citations(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ExaAgent hits real local socket, parses exa fixture."""
        monkeypatch.setenv("EXA_API_KEY", "fake-exa-key")
        fixture = _load_fixture("exa_search_response.json")

        with mock_server_ctx() as (base_url, _app):
            agent = ExaAgent(name=AgentName.SCHOLAR, base_url=base_url + "/exa")
            assert agent._mock is False

            result = await agent.search("machine learning transformers")

        assert result.status is StepStatus.SUCCEEDED
        assert result.agent is AgentName.SCHOLAR
        assert len(result.citations) == len(fixture["results"])
        assert result.citations[0].url == fixture["results"][0]["url"]
        assert result.citations[0].title == fixture["results"][0]["title"]
        # snippet is text[:500]
        assert result.citations[0].snippet == fixture["results"][0]["text"][:500]

    async def test_research_paper_category_returns_academic_results(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ExaAgent with category='research paper' receives the academic fixture."""
        monkeypatch.setenv("EXA_API_KEY", "fake-exa-key")
        fixture = _load_fixture("exa_research_paper_response.json")

        with mock_server_ctx() as (base_url, _app):
            agent = ExaAgent(
                name=AgentName.SCHOLAR,
                category="research paper",
                base_url=base_url + "/exa",
            )
            result = await agent.search("neural network architectures")

        assert result.status is StepStatus.SUCCEEDED
        assert len(result.citations) == len(fixture["results"])
        urls = [c.url for c in result.citations]
        assert any("arxiv.org" in u for u in urls)

    async def test_sends_api_key_header(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ExaAgent must send x-api-key header; mock returns 401 if absent."""
        monkeypatch.setenv("EXA_API_KEY", "fake-exa-key")

        with mock_server_ctx() as (base_url, _app):
            agent = ExaAgent(name=AgentName.SCHOLAR, base_url=base_url + "/exa")
            result = await agent.search("auth header test")

        assert result.status is StepStatus.SUCCEEDED

    async def test_retry_on_5xx_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ExaAgent retries once on 5xx and returns success."""
        monkeypatch.setenv("EXA_API_KEY", "fake-exa-key")

        with mock_server_ctx() as (base_url, _app):
            agent = ExaAgent(name=AgentName.SCHOLAR, base_url=base_url + "/exa")
            agent._endpoint = agent._endpoint + "?force=500"

            async def _no_sleep(_: float) -> None:
                pass

            monkeypatch.setattr(asyncio, "sleep", _no_sleep)

            result = await agent.search("retry test")

        assert result.status is StepStatus.SUCCEEDED


# ---------------------------------------------------------------------------
# base_url env-var override tests
# ---------------------------------------------------------------------------


class TestBaseUrlEnvOverride:
    """Verify TAVILY_API_BASE_URL / BRAVE_API_BASE_URL / EXA_API_BASE_URL redirect
    the adapter to the local server without constructor args."""

    async def test_tavily_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TAVILY_API_KEY", "fake-key")
        with mock_server_ctx() as (base_url, _app):
            monkeypatch.setenv("TAVILY_API_BASE_URL", base_url + "/tavily")
            agent = TavilyAgent(name=AgentName.WEB_SEARCH)
            assert agent._endpoint == base_url + "/tavily/search"
            result = await agent.search("env override test")
        assert result.status is StepStatus.SUCCEEDED

    async def test_brave_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BRAVE_API_KEY", "fake-key")
        with mock_server_ctx() as (base_url, _app):
            monkeypatch.setenv("BRAVE_API_BASE_URL", base_url + "/brave")
            agent = BraveAgent(name=AgentName.WEB_SEARCH, endpoint="web")
            assert agent._url == base_url + "/brave/web/search"
            result = await agent.search("env override test")
        assert result.status is StepStatus.SUCCEEDED

    async def test_exa_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EXA_API_KEY", "fake-key")
        with mock_server_ctx() as (base_url, _app):
            monkeypatch.setenv("EXA_API_BASE_URL", base_url + "/exa")
            agent = ExaAgent(name=AgentName.SCHOLAR)
            assert agent._endpoint == base_url + "/exa/search"
            result = await agent.search("env override test")
        assert result.status is StepStatus.SUCCEEDED


# ---------------------------------------------------------------------------
# Mock server smoke test — verify the server itself is reachable
# ---------------------------------------------------------------------------


class TestMockServerSmoke:
    async def test_missing_auth_returns_401(self) -> None:
        """Without auth headers, the mock returns 401 — exercises the server validation."""
        with mock_server_ctx() as (base_url, _app):
            async with httpx.AsyncClient() as client:
                # POST to Tavily without Authorization header.
                resp = await client.post(f"{base_url}/tavily/search", json={"query": "test"})
            assert resp.status_code == 401

    async def test_exa_category_routing(self) -> None:
        """Server returns research_paper fixture when category='research paper' is sent."""
        paper_fixture = _load_fixture("exa_research_paper_response.json")
        plain_fixture = _load_fixture("exa_search_response.json")

        with mock_server_ctx() as (base_url, _app):
            async with httpx.AsyncClient() as client:
                plain = await client.post(
                    f"{base_url}/exa/search",
                    json={"query": "test"},
                    headers={"x-api-key": "fake"},
                )
                paper = await client.post(
                    f"{base_url}/exa/search",
                    json={"query": "test", "category": "research paper"},
                    headers={"x-api-key": "fake"},
                )

        assert plain.status_code == 200
        assert plain.json()["requestId"] == plain_fixture["requestId"]

        assert paper.status_code == 200
        assert paper.json()["requestId"] == paper_fixture["requestId"]

    async def test_force_500_cycles(self) -> None:
        """?force=500 returns 500 on first call, 200 on second — exercises retry fixture."""
        with mock_server_ctx() as (base_url, _app):
            async with httpx.AsyncClient() as client:
                r1 = await client.post(
                    f"{base_url}/tavily/search?force=500",
                    json={"query": "test"},
                    headers={"Authorization": "Bearer fake"},
                )
                r2 = await client.post(
                    f"{base_url}/tavily/search?force=500",
                    json={"query": "test"},
                    headers={"Authorization": "Bearer fake"},
                )

        assert r1.status_code == 500
        assert r2.status_code == 200
