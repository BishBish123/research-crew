"""Tests for the three real search-API adapters: TavilyAgent, BraveAgent, ExaAgent.

Structure per adapter
---------------------
1. Mock-mode default test  — no env key set → ``_mock=True``, result is valid shape.
2. Mock-mode determinism    — same question → same URLs across two instances.
3. Mock-mode uniqueness     — the three adapters' mock outputs differ from each other.
4. Real-mode parse test     — ``httpx.AsyncClient`` patched with a fixture JSON
                              response; assert the adapter maps it to the expected shape.
5. Real-mode auth error     — 401 response → fail-soft to mock result.
6. Real-mode 5xx error      — 503 response → AgentResult with FAILED status.
7. Real-mode request error  — ``httpx.TimeoutException`` → fail-soft to mock result.

No real HTTP calls are made: the real-mode tests patch ``httpx.AsyncClient`` via
``unittest.mock.patch`` / ``respx`` — whichever the test needs.  We use
``unittest.mock.patch`` here since it's already available in the dev deps without
adding a new dependency.

TestRealAPIResponseParsing
--------------------------
Uses JSON fixtures from tests/fixtures/api_responses/ that mirror the documented
response shapes from each provider's API reference (verified 2026-05-06). These
tests catch schema-drift bugs — when a provider silently renames or removes a
field, the assertion failures surface before the change reaches production.

Fixtures and their sources:
  tavily_search_response.json    — https://docs.tavily.com/documentation/api-reference/endpoint/search
  brave_web_response.json        — https://api-dashboard.search.brave.com/app/documentation/web-search/responses
  brave_news_response.json       — https://api-dashboard.search.brave.com/app/documentation/news-search/get-started
  exa_search_response.json       — https://exa.ai/docs/reference/search
  exa_research_paper_response.json — https://exa.ai/docs/reference/search (category="research paper")
"""

from __future__ import annotations

import json
import pathlib
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from research_crew.agents import default_agents
from research_crew.agents.base import MockAgent
from research_crew.agents.real import BraveAgent, ExaAgent, TavilyAgent
from research_crew.models import AgentName, StepStatus

# ---------------------------------------------------------------------------
# Fixture loading helpers
# ---------------------------------------------------------------------------

_FIXTURE_DIR = pathlib.Path(__file__).parent / "fixtures" / "api_responses"


def _load_fixture(filename: str) -> dict[str, Any]:
    """Load a JSON fixture file from tests/fixtures/api_responses/."""
    return json.loads((_FIXTURE_DIR / filename).read_text())  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Shared fixture responses (based on official docs examples)
# ---------------------------------------------------------------------------

TAVILY_FIXTURE: dict[str, Any] = {
    "query": "what is asyncio",
    "results": [
        {
            "title": "Python asyncio documentation",
            "url": "https://docs.python.org/3/library/asyncio.html",
            "content": "asyncio is a library to write concurrent code using async/await.",
            "score": 0.95,
        },
        {
            "title": "Real Python asyncio guide",
            "url": "https://realpython.com/async-io-python/",
            "content": "An introduction to the asyncio module.",
            "score": 0.88,
        },
    ],
}

BRAVE_FIXTURE: dict[str, Any] = {
    "query": {"original": "what is asyncio"},
    "web": {
        "results": [
            {
                "title": "asyncio — Asynchronous I/O",
                "url": "https://docs.python.org/3/library/asyncio.html",
                "description": "asyncio is used as a foundation for multiple Python async frameworks.",
            },
            {
                "title": "Brave search result",
                "url": "https://brave.com/search/example",
                "description": "Another result about asyncio.",
            },
        ]
    },
}

BRAVE_NEWS_FIXTURE: dict[str, Any] = {
    "query": {"original": "python asyncio news"},
    "news": {
        "results": [
            {
                "title": "Python asyncio gets major upgrade",
                "url": "https://news.example.com/python-asyncio-2026",
                "description": "The asyncio library received significant performance improvements.",
                "age": "1 day ago",
            },
            {
                "title": "Async Python news roundup",
                "url": "https://news.example.com/async-roundup",
                "description": "Weekly roundup of async Python developments.",
                "age": "3 days ago",
            },
        ]
    },
}

EXA_FIXTURE: dict[str, Any] = {
    "requestId": "req-abc123",
    "results": [
        {
            "id": "exa-1",
            "title": "asyncio basics",
            "url": "https://docs.python.org/3/library/asyncio.html",
            "text": "asyncio provides infrastructure for writing single-threaded concurrent code.",
            "publishedDate": "2024-01-01",
        },
        {
            "id": "exa-2",
            "title": "Async Python",
            "url": "https://realpython.com/async-io-python/",
            "text": "An overview of the asyncio ecosystem.",
            "publishedDate": "2024-03-15",
        },
    ],
    "searchType": "neural",
}


# ---------------------------------------------------------------------------
# Helpers for patching httpx
# ---------------------------------------------------------------------------


def _mock_http_response(status_code: int, body: dict[str, Any]) -> MagicMock:
    """Build a MagicMock that looks like an httpx.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.is_success = 200 <= status_code < 300
    resp.json.return_value = body
    return resp


def _make_async_client_ctx(response: MagicMock) -> MagicMock:
    """Return a MagicMock context manager whose ``post`` / ``get`` return ``response``."""
    client = MagicMock()
    client.post = AsyncMock(return_value=response)
    client.get = AsyncMock(return_value=response)
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=client)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return ctx


# ===========================================================================
# TavilyAgent
# ===========================================================================


class TestTavilyAgentMockMode:
    def test_no_key_sets_mock_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        agent = TavilyAgent(name=AgentName.WEB_SEARCH)
        assert agent._mock is True

    async def test_mock_mode_returns_valid_shape(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        agent = TavilyAgent(name=AgentName.WEB_SEARCH)
        result = await agent.search("what is asyncio")
        assert result.status is StepStatus.SUCCEEDED
        assert result.agent is AgentName.WEB_SEARCH
        assert result.summary
        assert len(result.citations) >= 1
        for c in result.citations:
            assert c.url.startswith("https://")
            assert c.title

    async def test_mock_mode_is_deterministic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        a1 = TavilyAgent(name=AgentName.WEB_SEARCH)
        a2 = TavilyAgent(name=AgentName.WEB_SEARCH)
        r1 = await a1.search("determinism check")
        r2 = await a2.search("determinism check")
        assert [c.url for c in r1.citations] == [c.url for c in r2.citations]


class TestTavilyAgentRealMode:
    async def test_parses_fixture_response(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-key")
        agent = TavilyAgent(name=AgentName.WEB_SEARCH)
        assert agent._mock is False

        resp_mock = _mock_http_response(200, TAVILY_FIXTURE)
        ctx = _make_async_client_ctx(resp_mock)

        with patch("research_crew.agents.real.httpx.AsyncClient", return_value=ctx):
            result = await agent.search("what is asyncio")

        assert result.status is StepStatus.SUCCEEDED
        assert result.agent is AgentName.WEB_SEARCH
        assert len(result.citations) == 2
        assert result.citations[0].url == "https://docs.python.org/3/library/asyncio.html"
        assert result.citations[0].title == "Python asyncio documentation"
        assert "asyncio" in result.citations[0].snippet.lower()

    async def test_auth_error_falls_back_to_mock(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TAVILY_API_KEY", "bad-key")
        agent = TavilyAgent(name=AgentName.WEB_SEARCH)

        resp_mock = _mock_http_response(401, {"detail": "Unauthorized"})
        ctx = _make_async_client_ctx(resp_mock)

        with patch("research_crew.agents.real.httpx.AsyncClient", return_value=ctx):
            result = await agent.search("auth fail question")

        # Fail-soft: returns a succeeded mock result, not a hard failure.
        assert result.status is StepStatus.SUCCEEDED

    async def test_5xx_returns_failed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-key")
        agent = TavilyAgent(name=AgentName.WEB_SEARCH)

        resp_mock = _mock_http_response(503, {})
        resp_mock.is_success = False
        ctx = _make_async_client_ctx(resp_mock)

        with patch("research_crew.agents.real.httpx.AsyncClient", return_value=ctx):
            result = await agent.search("server error question")

        assert result.status is StepStatus.FAILED
        assert result.error and "503" in result.error

    async def test_timeout_falls_back_to_mock(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-key")
        agent = TavilyAgent(name=AgentName.WEB_SEARCH)

        client_mock = MagicMock()
        client_mock.post = AsyncMock(side_effect=httpx.TimeoutException("timed out"))
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=client_mock)
        ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("research_crew.agents.real.httpx.AsyncClient", return_value=ctx):
            result = await agent.search("timeout question")

        assert result.status is StepStatus.SUCCEEDED  # fail-soft


# ===========================================================================
# BraveAgent
# ===========================================================================


class TestBraveAgentMockMode:
    def test_no_key_sets_mock_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("BRAVE_API_KEY", raising=False)
        agent = BraveAgent(name=AgentName.NEWS)
        assert agent._mock is True

    async def test_mock_mode_returns_valid_shape(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("BRAVE_API_KEY", raising=False)
        agent = BraveAgent(name=AgentName.NEWS)
        result = await agent.search("what is asyncio")
        assert result.status is StepStatus.SUCCEEDED
        assert result.agent is AgentName.NEWS
        assert result.summary
        assert len(result.citations) >= 1
        for c in result.citations:
            assert c.url.startswith("https://")
            assert c.title

    async def test_mock_mode_is_deterministic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("BRAVE_API_KEY", raising=False)
        a1 = BraveAgent(name=AgentName.NEWS)
        a2 = BraveAgent(name=AgentName.NEWS)
        r1 = await a1.search("determinism check")
        r2 = await a2.search("determinism check")
        assert [c.url for c in r1.citations] == [c.url for c in r2.citations]


class TestBraveAgentRealMode:
    async def test_parses_fixture_response(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BRAVE_API_KEY", "brave-test-key")
        agent = BraveAgent(name=AgentName.NEWS)
        assert agent._mock is False

        resp_mock = _mock_http_response(200, BRAVE_FIXTURE)
        ctx = _make_async_client_ctx(resp_mock)

        with patch("research_crew.agents.real.httpx.AsyncClient", return_value=ctx):
            result = await agent.search("what is asyncio")

        assert result.status is StepStatus.SUCCEEDED
        assert result.agent is AgentName.NEWS
        assert len(result.citations) == 2
        assert result.citations[0].url == "https://docs.python.org/3/library/asyncio.html"
        assert result.citations[0].title == "asyncio — Asynchronous I/O"
        assert "asyncio" in result.citations[0].snippet.lower()

    async def test_auth_error_falls_back_to_mock(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BRAVE_API_KEY", "bad-key")
        agent = BraveAgent(name=AgentName.NEWS)

        resp_mock = _mock_http_response(401, {"message": "Unauthorized"})
        ctx = _make_async_client_ctx(resp_mock)

        with patch("research_crew.agents.real.httpx.AsyncClient", return_value=ctx):
            result = await agent.search("auth fail")

        assert result.status is StepStatus.SUCCEEDED

    async def test_5xx_returns_failed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BRAVE_API_KEY", "brave-test-key")
        agent = BraveAgent(name=AgentName.NEWS)

        resp_mock = _mock_http_response(503, {})
        resp_mock.is_success = False
        ctx = _make_async_client_ctx(resp_mock)

        with patch("research_crew.agents.real.httpx.AsyncClient", return_value=ctx):
            result = await agent.search("server error")

        assert result.status is StepStatus.FAILED
        assert result.error and "503" in result.error

    async def test_timeout_falls_back_to_mock(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BRAVE_API_KEY", "brave-test-key")
        agent = BraveAgent(name=AgentName.NEWS)

        client_mock = MagicMock()
        client_mock.get = AsyncMock(side_effect=httpx.TimeoutException("timed out"))
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=client_mock)
        ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("research_crew.agents.real.httpx.AsyncClient", return_value=ctx):
            result = await agent.search("timeout")

        assert result.status is StepStatus.SUCCEEDED


# ===========================================================================
# ExaAgent
# ===========================================================================


class TestExaAgentMockMode:
    def test_no_key_sets_mock_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("EXA_API_KEY", raising=False)
        agent = ExaAgent(name=AgentName.SCHOLAR)
        assert agent._mock is True

    async def test_mock_mode_returns_valid_shape(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("EXA_API_KEY", raising=False)
        agent = ExaAgent(name=AgentName.SCHOLAR)
        result = await agent.search("what is asyncio")
        assert result.status is StepStatus.SUCCEEDED
        assert result.agent is AgentName.SCHOLAR
        assert result.summary
        assert len(result.citations) >= 1
        for c in result.citations:
            assert c.url.startswith("https://")
            assert c.title

    async def test_mock_mode_is_deterministic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("EXA_API_KEY", raising=False)
        a1 = ExaAgent(name=AgentName.SCHOLAR)
        a2 = ExaAgent(name=AgentName.SCHOLAR)
        r1 = await a1.search("determinism check")
        r2 = await a2.search("determinism check")
        assert [c.url for c in r1.citations] == [c.url for c in r2.citations]


class TestExaAgentRealMode:
    async def test_parses_fixture_response(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EXA_API_KEY", "exa-test-key")
        agent = ExaAgent(name=AgentName.SCHOLAR)
        assert agent._mock is False

        resp_mock = _mock_http_response(200, EXA_FIXTURE)
        ctx = _make_async_client_ctx(resp_mock)

        with patch("research_crew.agents.real.httpx.AsyncClient", return_value=ctx):
            result = await agent.search("what is asyncio")

        assert result.status is StepStatus.SUCCEEDED
        assert result.agent is AgentName.SCHOLAR
        assert len(result.citations) == 2
        assert result.citations[0].url == "https://docs.python.org/3/library/asyncio.html"
        assert result.citations[0].title == "asyncio basics"
        assert "asyncio" in result.citations[0].snippet.lower()

    async def test_auth_error_falls_back_to_mock(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EXA_API_KEY", "bad-key")
        agent = ExaAgent(name=AgentName.SCHOLAR)

        resp_mock = _mock_http_response(401, {"error": "Unauthorized"})
        ctx = _make_async_client_ctx(resp_mock)

        with patch("research_crew.agents.real.httpx.AsyncClient", return_value=ctx):
            result = await agent.search("auth fail")

        assert result.status is StepStatus.SUCCEEDED

    async def test_5xx_returns_failed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EXA_API_KEY", "exa-test-key")
        agent = ExaAgent(name=AgentName.SCHOLAR)

        resp_mock = _mock_http_response(503, {})
        resp_mock.is_success = False
        ctx = _make_async_client_ctx(resp_mock)

        with patch("research_crew.agents.real.httpx.AsyncClient", return_value=ctx):
            result = await agent.search("server error")

        assert result.status is StepStatus.FAILED
        assert result.error and "503" in result.error

    async def test_timeout_falls_back_to_mock(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EXA_API_KEY", "exa-test-key")
        agent = ExaAgent(name=AgentName.SCHOLAR)

        client_mock = MagicMock()
        client_mock.post = AsyncMock(side_effect=httpx.TimeoutException("timed out"))
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=client_mock)
        ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("research_crew.agents.real.httpx.AsyncClient", return_value=ctx):
            result = await agent.search("timeout")

        assert result.status is StepStatus.SUCCEEDED


# ===========================================================================
# BraveAgent: endpoint parameterisation
# ===========================================================================


class TestBraveAgentEndpointParam:
    def test_default_endpoint_is_web(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("BRAVE_API_KEY", raising=False)
        agent = BraveAgent(name=AgentName.NEWS)
        assert agent.endpoint == "web"
        assert agent._url == "https://api.search.brave.com/res/v1/web/search"

    def test_news_endpoint_sets_correct_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("BRAVE_API_KEY", raising=False)
        agent = BraveAgent(name=AgentName.NEWS, endpoint="news")
        assert agent._url == "https://api.search.brave.com/res/v1/news/search"

    def test_invalid_endpoint_raises_value_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("BRAVE_API_KEY", raising=False)
        with pytest.raises(ValueError, match="endpoint must be one of"):
            BraveAgent(name=AgentName.NEWS, endpoint="video")

    async def test_news_endpoint_parses_news_fixture(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """news endpoint reads results from response["news"]["results"]."""
        monkeypatch.setenv("BRAVE_API_KEY", "brave-test-key")
        agent = BraveAgent(name=AgentName.NEWS, endpoint="news")
        assert agent._mock is False

        resp_mock = _mock_http_response(200, BRAVE_NEWS_FIXTURE)
        ctx = _make_async_client_ctx(resp_mock)

        with patch("research_crew.agents.real.httpx.AsyncClient", return_value=ctx):
            result = await agent.search("python asyncio news")

        assert result.status is StepStatus.SUCCEEDED
        assert result.agent is AgentName.NEWS
        assert len(result.citations) == 2
        assert result.citations[0].url == "https://news.example.com/python-asyncio-2026"
        assert result.citations[0].title == "Python asyncio gets major upgrade"
        assert "asyncio" in result.citations[0].snippet.lower()

    async def test_web_endpoint_ignores_news_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """web endpoint (default) reads from "web" key; "news" key in response is ignored."""
        monkeypatch.setenv("BRAVE_API_KEY", "brave-test-key")
        agent = BraveAgent(name=AgentName.NEWS, endpoint="web")

        # Response has both keys; web endpoint should use "web".
        combined: dict[str, Any] = {**BRAVE_FIXTURE, **BRAVE_NEWS_FIXTURE}
        resp_mock = _mock_http_response(200, combined)
        ctx = _make_async_client_ctx(resp_mock)

        with patch("research_crew.agents.real.httpx.AsyncClient", return_value=ctx):
            result = await agent.search("what is asyncio")

        assert result.citations[0].url == "https://docs.python.org/3/library/asyncio.html"


# ===========================================================================
# ExaAgent: category parameterisation
# ===========================================================================


class TestExaAgentCategoryParam:
    def test_default_category_is_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("EXA_API_KEY", raising=False)
        agent = ExaAgent(name=AgentName.SCHOLAR)
        assert agent.category is None

    def test_category_research_paper_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("EXA_API_KEY", raising=False)
        agent = ExaAgent(name=AgentName.SCHOLAR, category="research paper")
        assert agent.category == "research paper"

    async def test_category_included_in_payload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When category is set, it is sent in the POST body."""
        monkeypatch.setenv("EXA_API_KEY", "exa-test-key")
        agent = ExaAgent(name=AgentName.SCHOLAR, category="research paper")

        resp_mock = _mock_http_response(200, EXA_FIXTURE)
        client_mock = MagicMock()
        client_mock.post = AsyncMock(return_value=resp_mock)
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=client_mock)
        ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("research_crew.agents.real.httpx.AsyncClient", return_value=ctx):
            result = await agent.search("neural networks")

        assert result.status is StepStatus.SUCCEEDED
        # Verify category was sent in the POST body.
        call_kwargs = client_mock.post.call_args.kwargs
        sent_json: dict[str, Any] = call_kwargs["json"]
        assert sent_json.get("category") == "research paper"

    async def test_no_category_omits_field_from_payload(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When category is None, 'category' key must not appear in the POST body."""
        monkeypatch.setenv("EXA_API_KEY", "exa-test-key")
        agent = ExaAgent(name=AgentName.SCHOLAR)  # category=None default

        resp_mock = _mock_http_response(200, EXA_FIXTURE)
        client_mock = MagicMock()
        client_mock.post = AsyncMock(return_value=resp_mock)
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=client_mock)
        ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("research_crew.agents.real.httpx.AsyncClient", return_value=ctx):
            result = await agent.search("python")

        assert result.status is StepStatus.SUCCEEDED
        call_kwargs = client_mock.post.call_args.kwargs
        sent_json = call_kwargs["json"]
        assert "category" not in sent_json

    async def test_mock_mode_unaffected_by_category(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Mock mode returns the standard mock result regardless of category."""
        monkeypatch.delenv("EXA_API_KEY", raising=False)
        agent = ExaAgent(name=AgentName.SCHOLAR, category="research paper")
        result = await agent.search("determinism check")
        assert result.status is StepStatus.SUCCEEDED
        assert result.agent is AgentName.SCHOLAR


# ===========================================================================
# Cross-adapter: mock outputs are distinguishable
# ===========================================================================


class TestMockDistinctness:
    async def test_three_mock_urls_are_all_different(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        monkeypatch.delenv("BRAVE_API_KEY", raising=False)
        monkeypatch.delenv("EXA_API_KEY", raising=False)

        q = "same question for all three"
        tavily = TavilyAgent(name=AgentName.WEB_SEARCH)
        brave = BraveAgent(name=AgentName.NEWS)
        exa = ExaAgent(name=AgentName.SCHOLAR)

        r_t = await tavily.search(q)
        r_b = await brave.search(q)
        r_e = await exa.search(q)

        urls_t = {c.url for c in r_t.citations}
        urls_b = {c.url for c in r_b.citations}
        urls_e = {c.url for c in r_e.citations}

        # No two adapters should produce the same mock URL set.
        assert urls_t != urls_b
        assert urls_t != urls_e
        assert urls_b != urls_e


# ===========================================================================
# default_agents() factory: env-key-driven wiring
# ===========================================================================


class TestDefaultAgentsFactory:
    def test_no_keys_returns_5_mock_agents(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When no API keys are set, default_agents() returns 5 MockAgent instances."""
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        monkeypatch.delenv("BRAVE_API_KEY", raising=False)
        monkeypatch.delenv("EXA_API_KEY", raising=False)

        agents = default_agents()

        assert len(agents) == 5
        assert all(isinstance(a, MockAgent) for a in agents)
        assert {a.name for a in agents} == set(AgentName)

    def test_tavily_key_returns_one_tavily_and_4_mock(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When only TAVILY_API_KEY is set, the WEB_SEARCH slot uses TavilyAgent."""
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-key")
        monkeypatch.delenv("BRAVE_API_KEY", raising=False)
        monkeypatch.delenv("EXA_API_KEY", raising=False)

        agents = default_agents()

        assert len(agents) == 5
        tavily_agents = [a for a in agents if isinstance(a, TavilyAgent)]
        mock_agents = [a for a in agents if isinstance(a, MockAgent)]
        assert len(tavily_agents) == 1
        assert tavily_agents[0].name is AgentName.WEB_SEARCH
        assert len(mock_agents) == 4

    def test_brave_key_wires_news_endpoint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """BRAVE_API_KEY → BraveAgent with endpoint='news' for the NEWS slot."""
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        monkeypatch.setenv("BRAVE_API_KEY", "brave-test-key")
        monkeypatch.delenv("EXA_API_KEY", raising=False)

        agents = default_agents()

        brave_agents = [a for a in agents if isinstance(a, BraveAgent)]
        assert len(brave_agents) == 1
        brave = brave_agents[0]
        assert brave.name is AgentName.NEWS
        assert brave.endpoint == "news"
        assert brave._url == "https://api.search.brave.com/res/v1/news/search"

    def test_exa_key_wires_research_paper_category(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """EXA_API_KEY → ExaAgent with category='research paper' for the SCHOLAR slot."""
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        monkeypatch.delenv("BRAVE_API_KEY", raising=False)
        monkeypatch.setenv("EXA_API_KEY", "exa-test-key")

        agents = default_agents()

        exa_agents = [a for a in agents if isinstance(a, ExaAgent)]
        assert len(exa_agents) == 1
        exa = exa_agents[0]
        assert exa.name is AgentName.SCHOLAR
        assert exa.category == "research paper"


# ===========================================================================
# TestRealAPIResponseParsing — fixture-shape tests against documented schemas
# ===========================================================================
# Each test loads a JSON fixture that mirrors the real provider API response
# shape (verified from API docs 2026-05-06), mocks httpx to return it, and
# asserts the adapter correctly extracts URLs/titles/snippets from the exact
# field paths the provider uses.  A field rename or removal by the provider
# will break the corresponding assertion here before it silently corrupts
# production results.
# ===========================================================================


class TestRealAPIResponseParsing:
    # -----------------------------------------------------------------------
    # Tavily — full documented response shape
    # -----------------------------------------------------------------------

    async def test_tavily_parses_full_response_shape(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Tavily fixture: results[].title/url/content → Citation; score field is read-through."""
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-fixture-key")
        agent = TavilyAgent(name=AgentName.WEB_SEARCH)

        fixture = _load_fixture("tavily_search_response.json")
        resp_mock = _mock_http_response(200, fixture)
        ctx = _make_async_client_ctx(resp_mock)

        with patch("research_crew.agents.real.httpx.AsyncClient", return_value=ctx):
            result = await agent.search("machine learning transformers")

        assert result.status is StepStatus.SUCCEEDED
        assert result.agent is AgentName.WEB_SEARCH
        # Fixture has 5 results, all with url set.
        assert len(result.citations) == 5

        # First result — highest scored academic paper
        c0 = result.citations[0]
        assert c0.url == "https://arxiv.org/abs/1706.03762"
        assert c0.title == "Attention Is All You Need - Vaswani et al."
        assert "attention" in c0.snippet.lower() or "Transformer" in c0.snippet

        # Second result — BERT paper
        c1 = result.citations[1]
        assert c1.url == "https://arxiv.org/abs/1810.04805"
        assert "BERT" in c1.title

        # Summary is built from the first 3 results' content joined by spaces.
        assert result.summary
        assert "Transformer" in result.summary or "transduction" in result.summary.lower()

        # raw_content=null in fixture → not used; content field drives snippet.
        assert (
            "attention mechanisms" in c0.snippet.lower()
            or "sequence transduction" in c0.snippet.lower()
        )

    async def test_tavily_handles_missing_optional_fields(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """score, raw_content, favicon, images absent → no crash; only url/title/content used."""
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-fixture-key")
        agent = TavilyAgent(name=AgentName.WEB_SEARCH)

        minimal_fixture: dict[str, Any] = {
            "query": "test query",
            "results": [
                {
                    "title": "Minimal Result",
                    "url": "https://example.com/minimal",
                    "content": "Some content without score or raw_content.",
                    # score, raw_content, favicon, images all absent
                },
            ],
            "response_time": 0.5,
        }
        resp_mock = _mock_http_response(200, minimal_fixture)
        ctx = _make_async_client_ctx(resp_mock)

        with patch("research_crew.agents.real.httpx.AsyncClient", return_value=ctx):
            result = await agent.search("test query")

        assert result.status is StepStatus.SUCCEEDED
        assert len(result.citations) == 1
        assert result.citations[0].url == "https://example.com/minimal"
        assert result.citations[0].title == "Minimal Result"
        assert "content without score" in result.citations[0].snippet

    async def test_tavily_handles_empty_results(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Empty results array → SUCCEEDED with no citations; summary falls back to query."""
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-fixture-key")
        agent = TavilyAgent(name=AgentName.WEB_SEARCH)

        empty_fixture: dict[str, Any] = {
            "query": "no results query",
            "results": [],
            "response_time": 0.1,
        }
        resp_mock = _mock_http_response(200, empty_fixture)
        ctx = _make_async_client_ctx(resp_mock)

        with patch("research_crew.agents.real.httpx.AsyncClient", return_value=ctx):
            result = await agent.search("no results query")

        assert result.status is StepStatus.SUCCEEDED
        assert result.citations == []
        # Summary falls back to "[tavily] <question>".
        assert "tavily" in result.summary or "no results query" in result.summary

    # -----------------------------------------------------------------------
    # Brave Web — full documented response shape
    # -----------------------------------------------------------------------

    async def test_brave_web_parses_full_response_shape(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Brave web fixture: web.results[].title/url/description → Citation."""
        monkeypatch.setenv("BRAVE_API_KEY", "brave-fixture-key")
        agent = BraveAgent(name=AgentName.WEB_SEARCH, endpoint="web")

        fixture = _load_fixture("brave_web_response.json")
        resp_mock = _mock_http_response(200, fixture)
        ctx = _make_async_client_ctx(resp_mock)

        with patch("research_crew.agents.real.httpx.AsyncClient", return_value=ctx):
            result = await agent.search("machine learning transformers")

        assert result.status is StepStatus.SUCCEEDED
        assert len(result.citations) == 3

        c0 = result.citations[0]
        assert c0.url == "https://www.ibm.com/topics/transformers-in-machine-learning"
        assert "Transformers in Machine Learning" in c0.title
        assert "neural network" in c0.snippet.lower()

        # extra_snippets and meta_url are NOT parsed into Citation — only
        # title/url/description are used.  This guards against a provider
        # switch that moves content into extra_snippets instead of description.
        assert "extra_snippets" not in c0.snippet
        assert "meta_url" not in c0.snippet

    async def test_brave_web_handles_missing_optional_fields(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """age, extra_snippets, meta_url, family_friendly absent → no crash."""
        monkeypatch.setenv("BRAVE_API_KEY", "brave-fixture-key")
        agent = BraveAgent(name=AgentName.WEB_SEARCH, endpoint="web")

        minimal_fixture: dict[str, Any] = {
            "query": {"original": "test"},
            "web": {
                "results": [
                    {
                        "title": "Bare Web Result",
                        "url": "https://example.com/bare",
                        "description": "Bare description without optional fields.",
                        # age, extra_snippets, meta_url, family_friendly all absent
                    }
                ]
            },
        }
        resp_mock = _mock_http_response(200, minimal_fixture)
        ctx = _make_async_client_ctx(resp_mock)

        with patch("research_crew.agents.real.httpx.AsyncClient", return_value=ctx):
            result = await agent.search("test")

        assert result.status is StepStatus.SUCCEEDED
        assert len(result.citations) == 1
        assert result.citations[0].url == "https://example.com/bare"

    async def test_brave_web_handles_empty_results(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Empty web.results → SUCCEEDED with no citations."""
        monkeypatch.setenv("BRAVE_API_KEY", "brave-fixture-key")
        agent = BraveAgent(name=AgentName.WEB_SEARCH, endpoint="web")

        empty_fixture: dict[str, Any] = {
            "query": {"original": "empty"},
            "web": {"results": []},
        }
        resp_mock = _mock_http_response(200, empty_fixture)
        ctx = _make_async_client_ctx(resp_mock)

        with patch("research_crew.agents.real.httpx.AsyncClient", return_value=ctx):
            result = await agent.search("empty")

        assert result.status is StepStatus.SUCCEEDED
        assert result.citations == []

    # -----------------------------------------------------------------------
    # Brave News — full documented response shape
    # -----------------------------------------------------------------------

    async def test_brave_news_parses_full_response_shape(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Brave news fixture: news.results[].title/url/description → Citation.
        Verifies the adapter uses the 'news' key, not 'web', for news endpoint results."""
        monkeypatch.setenv("BRAVE_API_KEY", "brave-fixture-key")
        agent = BraveAgent(name=AgentName.NEWS, endpoint="news")

        fixture = _load_fixture("brave_news_response.json")
        resp_mock = _mock_http_response(200, fixture)
        ctx = _make_async_client_ctx(resp_mock)

        with patch("research_crew.agents.real.httpx.AsyncClient", return_value=ctx):
            result = await agent.search("machine learning transformers 2026")

        assert result.status is StepStatus.SUCCEEDED
        assert len(result.citations) == 3

        c0 = result.citations[0]
        assert c0.url == "https://techcrunch.com/2026/04/15/deepmind-next-gen-transformer"
        assert "DeepMind" in c0.title
        assert "transformer" in c0.snippet.lower()

        # news-specific field: page_age and thumbnail are NOT mapped to Citation.
        # If a drift moves description → something else, this fails loudly.
        assert "page_age" not in c0.snippet
        assert "thumbnail" not in c0.snippet

    async def test_brave_news_uses_news_key_not_web_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """news endpoint reads from news.results; a response with a 'web' key only → 0 citations."""
        monkeypatch.setenv("BRAVE_API_KEY", "brave-fixture-key")
        agent = BraveAgent(name=AgentName.NEWS, endpoint="news")

        # Response only has 'web' key — news endpoint should look for 'news' key.
        web_only_fixture: dict[str, Any] = {
            "query": {"original": "test"},
            "web": {
                "results": [
                    {
                        "title": "Wrong key result",
                        "url": "https://example.com/wrong",
                        "description": "d",
                    }
                ]
            },
        }
        resp_mock = _mock_http_response(200, web_only_fixture)
        ctx = _make_async_client_ctx(resp_mock)

        with patch("research_crew.agents.real.httpx.AsyncClient", return_value=ctx):
            result = await agent.search("test")

        # The news endpoint looks for data["news"]["results"], not data["web"]["results"].
        assert result.status is StepStatus.SUCCEEDED
        assert result.citations == []  # 'news' key absent → no results parsed

    async def test_brave_news_handles_missing_optional_fields(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """age, page_age, thumbnail absent from news result → no crash."""
        monkeypatch.setenv("BRAVE_API_KEY", "brave-fixture-key")
        agent = BraveAgent(name=AgentName.NEWS, endpoint="news")

        minimal_news_fixture: dict[str, Any] = {
            "query": {"original": "news test"},
            "news": {
                "results": [
                    {
                        "title": "Bare News Result",
                        "url": "https://news.example.com/bare",
                        "description": "A news article without age or thumbnail.",
                        # age, page_age, thumbnail, meta_url all absent
                    }
                ]
            },
        }
        resp_mock = _mock_http_response(200, minimal_news_fixture)
        ctx = _make_async_client_ctx(resp_mock)

        with patch("research_crew.agents.real.httpx.AsyncClient", return_value=ctx):
            result = await agent.search("news test")

        assert result.status is StepStatus.SUCCEEDED
        assert len(result.citations) == 1
        assert result.citations[0].url == "https://news.example.com/bare"

    async def test_brave_news_handles_empty_results(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Empty news.results → SUCCEEDED with no citations."""
        monkeypatch.setenv("BRAVE_API_KEY", "brave-fixture-key")
        agent = BraveAgent(name=AgentName.NEWS, endpoint="news")

        empty_fixture: dict[str, Any] = {
            "query": {"original": "empty news"},
            "news": {"results": []},
        }
        resp_mock = _mock_http_response(200, empty_fixture)
        ctx = _make_async_client_ctx(resp_mock)

        with patch("research_crew.agents.real.httpx.AsyncClient", return_value=ctx):
            result = await agent.search("empty news")

        assert result.status is StepStatus.SUCCEEDED
        assert result.citations == []

    # -----------------------------------------------------------------------
    # Exa — full documented response shape
    # -----------------------------------------------------------------------

    async def test_exa_parses_full_response_shape(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Exa fixture: results[].id/url/title/text/score/publishedDate → Citation."""
        monkeypatch.setenv("EXA_API_KEY", "exa-fixture-key")
        agent = ExaAgent(name=AgentName.SCHOLAR)

        fixture = _load_fixture("exa_search_response.json")
        resp_mock = _mock_http_response(200, fixture)
        ctx = _make_async_client_ctx(resp_mock)

        with patch("research_crew.agents.real.httpx.AsyncClient", return_value=ctx):
            result = await agent.search("machine learning transformers")

        assert result.status is StepStatus.SUCCEEDED
        assert result.agent is AgentName.SCHOLAR
        assert len(result.citations) == 3

        c0 = result.citations[0]
        assert c0.url == "https://arxiv.org/abs/1706.03762"
        assert c0.title == "Attention Is All You Need"
        # snippet is text[:500]
        assert "attention mechanisms" in c0.snippet.lower() or "Transformer" in c0.snippet

        # score/publishedDate/author/highlights are NOT mapped to Citation fields;
        # only url/title/text are used.
        assert len(c0.snippet) <= 500

        # requestId, searchType, costDollars are top-level — not used in result.
        assert result.summary  # built from text of first 3 results

    async def test_exa_research_filter_in_request_body(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """category='research paper' fixture: request body must include category field."""
        monkeypatch.setenv("EXA_API_KEY", "exa-fixture-key")
        agent = ExaAgent(name=AgentName.SCHOLAR, category="research paper")

        fixture = _load_fixture("exa_research_paper_response.json")
        resp_mock = _mock_http_response(200, fixture)
        client_mock = MagicMock()
        client_mock.post = AsyncMock(return_value=resp_mock)
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=client_mock)
        ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("research_crew.agents.real.httpx.AsyncClient", return_value=ctx):
            result = await agent.search("neural network architectures")

        assert result.status is StepStatus.SUCCEEDED
        # Verify the POST body sent to Exa includes category.
        call_kwargs = client_mock.post.call_args.kwargs
        sent_json: dict[str, Any] = call_kwargs["json"]
        assert sent_json.get("category") == "research paper"
        # Also verify numResults and contents are present.
        assert sent_json.get("numResults") == 5
        assert sent_json.get("contents") == {"text": True}

    async def test_exa_research_paper_parses_academic_urls(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Exa research paper fixture has academic URLs (arxiv, nature, pubmed)."""
        monkeypatch.setenv("EXA_API_KEY", "exa-fixture-key")
        agent = ExaAgent(name=AgentName.SCHOLAR, category="research paper")

        fixture = _load_fixture("exa_research_paper_response.json")
        resp_mock = _mock_http_response(200, fixture)
        ctx = _make_async_client_ctx(resp_mock)

        with patch("research_crew.agents.real.httpx.AsyncClient", return_value=ctx):
            result = await agent.search("transformer models")

        assert result.status is StepStatus.SUCCEEDED
        # Fixture has 4 results; the one with score=null still has a url → parsed.
        assert len(result.citations) == 4
        urls = [c.url for c in result.citations]
        assert any("arxiv.org" in u for u in urls)
        assert any("nature.com" in u for u in urls)
        assert any("pubmed" in u for u in urls)

    async def test_exa_handles_missing_optional_fields(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """score, publishedDate, author, highlights absent → no crash."""
        monkeypatch.setenv("EXA_API_KEY", "exa-fixture-key")
        agent = ExaAgent(name=AgentName.SCHOLAR)

        minimal_fixture: dict[str, Any] = {
            "requestId": "req-minimal",
            "results": [
                {
                    "id": "https://example.com/paper",
                    "url": "https://example.com/paper",
                    "title": "A Minimal Paper",
                    "text": "This paper has only the required fields set.",
                    # score, publishedDate, author, highlights, highlightScores all absent
                }
            ],
        }
        resp_mock = _mock_http_response(200, minimal_fixture)
        ctx = _make_async_client_ctx(resp_mock)

        with patch("research_crew.agents.real.httpx.AsyncClient", return_value=ctx):
            result = await agent.search("minimal")

        assert result.status is StepStatus.SUCCEEDED
        assert len(result.citations) == 1
        assert result.citations[0].url == "https://example.com/paper"
        assert result.citations[0].title == "A Minimal Paper"

    async def test_exa_handles_empty_results(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Empty results array → SUCCEEDED with no citations; summary falls back to query."""
        monkeypatch.setenv("EXA_API_KEY", "exa-fixture-key")
        agent = ExaAgent(name=AgentName.SCHOLAR)

        empty_fixture: dict[str, Any] = {
            "requestId": "req-empty",
            "results": [],
        }
        resp_mock = _mock_http_response(200, empty_fixture)
        ctx = _make_async_client_ctx(resp_mock)

        with patch("research_crew.agents.real.httpx.AsyncClient", return_value=ctx):
            result = await agent.search("empty query")

        assert result.status is StepStatus.SUCCEEDED
        assert result.citations == []
        assert "exa" in result.summary or "empty query" in result.summary

    async def test_exa_snippet_truncated_to_500_chars(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """text field longer than 500 chars is truncated to 500 in the Citation snippet."""
        monkeypatch.setenv("EXA_API_KEY", "exa-fixture-key")
        agent = ExaAgent(name=AgentName.SCHOLAR)

        long_text = "x" * 1000
        fixture: dict[str, Any] = {
            "requestId": "req-long",
            "results": [
                {
                    "id": "https://example.com/long",
                    "url": "https://example.com/long",
                    "title": "Long Text Paper",
                    "text": long_text,
                }
            ],
        }
        resp_mock = _mock_http_response(200, fixture)
        ctx = _make_async_client_ctx(resp_mock)

        with patch("research_crew.agents.real.httpx.AsyncClient", return_value=ctx):
            result = await agent.search("long")

        assert len(result.citations[0].snippet) == 500
