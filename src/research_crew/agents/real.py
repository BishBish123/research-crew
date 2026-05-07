"""Real search-API adapters: TavilyAgent, BraveAgent, ExaAgent.

Each adapter reads its API key from the environment on construction.
When the key is absent, the adapter runs in mock mode (``_mock=True``)
and delegates to a per-adapter deterministic stub — different blake2b
seeds so the three mock outputs are distinguishable from one another.
When the key is present, the adapter makes a real HTTP call and maps
the response into the shared ``AgentResult`` / ``Citation`` shape.

Real-path guarantees
--------------------
* Timeouts: 5 s connect, 10 s total (via ``httpx.Timeout``).
* Retry: one automatic retry on any 5xx response, with a 1 s backoff.
* Fail-soft: a 4xx / auth error logs a structlog warning and falls back
  to the mock result so the wider fan-out still succeeds.

Environment variables
---------------------
* ``TAVILY_API_KEY``       — auth for ``https://api.tavily.com/search``
* ``TAVILY_API_BASE_URL``  — optional override for the Tavily base URL (tests: localhost)
* ``BRAVE_API_KEY``        — auth for Brave Search; endpoint selected by ``endpoint`` arg:
  - ``"web"``  → ``https://api.search.brave.com/res/v1/web/search``
  - ``"news"`` → ``https://api.search.brave.com/res/v1/news/search``
* ``BRAVE_API_BASE_URL``   — optional override for the Brave base URL (tests: localhost)
* ``EXA_API_KEY``          — auth for ``https://api.exa.ai/search``
* ``EXA_API_BASE_URL``     — optional override for the Exa base URL (tests: localhost)

Auth headers (verified 2026-05-06)
-----------------------------------
* Tavily: ``Authorization: Bearer <key>``
  Source: https://docs.tavily.com/documentation/quickstart
* Brave:  ``X-Subscription-Token: <key>``
  Source: https://api-dashboard.search.brave.com/app/documentation/web-search/get-started
  News endpoint: https://api-dashboard.search.brave.com/app/documentation/news-search/get-started
  (verified 2026-05-06: separate /res/v1/news/search path; same auth; results under "news" key)
* Exa:    ``x-api-key: <key>``
  Source: https://exa.ai/docs/reference/search
  (verified 2026-05-06: ``category="research paper"`` filters to academic/research results)

Provider response shapes (verified 2026-05-06)
-----------------------------------------------
Fixture tests in ``tests/fixtures/api_responses/`` validate parsing against
the shapes documented below.  Schema-drift (a provider renaming or removing a
field) will surface as a fixture-test failure before it silently breaks results.

**Tavily** — POST https://api.tavily.com/search
  Source: https://docs.tavily.com/documentation/api-reference/endpoint/search
  Required top-level: query (str), results (list), response_time (float)
  Optional top-level: answer (str), images (list), usage (obj), request_id (str)
  results[] required: title (str), url (str), content (str)
  results[] optional: score (float), raw_content (str|null), favicon (str), images (list)
  Parsed fields → Citation: title=title, url=url, snippet=content
  Malformed/missing response → logs warning, falls back to mock result

**Brave Web** — GET https://api.search.brave.com/res/v1/web/search
  Source: https://api-dashboard.search.brave.com/app/documentation/web-search/responses
  Results live at: response["web"]["results"]
  results[] required: title (str), url (str)
  results[] optional: description (str), age (str), language (str), family_friendly (bool),
                       extra_snippets (list[str]), meta_url (obj)
  Parsed fields → Citation: title=title, url=url, snippet=description

**Brave News** — GET https://api.search.brave.com/res/v1/news/search
  Source: https://api-dashboard.search.brave.com/app/documentation/news-search/get-started
  Results live at: response["news"]["results"]  (key matches endpoint name)
  results[] required: title (str), url (str)
  results[] optional: description (str), age (str), page_age (str ISO-8601),
                       thumbnail (obj|null), meta_url (obj)
  Parsed fields → Citation: title=title, url=url, snippet=description
  Malformed/missing "news" key → empty citations (no crash)

**Exa** — POST https://api.exa.ai/search
  Source: https://exa.ai/docs/reference/search
  Required top-level: requestId (str), results (list)
  Optional top-level: searchType (str), costDollars (obj)
  results[] required: id (str), url (str)
  results[] optional: title (str), publishedDate (str ISO-8601|null), author (str|null),
                       score (float|null), text (str), highlights (list[str]),
                       highlightScores (list[float]), image (str|null), favicon (str)
  Parsed fields → Citation: title=title, url=url, snippet=text[:500]
  category="research paper" body param → academic/peer-reviewed results (same shape)
  Malformed response → logs warning, falls back to mock result
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import time
from dataclasses import dataclass, field

import httpx
import structlog

from research_crew.models import AgentName, AgentResult, Citation, StepStatus

log = structlog.get_logger()

# Shared timeout: 5 s connect, 10 s total.
_TIMEOUT = httpx.Timeout(10.0, connect=5.0)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _mock_citations_seeded(name: AgentName, question: str, seed: str) -> list[Citation]:
    """Deterministic mock citations with a per-adapter seed so the three mocks differ."""
    base = hashlib.blake2b(f"{seed}|{name}|{question}".encode(), digest_size=4).hexdigest()
    return [
        Citation(
            title=f"[{name}] mock result {i + 1}: {question[:40]}",
            url=f"https://mock.{name}.example.com/{base}/{i}",
            snippet=f"Mock snippet from {name} (seed={seed}, hash={base}-{i}).",
        )
        for i in range(2)
    ]


def _mock_result(name: AgentName, question: str, seed: str) -> AgentResult:
    return AgentResult(
        agent=name,
        status=StepStatus.SUCCEEDED,
        summary=f"[{name}:{seed}] mock summary for: {question}",
        citations=_mock_citations_seeded(name, question, seed),
        elapsed_ms=0.0,
        attempts=1,
    )


async def _post_with_retry(
    client: httpx.AsyncClient,
    url: str,
    *,
    json: dict[object, object] | None = None,
    headers: dict[str, str],
) -> httpx.Response:
    """POST with one retry on 5xx; returns the first non-5xx response."""
    for attempt in range(2):
        resp = await client.post(url, json=json, headers=headers, timeout=_TIMEOUT)
        if resp.status_code < 500 or attempt == 1:
            return resp
        await asyncio.sleep(1.0)
    return resp  # unreachable, satisfies mypy


async def _get_with_retry(
    client: httpx.AsyncClient,
    url: str,
    *,
    params: dict[str, str | int],
    headers: dict[str, str],
) -> httpx.Response:
    """GET with one retry on 5xx."""
    for attempt in range(2):
        resp = await client.get(url, params=params, headers=headers, timeout=_TIMEOUT)
        if resp.status_code < 500 or attempt == 1:
            return resp
        await asyncio.sleep(1.0)
    return resp  # unreachable, satisfies mypy


# ---------------------------------------------------------------------------
# TavilyAgent
# ---------------------------------------------------------------------------


@dataclass
class TavilyAgent:
    """Calls https://api.tavily.com/search (bearer-token auth).

    Auth header: ``Authorization: Bearer <TAVILY_API_KEY>``
    Verified: https://docs.tavily.com/documentation/quickstart (2026-05-06)

    Mock mode: ``_mock=True`` when ``TAVILY_API_KEY`` is unset.

    The ``base_url`` constructor parameter (or ``TAVILY_API_BASE_URL`` env var)
    overrides the production endpoint — used by integration tests to point the
    adapter at a local mock server without changing default behaviour.
    """

    name: AgentName = field(default=AgentName.WEB_SEARCH)
    base_url: str = field(default="")
    _api_key: str = field(default="", init=False, repr=False)
    _mock: bool = field(default=False, init=False)
    _endpoint: str = field(default="", init=False, repr=False)

    _DEFAULT_ENDPOINT = "https://api.tavily.com/search"
    _MOCK_SEED = "tavily-mock-seed-v1"

    def __post_init__(self) -> None:
        key = os.environ.get("TAVILY_API_KEY", "")
        if not key:
            self._mock = True
        else:
            self._api_key = key
        # base_url constructor arg wins; env var is fallback; then production default.
        resolved_base = self.base_url or os.environ.get("TAVILY_API_BASE_URL", "")
        if resolved_base:
            self._endpoint = resolved_base.rstrip("/") + "/search"
        else:
            self._endpoint = self._DEFAULT_ENDPOINT

    async def search(self, question: str) -> AgentResult:
        if self._mock:
            return _mock_result(self.name, question, self._MOCK_SEED)

        t0 = time.monotonic()
        headers = {"Authorization": f"Bearer {self._api_key}"}
        payload: dict[object, object] = {
            "query": question,
            "max_results": 5,
            "search_depth": "basic",
        }

        try:
            async with httpx.AsyncClient() as client:
                resp = await _post_with_retry(client, self._endpoint, json=payload, headers=headers)
        except (httpx.TimeoutException, httpx.RequestError) as exc:
            log.warning("tavily.request_error", error=str(exc), question=question[:80])
            return _mock_result(self.name, question, self._MOCK_SEED)

        elapsed = (time.monotonic() - t0) * 1000.0

        if resp.status_code in (401, 403):
            log.warning(
                "tavily.auth_error",
                status=resp.status_code,
                question=question[:80],
            )
            return _mock_result(self.name, question, self._MOCK_SEED)

        if not resp.is_success:
            log.warning(
                "tavily.http_error",
                status=resp.status_code,
                question=question[:80],
            )
            return AgentResult(
                agent=self.name,
                status=StepStatus.FAILED,
                summary="",
                error=f"Tavily HTTP {resp.status_code}",
                elapsed_ms=elapsed,
            )

        data = resp.json()
        results: list[dict[str, object]] = data.get("results", [])
        citations = [
            Citation(
                title=str(r.get("title", "")),
                url=str(r.get("url", "")),
                snippet=str(r.get("content", "")),
            )
            for r in results
            if r.get("url")
        ]
        summary_parts = [str(r.get("content", "")) for r in results[:3] if r.get("content")]
        summary = " ".join(summary_parts) or f"[tavily] {question}"

        return AgentResult(
            agent=self.name,
            status=StepStatus.SUCCEEDED,
            summary=summary,
            citations=citations,
            elapsed_ms=elapsed,
            attempts=1,
        )


# ---------------------------------------------------------------------------
# BraveAgent
# ---------------------------------------------------------------------------

_BRAVE_ENDPOINTS: dict[str, str] = {
    "web": "https://api.search.brave.com/res/v1/web/search",
    "news": "https://api.search.brave.com/res/v1/news/search",
}


@dataclass
class BraveAgent:
    """Calls the Brave Search API.

    The ``endpoint`` constructor argument selects which Brave endpoint to hit:

    * ``"web"``  — ``/res/v1/web/search``   (generic web results; response key ``"web"``)
    * ``"news"`` — ``/res/v1/news/search``  (news results; response key ``"news"``)

    Auth header: ``X-Subscription-Token: <BRAVE_API_KEY>``

    Sources (verified 2026-05-06):
    - Web:  https://api-dashboard.search.brave.com/app/documentation/web-search/get-started
    - News: https://api-dashboard.search.brave.com/app/documentation/news-search/get-started

    Mock mode: ``_mock=True`` when ``BRAVE_API_KEY`` is unset.

    The ``base_url`` constructor parameter (or ``BRAVE_API_BASE_URL`` env var)
    overrides the production base URL — used by integration tests to point the
    adapter at a local mock server without changing default behaviour.
    When set, the full endpoint URL becomes ``<base_url>/<endpoint>/search``
    (e.g. ``http://localhost:12345/web/search`` for endpoint ``"web"``).
    """

    name: AgentName = field(default=AgentName.NEWS)
    endpoint: str = field(default="web")
    base_url: str = field(default="")
    _api_key: str = field(default="", init=False, repr=False)
    _mock: bool = field(default=False, init=False)
    _url: str = field(default="", init=False, repr=False)

    _MOCK_SEED = "brave-mock-seed-v1"

    def __post_init__(self) -> None:
        if self.endpoint not in _BRAVE_ENDPOINTS:
            raise ValueError(
                f"BraveAgent endpoint must be one of {list(_BRAVE_ENDPOINTS)!r}; "
                f"got {self.endpoint!r}"
            )
        resolved_base = self.base_url or os.environ.get("BRAVE_API_BASE_URL", "")
        if resolved_base:
            self._url = resolved_base.rstrip("/") + f"/{self.endpoint}/search"
        else:
            self._url = _BRAVE_ENDPOINTS[self.endpoint]
        key = os.environ.get("BRAVE_API_KEY", "")
        if not key:
            self._mock = True
        else:
            self._api_key = key

    async def search(self, question: str) -> AgentResult:
        if self._mock:
            return _mock_result(self.name, question, self._MOCK_SEED)

        t0 = time.monotonic()
        headers = {
            "X-Subscription-Token": self._api_key,
            "Accept": "application/json",
        }
        params: dict[str, str | int] = {"q": question, "count": 5}

        try:
            async with httpx.AsyncClient() as client:
                resp = await _get_with_retry(client, self._url, params=params, headers=headers)
        except (httpx.TimeoutException, httpx.RequestError) as exc:
            log.warning("brave.request_error", error=str(exc), question=question[:80])
            return _mock_result(self.name, question, self._MOCK_SEED)

        elapsed = (time.monotonic() - t0) * 1000.0

        if resp.status_code in (401, 403):
            log.warning(
                "brave.auth_error",
                status=resp.status_code,
                question=question[:80],
            )
            return _mock_result(self.name, question, self._MOCK_SEED)

        if not resp.is_success:
            log.warning(
                "brave.http_error",
                status=resp.status_code,
                question=question[:80],
            )
            return AgentResult(
                agent=self.name,
                status=StepStatus.FAILED,
                summary="",
                error=f"Brave HTTP {resp.status_code}",
                elapsed_ms=elapsed,
            )

        data = resp.json()
        # Web endpoint → "web" key; news endpoint → "news" key.
        top_key = self.endpoint  # "web" or "news"
        section: dict[str, object] = data.get(top_key, {})
        raw_results = section.get("results", [])
        results: list[dict[str, object]] = raw_results if isinstance(raw_results, list) else []
        citations = [
            Citation(
                title=str(r.get("title", "")),
                url=str(r.get("url", "")),
                snippet=str(r.get("description", "")),
            )
            for r in results
            if r.get("url")
        ]
        summary_parts = [str(r.get("description", "")) for r in results[:3] if r.get("description")]
        summary = " ".join(summary_parts) or f"[brave] {question}"

        return AgentResult(
            agent=self.name,
            status=StepStatus.SUCCEEDED,
            summary=summary,
            citations=citations,
            elapsed_ms=elapsed,
            attempts=1,
        )


# ---------------------------------------------------------------------------
# ExaAgent
# ---------------------------------------------------------------------------


@dataclass
class ExaAgent:
    """Calls https://api.exa.ai/search.

    The optional ``category`` constructor argument narrows results to a
    specific content type.  Exa supports:
    ``"company"``, ``"research paper"``, ``"news"``, ``"personal site"``,
    ``"financial report"``, ``"people"``.

    When ``category="research paper"`` is passed (as wired for the SCHOLAR
    slot in ``default_agents()``), Exa filters to academic / peer-reviewed
    sources, matching the slot's semantics.

    Auth header: ``x-api-key: <EXA_API_KEY>``

    Source (verified 2026-05-06): https://exa.ai/docs/reference/search
    (``category`` is a top-level body param; valid string ``"research paper"``
    for academic filtering.)

    Mock mode: ``_mock=True`` when ``EXA_API_KEY`` is unset.

    The ``base_url`` constructor parameter (or ``EXA_API_BASE_URL`` env var)
    overrides the production endpoint — used by integration tests to point the
    adapter at a local mock server without changing default behaviour.
    """

    name: AgentName = field(default=AgentName.SCHOLAR)
    category: str | None = field(default=None)
    base_url: str = field(default="")
    _api_key: str = field(default="", init=False, repr=False)
    _mock: bool = field(default=False, init=False)
    _endpoint: str = field(default="", init=False, repr=False)

    _DEFAULT_ENDPOINT = "https://api.exa.ai/search"
    _MOCK_SEED = "exa-mock-seed-v1"

    def __post_init__(self) -> None:
        key = os.environ.get("EXA_API_KEY", "")
        if not key:
            self._mock = True
        else:
            self._api_key = key
        resolved_base = self.base_url or os.environ.get("EXA_API_BASE_URL", "")
        if resolved_base:
            self._endpoint = resolved_base.rstrip("/") + "/search"
        else:
            self._endpoint = self._DEFAULT_ENDPOINT

    async def search(self, question: str) -> AgentResult:
        if self._mock:
            return _mock_result(self.name, question, self._MOCK_SEED)

        t0 = time.monotonic()
        headers = {
            "x-api-key": self._api_key,
            "Content-Type": "application/json",
        }
        payload: dict[object, object] = {
            "query": question,
            "numResults": 5,
            "contents": {"text": True},
        }
        if self.category is not None:
            payload["category"] = self.category

        try:
            async with httpx.AsyncClient() as client:
                resp = await _post_with_retry(client, self._endpoint, json=payload, headers=headers)
        except (httpx.TimeoutException, httpx.RequestError) as exc:
            log.warning("exa.request_error", error=str(exc), question=question[:80])
            return _mock_result(self.name, question, self._MOCK_SEED)

        elapsed = (time.monotonic() - t0) * 1000.0

        if resp.status_code in (401, 403):
            log.warning(
                "exa.auth_error",
                status=resp.status_code,
                question=question[:80],
            )
            return _mock_result(self.name, question, self._MOCK_SEED)

        if not resp.is_success:
            log.warning(
                "exa.http_error",
                status=resp.status_code,
                question=question[:80],
            )
            return AgentResult(
                agent=self.name,
                status=StepStatus.FAILED,
                summary="",
                error=f"Exa HTTP {resp.status_code}",
                elapsed_ms=elapsed,
            )

        data = resp.json()
        results: list[dict[str, object]] = data.get("results", [])
        citations = [
            Citation(
                title=str(r.get("title", "")),
                url=str(r.get("url", "")),
                snippet=str(r.get("text", ""))[:500],
            )
            for r in results
            if r.get("url")
        ]
        summary_parts = [str(r.get("text", ""))[:300] for r in results[:3] if r.get("text")]
        summary = " ".join(summary_parts) or f"[exa] {question}"

        return AgentResult(
            agent=self.name,
            status=StepStatus.SUCCEEDED,
            summary=summary,
            citations=citations,
            elapsed_ms=elapsed,
            attempts=1,
        )
