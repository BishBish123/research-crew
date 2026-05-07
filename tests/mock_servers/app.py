"""FastAPI mock server that responds with documented provider JSON shapes.

Each endpoint:
- Validates that the auth header is present (any non-empty value).
- Returns the fixture JSON corresponding to the documented API shape.
- Supports ``?force=500`` query parameter to trigger one 5xx response then
  succeed on the next call — exercises the adapter retry path.
"""

from __future__ import annotations

import json
import pathlib
from collections import defaultdict
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Query, Request, Response

_FIXTURE_DIR = pathlib.Path(__file__).parent.parent / "fixtures" / "api_responses"


def _load(filename: str) -> dict[str, Any]:
    return json.loads((_FIXTURE_DIR / filename).read_text())  # type: ignore[no-any-return]


def _maybe_force_500(app: FastAPI, key: str, force: str | None) -> Response | None:
    """Return a 500 Response the first time ``?force=500`` is seen for *key*,
    None afterwards (counter is reset so cycles repeat).
    """
    if force != "500":
        return None
    counters: dict[str, int] = app.state.force_500_counters
    counters[key] += 1
    if counters[key] == 1:
        return Response(
            content='{"error":"forced 500"}',
            status_code=500,
            media_type="application/json",
        )
    counters[key] = 0
    return None


def _json_response(fixture: dict[str, Any]) -> Response:
    return Response(content=json.dumps(fixture), status_code=200, media_type="application/json")


def create_mock_app() -> FastAPI:
    """Return a fresh FastAPI app with all mock endpoints mounted.

    The app carries a ``force_500_counters`` dict (keyed by route) that
    tracks how many times each ``?force=500`` endpoint has been called this
    process lifetime, so the first call returns 500 and the second returns 200.
    """
    app = FastAPI(title="research-crew-mock-api")
    app.state.force_500_counters: dict[str, int] = defaultdict(int)  # type: ignore[assignment]

    # -----------------------------------------------------------------------
    # Tavily — POST /tavily/search
    # -----------------------------------------------------------------------

    @app.post("/tavily/search")
    async def tavily_search(
        request: Request,
        force: str | None = Query(default=None),
        authorization: str | None = Header(default=None),
    ) -> Response:
        if not authorization:
            raise HTTPException(status_code=401, detail="Missing Authorization header")
        forced = _maybe_force_500(app, "tavily_search", force)
        if forced is not None:
            return forced
        return _json_response(_load("tavily_search_response.json"))

    # -----------------------------------------------------------------------
    # Brave — GET /brave/web/search and GET /brave/news/search
    # -----------------------------------------------------------------------

    @app.get("/brave/web/search")
    async def brave_web_search(
        request: Request,
        force: str | None = Query(default=None),
        x_subscription_token: str | None = Header(default=None, alias="x-subscription-token"),
    ) -> Response:
        if not x_subscription_token:
            raise HTTPException(status_code=401, detail="Missing X-Subscription-Token header")
        forced = _maybe_force_500(app, "brave_web_search", force)
        if forced is not None:
            return forced
        return _json_response(_load("brave_web_response.json"))

    @app.get("/brave/news/search")
    async def brave_news_search(
        request: Request,
        force: str | None = Query(default=None),
        x_subscription_token: str | None = Header(default=None, alias="x-subscription-token"),
    ) -> Response:
        if not x_subscription_token:
            raise HTTPException(status_code=401, detail="Missing X-Subscription-Token header")
        forced = _maybe_force_500(app, "brave_news_search", force)
        if forced is not None:
            return forced
        return _json_response(_load("brave_news_response.json"))

    # -----------------------------------------------------------------------
    # Exa — POST /exa/search
    # Returns exa_research_paper_response.json when category="research paper"
    # is in the request body; otherwise exa_search_response.json.
    # -----------------------------------------------------------------------

    @app.post("/exa/search")
    async def exa_search(
        request: Request,
        force: str | None = Query(default=None),
        x_api_key: str | None = Header(default=None, alias="x-api-key"),
    ) -> Response:
        if not x_api_key:
            raise HTTPException(status_code=401, detail="Missing x-api-key header")
        forced = _maybe_force_500(app, "exa_search", force)
        if forced is not None:
            return forced
        try:
            body = await request.json()
        except Exception:
            body = {}
        if isinstance(body, dict) and body.get("category") == "research paper":
            return _json_response(_load("exa_research_paper_response.json"))
        return _json_response(_load("exa_search_response.json"))

    # -----------------------------------------------------------------------
    # Langfuse — POST /langfuse/api/public/ingestion
    # -----------------------------------------------------------------------

    @app.post("/langfuse/api/public/ingestion")
    async def langfuse_ingestion(
        request: Request,
        x_langfuse_public_key: str | None = Header(default=None, alias="x-langfuse-public-key"),
        authorization: str | None = Header(default=None),
    ) -> Response:
        if not x_langfuse_public_key and not authorization:
            raise HTTPException(status_code=401, detail="Missing auth header")
        try:
            body = await request.json()
        except Exception:
            body = {}
        batch = body.get("batch", []) if isinstance(body, dict) else []
        successes = [{"id": item.get("id", "unknown"), "status": 201} for item in batch]
        payload: dict[str, Any] = {"successes": successes, "errors": []}
        return Response(
            content=json.dumps(payload),
            status_code=207,
            media_type="application/json",
        )

    return app
