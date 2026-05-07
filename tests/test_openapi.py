"""Tests for OpenAPI spec, Postman collection, and Swagger UI endpoint.

Validates:
- docs/openapi.json is a valid OpenAPI 3.x document
- All API endpoints declared in api.py appear in the spec
- docs/postman_collection.json is a valid Postman v2.1 collection
- The Postman collection has at least one request per endpoint
- GET /docs returns Swagger UI HTML
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path

import fakeredis.aioredis as fake_aioredis
import pytest
from httpx import ASGITransport, AsyncClient

from research_crew.api import app
from research_crew.store import RedisRunStore

# ---------------------------------------------------------------------------
# Paths to generated artifacts
# ---------------------------------------------------------------------------

_DOCS_DIR = Path(__file__).parent.parent / "docs"
_OPENAPI_PATH = _DOCS_DIR / "openapi.json"
_POSTMAN_PATH = _DOCS_DIR / "postman_collection.json"

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    fake = fake_aioredis.FakeRedis(decode_responses=True)
    app.state.redis = fake
    app.state.store = RedisRunStore(fake)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as c:
        yield c
    await fake.aclose()


# ---------------------------------------------------------------------------
# Helper: known API paths from api.py (non-static routes)
# ---------------------------------------------------------------------------

_EXPECTED_PATHS = {"/health", "/research", "/runs/{run_id}"}


# ---------------------------------------------------------------------------
# OpenAPI spec tests
# ---------------------------------------------------------------------------


class TestOpenAPISpec:
    def test_openapi_json_file_exists(self) -> None:
        assert _OPENAPI_PATH.exists(), f"Missing {_OPENAPI_PATH}; run `make openapi` to generate"

    def test_openapi_json_is_valid_json(self) -> None:
        text = _OPENAPI_PATH.read_text()
        data = json.loads(text)
        assert isinstance(data, dict)

    def test_openapi_version_is_3x(self) -> None:
        data = json.loads(_OPENAPI_PATH.read_text())
        version = data.get("openapi", "")
        assert str(version).startswith("3."), f"Expected OpenAPI 3.x, got: {version}"

    def test_openapi_has_info_block(self) -> None:
        data = json.loads(_OPENAPI_PATH.read_text())
        assert "info" in data
        assert "title" in data["info"]
        assert "version" in data["info"]

    def test_openapi_has_paths(self) -> None:
        data = json.loads(_OPENAPI_PATH.read_text())
        assert "paths" in data
        assert len(data["paths"]) > 0

    def test_all_api_endpoints_in_spec(self) -> None:
        data = json.loads(_OPENAPI_PATH.read_text())
        spec_paths = set(data.get("paths", {}).keys())
        for expected in _EXPECTED_PATHS:
            assert expected in spec_paths, f"Path {expected!r} missing from OpenAPI spec"

    def test_openapi_has_schemas(self) -> None:
        data = json.loads(_OPENAPI_PATH.read_text())
        schemas = data.get("components", {}).get("schemas", {})
        assert len(schemas) > 0, "Expected at least one schema in components"

    def test_research_endpoint_has_request_body(self) -> None:
        data = json.loads(_OPENAPI_PATH.read_text())
        post_op = data["paths"]["/research"]["post"]
        assert "requestBody" in post_op

    def test_runs_endpoint_has_path_param(self) -> None:
        data = json.loads(_OPENAPI_PATH.read_text())
        get_op = data["paths"]["/runs/{run_id}"]["get"]
        params = get_op.get("parameters", [])
        param_names = {p["name"] for p in params}
        assert "run_id" in param_names


# ---------------------------------------------------------------------------
# Postman collection tests
# ---------------------------------------------------------------------------


class TestPostmanCollection:
    def test_postman_json_file_exists(self) -> None:
        assert _POSTMAN_PATH.exists(), f"Missing {_POSTMAN_PATH}; run `make openapi` to generate"

    def test_postman_json_is_valid_json(self) -> None:
        text = _POSTMAN_PATH.read_text()
        data = json.loads(text)
        assert isinstance(data, dict)

    def test_postman_schema_is_v21(self) -> None:
        data = json.loads(_POSTMAN_PATH.read_text())
        schema_url = data.get("info", {}).get("schema", "")
        assert "v2.1" in schema_url, f"Expected Postman v2.1 schema, got: {schema_url}"

    def test_postman_has_info_block(self) -> None:
        data = json.loads(_POSTMAN_PATH.read_text())
        assert "info" in data
        assert "name" in data["info"]

    def test_postman_has_items(self) -> None:
        data = json.loads(_POSTMAN_PATH.read_text())
        items = data.get("item", [])
        assert len(items) > 0, "Postman collection has no requests"

    def test_postman_has_at_least_one_request_per_api_endpoint(self) -> None:
        data = json.loads(_POSTMAN_PATH.read_text())
        # Each item must have a method and a URL
        items = data.get("item", [])
        assert len(items) >= len(_EXPECTED_PATHS), (
            f"Expected at least {len(_EXPECTED_PATHS)} Postman requests, got {len(items)}"
        )
        for item in items:
            req = item.get("request", {})
            assert "method" in req, f"Item {item.get('name')!r} missing 'method'"
            assert "url" in req, f"Item {item.get('name')!r} missing 'url'"

    def test_postman_has_bearer_auth(self) -> None:
        data = json.loads(_POSTMAN_PATH.read_text())
        auth = data.get("auth", {})
        assert auth.get("type") == "bearer", "Postman collection should have bearer auth at root"

    def test_postman_research_item_has_body(self) -> None:
        data = json.loads(_POSTMAN_PATH.read_text())
        items = data.get("item", [])
        post_items = [it for it in items if it.get("request", {}).get("method") == "POST"]
        assert post_items, "No POST requests in Postman collection"
        for item in post_items:
            body = item["request"].get("body")
            assert body is not None, f"POST item {item.get('name')!r} has no body"
            assert body.get("mode") == "raw"
            raw = body.get("raw", "")
            parsed = json.loads(raw)
            assert isinstance(parsed, dict)


# ---------------------------------------------------------------------------
# Swagger UI / /docs endpoint test
# ---------------------------------------------------------------------------


class TestSwaggerUI:
    async def test_docs_endpoint_returns_html(self, client: AsyncClient) -> None:
        resp = await client.get("/docs")
        assert resp.status_code == 200
        content_type = resp.headers.get("content-type", "")
        assert "text/html" in content_type, f"Expected HTML, got: {content_type}"

    async def test_docs_html_contains_swagger(self, client: AsyncClient) -> None:
        resp = await client.get("/docs")
        body = resp.text
        # Swagger UI HTML always contains this string
        assert "swagger" in body.lower(), "GET /docs did not return Swagger UI HTML"

    async def test_redoc_endpoint_returns_html(self, client: AsyncClient) -> None:
        resp = await client.get("/redoc")
        assert resp.status_code == 200
        content_type = resp.headers.get("content-type", "")
        assert "text/html" in content_type

    async def test_openapi_json_endpoint_is_valid(self, client: AsyncClient) -> None:
        resp = await client.get("/openapi.json")
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("openapi", "").startswith("3.")
        assert "paths" in data
