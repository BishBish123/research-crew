"""Export OpenAPI spec and Postman collection from the FastAPI app.

Usage:
    uv run python scripts/export_openapi.py

Writes:
    docs/openapi.json           -- OpenAPI 3.x spec (pretty-printed)
    docs/postman_collection.json -- Postman v2.1 collection

Only stdlib JSON used for transformations; no third-party converters.
"""

from __future__ import annotations

import json
import os
import re
import sys
import uuid
from pathlib import Path

# Ensure src/ is importable when run from any cwd.
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

# Suppress the loud "auth disabled" warning during schema generation.
os.environ.setdefault("RESEARCH_DEV_MODE", "1")

from research_crew.api import app as _api_app  # noqa: E402, I001


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _openapi_type_to_postman_example(schema: dict) -> object:
    """Return a placeholder value for a JSON schema node."""
    t = schema.get("type")
    _scalar: dict[str, object] = {"integer": 0, "number": 0.0, "boolean": True}
    if t in _scalar:
        return _scalar[t]
    if t == "string":
        return "2024-01-01T00:00:00Z" if schema.get("format") == "date-time" else "string"
    if t == "array":
        return [_openapi_type_to_postman_example(schema.get("items", {}))]
    if t == "object":
        props = schema.get("properties", {})
        return {k: _openapi_type_to_postman_example(v) for k, v in props.items()}
    return None


def _resolve_ref(ref: str, components: dict) -> dict:
    """Resolve a $ref like '#/components/schemas/Foo'."""
    parts = ref.lstrip("#/").split("/")
    node: dict = components  # type: ignore[assignment]
    for part in parts[1:]:  # skip 'components'
        node = node[part]
    return node


def _build_example_body(
    schema: dict | None, components: dict, visited: set[str] | None = None
) -> object:
    """Recursively build an example request body from an OpenAPI schema node."""
    if schema is None:
        return {}
    if visited is None:
        visited = set()
    if "$ref" in schema:
        ref = schema["$ref"]
        if ref in visited:
            return {}
        visited = visited | {ref}
        schema = _resolve_ref(ref, {"schemas": components.get("schemas", {})})
    props = schema.get("properties", {})
    required = set(schema.get("required", []))
    if not props:
        return _openapi_type_to_postman_example(schema)
    return {
        k: _build_example_body(v, components, visited)
        for k, v in props.items()
        if k in required or not schema.get("additionalProperties", True)
    } or {k: _build_example_body(v, components, visited) for k, v in props.items()}


def _postman_url(base_url: str, path: str) -> dict:
    """Convert an OpenAPI path like '/runs/{run_id}' to a Postman URL object."""
    # Replace {param} with :param (Postman style)
    pm_path = path
    variables = []
    for match in re.finditer(r"\{(\w+)\}", path):
        name = match.group(1)
        pm_path = pm_path.replace(f"{{{name}}}", f":{name}")
        variables.append({"key": name, "value": f"<{name}>", "description": ""})

    host = base_url.rstrip("/")
    raw = host + pm_path
    # Split path segments
    segments = [s for s in pm_path.split("/") if s]

    url: dict = {
        "raw": raw,
        "host": [host],
        "path": segments,
    }
    if variables:
        url["variable"] = variables
    return url


def openapi_to_postman(spec: dict, base_url: str = "http://localhost:8000") -> dict:
    """Build a minimal Postman v2.1 collection from an OpenAPI 3.x spec.

    One request per (path, method) pair. Request bodies are inferred from
    the requestBody schema. A Bearer auth template is included at the
    collection level; individual requests inherit it.
    """
    info = spec.get("info", {})
    components = spec.get("components", {})

    collection: dict = {
        "info": {
            "_postman_id": str(uuid.uuid4()),
            "name": info.get("title", "API"),
            "description": info.get("description", ""),
            "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json",
        },
        "auth": {
            "type": "bearer",
            "bearer": [{"key": "token", "value": "{{API_TOKEN}}", "type": "string"}],
        },
        "variable": [
            {"key": "base_url", "value": base_url, "type": "string"},
            {"key": "API_TOKEN", "value": "", "type": "string"},
        ],
        "item": [],
    }

    paths = spec.get("paths", {})
    for path, methods in paths.items():
        if not isinstance(methods, dict):
            continue
        for method, operation in methods.items():
            if method.lower() not in {"get", "post", "put", "patch", "delete", "head", "options"}:
                continue
            if not isinstance(operation, dict):
                continue

            name = (
                operation.get("summary")
                or operation.get("operationId")
                or f"{method.upper()} {path}"
            )

            # Build request body if present
            body_obj: dict | None = None
            req_body = operation.get("requestBody", {})
            if req_body:
                content = req_body.get("content", {})
                json_content = content.get("application/json", {})
                body_schema = json_content.get("schema")
                if body_schema:
                    example_data = _build_example_body(body_schema, components)
                    body_obj = {
                        "mode": "raw",
                        "raw": json.dumps(example_data, indent=2),
                        "options": {"raw": {"language": "json"}},
                    }

            # Build headers
            headers = [{"key": "Content-Type", "value": "application/json"}]

            request: dict = {
                "method": method.upper(),
                "header": headers,
                "url": _postman_url("{{base_url}}", path),
                "description": operation.get("description", ""),
            }
            if body_obj:
                request["body"] = body_obj

            item: dict = {
                "name": name,
                "request": request,
                "response": [],
            }
            collection["item"].append(item)

    return collection


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    schema = _api_app.openapi()

    docs_dir = _ROOT / "docs"
    docs_dir.mkdir(exist_ok=True)

    # Write OpenAPI spec
    openapi_path = docs_dir / "openapi.json"
    openapi_path.write_text(json.dumps(schema, indent=2) + "\n")
    print(f"Wrote {openapi_path}")

    # Write Postman collection
    postman = openapi_to_postman(schema)
    postman_path = docs_dir / "postman_collection.json"
    postman_path.write_text(json.dumps(postman, indent=2) + "\n")
    print(f"Wrote {postman_path}")

    paths_count = len(schema.get("paths", {}))
    schemas_count = len(schema.get("components", {}).get("schemas", {}))
    items_count = len(postman["item"])
    print(f"OpenAPI: {paths_count} paths, {schemas_count} schemas")
    print(f"Postman: {items_count} requests")


if __name__ == "__main__":
    main()
