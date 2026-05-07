"""Mock API server package for real-mode adapter integration tests.

Exposes ``create_mock_app()`` — a FastAPI application that mounts documented
response shapes for Tavily, Brave, Exa, and Langfuse.  Each handler:

- Validates that the auth header is present (any non-empty value passes).
- Returns the fixture JSON that mirrors the real provider response shape.
- Supports ``?force=500`` to simulate one 5xx then success (retry path).
"""

from tests.mock_servers.app import create_mock_app

__all__ = ["create_mock_app"]
