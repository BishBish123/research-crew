"""Auth + rate-limit tests for the public API.

The product surface is intentionally small:

* `RESEARCH_API_TOKEN` env var, when set, gates `/research` and
  `/runs/{id}` behind a bearer token. `/health` stays open so a load
  balancer can probe without provisioning a token.
* When the token is unset, the service runs unauthenticated and the
  lifespan logs a loud warning — the dev-loop default.
* `RESEARCH_RATE_LIMIT_PER_MIN` (default: 10) is a per-IP sliding-window
  counter on POST `/research`; reaching the cap returns 429 with
  `Retry-After`.

These tests poke `app.state.api_token` / `app.state.rate_limiter`
directly because the ASGI transport bypasses the FastAPI lifespan;
that's the same shape the existing API tests use for `app.state.store`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import fakeredis.aioredis as fake_aioredis
import pytest
from httpx import ASGITransport, AsyncClient

from research_crew.api import _RateLimiter, app
from research_crew.store import RedisRunStore


@pytest.fixture
async def auth_client() -> AsyncIterator[AsyncClient]:
    """Client with a known bearer token wired into app.state.

    Mirrors the test_api.py fixture; the only delta is `api_token`.
    """
    fake = fake_aioredis.FakeRedis(decode_responses=True)
    app.state.redis = fake
    app.state.store = RedisRunStore(fake)
    app.state.api_token = "secret-test-token"  # noqa: S105 — test fixture token
    # A generous limit so the auth tests aren't accidentally rate-limited.
    app.state.rate_limiter = _RateLimiter(limit_per_min=1000)
    if hasattr(app.state, "terminal_shadow") and hasattr(app.state.terminal_shadow, "clear"):
        app.state.terminal_shadow.clear()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://t") as c:
        yield c
    app.state.api_token = None
    app.state.rate_limiter = None
    await fake.aclose()


@pytest.fixture
async def open_client() -> AsyncIterator[AsyncClient]:
    """Client with no bearer token configured (the dev path)."""
    fake = fake_aioredis.FakeRedis(decode_responses=True)
    app.state.redis = fake
    app.state.store = RedisRunStore(fake)
    app.state.api_token = None
    app.state.rate_limiter = _RateLimiter(limit_per_min=1000)
    if hasattr(app.state, "terminal_shadow") and hasattr(app.state.terminal_shadow, "clear"):
        app.state.terminal_shadow.clear()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://t") as c:
        yield c
    app.state.rate_limiter = None
    await fake.aclose()


class TestAuthEnforced:
    async def test_research_requires_token_when_set(self, auth_client: AsyncClient) -> None:
        """No header → 401. Correct token → 202."""
        unauth = await auth_client.post("/research", json={"question": "what is python"})
        assert unauth.status_code == 401
        assert "authorization" in unauth.json()["detail"].lower()

        ok = await auth_client.post(
            "/research",
            json={"question": "what is python"},
            headers={"Authorization": "Bearer secret-test-token"},
        )
        assert ok.status_code == 202

    async def test_research_rejects_wrong_token(self, auth_client: AsyncClient) -> None:
        resp = await auth_client.post(
            "/research",
            json={"question": "what is python"},
            headers={"Authorization": "Bearer not-the-real-one"},
        )
        assert resp.status_code == 401
        assert "invalid" in resp.json()["detail"].lower()

    async def test_research_rejects_malformed_header(self, auth_client: AsyncClient) -> None:
        resp = await auth_client.post(
            "/research",
            json={"question": "what is python"},
            headers={"Authorization": "secret-test-token"},  # missing Bearer prefix
        )
        assert resp.status_code == 401

    async def test_get_run_requires_token_when_set(self, auth_client: AsyncClient) -> None:
        resp = await auth_client.get("/runs/anything")
        assert resp.status_code == 401

    async def test_research_health_unauthenticated_with_token_set(
        self, auth_client: AsyncClient
    ) -> None:
        """/health stays open even when the token is configured — load
        balancer probes mustn't need to know the token.
        """
        resp = await auth_client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestAuthOpen:
    async def test_research_open_when_token_unset(self, open_client: AsyncClient) -> None:
        """Token unset (dev path) → unauthenticated POST works."""
        resp = await open_client.post("/research", json={"question": "what is python"})
        assert resp.status_code == 202

    async def test_get_run_open_when_token_unset(self, open_client: AsyncClient) -> None:
        resp = await open_client.get("/runs/anything")
        assert resp.status_code == 404


class TestOpenApiSecurityScheme:
    """The OpenAPI doc must advertise the bearer scheme + a 401 response
    on the protected routes so generated clients know to send the
    `Authorization: Bearer …` header.
    """

    async def test_openapi_advertises_bearer_scheme(self, open_client: AsyncClient) -> None:
        r = await open_client.get("/openapi.json")
        assert r.status_code == 200
        spec = r.json()

        # Security scheme is declared.
        schemes = spec.get("components", {}).get("securitySchemes", {})
        assert any(
            s.get("type") == "http" and s.get("scheme") == "bearer"
            for s in schemes.values()
        ), f"expected an HTTP bearer security scheme; got {schemes}"

        # 401 response is documented on /research and /runs/{run_id}.
        post_research = spec["paths"]["/research"]["post"]
        get_run = spec["paths"]["/runs/{run_id}"]["get"]
        assert "401" in post_research["responses"]
        assert "401" in get_run["responses"]


class TestRateLimit:
    async def test_rate_limit_returns_429(self, open_client: AsyncClient) -> None:
        """The (N+1)th request inside the window returns 429 with
        a Retry-After header; the first N succeed."""
        # Tighten the limiter for this test only.
        app.state.rate_limiter = _RateLimiter(limit_per_min=10)

        for _ in range(10):
            r = await open_client.post("/research", json={"question": "what is python"})
            assert r.status_code == 202
        rejected = await open_client.post("/research", json={"question": "what is python"})
        assert rejected.status_code == 429
        assert rejected.headers.get("Retry-After") is not None
        retry_after = int(rejected.headers["Retry-After"])
        assert retry_after >= 1
        assert "rate limit" in rejected.json()["detail"].lower()


class TestRateLimiterUnit:
    """Direct unit tests on `_RateLimiter` so the math is pinned down
    without an HTTP roundtrip per assertion."""

    def test_first_n_allowed_then_rejected(self) -> None:
        rl = _RateLimiter(limit_per_min=3)
        assert rl.check("1.2.3.4", now=100.0) == (True, 0.0)
        assert rl.check("1.2.3.4", now=100.5) == (True, 0.0)
        assert rl.check("1.2.3.4", now=101.0) == (True, 0.0)
        allowed, retry = rl.check("1.2.3.4", now=101.5)
        assert allowed is False
        # First entry was at t=100, window=60, asked at t=101.5 → wait ~58.5s
        assert 58.0 < retry <= 60.0

    def test_different_ips_have_independent_buckets(self) -> None:
        rl = _RateLimiter(limit_per_min=1)
        assert rl.check("1.1.1.1", now=10.0) == (True, 0.0)
        # Different IP, same window — must still get its first slot.
        assert rl.check("2.2.2.2", now=10.0) == (True, 0.0)
        # Original IP exhausted.
        allowed, _ = rl.check("1.1.1.1", now=10.0)
        assert allowed is False

    def test_window_expiry_replenishes(self) -> None:
        rl = _RateLimiter(limit_per_min=1, window_s=60.0)
        assert rl.check("ip", now=0.0) == (True, 0.0)
        assert rl.check("ip", now=30.0)[0] is False
        # After the window slides past 60s the bucket is empty again.
        assert rl.check("ip", now=61.0) == (True, 0.0)

    def test_invalid_limit_rejected(self) -> None:
        with pytest.raises(ValueError):
            _RateLimiter(limit_per_min=0)
        with pytest.raises(ValueError):
            _RateLimiter(limit_per_min=-1)


class TestTrustedProxyXFF:
    """When the immediate peer is a trusted proxy the rate limiter
    keys on the originating client IP from `X-Forwarded-For`. When
    it is *not* trusted, XFF is ignored — otherwise any caller could
    spoof the header to dodge the per-IP cap.
    """

    async def test_rate_limit_uses_xff_when_proxy_trusted(
        self, open_client: AsyncClient
    ) -> None:
        # A tight limiter so we can prove two distinct client IPs each
        # get their own bucket *through* the same trusted proxy.
        app.state.rate_limiter = _RateLimiter(limit_per_min=1)
        # The test ASGI transport reports request.client.host as
        # something like "127.0.0.1" — pin the peer as trusted so the
        # XFF walk runs.
        app.state.trusted_proxies = {"127.0.0.1"}
        try:
            r1 = await open_client.post(
                "/research",
                json={"question": "what is python"},
                headers={"X-Forwarded-For": "10.0.0.1"},
            )
            assert r1.status_code == 202
            # Same proxy, different client IP — must get its own slot.
            r2 = await open_client.post(
                "/research",
                json={"question": "what is python"},
                headers={"X-Forwarded-For": "10.0.0.2"},
            )
            assert r2.status_code == 202
            # Same client IP again — must hit the cap.
            r3 = await open_client.post(
                "/research",
                json={"question": "what is python"},
                headers={"X-Forwarded-For": "10.0.0.1"},
            )
            assert r3.status_code == 429
        finally:
            app.state.trusted_proxies = set()

    async def test_rate_limit_ignores_xff_when_proxy_not_trusted(
        self, open_client: AsyncClient
    ) -> None:
        """No trusted-proxy config -> all requests share the peer's
        bucket regardless of XFF. A spoofer can't get a fresh slot by
        rotating the header value.
        """
        app.state.rate_limiter = _RateLimiter(limit_per_min=1)
        app.state.trusted_proxies = set()
        r1 = await open_client.post(
            "/research",
            json={"question": "what is python"},
            headers={"X-Forwarded-For": "10.0.0.1"},
        )
        assert r1.status_code == 202
        # Different XFF; same peer -> still the same bucket -> 429.
        r2 = await open_client.post(
            "/research",
            json={"question": "what is python"},
            headers={"X-Forwarded-For": "10.0.0.99"},
        )
        assert r2.status_code == 429
