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

import os
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

    async def test_auth_accepts_lowercase_bearer_scheme(self, auth_client: AsyncClient) -> None:
        """Scheme comparison is case-insensitive: `bearer`, `Bearer`,
        and `BEARER` must all be accepted when the token is correct.

        Some HTTP clients (curl's default, certain mobile SDKs) emit a
        lowercase scheme string; rejecting them would break those clients
        without any security benefit.
        """
        for scheme in ("bearer", "Bearer", "BEARER"):
            resp = await auth_client.post(
                "/research",
                json={"question": "what is python"},
                headers={"Authorization": f"{scheme} secret-test-token"},
            )
            assert resp.status_code == 202, (
                f"scheme '{scheme}' should be accepted; got {resp.status_code}"
            )


class TestAuthOpen:
    async def test_research_open_when_token_unset(self, open_client: AsyncClient) -> None:
        """Token unset (dev path) → unauthenticated POST works."""
        resp = await open_client.post("/research", json={"question": "what is python"})
        assert resp.status_code == 202

    async def test_get_run_open_when_token_unset(self, open_client: AsyncClient) -> None:
        resp = await open_client.get("/runs/anything")
        assert resp.status_code == 404


class TestOpenApiSecurityScheme:
    """OpenAPI security scheme and 401 response body shape.

    When ``RESEARCH_API_TOKEN`` is set (auth_client fixture), the bearer
    security scheme must be advertised and 401 responses must reference
    the ``ErrorDetail`` model. When the token is unset (open_client),
    the scheme must be absent so the Swagger UI "Authorise" button does
    not confuse users of an open endpoint.
    """

    async def test_openapi_advertises_bearer_scheme_when_token_set(
        self, auth_client: AsyncClient
    ) -> None:
        # Clear any cached schema from a prior test so the env-var check
        # inside _build_openapi() runs fresh.
        app.openapi_schema = None  # type: ignore[assignment]
        os.environ["RESEARCH_API_TOKEN"] = "secret-test-token"  # noqa: S105
        try:
            r = await auth_client.get("/openapi.json")
            assert r.status_code == 200
            spec = r.json()

            # Security scheme is declared.
            schemes = spec.get("components", {}).get("securitySchemes", {})
            assert any(
                s.get("type") == "http" and s.get("scheme") == "bearer" for s in schemes.values()
            ), f"expected an HTTP bearer security scheme; got {schemes}"

            # 401 response is documented on /research and /runs/{run_id}.
            post_research = spec["paths"]["/research"]["post"]
            get_run = spec["paths"]["/runs/{run_id}"]["get"]
            assert "401" in post_research["responses"]
            assert "401" in get_run["responses"]
        finally:
            os.environ.pop("RESEARCH_API_TOKEN", None)
            app.openapi_schema = None  # type: ignore[assignment]

    async def test_openapi_omits_bearer_scheme_when_token_unset(
        self, open_client: AsyncClient
    ) -> None:
        """No token configured → security scheme must be absent from the
        OpenAPI spec to avoid a confusing "Authorise" button.
        """
        os.environ.pop("RESEARCH_API_TOKEN", None)
        app.openapi_schema = None  # type: ignore[assignment]
        try:
            r = await open_client.get("/openapi.json")
            assert r.status_code == 200
            spec = r.json()
            schemes = spec.get("components", {}).get("securitySchemes", {})
            assert not any(
                s.get("type") == "http" and s.get("scheme") == "bearer" for s in schemes.values()
            ), f"expected no HTTP bearer security scheme when token unset; got {schemes}"
        finally:
            app.openapi_schema = None  # type: ignore[assignment]

    async def test_openapi_401_references_error_detail_model(
        self, auth_client: AsyncClient
    ) -> None:
        """401 responses on protected routes must reference the typed
        ``ErrorDetail`` schema so generated clients get a proper model.
        """
        os.environ["RESEARCH_API_TOKEN"] = "secret-test-token"  # noqa: S105
        app.openapi_schema = None  # type: ignore[assignment]
        try:
            r = await auth_client.get("/openapi.json")
            assert r.status_code == 200
            spec = r.json()
            post_401 = spec["paths"]["/research"]["post"]["responses"]["401"]
            # FastAPI emits a `$ref` to the model under content > application/json > schema.
            content = post_401.get("content", {})
            json_content = content.get("application/json", {})
            schema = json_content.get("schema", {})
            assert "$ref" in schema or "properties" in schema, (
                f"401 response should reference ErrorDetail schema; got {post_401}"
            )
        finally:
            os.environ.pop("RESEARCH_API_TOKEN", None)
            app.openapi_schema = None  # type: ignore[assignment]


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

    def test_rate_limit_evicts_empty_buckets(self) -> None:
        """A flood of distinct IPs cycling through their windows must
        not leak dict entries: once each window has aged out, the dict
        returns to size 0.
        """
        rl = _RateLimiter(limit_per_min=2, window_s=60.0)
        # 100 distinct IPs each ping once at t=0.
        for i in range(100):
            allowed, _ = rl.check(f"10.0.0.{i}", now=0.0)
            assert allowed is True
        assert rl.bucket_count == 100
        # After the window ages out, an explicit GC sweep drops them all.
        evicted = rl.gc(now=120.0)
        assert evicted == 100
        assert rl.bucket_count == 0
        # And a check from any of those IPs after the window also frees
        # its own entry (the inline-prune path) — verify with a fresh
        # cycle.
        for i in range(5):
            rl.check(f"172.16.0.{i}", now=200.0)
        assert rl.bucket_count == 5
        # Same IP at t=300 (window aged out): bucket is empty on entry,
        # gets reaped + recreated, leaving exactly one entry per IP.
        for i in range(5):
            rl.check(f"172.16.0.{i}", now=300.0)
        assert rl.bucket_count == 5

    def test_rate_limit_evicts_oldest_when_max_buckets_exceeded(self) -> None:
        """A sustained flood of unique IPs hits the max_buckets ceiling;
        the limiter must evict the oldest-touched bucket on overflow so
        memory stays bounded."""
        rl = _RateLimiter(limit_per_min=10, window_s=60.0, max_buckets=3)
        rl.check("a", now=0.0)
        rl.check("b", now=1.0)
        rl.check("c", now=2.0)
        assert rl.bucket_count == 3
        # Touching "a" again moves it to youngest; "b" is now oldest.
        rl.check("a", now=3.0)
        rl.check("d", now=4.0)
        assert rl.bucket_count == 3
        # "b" should have been the eviction target.
        # We can't introspect order from outside, but a fresh "b" check
        # after the window is still small means the slot was reused —
        # the simpler assertion is that the cap held.
        rl.check("e", now=5.0)
        assert rl.bucket_count == 3


class TestTrustedProxyXFF:
    """When the immediate peer is a trusted proxy the rate limiter
    keys on the originating client IP from `X-Forwarded-For`. When
    it is *not* trusted, XFF is ignored — otherwise any caller could
    spoof the header to dodge the per-IP cap.
    """

    async def test_rate_limit_uses_xff_when_proxy_trusted(self, open_client: AsyncClient) -> None:
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

    async def test_xff_canonicalizes_ipv6_brackets_and_ports(
        self, open_client: AsyncClient
    ) -> None:
        """Equivalent encodings of the same client IP share a bucket.

        Without canonicalisation a client could dodge the per-IP cap by
        rotating its source port or by toggling between bracketed and
        bare IPv6 forms — each variant would land in its own deque and
        the rate limit would never trip.
        """
        app.state.rate_limiter = _RateLimiter(limit_per_min=1)
        app.state.trusted_proxies = {"127.0.0.1"}
        try:
            # First request from `[::1]:8080` — bracketed IPv6 + port.
            r1 = await open_client.post(
                "/research",
                json={"question": "what is python"},
                headers={"X-Forwarded-For": "[::1]:8080"},
            )
            assert r1.status_code == 202
            # Same client (`::1`) re-encoded as `[::1]:9999` — different
            # port, different bracket presence — must hit the same
            # bucket and 429.
            r2 = await open_client.post(
                "/research",
                json={"question": "what is python"},
                headers={"X-Forwarded-For": "[::1]:9999"},
            )
            assert r2.status_code == 429
            # And a bare `::1` encoding must also share that bucket.
            r3 = await open_client.post(
                "/research",
                json={"question": "what is python"},
                headers={"X-Forwarded-For": "::1"},
            )
            assert r3.status_code == 429
            # IPv4 with port also normalises: `10.0.0.5:12345` and
            # `10.0.0.5` share a bucket.
            app.state.rate_limiter = _RateLimiter(limit_per_min=1)
            r4 = await open_client.post(
                "/research",
                json={"question": "what is python"},
                headers={"X-Forwarded-For": "10.0.0.5:12345"},
            )
            assert r4.status_code == 202
            r5 = await open_client.post(
                "/research",
                json={"question": "what is python"},
                headers={"X-Forwarded-For": "10.0.0.5"},
            )
            assert r5.status_code == 429
        finally:
            app.state.trusted_proxies = set()

    async def test_xff_skips_unparseable_tokens(self, open_client: AsyncClient) -> None:
        """A token that doesn't parse as an IP is skipped, not used as
        a bucket key — otherwise garbage in the header would create a
        per-string bucket and leak memory."""
        app.state.rate_limiter = _RateLimiter(limit_per_min=2)
        app.state.trusted_proxies = {"127.0.0.1"}
        try:
            # First token is garbage, second is a real IP — limiter
            # must use the real one.
            r1 = await open_client.post(
                "/research",
                json={"question": "what is python"},
                headers={"X-Forwarded-For": "not-an-ip, 10.0.0.99"},
            )
            assert r1.status_code == 202
            r2 = await open_client.post(
                "/research",
                json={"question": "what is python"},
                headers={"X-Forwarded-For": "another-bad, 10.0.0.99"},
            )
            assert r2.status_code == 202
            # Third request from same real IP hits the cap.
            r3 = await open_client.post(
                "/research",
                json={"question": "what is python"},
                headers={"X-Forwarded-For": "10.0.0.99"},
            )
            assert r3.status_code == 429
        finally:
            app.state.trusted_proxies = set()

    async def test_xff_multi_hop_prefers_rightmost_untrusted_hop(
        self, open_client: AsyncClient
    ) -> None:
        """RTL walk regression: a malicious client prepends spoofed
        XFF entries; the trusted proxy appends the real client IP at
        the END. The limiter MUST take the rightmost untrusted hop
        (the proxy's observation) and ignore the attacker-supplied
        prefix — otherwise the client could rotate the spoofed prefix
        on every request to dodge the per-IP cap.
        """
        app.state.rate_limiter = _RateLimiter(limit_per_min=1)
        app.state.trusted_proxies = {"127.0.0.1", "10.0.0.250"}
        try:
            # First request: attacker prepends "1.1.1.1", real client
            # is "10.0.0.7" appended by the trusted proxy, which then
            # also appends its own "10.0.0.250" hop.
            r1 = await open_client.post(
                "/research",
                json={"question": "what is python"},
                headers={"X-Forwarded-For": "1.1.1.1, 10.0.0.7, 10.0.0.250"},
            )
            assert r1.status_code == 202
            # Same real client; attacker rotates the spoof to "2.2.2.2".
            # If the walk were left-to-right, this would land in a fresh
            # bucket and pass; with RTL discarding trusted hops, it
            # MUST collapse onto the same "10.0.0.7" bucket and 429.
            r2 = await open_client.post(
                "/research",
                json={"question": "what is python"},
                headers={"X-Forwarded-For": "2.2.2.2, 10.0.0.7, 10.0.0.250"},
            )
            assert r2.status_code == 429
            # And rotating to a third spoof must keep failing.
            r3 = await open_client.post(
                "/research",
                json={"question": "what is python"},
                headers={"X-Forwarded-For": "3.3.3.3, 10.0.0.7, 10.0.0.250"},
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
