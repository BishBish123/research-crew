"""FastAPI service: POST /research, GET /runs/{id}, health probe."""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import json
import os
import re
import secrets
import time
import uuid
from collections import OrderedDict, deque
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime

import redis.asyncio as aioredis
import structlog
import typer
import uvicorn
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from redis.exceptions import RedisError

from research_crew.agents import default_agents
from research_crew.errors import StoreUnavailableError
from research_crew.models import (
    ResearchReport,
    ResearchRequest,
    RunStatus,
    StepStatus,
)
from research_crew.store import RedisRunStore, RunStore, migrate_run_blob
from research_crew.synthesizer import StitchSynthesizer
from research_crew.workflow import WorkflowEngine

_log = structlog.get_logger(__name__)


# Default ceiling on the in-process terminal-state shadow. The shadow only
# accumulates entries while the run store is unavailable, so 10k is many
# orders of magnitude more than a typical degraded-mode window. Operators
# can override via the RESEARCH_SHADOW_MAX env var.
_DEFAULT_SHADOW_MAX = 10_000

# Default per-IP rate-limit budget for POST /research. The cost of a
# research run is asymmetric (one HTTP call → 5 fan-out agents + Redis
# writes), so a conservative default protects the upstream search APIs
# from a single misbehaving client. Operators can override via
# RESEARCH_RATE_LIMIT_PER_MIN.
_DEFAULT_RATE_LIMIT_PER_MIN = 10
_RATE_LIMIT_WINDOW_S = 60.0

# Hard ceiling on the number of distinct IPs the limiter tracks at any
# one time. Without this, a sustained stream of unique source IPs (NAT
# churn, scanners, DDoS) would grow the dict unboundedly even after
# every per-IP window has aged out. On overflow we evict the oldest-
# touched bucket — see ``_RateLimiter.check``.
_DEFAULT_MAX_BUCKETS = 10_000

# Heartbeat cadence and staleness threshold for orphan reconciliation.
# The background task refreshes ``RunStatus.heartbeat_at`` every
# ``_HEARTBEAT_INTERVAL_S`` seconds while the run is RUNNING. The
# lifespan reconciler only flips a RUNNING blob to FAILED if its
# heartbeat is older than ``_DEFAULT_STALE_HEARTBEAT_S``; this prevents
# a freshly-started instance from killing peer instances' still-live
# work in a multi-process deployment. Operators can override via
# ``RESEARCH_HEARTBEAT_STALE_S``.
_HEARTBEAT_INTERVAL_S = 30.0
_DEFAULT_STALE_HEARTBEAT_S = 120

# Whitespace-tolerant pre-filter for the JSON `state: running` field.
# A naive substring check on the literal `"state":"running"` misses
# blobs serialised with default `json.dumps` spacing (which inserts a
# space after the colon), silently undercounting RUNNING records. We
# tolerate any inter-token whitespace before parsing, then re-check on
# the parsed payload to avoid false positives in nested fields.
_RUNNING_STATE_RE = re.compile(r'"state"\s*:\s*"running"')

# Process-wide identifier stamped on every RUNNING run claimed by this
# worker. A peer instance reading a foreign ``owner_id`` knows the run
# is not its own to reconcile until the heartbeat goes stale.
_WORKER_ID = uuid.uuid4().hex


class _RateLimiter:
    """Per-IP sliding-window counter for POST /research.

    The window is a `deque` of timestamps newer than `now - 60s`. When
    the window already holds `limit` entries, the next request is
    rejected with a Retry-After hint computed from the oldest pending
    timestamp.

    Memory hygiene: ``_buckets`` is an ``OrderedDict`` keyed by IP. We
    rely on insertion-order semantics so the oldest-touched bucket can
    be evicted in O(1) when ``len > max_buckets``. After expired
    timestamps drain a bucket to empty, we delete the bucket entirely
    so a flood of distinct one-shot IPs cannot leak memory across
    windows. ``move_to_end`` is called on every access so "oldest" is
    measured by last-touched time, not insertion time.

    Concurrency note: all reads/writes happen on the FastAPI event loop,
    so dict/deque mutations are not interleaved at the bytecode level.
    No explicit lock is needed.
    """

    def __init__(
        self,
        limit_per_min: int,
        window_s: float = _RATE_LIMIT_WINDOW_S,
        max_buckets: int = _DEFAULT_MAX_BUCKETS,
    ) -> None:
        if limit_per_min <= 0:
            raise ValueError("limit_per_min must be positive")
        if max_buckets <= 0:
            raise ValueError("max_buckets must be positive")
        self._limit = limit_per_min
        self._window = window_s
        self._max_buckets = max_buckets
        self._buckets: OrderedDict[str, deque[float]] = OrderedDict()

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def bucket_count(self) -> int:
        """Test-visible size accessor (no public surface area otherwise)."""
        return len(self._buckets)

    def check(self, ip: str, *, now: float | None = None) -> tuple[bool, float]:
        """Record a request for `ip`. Returns ``(allowed, retry_after_s)``.

        On allow, ``retry_after_s`` is 0. On reject, it's the number of
        seconds the caller would need to wait before the oldest in-window
        timestamp ages out.

        Side effect: empty buckets are pruned, and on insert beyond
        ``max_buckets`` the oldest-touched bucket is evicted.
        """
        ts = time.monotonic() if now is None else now
        cutoff = ts - self._window
        bucket = self._buckets.get(ip)
        if bucket is None:
            bucket = deque()
            # Enforce the bucket-count ceiling *before* inserting so a
            # sustained unique-IP flood can't temporarily blow the cap.
            while len(self._buckets) >= self._max_buckets:
                self._buckets.popitem(last=False)
            self._buckets[ip] = bucket
        else:
            # Touching an existing bucket bumps it to the youngest
            # position so eviction targets truly idle entries.
            self._buckets.move_to_end(ip)
        # Drop expired entries off the front before counting.
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        if not bucket:
            # Window fully aged out; the bucket itself can go too. The
            # caller's request will re-create it below if it's allowed.
            del self._buckets[ip]
            bucket = deque()
            while len(self._buckets) >= self._max_buckets:
                self._buckets.popitem(last=False)
            self._buckets[ip] = bucket
        if len(bucket) >= self._limit:
            retry_after = max(0.0, bucket[0] + self._window - ts)
            return False, retry_after
        bucket.append(ts)
        return True, 0.0

    def gc(self, *, now: float | None = None) -> int:
        """Drop empty / fully-aged-out buckets. Returns number evicted.

        Only the limiter's own ``check`` calls trigger inline pruning;
        a bucket whose owning IP never returns to the service stays
        non-empty until its window ages out — but it can also be
        proactively GC'd by tests or a future periodic task.
        """
        ts = time.monotonic() if now is None else now
        cutoff = ts - self._window
        evicted = 0
        for ip in list(self._buckets):
            bucket = self._buckets[ip]
            while bucket and bucket[0] < cutoff:
                bucket.popleft()
            if not bucket:
                del self._buckets[ip]
                evicted += 1
        return evicted

    def reset(self) -> None:
        """Test-only hook: drop all buckets."""
        self._buckets.clear()


# Header name carrying the bearer token. The actual auth check is
# manual (`_require_auth`) so we can keep the dev-mode "no token set →
# unauthenticated" path without surfacing it as a special case in the
# OpenAPI doc. We *also* declare the FastAPI `HTTPBearer` security
# scheme as a dependency so /openapi.json advertises the bearer
# requirement when ``RESEARCH_API_TOKEN`` is configured. When the token
# is unset, a custom OpenAPI override strips the security declarations
# from the spec so the "Authorise" button doesn't appear for an open
# endpoint.
_AUTH_HEADER = "Authorization"
_BEARER_PREFIX = "Bearer "
_bearer_scheme = HTTPBearer(auto_error=False, description="Bearer token issued out-of-band.")


class _TerminalShadow:
    """Bounded FIFO shadow of terminal RunStatuses.

    Used only on the bg-task store-outage path. Behaves like a `dict` for
    the read sites but evicts the oldest entry on insert once `max_size`
    is exceeded — without this cap, a long store outage with continuous
    submits would grow the process heap unboundedly.

    Concurrency note: shadow reads (GET handler) and shadow writes (bg
    task) both run in the same event loop, so dict mutations are not
    interleaved at the bytecode level. No explicit lock is needed.
    """

    def __init__(self, max_size: int = _DEFAULT_SHADOW_MAX) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        self._items: OrderedDict[str, RunStatus] = OrderedDict()
        self._max_size = max_size

    def __setitem__(self, run_id: str, run: RunStatus) -> None:
        if run_id in self._items:
            # Refresh insertion order so the newest write is the youngest.
            self._items.move_to_end(run_id)
        self._items[run_id] = run
        while len(self._items) > self._max_size:
            evicted_id, _ = self._items.popitem(last=False)
            _log.warning(
                "api.terminal_shadow_evicted",
                run_id=evicted_id,
                max_size=self._max_size,
                reason="shadow_full",
            )

    def __getitem__(self, run_id: str) -> RunStatus:
        return self._items[run_id]

    def __contains__(self, run_id: object) -> bool:
        return run_id in self._items

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)

    def get(self, run_id: str, default: RunStatus | None = None) -> RunStatus | None:
        return self._items.get(run_id, default)

    def pop(self, run_id: str, default: RunStatus | None = None) -> RunStatus | None:
        return self._items.pop(run_id, default)

    def clear(self) -> None:
        self._items.clear()


def _redis_url() -> str:
    return os.environ.get("REDIS_URL", "redis://localhost:6379/0")


@contextlib.asynccontextmanager
async def _lifespan(app_: FastAPI) -> AsyncIterator[None]:
    """Open the Redis client on startup; close it cleanly on shutdown.

    Tests can override `app_.state.redis` / `app_.state.store` *before*
    issuing requests because the lifespan sees ``getattr(..., None)`` and
    only initialises what's missing.

    Also initialises the in-process terminal-state shadow cache: when the
    store rejects a terminal-state write from the background task (the
    whole point of that path is "store is down"), the bg task stashes the
    final RunStatus here so a subsequent `GET /runs/{run_id}` can still
    surface a terminal state instead of "running forever".
    """
    if getattr(app_.state, "redis", None) is None:
        client = aioredis.from_url(_redis_url(), decode_responses=True)
        app_.state.redis = client
        # Per-deployment key prefix lets multiple environments share
        # one Redis without cross-talk; defaults to "research" via the
        # store constructor.
        prefix = os.environ.get("RESEARCH_REDIS_PREFIX", "").strip()
        app_.state.store = RedisRunStore(client, prefix=prefix) if prefix else RedisRunStore(client)
    if getattr(app_.state, "terminal_shadow", None) is None:
        app_.state.terminal_shadow = _TerminalShadow(
            max_size=int(os.environ.get("RESEARCH_SHADOW_MAX", _DEFAULT_SHADOW_MAX))
        )
    # Auth token — when unset, run open and log a loud DEV-ONLY warning
    # so operators don't ship to prod accidentally. Local dev loops can
    # opt out of the WARNING noise by exporting ``RESEARCH_DEV_MODE=1``,
    # which demotes the same event to INFO; the message body is identical
    # so a log scrape still surfaces it on the dev path.
    token = os.environ.get("RESEARCH_API_TOKEN")
    if not token:
        _auth_disabled_msg = (
            "RESEARCH_API_TOKEN not set; running unauthenticated — DEV ONLY; "
            "set RESEARCH_API_TOKEN=<token> to enable"
        )
        if _is_dev_mode():
            _log.info(
                "api.auth_disabled",
                message=_auth_disabled_msg,
                dev_mode=True,
            )
        else:
            _log.warning(
                "api.auth_disabled",
                message=_auth_disabled_msg,
            )
    app_.state.api_token = token or None
    # Rate limiter is always wired up; if RESEARCH_RATE_LIMIT_PER_MIN
    # is unset we use the conservative default. Recreate per lifespan
    # so `make api && CTRL+C && make api` doesn't carry counters across
    # the (logical) restart.
    rate_per_min = int(os.environ.get("RESEARCH_RATE_LIMIT_PER_MIN", _DEFAULT_RATE_LIMIT_PER_MIN))
    app_.state.rate_limiter = _RateLimiter(limit_per_min=rate_per_min)
    # Trusted-proxy set for X-Forwarded-For honouring. CSV of bare IPs
    # (CIDR is intentionally NOT supported here — keeping the surface
    # small until there's a concrete need; an operator running on AWS
    # ALB enumerates the front-end IPs in their config). Empty default
    # means XFF is ignored, which is the safe behaviour for direct
    # exposure.
    raw_proxies = os.environ.get("RESEARCH_TRUSTED_PROXIES", "").strip()
    app_.state.trusted_proxies = (
        {p.strip() for p in raw_proxies.split(",") if p.strip()} if raw_proxies else set()
    )
    # Best-effort reconciliation of any RUNNING runs left behind by a
    # previous process. Background execution is bound to the accepting
    # instance, so a process restart strands those runs in RUNNING with
    # no worker — without reconciliation, callers would poll forever.
    # A durable worker queue is the proper fix; this is the smallest
    # defensible interim. See README "Limitations".
    await _reconcile_orphan_runs(app_)
    try:
        yield
    finally:
        if getattr(app_.state, "redis", None) is not None:
            with contextlib.suppress(Exception):
                await app_.state.redis.aclose()


async def _reconcile_orphan_runs(app_: FastAPI) -> None:
    """Mark abandoned RUNNING run records FAILED on startup.

    Background execution is in-process, so a previous process that
    crashed or was killed mid-run leaves RUNNING records in Redis with
    no worker behind them. The reconciler distinguishes "abandoned"
    from "still being executed by a peer instance" via the run's
    ``heartbeat_at`` field: only runs whose heartbeat is older than
    ``_stale_heartbeat_seconds()`` (default 120s, env-overridable) are
    flipped. Active peers refresh the heartbeat every
    ``_HEARTBEAT_INTERVAL_S`` seconds, so they will never look stale to
    a sibling lifespan running concurrently.

    Legacy blobs without ``heartbeat_at`` (older deploys, or runs
    written before the heartbeat field landed) are also flipped, since
    we have no other signal to tell live work from a true orphan.

    This is best-effort: we swallow any scan/parse error so a transient
    Redis hiccup at startup doesn't block the service from coming up.
    A durable worker queue (Inngest, Temporal, Redis Streams consumer
    group) is the right long-term fix; tracked under "Limitations" in
    README.
    """
    store = getattr(app_.state, "store", None)
    redis_client = getattr(app_.state, "redis", None)
    if redis_client is None or not isinstance(store, RedisRunStore):
        return
    pattern = f"{store.prefix}:run:*"
    stale_after_s = _stale_heartbeat_seconds()
    now = datetime.now(UTC)
    reconciled = 0
    skipped_live = 0
    try:
        # SCAN to avoid blocking the loop on a large keyspace; the same
        # shape /health uses so the cost profile is already understood.
        async for key in redis_client.scan_iter(match=pattern, count=200):
            if key.endswith(":steps"):
                continue
            outcome = await _reconcile_one(
                redis_client, store, key, now=now, stale_after_s=stale_after_s
            )
            if outcome == "reconciled":
                reconciled += 1
            elif outcome == "skipped_live":
                skipped_live += 1
    except Exception as exc:
        _log.warning("api.reconcile_scan_failed", error=str(exc))
        return
    if reconciled or skipped_live:
        _log.warning(
            "api.reconciled_orphan_runs",
            reconciled=reconciled,
            skipped_live=skipped_live,
        )


async def _reconcile_one(
    redis_client: aioredis.Redis,
    store: RedisRunStore,
    key: str,
    *,
    now: datetime,
    stale_after_s: int,
) -> str:
    """Process a single ``{prefix}:run:*`` key for the lifespan reconciler.

    Returns ``"reconciled"`` on flip, ``"skipped_live"`` if a fresh
    heartbeat kept the run alive, ``"noop"`` for everything else
    (terminal, parse error, transient outage). Caller logs the rollup.
    """
    run = await _load_running_run(redis_client, key)
    if run is None:
        return "noop"
    abandon_reason = _abandonment_reason(run, now=now, stale_after_s=stale_after_s)
    if abandon_reason is None:
        return "skipped_live"
    # Capture the heartbeat observed *before* the flip so the CAS can
    # verify nothing changed between our read and our write.
    observed_heartbeat = run.heartbeat_at
    run.state = StepStatus.FAILED
    run.finished_at = now
    run.abandoned_at = now
    run.error = abandon_reason
    try:
        swapped = await store.cas_reconcile_run(
            run.run_id,
            expected_state=StepStatus.RUNNING,
            expected_heartbeat_at=observed_heartbeat,
            new_run=run,
        )
    except Exception as exc:
        _log.warning("api.reconcile_put_failed", run_id=run.run_id, error=str(exc))
        return "noop"
    if not swapped:
        _log.info(
            "api.reconcile_skipped_concurrent_change",
            run_id=run.run_id,
            reason="skipped: concurrent change",
        )
        return "noop"
    return "reconciled"


async def _load_running_run(redis_client: aioredis.Redis, key: str) -> RunStatus | None:
    """Read + parse a RUNNING blob; return ``None`` for any reason it
    isn't a candidate (transient read error, missing, terminal,
    unparseable, version too new, or state not actually RUNNING).

    Routes the JSON through ``_migrate_run_blob`` so a pre-versioning
    blob still parses and a newer-version blob is logged + skipped.
    """
    try:
        raw = await redis_client.get(key)
    except Exception as exc:  # transient — log and skip this key
        _log.warning("api.reconcile_get_failed", key=key, error=str(exc))
        return None
    if raw is None or not _RUNNING_STATE_RE.search(raw):
        return None
    try:
        payload = json.loads(raw)
    except Exception as exc:
        _log.warning("api.reconcile_parse_failed", key=key, error=str(exc))
        return None
    migrated = migrate_run_blob(payload, key=key)
    if migrated is None:
        return None
    try:
        run = RunStatus.model_validate(migrated)
    except Exception as exc:
        _log.warning("api.reconcile_parse_failed", key=key, error=str(exc))
        return None
    return run if run.state is StepStatus.RUNNING else None


def _abandonment_reason(run: RunStatus, *, now: datetime, stale_after_s: int) -> str | None:
    """Decide whether a RUNNING run should be flipped to FAILED.

    Returns the human-readable abandonment reason, or ``None`` if the
    run is still live (fresh heartbeat) and must not be touched.
    """
    if run.heartbeat_at is None:
        # Legacy / pre-heartbeat blobs: no signal to tell live work
        # from a true orphan, so treat as abandoned.
        return "abandoned by previous process"
    age_s = (now - run.heartbeat_at).total_seconds()
    if age_s <= stale_after_s:
        _log.info(
            "api.reconcile_skipped_live",
            run_id=run.run_id,
            owner_id=run.owner_id,
            heartbeat_age_s=age_s,
            stale_after_s=stale_after_s,
        )
        return None
    return f"abandoned: no heartbeat for {int(age_s)}s (threshold {stale_after_s}s)"


def _is_dev_mode() -> bool:
    """``RESEARCH_DEV_MODE`` truthiness check.

    Accepts the usual truthy strings (``1``, ``true``, ``yes``, ``on``)
    case-insensitively; everything else (unset, ``0``, ``false``, garbage)
    is false. Kept narrow on purpose — this only gates the auth-disabled
    log level, not actual auth enforcement.
    """
    raw = os.environ.get("RESEARCH_DEV_MODE", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _stale_heartbeat_seconds() -> int:
    """Resolve the heartbeat staleness threshold from the environment.

    Falls back to ``_DEFAULT_STALE_HEARTBEAT_S`` on missing / unparseable
    / non-positive values so an operator typo can't silently disable
    reconciliation.
    """
    raw = os.environ.get("RESEARCH_HEARTBEAT_STALE_S")
    if not raw:
        return _DEFAULT_STALE_HEARTBEAT_S
    try:
        parsed = int(raw)
    except ValueError:
        return _DEFAULT_STALE_HEARTBEAT_S
    return parsed if parsed > 0 else _DEFAULT_STALE_HEARTBEAT_S


app = FastAPI(
    title="research-crew",
    version="0.1.0",
    summary="Concurrent multi-agent research service.",
    description=(
        "5 specialist agents fan out in parallel via a Redis-backed durable "
        "workflow with idempotent step semantics + bounded exponential-backoff "
        "retries; results merge through a Synthesizer into a single "
        "citation-grounded report."
    ),
    lifespan=_lifespan,
)


def _build_openapi() -> dict[str, object]:
    """Custom OpenAPI schema builder.

    When ``RESEARCH_API_TOKEN`` is not set the service runs
    unauthenticated; in that case we strip the bearer security scheme
    and per-route ``security`` declarations from the generated spec so
    the Swagger UI "Authorise" button doesn't appear and generated
    clients don't add an empty ``Authorization`` header. The routes'
    ``responses`` dict still lists 401 (our ``_AUTH_RESPONSES``
    reference includes it), which is accurate — if an operator later
    sets the token without redeploying the service will start enforcing
    it.
    """
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title=app.title,
        version=app.version,
        summary=app.summary,
        description=app.description,
        routes=app.routes,
    )
    if not os.environ.get("RESEARCH_API_TOKEN"):
        # Strip bearer security scheme.
        schema.get("components", {}).pop("securitySchemes", None)
        # Strip per-route security declarations.
        for _path, methods in schema.get("paths", {}).items():
            for _method, op in methods.items():
                if isinstance(op, dict):
                    op.pop("security", None)
    app.openapi_schema = schema
    return schema


app.openapi = _build_openapi  # type: ignore[method-assign]


@app.exception_handler(StoreUnavailableError)
async def _store_unavailable_handler(request: Request, exc: StoreUnavailableError) -> JSONResponse:
    """Map any leaked StoreUnavailableError to a typed 503."""
    _log.warning("api.store_unavailable", path=request.url.path, error=str(exc))
    return JSONResponse(status_code=503, content={"detail": f"run store unavailable: {exc}"})


def _require_auth(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None,
) -> None:
    """Enforce bearer-token auth on the protected routes.

    Uses the pre-parsed ``HTTPAuthorizationCredentials`` from the
    ``HTTPBearer`` dependency so scheme extraction is handled by FastAPI
    rather than a manual string split.  The scheme is compared
    case-insensitively so ``bearer``, ``Bearer``, and ``BEARER`` are
    all accepted.

    No-ops when ``app.state.api_token`` is unset (the dev path) — the
    lifespan logs a loud warning in that case so unauthenticated mode
    is loud, not silent.
    """
    expected: str | None = getattr(request.app.state, "api_token", None)
    if not expected:
        return
    if credentials is None or credentials.scheme.lower() != "bearer":
        # `HTTPBearer(auto_error=False)` returns ``None`` for both a
        # missing header and a parsable-but-wrong-scheme header
        # (e.g. ``Basic ...``). Inspect the raw header so we can return
        # a more specific message when the scheme is the issue. Only
        # treat it as a wrong-scheme error when the header actually has
        # a "<scheme> <token>" shape with a non-Bearer scheme; a header
        # with no whitespace separator is malformed, not "wrong scheme".
        raw_header = request.headers.get(_AUTH_HEADER, "").strip()
        parts = raw_header.split(" ", 1) if raw_header else []
        if len(parts) == 2 and parts[0].lower() != "bearer":
            raise HTTPException(
                status_code=401, detail="expected Bearer scheme in Authorization header"
            )
        raise HTTPException(status_code=401, detail="missing or malformed Authorization header")
    # Constant-time compare so a remote caller can't infer the token via
    # response-timing across many guesses. `compare_digest` short-circuits
    # only on length mismatch, which is the documented limit.
    if not secrets.compare_digest(credentials.credentials, expected):
        raise HTTPException(status_code=401, detail="invalid bearer token")


def _client_ip(request: Request) -> str:
    """Best-effort client IP for the rate-limit key.

    Honour the ``X-Forwarded-For`` header only when the immediate peer
    (``request.client.host``) is in the operator-configured trusted-
    proxy set (``RESEARCH_TRUSTED_PROXIES`` CSV). Without that gate a
    raw client could spoof XFF to evade per-IP limits; with it, real
    deployments behind ALB/nginx/Cloudflare see the originating
    client IP.

    Each XFF hop is canonicalised through ``_canonical_ip`` before
    bucket lookup so equivalent representations of the same client
    (``10.0.0.1`` vs ``10.0.0.1:12345``, ``::1`` vs ``[::1]:8080``)
    share a bucket instead of fanning into N independent ones — the
    latter would let an attacker dodge the per-IP cap by rotating the
    port suffix.

    The ASGI transport used in tests doesn't always populate
    ``request.client``; fall back to a sentinel so the test path still
    exercises the limiter (one bucket for everyone) rather than
    short-circuiting on ``None``.
    """
    peer = request.client.host if request.client is not None and request.client.host else None
    trusted = getattr(request.app.state, "trusted_proxies", None) or set()
    if peer is not None and peer in trusted:
        # Walk the XFF chain from RIGHT to LEFT. Proxies append the
        # peer they observed to the END of the header, so the
        # rightmost hop is the most recent — and therefore the most
        # trusted — address. We discard trusted-proxy hops as we walk
        # leftward and take the first untrusted hop adjacent to the
        # trusted chain as the real client. A left-to-right walk would
        # let a raw client prepend spoofed entries (the request arrives
        # already containing ``X-Forwarded-For: 1.2.3.4``; the proxy
        # appends its own observation) and the leftmost-first reader
        # would happily accept the attacker's value, defeating the
        # per-IP rate limit. RFC 7239 `Forwarded: for=...` is the
        # modern alias; XFF is the de-facto standard most LBs emit.
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            for raw in reversed(forwarded.split(",")):
                candidate = _canonical_ip(raw)
                if candidate is None:
                    # Garbage hop — keep walking so a single malformed
                    # entry doesn't make us fall through to ``peer``
                    # (which would wrongly bucket every client behind
                    # the trusted proxy together).
                    continue
                if candidate in trusted:
                    continue
                return candidate
    if peer:
        return peer
    return "unknown"


def _canonical_ip(raw: str) -> str | None:
    """Normalise an XFF hop to a canonical IP string, or ``None``.

    Real-world XFF emitters mix several encodings for the same client:

    * ``"10.0.0.1"`` — bare IPv4
    * ``"10.0.0.1:12345"`` — IPv4 + ephemeral port (some Envoy configs)
    * ``"::1"`` — bare IPv6
    * ``"[::1]:8080"`` — IPv6 + port (RFC 3986 bracketed form)
    * ``"  10.0.0.1  "`` — surrounding whitespace from CSV split

    Without canonicalisation each variant lands in its own
    ``_RateLimiter`` bucket, which is both a memory leak and a
    correctness bug — a client cycling its source port could dodge
    the per-IP cap. We strip brackets / ports, validate via the
    stdlib ``ipaddress`` module, and return the IPv4/IPv6 string in
    its compressed canonical form. Unparseable tokens return ``None``
    so the caller can skip them rather than treat an empty / garbage
    string as a real client identity.
    """
    token = raw.strip()
    if not token:
        return None
    # Bracketed IPv6 with optional port: `[::1]` or `[::1]:8080`.
    if token.startswith("["):
        end = token.find("]")
        if end == -1:
            return None
        host = token[1:end]
    elif token.count(":") == 1:
        # Single colon → almost certainly IPv4:port (bare IPv6 like
        # `::1` always has multiple colons or a leading bracket).
        host, _, _port = token.partition(":")
    else:
        host = token
    try:
        return ipaddress.ip_address(host).compressed
    except ValueError:
        return None


def _enforce_rate_limit(request: Request) -> None:
    limiter: _RateLimiter | None = getattr(request.app.state, "rate_limiter", None)
    if limiter is None:
        return
    ip = _client_ip(request)
    allowed, retry_after = limiter.check(ip)
    if not allowed:
        # Round up to a whole second so the Retry-After header is a
        # well-formed integer (the RFC permits a delta-seconds *or* an
        # HTTP-date, integer is the easier contract for clients).
        retry_secs = max(1, int(retry_after) + 1)
        _log.warning(
            "api.rate_limited",
            ip=ip,
            limit_per_min=limiter.limit,
            retry_after_s=retry_secs,
        )
        raise HTTPException(
            status_code=429,
            detail=f"rate limit exceeded: {limiter.limit} requests/min",
            headers={"Retry-After": str(retry_secs)},
        )


@app.get("/health")
async def health(request: Request) -> dict[str, object]:
    """Liveness + worker-load probe.

    Adds two workload-shape fields beyond the basic Redis ping:

    * ``active_runs`` — count of run records whose state is RUNNING,
      derived via a SCAN over `{prefix}:run:*` keys (skipping the
      `:steps` lists). Falls back to ``None`` if the count couldn't be
      computed; the probe still returns 200 in that case so a brief
      keyspace hiccup doesn't flap a load-balancer health check.
    * ``shadow_size`` — number of terminal RunStatus entries currently
      held in the in-process shadow. Non-zero means the bg task has
      been recovering from a store outage; operators want to see this.
    """
    redis_client = getattr(request.app.state, "redis", None)
    if redis_client is None:
        raise HTTPException(status_code=503, detail="redis client not initialised")
    try:
        pong = await redis_client.ping()
    except Exception as exc:  # surface any connection error as 503
        _log.warning("api.health_redis_down", error=str(exc))
        raise HTTPException(status_code=503, detail=f"redis unavailable: {exc}") from exc

    body: dict[str, object] = {
        "status": "ok",
        "redis": "up" if pong else "down",
        "active_runs": await _count_active_runs(request),
        "shadow_size": _shadow_size(request),
    }
    return body


async def _count_active_runs(request: Request) -> int | None:
    """Best-effort count of RUNNING runs in Redis.

    Uses SCAN (not KEYS) so a large keyspace doesn't block the event
    loop. Returns ``None`` if the store isn't a `RedisRunStore`
    instance (e.g. tests using `InMemoryRunStore`) or if the SCAN/get
    pipeline fails — `/health` must not 5xx because of a load-info
    side effect.
    """
    store = getattr(request.app.state, "store", None)
    redis_client = getattr(request.app.state, "redis", None)
    if redis_client is None or not isinstance(store, RedisRunStore):
        return None
    pattern = f"{store.prefix}:run:*"
    try:
        running = 0
        async for key in redis_client.scan_iter(match=pattern, count=200):
            # Skip the per-run `:steps` lists so we count run records
            # exactly once. The patterns overlap because both share the
            # `{prefix}:run:` stem.
            if key.endswith(":steps"):
                continue
            raw = await redis_client.get(key)
            if raw is None:
                continue
            # Parsing every record on every probe would be wasteful; a
            # cheap regex pre-filter on the JSON `state` field is
            # enough for this counter. The regex tolerates the
            # whitespace `json.dumps` inserts after the colon — the
            # earlier literal-substring check missed those records and
            # silently undercounted RUNNING runs. False positives would
            # require a run whose `question` literally contains
            # something matching ``"state": "running"`` after
            # serialisation, which Pydantic would reject on submit.
            if _RUNNING_STATE_RE.search(raw):
                running += 1
        return running
    except Exception as exc:
        _log.warning("api.health_active_runs_scan_failed", error=str(exc))
        return None


def _shadow_size(request: Request) -> int:
    shadow: _TerminalShadow | None = getattr(request.app.state, "terminal_shadow", None)
    return 0 if shadow is None else len(shadow)


def _store(request: Request) -> RunStore:
    store = getattr(request.app.state, "store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="run store not initialised")
    return store  # type: ignore[no-any-return]


class ErrorDetail(BaseModel):
    """Standard error body shape for 4xx / 5xx responses.

    Declared here so every protected route can reference it in its
    ``responses`` dict, making the OpenAPI schema emit a typed model
    reference instead of a plain ``{}`` schema for error responses.
    """

    detail: str


_AUTH_RESPONSES: dict[int | str, dict[str, object]] = {
    401: {"model": ErrorDetail, "description": "Missing or invalid bearer token."},
    422: {"description": "Validation error."},
    429: {"description": "Rate limit exceeded."},
    503: {"description": "Run store unavailable."},
}


@app.post(
    "/research",
    status_code=202,
    responses=_AUTH_RESPONSES,
)
async def submit_research(
    payload: ResearchRequest,
    background: BackgroundTasks,
    request: Request,
    _credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> dict[str, str]:
    """Enqueue a research run. Returns immediately with the run_id."""
    _require_auth(request, _credentials)
    _enforce_rate_limit(request)
    run_id = uuid.uuid4().hex
    store = _store(request)
    # Stamp ownership + an initial heartbeat so the lifespan reconciler
    # of any peer instance treats this run as live until our heartbeat
    # actually goes stale.
    run = RunStatus(
        run_id=run_id,
        question=payload.question,
        state=StepStatus.RUNNING,
        owner_id=_WORKER_ID,
        heartbeat_at=datetime.now(UTC),
    )
    try:
        await store.put_run(run)
    except (StoreUnavailableError, RedisError) as exc:
        _log.warning("api.store_down_on_submit", run_id=run_id, error=str(exc))
        raise HTTPException(status_code=503, detail=f"run store unavailable: {exc}") from exc
    _log.info("api.run_submitted", run_id=run_id, question_len=len(payload.question))
    background.add_task(_execute_run, store, _terminal_shadow(request), run_id, payload)
    # Absolute URL so callers behind a proxy / different host can poll
    # without re-deriving the base. ``request.url_for`` resolves the
    # named route (`get_run`) against the live request scope, so it
    # picks up ``X-Forwarded-*`` overrides if uvicorn is run with
    # ``--forwarded-allow-ips``.
    status_url = str(request.url_for("get_run", run_id=run_id))
    return {"run_id": run_id, "status_url": status_url}


_TERMINAL_STATES = (StepStatus.SUCCEEDED, StepStatus.FAILED)


def _terminal_shadow(request: Request) -> _TerminalShadow:
    shadow: _TerminalShadow | None = getattr(request.app.state, "terminal_shadow", None)
    if shadow is None:
        # Lifespan didn't run (some tests poke app.state directly); make
        # the shadow lookup a no-op rather than crash on missing attr.
        shadow = _TerminalShadow()
        request.app.state.terminal_shadow = shadow
    return shadow


@app.get(
    "/runs/{run_id}",
    responses=_AUTH_RESPONSES,
)
async def get_run(
    run_id: str,
    request: Request,
    _credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> RunStatus:
    _require_auth(request, _credentials)
    """Return the latest known state for `run_id`.

    Lookup precedence (the shadow exists for the bg-task store-outage path
    where the run *did* finish but the store rejected our terminal write):

    1. Store has a terminal record (SUCCEEDED/FAILED) → return it.
    2. Store has a still-RUNNING record but the shadow has a terminal one
       → return the shadow (the bg task hit a store outage and recovered
       to in-process state, while a stale RUNNING blob lingers in Redis).
    3. Store says 404 but the shadow has it → return the shadow.
    4. Store outage → return the shadow if present, else 503.
    """
    store = _store(request)
    shadow = _terminal_shadow(request)
    try:
        run = await store.get_run(run_id)
    except (StoreUnavailableError, RedisError) as exc:
        shadow_run = shadow.get(run_id)
        if shadow_run is not None:
            _log.info("api.get_run_served_from_shadow_on_outage", run_id=run_id)
            return await _hydrate_steps_best_effort(store, run_id, shadow_run)
        _log.warning("api.store_down_on_get", run_id=run_id, error=str(exc))
        raise HTTPException(status_code=503, detail=f"run store unavailable: {exc}") from exc

    if run is None:
        shadow_run = shadow.get(run_id)
        if shadow_run is not None:
            _log.info("api.get_run_served_from_shadow_on_404", run_id=run_id)
            return await _hydrate_steps_best_effort(store, run_id, shadow_run)
        raise HTTPException(status_code=404, detail=f"run {run_id} not found")

    if run.state not in _TERMINAL_STATES:
        shadow_run = shadow.get(run_id)
        if shadow_run is not None:
            _log.info("api.get_run_shadow_overrides_running", run_id=run_id)
            return await _hydrate_steps_best_effort(store, run_id, shadow_run)

    try:
        run.steps = await store.list_steps(run_id)
    except (StoreUnavailableError, RedisError) as exc:
        _log.warning("api.store_down_on_list_steps", run_id=run_id, error=str(exc))
        raise HTTPException(status_code=503, detail=f"run store unavailable: {exc}") from exc
    return run


async def _hydrate_steps_best_effort(store: RunStore, run_id: str, run: RunStatus) -> RunStatus:
    """When serving from the shadow, try once to populate `steps` from
    the store. If the store is unreachable (the same outage that drove
    us to the shadow in the first place), log and return the shadow as
    -is with empty steps — surfacing a terminal state with no audit is
    strictly better than 503ing.

    The shadow is constructed before the workflow appends step rows, so
    without this hydration GET would always return `steps=[]` even when
    the store *does* have them — see FIX 5.
    """
    try:
        steps = await store.list_steps(run_id)
    except Exception as exc:  # outage that put us on the shadow path
        _log.warning(
            "api.shadow_step_hydrate_failed",
            run_id=run_id,
            exc_type=type(exc).__name__,
            error=str(exc),
        )
        return run
    # Mutate a copy so a future shadow read still sees the cached entry
    # without the steps populated (cheaper than re-list on every GET).
    return run.model_copy(update={"steps": steps})


# ---------------------------------------------------------------------------
# Background execution
# ---------------------------------------------------------------------------


async def _persist_terminal(
    store: RunStore,
    shadow: _TerminalShadow,
    run: RunStatus,
    *,
    agent_label: str,
) -> None:
    """Write a terminal RunStatus to the store; on store outage stash in
    the in-process shadow so a subsequent GET still surfaces a terminal
    state.

    The shadow path is only for *expected* outage exceptions
    (`StoreUnavailableError`, `RedisError`). Programmer / serialization
    bugs propagate to the caller's outer handler so they actually get
    fixed instead of getting silently downgraded into a shadow write.

    A successful put leaves `shadow[run_id]` untouched so happy-path
    runs never pollute the in-process cache.
    """
    try:
        await store.put_run(run)
    except (StoreUnavailableError, RedisError) as exc:
        _log.error(
            "api.background_terminal_write_failed",
            run_id=run.run_id,
            agent=agent_label,
            exc_type=type(exc).__name__,
            error=str(exc),
        )
        shadow[run.run_id] = run


async def _stop_task(task: asyncio.Task[None]) -> None:
    """Cancel a sibling task and await it, swallowing the propagated
    ``CancelledError`` (and any non-cancellation exit) so the caller
    can proceed with its terminal write without re-entering the
    cancellation handler."""
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await task


async def _heartbeat_loop(
    store: RunStore, run_id: str, *, interval_s: float = _HEARTBEAT_INTERVAL_S
) -> None:
    """Refresh ``RunStatus.heartbeat_at`` on a fixed cadence.

    The loop is meant to be spawned as an `asyncio.Task` alongside the
    workflow body and cancelled once the workflow returns. It tolerates
    transient store outages (RedisError / StoreUnavailableError) by
    logging + continuing — the next tick will retry. Any other
    exception propagates so the bug surfaces in the logs.
    """
    while True:
        try:
            await asyncio.sleep(interval_s)
            existing = await store.get_run(run_id)
            if existing is None or existing.state is not StepStatus.RUNNING:
                # Terminal write already landed (or the run record was
                # evicted). Either way, no further heartbeats apply.
                return
            existing.heartbeat_at = datetime.now(UTC)
            await store.put_run(existing)
        except asyncio.CancelledError:
            raise
        except (StoreUnavailableError, RedisError) as exc:
            # Heartbeat best-effort: a brief Redis hiccup must not kill
            # the run. The next iteration will retry.
            _log.warning(
                "api.heartbeat_store_outage",
                run_id=run_id,
                exc_type=type(exc).__name__,
                error=str(exc),
            )


async def _run_workflow_and_persist(
    store: RunStore,
    shadow: _TerminalShadow,
    run_id: str,
    payload: ResearchRequest,
    run_started_at: float,
    heartbeat_task: asyncio.Task[None],
) -> tuple[int, int, int]:
    """Happy-path body of `_execute_run`. Extracted so the outer function
    stays under the per-function statement budget while keeping the
    cancellation/exception branches inline where they're easiest to
    read.

    Returns ``(agent_count, succeeded_agents, failed_agents)`` so the
    caller can pass through accurate counters to the
    ``api.run_completed`` event from the cancellation/exception paths
    if work was already partially done.
    """
    agents = default_agents()
    if payload.agents:
        wanted = set(payload.agents)
        agents = [a for a in agents if a.name in wanted]
    agent_count = len(agents)
    engine = WorkflowEngine(
        run_id=run_id,
        record_step=store.append_step,
        cache_get=store.cache_get,
        cache_put=store.cache_put,
    )
    results = await engine.run_parallel(agents, payload.question)
    succeeded_agents = sum(1 for r in results if r.status is not StepStatus.FAILED)
    failed_agents = sum(1 for r in results if r.status is StepStatus.FAILED)
    report: ResearchReport = await StitchSynthesizer().synthesize(run_id, payload.question, results)
    await _stop_task(heartbeat_task)
    finished = await store.get_run(run_id)
    if finished is None:
        # The run record vanished between submit and finalisation —
        # most likely a TTL eviction or a peer process clobbering the
        # key. We can't return success without a record (the caller
        # would never see a terminal state), so synthesise a minimal
        # FAILED RunStatus from what we still know in-process and
        # persist it through the same shadow-fallback path the regular
        # terminal write uses. Without this, callers polling
        # /runs/{id} would 404 forever and the in-process shadow would
        # never get populated either.
        _log.warning("api.run_record_missing_at_finalization", run_id=run_id)
        synthesised = RunStatus(
            run_id=run_id,
            question=payload.question,
            state=StepStatus.FAILED,
            finished_at=datetime.now(UTC),
            total_latency_ms=(time.monotonic() - run_started_at) * 1000.0,
            error="run record missing at finalization",
            report=report,
        )
        await _persist_terminal(store, shadow, synthesised, agent_label="workflow_record_missing")
        _emit_run_completed(
            run_id=run_id,
            run_started_at=run_started_at,
            terminal_state=StepStatus.FAILED,
            agent_count=agent_count,
            succeeded_agents=succeeded_agents,
            failed_agents=failed_agents,
            note="run_record_missing_at_finalization",
        )
        return agent_count, succeeded_agents, failed_agents
    finished.state = (
        StepStatus.FAILED
        if all(r.status is StepStatus.FAILED for r in results)
        else StepStatus.SUCCEEDED
    )
    finished.finished_at = datetime.now(UTC)
    finished.report = report
    finished.total_latency_ms = (time.monotonic() - run_started_at) * 1000.0
    await _persist_terminal(store, shadow, finished, agent_label="workflow_done")
    _emit_run_completed(
        run_id=run_id,
        run_started_at=run_started_at,
        terminal_state=finished.state,
        agent_count=agent_count,
        succeeded_agents=succeeded_agents,
        failed_agents=failed_agents,
    )
    return agent_count, succeeded_agents, failed_agents


async def _execute_run(
    store: RunStore,
    shadow: _TerminalShadow,
    run_id: str,
    payload: ResearchRequest,
) -> None:
    """Run the workflow and persist the terminal state.

    Wrapped so any uncaught failure (workflow bug, store outage,
    synthesizer crash) leaves the run in a terminal FAILED state with
    an error message instead of stuck at `running` forever. When the
    store itself is the cause of the outage, the terminal RunStatus is
    written to the in-process `shadow` cache so `GET /runs/{id}` can
    still report a terminal state — the next-best thing to a durable
    write.

    A sibling heartbeat task runs in the background while the workflow
    executes; it bumps ``RunStatus.heartbeat_at`` every
    ``_HEARTBEAT_INTERVAL_S`` so a peer instance's lifespan reconciler
    can tell live work from a true orphan.

    On every terminal write we also emit `api.run_completed` with the
    end-to-end latency, agent counts, and terminal state so operators
    have a single log line to grep for run-level SLO tracking.
    """
    run_started_at = time.monotonic()
    succeeded_agents = 0
    failed_agents = 0
    agent_count = 0
    heartbeat_task = asyncio.create_task(_heartbeat_loop(store, run_id))
    try:
        agent_count, succeeded_agents, failed_agents = await _run_workflow_and_persist(
            store, shadow, run_id, payload, run_started_at, heartbeat_task
        )
    except asyncio.CancelledError:
        # Shutdown-time cancellation must still leave a terminal record
        # behind — otherwise a polling client sees RUNNING forever and
        # only the next-startup orphan sweep cleans it up. Persist a
        # FAILED RunStatus with a clear reason, then let the cancellation
        # propagate so the surrounding task tree unwinds cleanly.
        _log.warning("api.background_run_cancelled", run_id=run_id)
        await _stop_task(heartbeat_task)
        cancelled = RunStatus(
            run_id=run_id,
            question=payload.question,
            state=StepStatus.FAILED,
            finished_at=datetime.now(UTC),
            total_latency_ms=(time.monotonic() - run_started_at) * 1000.0,
            error="cancelled during shutdown",
        )
        await _persist_terminal(store, shadow, cancelled, agent_label="workflow_cancelled")
        _emit_run_completed(
            run_id=run_id,
            run_started_at=run_started_at,
            terminal_state=StepStatus.FAILED,
            agent_count=agent_count,
            succeeded_agents=succeeded_agents,
            failed_agents=max(failed_agents, agent_count - succeeded_agents),
            note="cancelled_during_shutdown",
        )
        raise
    except Exception as exc:
        _log.error(
            "api.background_run_failed",
            run_id=run_id,
            exc_type=type(exc).__name__,
            error=str(exc),
        )
        await _stop_task(heartbeat_task)
        # Best-effort: mark the run FAILED so callers see a terminal state.
        # If the store itself is the cause of the outage, both the read
        # and the write may raise — fall back to the in-process shadow
        # so the GET path can still surface FAILED.
        try:
            existing = await store.get_run(run_id)
        except Exception as recovery_exc:
            _log.error(
                "api.background_failed_state_read_failed",
                run_id=run_id,
                exc_type=type(recovery_exc).__name__,
                error=str(recovery_exc),
            )
            existing = None

        # Build the FAILED RunStatus to record. If we couldn't even read
        # the original record, synthesise a minimal one from what we know
        # so the shadow still has a terminal entry to serve.
        failed = (
            existing
            if existing is not None
            else RunStatus(
                run_id=run_id,
                question=payload.question,
                state=StepStatus.RUNNING,
            )
        )
        failed.state = StepStatus.FAILED
        failed.finished_at = datetime.now(UTC)
        failed.total_latency_ms = (time.monotonic() - run_started_at) * 1000.0
        await _persist_terminal(store, shadow, failed, agent_label="workflow_failed")
        _emit_run_completed(
            run_id=run_id,
            run_started_at=run_started_at,
            terminal_state=StepStatus.FAILED,
            agent_count=agent_count,
            succeeded_agents=succeeded_agents,
            failed_agents=max(failed_agents, agent_count - succeeded_agents),
            note="exception_path",
        )
    finally:
        # Belt-and-braces: every successful path already cancelled the
        # heartbeat above, but if a new exception path is added later
        # this guarantees the task never outlives the workflow.
        await _stop_task(heartbeat_task)


def _emit_run_completed(
    *,
    run_id: str,
    run_started_at: float,
    terminal_state: StepStatus,
    agent_count: int,
    succeeded_agents: int,
    failed_agents: int,
    note: str | None = None,
) -> None:
    """Single source of truth for the `api.run_completed` event.

    Centralised so the happy path, the missing-record edge, and the
    exception path all emit the same fields with the same names — a
    single grep covers run-level SLO observability.
    """
    payload: dict[str, object] = {
        "run_id": run_id,
        "total_latency_ms": (time.monotonic() - run_started_at) * 1000.0,
        "agent_count": agent_count,
        "succeeded_agents": succeeded_agents,
        "failed_agents": failed_agents,
        "terminal_state": terminal_state.value,
    }
    if note:
        payload["note"] = note
    _log.info("api.run_completed", **payload)


# ---------------------------------------------------------------------------
# CLI entry: `research-api`
# ---------------------------------------------------------------------------


_cli = typer.Typer(name="research-api", add_completion=False)


@_cli.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind host."),
    port: int = typer.Option(8000, help="Bind port."),
    reload: bool = typer.Option(False, "--reload", help="uvicorn auto-reload"),
    forwarded_allow_ips: str = typer.Option(
        "127.0.0.1",
        "--forwarded-allow-ips",
        help=(
            "Comma-separated IPs / CIDRs whose X-Forwarded-* headers uvicorn should trust. "
            "Required for the absolute `status_url` in API responses to use the public "
            "scheme/host when running behind a reverse proxy. "
            "Use `*` to trust any peer (only on a private network). "
            "Default trusts only loopback."
        ),
    ),
) -> None:
    """Run the FastAPI service."""
    uvicorn.run(
        "research_crew.api:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
        forwarded_allow_ips=forwarded_allow_ips,
        proxy_headers=True,
    )


def main() -> None:  # pragma: no cover
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    _cli()


if __name__ == "__main__":  # pragma: no cover
    main()
