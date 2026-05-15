"""Redis-backed run / step / cache store.

Three keyspaces, one Redis client; every key is namespaced under a
``prefix`` (default ``"research"``) so multiple deployments or tenants
on the same Redis instance never collide:

* ``{prefix}:run:{run_id}``        — JSON RunStatus blob (latest snapshot)
* ``{prefix}:run:{run_id}:steps``  — list of JSON StepRecord pushed in order
* ``{prefix}:step:{dedup_key}``    — JSON AgentResult (idempotency cache, TTL 1h)

The prefix is read from ``RESEARCH_REDIS_PREFIX`` at API lifespan; tests
construct ``RedisRunStore`` with an explicit prefix to assert the raw
keys land where expected. See ADR-001 for the full rationale.

Schema versioning: persisted ``RunStatus`` / ``StepRecord`` blobs carry
a ``schema_version`` field. Readers consult
``research_crew.models.CURRENT_SCHEMA_VERSION``: blobs whose declared
version exceeds the reader's are logged and skipped (a newer deploy
running ahead of this one shouldn't crash older readers); blobs whose
version is older are migrated by ``migrate_run_blob`` /
``_migrate_step_blob`` before validation.

Designed to drop in next to a real Inngest deployment: the run / step
keys mirror what a workflow engine would write, so a future commit can
replace this with `inngest.step.run(...)` without touching the API.
"""

from __future__ import annotations

import json
from typing import Any, Protocol

import redis.asyncio as aioredis
import structlog

from research_crew.models import (
    CURRENT_SCHEMA_VERSION,
    AgentResult,
    ResearchReport,
    RunStatus,
    StepRecord,
    StepStatus,
)

_log = structlog.get_logger(__name__)

DEFAULT_REDIS_PREFIX = "research"


class RunStore(Protocol):
    async def get_run(self, run_id: str) -> RunStatus | None: ...

    async def put_run(self, run: RunStatus) -> None: ...

    async def append_step(self, step: StepRecord) -> None: ...

    async def list_steps(self, run_id: str) -> list[StepRecord]: ...

    async def cache_get(self, dedup_key: str) -> AgentResult | None: ...

    async def cache_put(self, dedup_key: str, result: AgentResult) -> None: ...


class RedisRunStore:
    """RunStore backed by `redis-py` async client.

    All keys are built under ``{prefix}:`` so multiple environments
    (dev/staging/prod) or logical tenants can share one Redis without
    cross-talk. The prefix is intentionally a constructor param rather
    than a global so tests can pin it without env-var manipulation.
    """

    def __init__(
        self,
        client: aioredis.Redis,
        ttl_seconds: int = 3600,
        prefix: str = DEFAULT_REDIS_PREFIX,
    ) -> None:
        self._r = client
        self._ttl = ttl_seconds
        # Strip a trailing colon if a caller passed one — the helpers
        # always add a colon between the prefix and the keyspace.
        self._prefix = prefix.rstrip(":")

    @property
    def prefix(self) -> str:
        return self._prefix

    def _run_key(self, run_id: str) -> str:
        return f"{self._prefix}:run:{run_id}"

    def _steps_key(self, run_id: str) -> str:
        return f"{self._prefix}:run:{run_id}:steps"

    def _step_cache_key(self, dedup_key: str) -> str:
        # The workflow engine already prepends ``step:`` to dedup_key;
        # collapse that into the prefixed namespace so the on-disk key
        # is `{prefix}:step:{digest}` not `{prefix}:step:step:{digest}`.
        tail = dedup_key[len("step:"):] if dedup_key.startswith("step:") else dedup_key
        return f"{self._prefix}:step:{tail}"

    async def get_run(self, run_id: str) -> RunStatus | None:
        raw = await self._r.get(self._run_key(run_id))
        if raw is None:
            return None
        payload = migrate_run_blob(json.loads(raw), key=self._run_key(run_id))
        if payload is None:
            return None
        return RunStatus.model_validate(payload)

    async def put_run(self, run: RunStatus) -> None:
        await self._r.set(self._run_key(run.run_id), run.model_dump_json(), ex=self._ttl * 24)

    async def append_step(self, step: StepRecord) -> None:
        key = self._steps_key(step.run_id)
        await self._r.rpush(key, step.model_dump_json())  # type: ignore[misc]
        await self._r.expire(key, self._ttl * 24)

    async def list_steps(self, run_id: str) -> list[StepRecord]:
        raws = await self._r.lrange(self._steps_key(run_id), 0, -1)  # type: ignore[misc]
        steps: list[StepRecord] = []
        for r in raws:
            payload = _migrate_step_blob(json.loads(r), key=self._steps_key(run_id))
            if payload is None:
                continue
            steps.append(StepRecord.model_validate(payload))
        return steps

    async def cache_get(self, dedup_key: str) -> AgentResult | None:
        raw = await self._r.get(self._step_cache_key(dedup_key))
        if raw is None:
            return None
        return AgentResult.model_validate(json.loads(raw))

    async def cache_put(self, dedup_key: str, result: AgentResult) -> None:
        await self._r.set(self._step_cache_key(dedup_key), result.model_dump_json(), ex=self._ttl)

    def _cas_matches(
        self,
        raw: str,
        key: str,
        expected_state: StepStatus,
        expected_heartbeat_at: object,
    ) -> RunStatus | None:
        """Parse ``raw`` and return the current RunStatus if it still
        matches ``expected_state`` + ``expected_heartbeat_at``, else
        ``None`` (caller should abort the swap).
        """
        try:
            payload = json.loads(raw)
        except Exception:
            return None
        migrated = migrate_run_blob(payload, key=key)
        if migrated is None:
            return None
        try:
            current = RunStatus.model_validate(migrated)
        except Exception:
            return None
        if current.state is not expected_state:
            return None
        if current.heartbeat_at != expected_heartbeat_at:
            return None
        return current

    async def cas_reconcile_run(
        self,
        run_id: str,
        expected_state: StepStatus,
        expected_heartbeat_at: object,
        new_run: RunStatus,
    ) -> bool:
        """Compare-and-swap a run record using Redis WATCH/MULTI/EXEC.

        WATCHes the run key, reads the current value, validates that
        ``state`` and ``heartbeat_at`` still match the observed values
        the caller checked before deciding to reconcile, then atomically
        writes ``new_run`` under MULTI/EXEC.

        Returns ``True`` on a successful swap, ``False`` if the WATCH
        fired (another writer changed the key concurrently) or if the
        current value no longer matches the expected state/heartbeat.
        Any other Redis error propagates to the caller.
        """
        key = self._run_key(run_id)
        async with self._r.pipeline() as pipe:
            try:
                await pipe.watch(key)
                raw = await pipe.get(key)
                if raw is None or self._cas_matches(
                    raw, key, expected_state, expected_heartbeat_at
                ) is None:
                    await pipe.reset()  # type: ignore[no-untyped-call]
                    return False
                # State and heartbeat match — attempt the atomic write.
                pipe.multi()  # type: ignore[no-untyped-call]
                pipe.set(key, new_run.model_dump_json(), ex=self._ttl * 24)
                await pipe.execute()
                return True
            except aioredis.WatchError:
                # Another writer changed the key between WATCH and EXEC.
                return False


def migrate_run_blob(payload: dict[str, Any], *, key: str) -> dict[str, Any] | None:
    """Bring a persisted RunStatus dict up to ``CURRENT_SCHEMA_VERSION``.

    * Missing / older ``schema_version`` is treated as v1 and the field
      is filled in so the strict-mode validator accepts it.
    * Newer ``schema_version`` (a peer instance is running ahead of us)
      is logged and the record is skipped — returning ``None`` makes
      the caller treat it as "no record" rather than crash.

    The raw blob is mutated in place; the dict is the same one passed
    in so callers can validate it directly without a re-serialise.
    """
    version = payload.get("schema_version")
    if version is None:
        # Pre-versioning blob: stamp v1 so model_validate accepts the
        # extra="forbid" contract.
        payload["schema_version"] = 1
        return payload
    if not isinstance(version, int) or version <= 0:
        _log.warning(
            "store.schema_version_invalid",
            key=key,
            schema_version=version,
        )
        return None
    if version > CURRENT_SCHEMA_VERSION:
        _log.warning(
            "store.schema_version_unsupported",
            key=key,
            schema_version=version,
            supported=CURRENT_SCHEMA_VERSION,
        )
        return None
    return payload


def _migrate_step_blob(payload: dict[str, Any], *, key: str) -> dict[str, Any] | None:
    """Same as ``migrate_run_blob`` for ``StepRecord`` rows.

    Kept separate so future per-model migrations (renaming a field,
    splitting one into two, etc.) can diverge cleanly.
    """
    return migrate_run_blob(payload, key=key)


class InMemoryRunStore:
    """Test-only RunStore. Nothing fancy; mirrors the Redis semantics."""

    def __init__(self) -> None:
        self._runs: dict[str, RunStatus] = {}
        self._steps: dict[str, list[StepRecord]] = {}
        self._cache: dict[str, AgentResult] = {}

    async def get_run(self, run_id: str) -> RunStatus | None:
        return self._runs.get(run_id)

    async def put_run(self, run: RunStatus) -> None:
        self._runs[run.run_id] = run

    async def append_step(self, step: StepRecord) -> None:
        self._steps.setdefault(step.run_id, []).append(step)

    async def list_steps(self, run_id: str) -> list[StepRecord]:
        return list(self._steps.get(run_id, []))

    async def cache_get(self, dedup_key: str) -> AgentResult | None:
        return self._cache.get(dedup_key)

    async def cache_put(self, dedup_key: str, result: AgentResult) -> None:
        self._cache[dedup_key] = result


__all__ = [
    "AgentResult",
    "InMemoryRunStore",
    "RedisRunStore",
    "ResearchReport",
    "RunStatus",
    "RunStore",
    "StepRecord",
    "StepStatus",
]
