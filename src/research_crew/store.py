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

Designed to drop in next to a real Inngest deployment: the run / step
keys mirror what a workflow engine would write, so a future commit can
replace this with `inngest.step.run(...)` without touching the API.
"""

from __future__ import annotations

import json
from typing import Protocol

import redis.asyncio as aioredis

from research_crew.models import (
    AgentResult,
    ResearchReport,
    RunStatus,
    StepRecord,
    StepStatus,
)

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
        return RunStatus.model_validate(json.loads(raw))

    async def put_run(self, run: RunStatus) -> None:
        await self._r.set(self._run_key(run.run_id), run.model_dump_json(), ex=self._ttl * 24)

    async def append_step(self, step: StepRecord) -> None:
        key = self._steps_key(step.run_id)
        await self._r.rpush(key, step.model_dump_json())  # type: ignore[misc]
        await self._r.expire(key, self._ttl * 24)

    async def list_steps(self, run_id: str) -> list[StepRecord]:
        raws = await self._r.lrange(self._steps_key(run_id), 0, -1)  # type: ignore[misc]
        return [StepRecord.model_validate(json.loads(r)) for r in raws]

    async def cache_get(self, dedup_key: str) -> AgentResult | None:
        raw = await self._r.get(self._step_cache_key(dedup_key))
        if raw is None:
            return None
        return AgentResult.model_validate(json.loads(raw))

    async def cache_put(self, dedup_key: str, result: AgentResult) -> None:
        await self._r.set(self._step_cache_key(dedup_key), result.model_dump_json(), ex=self._ttl)


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
