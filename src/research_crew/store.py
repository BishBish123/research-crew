"""Redis-backed run / step / cache store.

Three keyspaces, one Redis client:

* `run:{run_id}`        — JSON RunStatus blob (latest snapshot)
* `run:{run_id}:steps`  — list of JSON StepRecord pushed in order
* `step:{dedup_key}`    — JSON AgentResult (idempotency cache, TTL 1h)

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


class RunStore(Protocol):
    async def get_run(self, run_id: str) -> RunStatus | None: ...

    async def put_run(self, run: RunStatus) -> None: ...

    async def append_step(self, step: StepRecord) -> None: ...

    async def list_steps(self, run_id: str) -> list[StepRecord]: ...

    async def cache_get(self, dedup_key: str) -> AgentResult | None: ...

    async def cache_put(self, dedup_key: str, result: AgentResult) -> None: ...


class RedisRunStore:
    """RunStore backed by `redis-py` async client."""

    def __init__(self, client: aioredis.Redis, ttl_seconds: int = 3600) -> None:
        self._r = client
        self._ttl = ttl_seconds

    async def get_run(self, run_id: str) -> RunStatus | None:
        raw = await self._r.get(f"run:{run_id}")
        if raw is None:
            return None
        return RunStatus.model_validate(json.loads(raw))

    async def put_run(self, run: RunStatus) -> None:
        await self._r.set(f"run:{run.run_id}", run.model_dump_json(), ex=self._ttl * 24)

    async def append_step(self, step: StepRecord) -> None:
        await self._r.rpush(f"run:{step.run_id}:steps", step.model_dump_json())  # type: ignore[misc]
        await self._r.expire(f"run:{step.run_id}:steps", self._ttl * 24)

    async def list_steps(self, run_id: str) -> list[StepRecord]:
        raws = await self._r.lrange(f"run:{run_id}:steps", 0, -1)  # type: ignore[misc]
        return [StepRecord.model_validate(json.loads(r)) for r in raws]

    async def cache_get(self, dedup_key: str) -> AgentResult | None:
        raw = await self._r.get(dedup_key)
        if raw is None:
            return None
        return AgentResult.model_validate(json.loads(raw))

    async def cache_put(self, dedup_key: str, result: AgentResult) -> None:
        await self._r.set(dedup_key, result.model_dump_json(), ex=self._ttl)


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
