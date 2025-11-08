"""Run / step / cache store — `RunStore` Protocol + in-memory backend.

A Redis-backed implementation lands in the next commit; the Protocol
exists first so the workflow runner can be written and tested against
the cheap in-memory backend before Redis is wired in.
"""

from __future__ import annotations

from typing import Protocol

from research_crew.models import (
    AgentResult,
    RunStatus,
    StepRecord,
)


class RunStore(Protocol):
    async def get_run(self, run_id: str) -> RunStatus | None: ...

    async def put_run(self, run: RunStatus) -> None: ...

    async def append_step(self, step: StepRecord) -> None: ...

    async def list_steps(self, run_id: str) -> list[StepRecord]: ...

    async def cache_get(self, dedup_key: str) -> AgentResult | None: ...

    async def cache_put(self, dedup_key: str, result: AgentResult) -> None: ...


class InMemoryRunStore:
    """Test-only RunStore. Mirrors the semantics the Redis backend must offer."""

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
    "RunStatus",
    "RunStore",
    "StepRecord",
]
