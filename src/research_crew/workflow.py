"""Durable workflow runner: parallel fan-out + retries + idempotency cache.

The runner owns the contract that distributed engines (Inngest,
Trigger.dev, Temporal) provide for free in production:

* **Idempotency** — every step has a stable `dedup_key = H(run_id, agent, question)`.
  Repeat calls return the cached AgentResult instead of re-running.
* **Bounded retries with exponential backoff** — `max_attempts=3` by
  default; per-attempt sleep is `base_backoff_s * 2**attempt`.
* **Per-step timeout** — wall-clock guard so a stuck agent can't block
  the whole run.
* **Parallel fan-out** — `asyncio.gather` runs every agent at once;
  partial failure does not abort the run.
* **Step records** — every attempt is persisted via the injected store
  so a `GET /runs/{id}` can render the full timeline.

The whole interface is a single `WorkflowEngine` class so swapping it
for a real Inngest call later is local — same input, same output.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

import structlog

from research_crew.agents import Agent
from research_crew.errors import AgentExecutionError, AgentTimeoutError
from research_crew.models import AgentResult, StepRecord, StepStatus

_log = structlog.get_logger(__name__)


@dataclass
class WorkflowConfig:
    max_attempts: int = 3
    base_backoff_s: float = 0.05
    per_step_timeout_s: float = 30.0


@dataclass
class WorkflowEngine:
    """Per-run engine. Cheap to construct; one instance per `/research` call."""

    run_id: str
    config: WorkflowConfig = field(default_factory=WorkflowConfig)
    record_step: Callable[[StepRecord], Awaitable[None]] | None = None
    cache_get: Callable[[str], Awaitable[AgentResult | None]] | None = None
    cache_put: Callable[[str, AgentResult], Awaitable[None]] | None = None
    # Per-dedup-key locks turn the read-then-write idempotency check into
    # an atomic critical section: two concurrent `run_one` calls with the
    # same key serialize, the loser observes the cache hit, and the agent
    # is invoked exactly once.
    _key_locks: dict[str, asyncio.Lock] = field(default_factory=dict, repr=False)
    _registry_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    async def _lock_for(self, dedup_key: str) -> asyncio.Lock:
        # The registry lock guards the dict, not the per-key critical
        # section — held only long enough to insert-or-fetch.
        async with self._registry_lock:
            lock = self._key_locks.get(dedup_key)
            if lock is None:
                lock = asyncio.Lock()
                self._key_locks[dedup_key] = lock
            return lock

    async def _release_key(self, dedup_key: str) -> None:
        # Drop the entry once the per-key critical section is done so the
        # registry doesn't grow unboundedly across long-lived engines.
        # Idempotent: a missing key is a no-op so the `finally` cleanup
        # path stays safe even if a previous branch already released.
        async with self._registry_lock:
            self._key_locks.pop(dedup_key, None)

    async def run_one(self, agent: Agent, question: str) -> AgentResult:
        """Execute one agent with the durability semantics above."""
        dedup_key = self._dedup_key(agent, question)
        log = _log.bind(run_id=self.run_id, agent=agent.name.value, dedup_key=dedup_key)
        # Capture the wall-clock start once. Each StepRecord we write is
        # stamped with this same `started_at`, so terminal records report
        # a real `finished_at - started_at` duration instead of stamping
        # `now()` for both ends.
        started_at = datetime.now(UTC)

        lock = await self._lock_for(dedup_key)
        async with lock:
            # The `finally` guarantees the registry entry is dropped on
            # every exit path — success, exhausted retries, *and* an
            # exception bubbling out of cache_get / record_step / cache_put
            # / agent.search. Without this, a store outage leaks one lock
            # entry per failed call and the registry grows without bound.
            try:
                if self.cache_get is not None:
                    cached = await self.cache_get(dedup_key)
                    if cached is not None:
                        log.info("workflow.cache_hit")
                        # Surface the cache-hit so the caller can distinguish.
                        return cached.model_copy(update={"status": StepStatus.CACHED})

                last_error: str | None = None
                for attempt in range(1, self.config.max_attempts + 1):
                    await self._record(agent, attempt, StepStatus.RUNNING, started_at=started_at)
                    t0 = time.perf_counter()
                    try:
                        result = await asyncio.wait_for(
                            agent.search(question), timeout=self.config.per_step_timeout_s
                        )
                    except TimeoutError:
                        err = AgentTimeoutError(agent.name.value, self.config.per_step_timeout_s)
                        last_error = str(err)
                        log.warning("workflow.timeout", attempt=attempt, timeout_s=err.timeout_s)
                        await self._record(
                            agent,
                            attempt,
                            StepStatus.FAILED,
                            error=last_error,
                            started_at=started_at,
                        )
                    except asyncio.CancelledError:
                        # Cooperative cancellation: record the failure, then let the
                        # cancellation propagate so the surrounding task tree unwinds.
                        last_error = "cancelled"
                        log.warning("workflow.cancelled", attempt=attempt)
                        await self._record(
                            agent,
                            attempt,
                            StepStatus.FAILED,
                            error=last_error,
                            started_at=started_at,
                        )
                        raise
                    except Exception as exc:  # agents are user code; wrap in AgentExecutionError
                        wrapped = AgentExecutionError(agent.name.value, exc)
                        last_error = f"{type(exc).__name__}: {exc}"
                        log.warning(
                            "workflow.agent_error",
                            attempt=attempt,
                            exc_type=type(exc).__name__,
                            error=str(exc),
                        )
                        await self._record(
                            agent,
                            attempt,
                            StepStatus.FAILED,
                            error=last_error,
                            started_at=started_at,
                        )
                        # Suppress chaining so loggers don't double-report; semantic
                        # info is preserved in `AgentExecutionError.original`.
                        del wrapped
                    else:
                        elapsed = (time.perf_counter() - t0) * 1000.0
                        if result.status is StepStatus.SUCCEEDED:
                            final = result.model_copy(
                                update={"attempts": attempt, "elapsed_ms": elapsed}
                            )
                            await self._record(
                                agent, attempt, StepStatus.SUCCEEDED, started_at=started_at
                            )
                            log.info("workflow.success", attempt=attempt, elapsed_ms=elapsed)
                            if self.cache_put is not None:
                                await self.cache_put(dedup_key, final)
                            return final
                        last_error = result.error or "agent reported FAILED"
                        log.warning(
                            "workflow.agent_returned_failed", attempt=attempt, error=last_error
                        )
                        await self._record(
                            agent,
                            attempt,
                            StepStatus.FAILED,
                            error=last_error,
                            started_at=started_at,
                        )
                    await self._sleep_backoff(attempt)

                # All attempts exhausted.
                log.warning(
                    "workflow.exhausted",
                    attempts=self.config.max_attempts,
                    last_error=last_error,
                )
                return AgentResult(
                    agent=agent.name,
                    status=StepStatus.FAILED,
                    summary="",
                    error=last_error or "exhausted retries",
                    attempts=self.config.max_attempts,
                )
            finally:
                await self._release_key(dedup_key)

    async def run_parallel(self, agents: list[Agent], question: str) -> list[AgentResult]:
        """Fan out to every agent concurrently. Partial failure does not abort."""
        return await asyncio.gather(*(self.run_one(a, question) for a in agents))

    # ---------- helpers ----------

    def _dedup_key(self, agent: Agent, question: str) -> str:
        digest = hashlib.blake2b(
            f"{self.run_id}|{agent.name}|{question}".encode(), digest_size=12
        ).hexdigest()
        return f"step:{digest}"

    async def _record(
        self,
        agent: Agent,
        attempt: int,
        status: StepStatus,
        error: str | None = None,
        started_at: datetime | None = None,
    ) -> None:
        if self.record_step is None:
            return
        # Preserve the real start time captured by `run_one` so terminal
        # records report a positive duration; fall back to `now` only if
        # a caller invokes `_record` directly without a start time.
        start = started_at if started_at is not None else datetime.now(UTC)
        await self.record_step(
            StepRecord(
                run_id=self.run_id,
                agent=agent.name,
                status=status,
                attempts=attempt,
                started_at=start,
                finished_at=datetime.now(UTC) if status is not StepStatus.RUNNING else None,
                error=error,
            )
        )

    async def _sleep_backoff(self, attempt: int) -> None:
        if attempt >= self.config.max_attempts:
            return
        await asyncio.sleep(self.config.base_backoff_s * (2 ** (attempt - 1)))
