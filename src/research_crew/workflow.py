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
import random
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
class _LockEntry:
    """Refcounted per-dedup-key lock entry.

    `refcount` tracks holders + waiters so the registry only evicts an
    entry when the last interested caller has released. Bumping the
    refcount under the registry guard before awaiting the lock pins the
    entry against eviction even when a holder finishes between two
    waiters arriving.
    """

    lock: asyncio.Lock
    refcount: int = 0


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
    #
    # Entries are refcounted — every caller that intends to acquire the
    # lock first bumps `refcount`, and the entry is only evicted when the
    # last holder/waiter has released. Without that, a `pop()` between
    # holder #1 releasing and waiter #2 acquiring would let a brand-new
    # caller create a *fresh* lock for the same key, holding two distinct
    # lock objects for the same key concurrently — exactly the race the
    # lock exists to close.
    _key_locks: dict[str, _LockEntry] = field(default_factory=dict, repr=False)
    _registry_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    async def _lock_for(self, dedup_key: str) -> asyncio.Lock:
        # The registry lock guards the dict + refcount mutations, not the
        # per-key critical section — held only long enough to insert-or-
        # bump-and-fetch. Bumping refcount BEFORE awaiting the lock pins
        # the entry against eviction by the releasing holder.
        async with self._registry_lock:
            entry = self._key_locks.get(dedup_key)
            if entry is None:
                entry = _LockEntry(lock=asyncio.Lock(), refcount=0)
                self._key_locks[dedup_key] = entry
            entry.refcount += 1
            return entry.lock

    async def _release_key(self, dedup_key: str) -> None:
        # Decrement refcount; only evict the entry when no other caller is
        # holding or waiting on it. `lock.locked()` is a defence-in-depth
        # check for the unlikely case that refcount drifts (it shouldn't,
        # because every acquire path goes through `_lock_for`).
        async with self._registry_lock:
            entry = self._key_locks.get(dedup_key)
            if entry is None:
                return
            entry.refcount -= 1
            if entry.refcount <= 0 and not entry.lock.locked():
                self._key_locks.pop(dedup_key, None)

    async def run_one(self, agent: Agent, question: str) -> AgentResult:
        """Execute one agent with the durability semantics above.

        Store-side hiccups (`cache_get`, `record_step`, `cache_put`) are
        intentionally swallowed here. The audit log and idempotency cache
        are observability/optimisation, not correctness — losing a step
        row or a cache write must never abort agent work that already
        succeeded. Only three classes of failure terminate `run_one`:

        1. The agent itself raised (handled by retry budget).
        2. The retry budget was exhausted on the agent.
        3. The per-step timeout fired.
        """
        dedup_key = self._dedup_key(agent, question)
        log = _log.bind(run_id=self.run_id, agent=agent.name.value, dedup_key=dedup_key)
        # Capture the wall-clock start once. Each StepRecord we write is
        # stamped with this same `started_at`, so terminal records report
        # a real `finished_at - started_at` duration instead of stamping
        # `now()` for both ends.
        started_at = datetime.now(UTC)

        lock = await self._lock_for(dedup_key)
        # Release the registry refcount AFTER the lock has been released
        # (i.e. once the `async with lock` block exits) so `lock.locked()`
        # reads False at eviction time. Otherwise we'd either evict while
        # waiters still hold the lock (the original race) or never evict
        # (defence-in-depth `lock.locked()` check would always pin).
        try:
            async with lock:
                result = await self._run_locked(agent, question, dedup_key, started_at, log)
            return result
        finally:
            await self._release_key(dedup_key)

    async def _run_locked(
        self,
        agent: Agent,
        question: str,
        dedup_key: str,
        started_at: datetime,
        log: structlog.stdlib.BoundLogger,
    ) -> AgentResult:
        """Body of `run_one` that executes under the per-key lock.

        Extracted so the lock-release ordering in `run_one` stays
        legible: acquire lock, do the work, drop lock, drop registry
        refcount — strictly in that order. The earlier inline form had
        `finally: _release_key()` nested *inside* `async with lock:`,
        which evicted the entry while the lock was still held and let
        a third caller create a parallel lock for the same key.
        """
        cached = await self._safe_cache_get(dedup_key, log)
        if cached is not None:
            log.info("workflow.cache_hit")
            # Surface the cache-hit so the caller can distinguish.
            return cached.model_copy(update={"status": StepStatus.CACHED})

        last_error: str | None = None
        for attempt in range(1, self.config.max_attempts + 1):
            await self._safe_record(
                agent, attempt, StepStatus.RUNNING, started_at=started_at, log=log
            )
            t0 = time.perf_counter()
            try:
                result = await asyncio.wait_for(
                    agent.search(question), timeout=self.config.per_step_timeout_s
                )
            except TimeoutError:
                err = AgentTimeoutError(agent.name.value, self.config.per_step_timeout_s)
                last_error = str(err)
                log.warning("workflow.timeout", attempt=attempt, timeout_s=err.timeout_s)
                await self._safe_record(
                    agent,
                    attempt,
                    StepStatus.FAILED,
                    error=last_error,
                    started_at=started_at,
                    log=log,
                )
            except asyncio.CancelledError:
                # Cooperative cancellation: record the failure, then let the
                # cancellation propagate so the surrounding task tree unwinds.
                last_error = "cancelled"
                log.warning("workflow.cancelled", attempt=attempt)
                await self._safe_record(
                    agent,
                    attempt,
                    StepStatus.FAILED,
                    error=last_error,
                    started_at=started_at,
                    log=log,
                )
                raise
            except Exception as exc:
                # Agent code is user-supplied. Wrap in `AgentExecutionError`
                # so callers (the API logger, future Inngest mapper) can
                # match on a typed handle for "the agent itself raised"
                # versus the other failure modes (timeout, agent reported
                # FAILED). Retry budget still applies — we report the
                # wrapped type in the error string and the log line so
                # the typed framing survives the catch loop.
                wrapped = AgentExecutionError(agent.name.value, exc)
                last_error = f"{type(wrapped).__name__}: {wrapped}"
                log.warning(
                    "workflow.agent_error",
                    attempt=attempt,
                    exc_type=type(wrapped).__name__,
                    inner_exc_type=type(exc).__name__,
                    error=str(wrapped),
                )
                await self._safe_record(
                    agent,
                    attempt,
                    StepStatus.FAILED,
                    error=last_error,
                    started_at=started_at,
                    log=log,
                )
            else:
                elapsed = (time.perf_counter() - t0) * 1000.0
                if result.status is StepStatus.SUCCEEDED:
                    final = result.model_copy(
                        update={"attempts": attempt, "elapsed_ms": elapsed}
                    )
                    await self._safe_record(
                        agent,
                        attempt,
                        StepStatus.SUCCEEDED,
                        started_at=started_at,
                        log=log,
                    )
                    log.info("workflow.success", attempt=attempt, elapsed_ms=elapsed)
                    await self._safe_cache_put(dedup_key, final, log)
                    return final
                last_error = result.error or "agent reported FAILED"
                log.warning(
                    "workflow.agent_returned_failed", attempt=attempt, error=last_error
                )
                await self._safe_record(
                    agent,
                    attempt,
                    StepStatus.FAILED,
                    error=last_error,
                    started_at=started_at,
                    log=log,
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

    async def _safe_cache_get(
        self, dedup_key: str, log: structlog.stdlib.BoundLogger
    ) -> AgentResult | None:
        """`cache_get` with store-failure suppressed.

        On any exception we log `workflow.cache_unavailable_skipping` and
        return ``None`` — i.e. behave exactly like a cache miss. The
        agent will then run from scratch, which is correct: a cache
        outage must never propagate as a run failure.
        """
        if self.cache_get is None:
            return None
        try:
            return await self.cache_get(dedup_key)
        except Exception as exc:
            log.warning(
                "workflow.cache_unavailable_skipping",
                exc_type=type(exc).__name__,
                error=str(exc),
            )
            return None

    async def _safe_cache_put(
        self, dedup_key: str, result: AgentResult, log: structlog.stdlib.BoundLogger
    ) -> None:
        """`cache_put` with store-failure suppressed.

        The agent already succeeded by the time this runs — losing the
        cache entry only costs us a re-run on the next identical request,
        not the result we already have. Log and return.
        """
        if self.cache_put is None:
            return
        try:
            await self.cache_put(dedup_key, result)
        except Exception as exc:
            log.warning(
                "workflow.cache_put_failed_succeeding_anyway",
                exc_type=type(exc).__name__,
                error=str(exc),
            )

    async def _safe_record(
        self,
        agent: Agent,
        attempt: int,
        status: StepStatus,
        *,
        started_at: datetime,
        error: str | None = None,
        log: structlog.stdlib.BoundLogger,
    ) -> None:
        """`_record` with store-failure suppressed.

        Step rows are an audit log, not the source of truth for run
        outcome. A dropped row is a known-loss observability event, not
        a reason to abort an otherwise successful agent run.
        """
        try:
            await self._record(agent, attempt, status, error=error, started_at=started_at)
        except Exception as exc:
            log.warning(
                "workflow.step_record_lost",
                attempt=attempt,
                step_status=status.value,
                exc_type=type(exc).__name__,
                error=str(exc),
            )

    async def _sleep_backoff(self, attempt: int) -> None:
        if attempt >= self.config.max_attempts:
            return
        await asyncio.sleep(self._backoff_delay(attempt))

    def _backoff_delay(self, attempt: int) -> float:
        """Compute the per-attempt backoff with ±25% uniform jitter.

        Without jitter, every parallel `run_one` retry fires at the same
        wall-clock instant after a failure, producing thundering-herd
        retry storms against whatever upstream is already struggling.
        Spreading retries across a 50%-wide window breaks that
        synchronisation while still preserving the exponential growth
        of the base backoff. See ADR-003.
        """
        base = self.config.base_backoff_s * (2 ** (attempt - 1))
        # `random.uniform` is uniform-inclusive on both ends; that's
        # fine — the property tests only need [0.75, 1.25] inclusive.
        # The PRNG is non-cryptographic on purpose: this is jitter
        # for thundering-herd avoidance, not a secret.
        jitter = float(random.uniform(0.75, 1.25))  # noqa: S311 — jitter, not crypto
        return float(base) * jitter
