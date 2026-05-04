"""Concurrency tests for the workflow runner.

The contract: two concurrent `run_one(...)` calls with the same
`(run_id, agent, question)` triple are atomic — the agent fires
exactly once, the loser observes a CACHED result. The per-key
lock in `WorkflowEngine` makes the read-then-write idempotency
check a single critical section.
"""

from __future__ import annotations

import asyncio

from research_crew.models import AgentName, AgentResult, StepStatus
from research_crew.store import InMemoryRunStore
from research_crew.workflow import WorkflowConfig, WorkflowEngine


class _Counter:
    name = AgentName.WEB_SEARCH

    def __init__(self) -> None:
        self.calls = 0

    async def search(self, question: str) -> AgentResult:
        self.calls += 1
        # Yield once so two pending tasks have a chance to interleave.
        await asyncio.sleep(0.01)
        return AgentResult(agent=self.name, status=StepStatus.SUCCEEDED, summary="ok")


class TestConcurrentSameKey:
    async def test_two_callers_same_key_run_agent_once(self) -> None:
        """Two `gather`-launched callers with the same dedup key must
        invoke the agent exactly once; the loser reads the cache."""
        store = InMemoryRunStore()
        engine = WorkflowEngine(
            run_id="r-conc",
            config=WorkflowConfig(max_attempts=1, base_backoff_s=0.0),
            record_step=store.append_step,
            cache_get=store.cache_get,
            cache_put=store.cache_put,
        )
        agent = _Counter()

        first, second = await asyncio.gather(engine.run_one(agent, "q"), engine.run_one(agent, "q"))
        assert {first.status, second.status} == {StepStatus.SUCCEEDED, StepStatus.CACHED}
        assert agent.calls == 1

    async def test_truly_parallel_callers_call_agent_exactly_once(self) -> None:
        """Even with no prior cache priming, the per-key lock guarantees
        a single agent invocation across two concurrent callers."""
        store = InMemoryRunStore()
        engine = WorkflowEngine(
            run_id="r-conc-2",
            config=WorkflowConfig(max_attempts=1, base_backoff_s=0.0),
            record_step=store.append_step,
            cache_get=store.cache_get,
            cache_put=store.cache_put,
        )
        agent = _Counter()
        a, b = await asyncio.gather(engine.run_one(agent, "q"), engine.run_one(agent, "q"))
        # One winner (SUCCEEDED), one loser (CACHED).
        assert {a.status, b.status} == {StepStatus.SUCCEEDED, StepStatus.CACHED}
        assert agent.calls == 1

    async def test_many_concurrent_callers_call_agent_exactly_once(self) -> None:
        """Stress: 16 simultaneous callers, still exactly one agent call."""
        store = InMemoryRunStore()
        engine = WorkflowEngine(
            run_id="r-conc-many",
            config=WorkflowConfig(max_attempts=1, base_backoff_s=0.0),
            record_step=store.append_step,
            cache_get=store.cache_get,
            cache_put=store.cache_put,
        )
        agent = _Counter()
        results = await asyncio.gather(*(engine.run_one(agent, "q") for _ in range(16)))
        statuses = [r.status for r in results]
        assert statuses.count(StepStatus.SUCCEEDED) == 1
        assert statuses.count(StepStatus.CACHED) == 15
        assert agent.calls == 1


class TestLockRegistryRefcounting:
    """The per-key lock registry must persist while waiters are queued.

    Round-2 review caught the case where `_release_key` ran inside the
    `async with lock` block, popping the entry while another waiter was
    still blocked on the same lock. A third arrival after the pop would
    create a *new* lock for the same key, violating the single-flight
    contract. The fix is refcounted entries that only evict on idle.
    """

    async def test_lock_persists_under_continuous_arrivals(self) -> None:
        """Stagger 50 same-key callers so the registry never drains
        between releases, and assert the agent still runs exactly once.

        If the registry evicted prematurely, a late arrival would build a
        fresh lock and the agent would fire a second time — this test
        is the regression guard for that race.
        """
        store = InMemoryRunStore()
        engine = WorkflowEngine(
            run_id="r-staggered",
            config=WorkflowConfig(max_attempts=1, base_backoff_s=0.0),
            record_step=store.append_step,
            cache_get=store.cache_get,
            cache_put=store.cache_put,
        )
        agent = _Counter()

        async def staggered(delay_s: float) -> None:
            if delay_s > 0:
                await asyncio.sleep(delay_s)
            await engine.run_one(agent, "q")

        # 50 callers fanned out over ~250ms; the first one holds the lock
        # while the remaining 49 queue up behind it.
        tasks = [asyncio.create_task(staggered(i * 0.005)) for i in range(50)]
        await asyncio.gather(*tasks)
        assert agent.calls == 1, (
            f"agent must fire exactly once across 50 staggered callers; got {agent.calls}"
        )

    async def test_lock_evicted_only_when_idle(self) -> None:
        """While a call is in-flight the registry holds exactly one entry
        for the key; once the last caller releases, the entry is evicted.
        """
        store = InMemoryRunStore()
        engine = WorkflowEngine(
            run_id="r-idle",
            config=WorkflowConfig(max_attempts=1, base_backoff_s=0.0),
            record_step=store.append_step,
            cache_get=store.cache_get,
            cache_put=store.cache_put,
        )

        gate = asyncio.Event()
        observed_size_during: int | None = None

        class GatedAgent:
            name = AgentName.WEB_SEARCH

            def __init__(self) -> None:
                self.calls = 0

            async def search(self, q: str) -> AgentResult:
                self.calls += 1
                # Hold the lock until the test releases the gate so we
                # can inspect the registry mid-flight.
                await gate.wait()
                return AgentResult(agent=self.name, status=StepStatus.SUCCEEDED, summary="ok")

        agent = GatedAgent()
        runner = asyncio.create_task(engine.run_one(agent, "q"))
        # Give the runner a chance to acquire the lock and enter agent.search.
        for _ in range(20):
            await asyncio.sleep(0.005)
            if engine._key_locks:
                observed_size_during = len(engine._key_locks)
                break
        gate.set()
        await runner

        assert observed_size_during == 1, (
            f"expected one entry while in-flight, got {observed_size_during}"
        )
        # On idle the registry drops back to empty.
        assert len(engine._key_locks) == 0, (
            f"expected empty registry after run, got {dict(engine._key_locks)}"
        )


class TestRunOnePerEngineIsolated:
    async def test_two_engines_different_run_ids_dont_share_cache(self) -> None:
        store = InMemoryRunStore()
        agent = _Counter()
        e1 = WorkflowEngine(
            run_id="run-A",
            config=WorkflowConfig(max_attempts=1, base_backoff_s=0.0),
            record_step=store.append_step,
            cache_get=store.cache_get,
            cache_put=store.cache_put,
        )
        e2 = WorkflowEngine(
            run_id="run-B",
            config=WorkflowConfig(max_attempts=1, base_backoff_s=0.0),
            record_step=store.append_step,
            cache_get=store.cache_get,
            cache_put=store.cache_put,
        )
        ra = await e1.run_one(agent, "same q")
        rb = await e2.run_one(agent, "same q")
        # Different run_ids ⇒ different dedup keys ⇒ both fired.
        assert ra.status is StepStatus.SUCCEEDED
        assert rb.status is StepStatus.SUCCEEDED
        assert agent.calls == 2
