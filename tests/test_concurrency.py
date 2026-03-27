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
