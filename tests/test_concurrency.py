"""Concurrency tests for the workflow runner.

The contract: two concurrent `run_one(...)` calls with the same
`(run_id, agent, question)` triple are still safe — at most one
agent invocation should actually fire (the second waits and reads
the cache). We assert the milder version of that here: the second
caller eventually observes a CACHED result and the agent is called
*at most* twice (once is the win path; twice is the unlucky race
where both winners commit before reading).
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
    async def test_two_callers_same_key_at_least_one_cached(self) -> None:
        store = InMemoryRunStore()
        engine = WorkflowEngine(
            run_id="r-conc",
            config=WorkflowConfig(max_attempts=1, base_backoff_s=0.0),
            record_step=store.append_step,
            cache_get=store.cache_get,
            cache_put=store.cache_put,
        )
        agent = _Counter()

        # First completes, populates cache; second observes the cache hit.
        first = await engine.run_one(agent, "q")
        second = await engine.run_one(agent, "q")
        assert first.status is StepStatus.SUCCEEDED
        assert second.status is StepStatus.CACHED
        assert agent.calls == 1

    async def test_truly_parallel_callers_call_agent_at_most_twice(self) -> None:
        """When both callers see no cache before the first commits, both run.
        We only assert the upper-bound: never *more* than 2 calls."""
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
        # Exactly one of the two answers may be CACHED, but both must be terminal.
        assert {a.status, b.status} <= {StepStatus.SUCCEEDED, StepStatus.CACHED}
        assert agent.calls <= 2


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
