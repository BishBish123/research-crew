"""Workflow runner tests — durability semantics on a single agent + parallel."""

from __future__ import annotations

import asyncio

from research_crew.agents import MockAgent
from research_crew.models import AgentName, AgentResult, StepStatus
from research_crew.store import InMemoryRunStore
from research_crew.workflow import WorkflowConfig, WorkflowEngine


def _engine(store: InMemoryRunStore, *, max_attempts: int = 3) -> WorkflowEngine:
    return WorkflowEngine(
        run_id="run-test",
        config=WorkflowConfig(max_attempts=max_attempts, base_backoff_s=0.0),
        record_step=store.append_step,
        cache_get=store.cache_get,
        cache_put=store.cache_put,
    )


class TestSingleStep:
    async def test_happy_path_records_success(self) -> None:
        store = InMemoryRunStore()
        engine = _engine(store)
        agent = MockAgent(name=AgentName.WEB_SEARCH, latency_ms=0)
        result = await engine.run_one(agent, "what is X")
        assert result.status is StepStatus.SUCCEEDED
        assert result.attempts == 1
        steps = await store.list_steps("run-test")
        # One RUNNING + one SUCCEEDED record per attempt.
        assert any(s.status is StepStatus.SUCCEEDED for s in steps)

    async def test_idempotency_returns_cached_on_second_call(self) -> None:
        store = InMemoryRunStore()
        engine = _engine(store)
        agent = MockAgent(name=AgentName.WEB_SEARCH, latency_ms=0)
        first = await engine.run_one(agent, "same question")
        second = await engine.run_one(agent, "same question")
        assert first.status is StepStatus.SUCCEEDED
        assert second.status is StepStatus.CACHED
        # Cached path doesn't re-record running.
        statuses = [s.status for s in await store.list_steps("run-test")]
        assert statuses.count(StepStatus.SUCCEEDED) == 1

    async def test_retries_then_succeeds(self) -> None:
        store = InMemoryRunStore()
        engine = _engine(store, max_attempts=5)
        # Use a deterministic counter on agent's attempt — this MockAgent
        # fails the first 2 attempts, succeeds on the 3rd.
        attempts: list[int] = []

        class FlakyAgent:
            name = AgentName.WEB_SEARCH

            async def search(self, question: str) -> AgentResult:
                attempts.append(1)
                if len(attempts) < 3:
                    return AgentResult(
                        agent=self.name, status=StepStatus.FAILED, summary="", error="boom"
                    )
                return AgentResult(agent=self.name, status=StepStatus.SUCCEEDED, summary="ok")

        result = await engine.run_one(FlakyAgent(), "q")
        assert result.status is StepStatus.SUCCEEDED
        assert result.attempts == 3
        assert len(attempts) == 3

    async def test_exhausts_retries_returns_failed(self) -> None:
        store = InMemoryRunStore()
        engine = _engine(store, max_attempts=2)

        class AlwaysFails:
            name = AgentName.WEB_SEARCH

            async def search(self, question: str) -> AgentResult:
                raise RuntimeError("permanent failure")

        result = await engine.run_one(AlwaysFails(), "q")
        assert result.status is StepStatus.FAILED
        assert result.attempts == 2
        assert "permanent failure" in (result.error or "")

    async def test_per_step_timeout(self) -> None:
        store = InMemoryRunStore()
        engine = WorkflowEngine(
            run_id="run-test",
            config=WorkflowConfig(max_attempts=1, base_backoff_s=0.0, per_step_timeout_s=0.05),
            record_step=store.append_step,
            cache_get=store.cache_get,
            cache_put=store.cache_put,
        )

        class SlowAgent:
            name = AgentName.WEB_SEARCH

            async def search(self, question: str) -> AgentResult:
                await asyncio.sleep(1.0)
                return AgentResult(agent=self.name, status=StepStatus.SUCCEEDED, summary="x")

        result = await engine.run_one(SlowAgent(), "q")
        assert result.status is StepStatus.FAILED
        assert "timeout" in (result.error or "").lower()


class TestParallel:
    async def test_runs_every_agent_concurrently(self) -> None:
        store = InMemoryRunStore()
        engine = _engine(store)
        agents = [
            MockAgent(name=AgentName.WEB_SEARCH, latency_ms=0),
            MockAgent(name=AgentName.SCHOLAR, latency_ms=0),
            MockAgent(name=AgentName.WIKIPEDIA, latency_ms=0),
        ]
        results = await engine.run_parallel(agents, "what is python")
        assert {r.agent for r in results} == {a.name for a in agents}
        assert all(r.status is StepStatus.SUCCEEDED for r in results)

    async def test_partial_failure_does_not_abort(self) -> None:
        store = InMemoryRunStore()
        engine = _engine(store, max_attempts=1)

        class BadAgent:
            name = AgentName.NEWS

            async def search(self, question: str) -> AgentResult:
                raise RuntimeError("nope")

        agents = [
            MockAgent(name=AgentName.WEB_SEARCH, latency_ms=0),
            BadAgent(),
            MockAgent(name=AgentName.WIKIPEDIA, latency_ms=0),
        ]
        results = await engine.run_parallel(agents, "q")
        states = {r.agent: r.status for r in results}
        assert states[AgentName.WEB_SEARCH] is StepStatus.SUCCEEDED
        assert states[AgentName.NEWS] is StepStatus.FAILED
        assert states[AgentName.WIKIPEDIA] is StepStatus.SUCCEEDED


class TestDedupKeyStability:
    async def test_dedup_key_changes_with_question(self) -> None:
        engine = WorkflowEngine(run_id="r")
        agent = MockAgent(name=AgentName.WEB_SEARCH)
        a = engine._dedup_key(agent, "q1")
        b = engine._dedup_key(agent, "q2")
        assert a != b

    async def test_dedup_key_changes_with_run_id(self) -> None:
        a = WorkflowEngine(run_id="r1")._dedup_key(MockAgent(name=AgentName.WEB_SEARCH), "q")
        b = WorkflowEngine(run_id="r2")._dedup_key(MockAgent(name=AgentName.WEB_SEARCH), "q")
        assert a != b

    async def test_dedup_key_stable_within_run(self) -> None:
        engine = WorkflowEngine(run_id="r")
        agent = MockAgent(name=AgentName.WEB_SEARCH)
        assert engine._dedup_key(agent, "q") == engine._dedup_key(agent, "q")
