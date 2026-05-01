"""Edge-case workflow tests beyond the happy path / single retry / single timeout.

Covers:

* Timeout-then-success on the next attempt (the timeout path *also* retries).
* Retry budget exhaustion records exactly `max_attempts` failure rows.
* Idempotency cache hit short-circuits before the agent ever runs.
* Cancellation propagates and still leaves a FAILED step record behind.
* Parallel fan-out keeps successful agents isolated from a single hard failure.
"""

from __future__ import annotations

import asyncio

import pytest

from research_crew.agents import MockAgent
from research_crew.models import AgentName, AgentResult, StepStatus
from research_crew.store import InMemoryRunStore
from research_crew.workflow import WorkflowConfig, WorkflowEngine


def _engine(
    store: InMemoryRunStore, *, max_attempts: int = 3, timeout_s: float = 30.0
) -> WorkflowEngine:
    return WorkflowEngine(
        run_id="run-edge",
        config=WorkflowConfig(
            max_attempts=max_attempts, base_backoff_s=0.0, per_step_timeout_s=timeout_s
        ),
        record_step=store.append_step,
        cache_get=store.cache_get,
        cache_put=store.cache_put,
    )


class TestTimeoutThenRetry:
    async def test_timeout_first_then_succeeds(self) -> None:
        store = InMemoryRunStore()
        engine = _engine(store, max_attempts=3, timeout_s=0.05)

        class FlakyTimeoutAgent:
            name = AgentName.WEB_SEARCH

            def __init__(self) -> None:
                self.calls = 0

            async def search(self, question: str) -> AgentResult:
                self.calls += 1
                if self.calls == 1:
                    await asyncio.sleep(1.0)  # exceeds 0.05s budget
                return AgentResult(agent=self.name, status=StepStatus.SUCCEEDED, summary="ok")

        agent = FlakyTimeoutAgent()
        result = await engine.run_one(agent, "q")
        assert result.status is StepStatus.SUCCEEDED
        assert result.attempts == 2
        assert agent.calls == 2

        steps = await store.list_steps("run-edge")
        timed_out = [s for s in steps if s.status is StepStatus.FAILED]
        assert len(timed_out) == 1
        assert "timed out" in (timed_out[0].error or "").lower()


class TestRetryBudget:
    async def test_records_one_failure_per_attempt(self) -> None:
        store = InMemoryRunStore()
        engine = _engine(store, max_attempts=3)

        class AlwaysBoom:
            name = AgentName.SCHOLAR

            async def search(self, question: str) -> AgentResult:
                raise RuntimeError("boom")

        result = await engine.run_one(AlwaysBoom(), "q")
        assert result.status is StepStatus.FAILED
        assert result.attempts == 3
        steps = await store.list_steps("run-edge")
        failed = [s for s in steps if s.status is StepStatus.FAILED]
        running = [s for s in steps if s.status is StepStatus.RUNNING]
        assert len(failed) == 3
        assert len(running) == 3

    async def test_max_attempts_one_skips_backoff(self) -> None:
        """With max_attempts=1 there is exactly one attempt and no backoff sleep."""
        store = InMemoryRunStore()
        engine = _engine(store, max_attempts=1)

        class AlwaysBoom:
            name = AgentName.NEWS

            async def search(self, question: str) -> AgentResult:
                raise RuntimeError("boom")

        loop = asyncio.get_running_loop()
        t0 = loop.time()
        result = await engine.run_one(AlwaysBoom(), "q")
        elapsed = loop.time() - t0
        assert result.attempts == 1
        # No backoff sleep should fire when there is no next attempt.
        assert elapsed < 0.1, f"single attempt should not back off, elapsed={elapsed}"


class TestIdempotencyCacheSkipsExecution:
    async def test_cache_hit_does_not_call_agent(self) -> None:
        store = InMemoryRunStore()
        engine = _engine(store)
        question = "expensive question"

        class CountingAgent:
            name = AgentName.WIKIPEDIA

            def __init__(self) -> None:
                self.calls = 0

            async def search(self, q: str) -> AgentResult:
                self.calls += 1
                return AgentResult(agent=self.name, status=StepStatus.SUCCEEDED, summary="hi")

        agent = CountingAgent()
        first = await engine.run_one(agent, question)
        second = await engine.run_one(agent, question)
        third = await engine.run_one(agent, question)

        assert first.status is StepStatus.SUCCEEDED
        assert second.status is StepStatus.CACHED
        assert third.status is StepStatus.CACHED
        # Agent must only be invoked once across three logical runs of the same key.
        assert agent.calls == 1


class TestCancellationCleanup:
    async def test_cancelling_propagates_and_records_failure(self) -> None:
        store = InMemoryRunStore()
        engine = _engine(store, max_attempts=1)
        started = asyncio.Event()

        class HangingAgent:
            name = AgentName.CODE

            async def search(self, q: str) -> AgentResult:
                started.set()
                await asyncio.sleep(10)  # will be cancelled
                return AgentResult(agent=self.name, status=StepStatus.SUCCEEDED, summary="x")

        task = asyncio.create_task(engine.run_one(HangingAgent(), "q"))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # The run-record must show the FAILED-cancelled row so a `GET /runs/{id}`
        # surfaces *why* the run looks incomplete.
        steps = await store.list_steps("run-edge")
        cancelled_rows = [s for s in steps if s.status is StepStatus.FAILED]
        assert len(cancelled_rows) == 1
        assert (cancelled_rows[0].error or "").lower() == "cancelled"


class TestParallelFanoutIsolation:
    async def test_one_failure_does_not_starve_the_others(self) -> None:
        store = InMemoryRunStore()
        engine = _engine(store, max_attempts=1)

        class HardFail:
            name = AgentName.NEWS

            async def search(self, q: str) -> AgentResult:
                raise RuntimeError("network exploded")

        agents = [
            MockAgent(name=AgentName.WEB_SEARCH, latency_ms=0),
            HardFail(),
            MockAgent(name=AgentName.SCHOLAR, latency_ms=0),
            MockAgent(name=AgentName.WIKIPEDIA, latency_ms=0),
            MockAgent(name=AgentName.CODE, latency_ms=0),
        ]
        results = await engine.run_parallel(agents, "q")
        states = {r.agent: r.status for r in results}
        assert states[AgentName.NEWS] is StepStatus.FAILED
        # Every other agent stays SUCCEEDED — no shared blast radius.
        for n in (AgentName.WEB_SEARCH, AgentName.SCHOLAR, AgentName.WIKIPEDIA, AgentName.CODE):
            assert states[n] is StepStatus.SUCCEEDED, f"{n} got {states[n]}"


class TestNoCacheStillRuns:
    async def test_engine_without_cache_callbacks_still_works(self) -> None:
        """`cache_get`/`cache_put` are optional; engine must not require them."""
        engine = WorkflowEngine(
            run_id="r",
            config=WorkflowConfig(max_attempts=1, base_backoff_s=0.0),
        )
        agent = MockAgent(name=AgentName.WEB_SEARCH, latency_ms=0)
        result = await engine.run_one(agent, "q")
        assert result.status is StepStatus.SUCCEEDED


class TestKeyRegistryCleanup:
    async def test_cache_put_failure_releases_lock_entry(self) -> None:
        """`cache_put` failures are now swallowed (see FIX 1), but the
        per-key lock registry must still drop the entry on the success
        exit path. Otherwise long-running engines leak entries.
        """
        store = InMemoryRunStore()

        class FlakyCache:
            def __init__(self) -> None:
                self.real = store

            async def cache_get(self, dedup_key: str) -> AgentResult | None:
                return await self.real.cache_get(dedup_key)

            async def cache_put(self, dedup_key: str, result: AgentResult) -> None:
                raise RuntimeError("redis ate it")

        flaky = FlakyCache()
        engine = WorkflowEngine(
            run_id="run-edge",
            config=WorkflowConfig(max_attempts=1, base_backoff_s=0.0),
            record_step=store.append_step,
            cache_get=flaky.cache_get,
            cache_put=flaky.cache_put,
        )
        agent = MockAgent(name=AgentName.WEB_SEARCH, latency_ms=0)

        # cache_put errors are observability-only; the agent succeeded.
        result = await engine.run_one(agent, "q")
        assert result.status is StepStatus.SUCCEEDED

        # Registry must be empty: the key cannot remain pinned even
        # when cache_put silently failed.
        assert engine._key_locks == {}, (
            f"expected empty lock registry after run, got {engine._key_locks}"
        )

    async def test_cache_get_failure_releases_lock_entry(self) -> None:
        """Same property when the failure is during `cache_get`. The
        engine treats the error as a cache miss and the agent runs;
        the lock registry must still be empty on exit.
        """
        store = InMemoryRunStore()

        async def boom_get(_: str) -> AgentResult | None:
            raise RuntimeError("get exploded")

        engine = WorkflowEngine(
            run_id="run-edge",
            config=WorkflowConfig(max_attempts=1, base_backoff_s=0.0),
            record_step=store.append_step,
            cache_get=boom_get,
            cache_put=store.cache_put,
        )
        agent = MockAgent(name=AgentName.WEB_SEARCH, latency_ms=0)

        result = await engine.run_one(agent, "q")
        assert result.status is StepStatus.SUCCEEDED

        assert engine._key_locks == {}


class TestStoreFailureDoesNotAbortRun:
    """Store-side hiccups (cache_get / append_step / cache_put) must not
    abort agent work that already succeeded — they are observability /
    optimisation, not correctness. See FIX 1 in the workflow runner.
    """

    async def test_cache_get_failure_treated_as_miss(self) -> None:
        store = InMemoryRunStore()

        async def boom_get(_: str) -> AgentResult | None:
            raise RuntimeError("redis get down")

        engine = WorkflowEngine(
            run_id="run-edge",
            config=WorkflowConfig(max_attempts=1, base_backoff_s=0.0),
            record_step=store.append_step,
            cache_get=boom_get,
            cache_put=store.cache_put,
        )
        agent = MockAgent(name=AgentName.WEB_SEARCH, latency_ms=0)

        result = await engine.run_one(agent, "q")

        assert result.status is StepStatus.SUCCEEDED, (
            "cache_get errors must be treated as a miss, not propagated as run failure"
        )

    async def test_step_record_failure_does_not_abort(self) -> None:
        store = InMemoryRunStore()

        async def boom_record(_step: object) -> None:
            raise RuntimeError("redis rpush down")

        engine = WorkflowEngine(
            run_id="run-edge",
            config=WorkflowConfig(max_attempts=1, base_backoff_s=0.0),
            record_step=boom_record,
            cache_get=store.cache_get,
            cache_put=store.cache_put,
        )
        agent = MockAgent(name=AgentName.WEB_SEARCH, latency_ms=0)

        result = await engine.run_one(agent, "q")

        assert result.status is StepStatus.SUCCEEDED, (
            "append_step errors must not abort the surrounding run; the audit "
            "log is observability, not correctness"
        )

    async def test_cache_put_failure_returns_result(self) -> None:
        store = InMemoryRunStore()

        async def boom_put(_key: str, _result: AgentResult) -> None:
            raise RuntimeError("redis set down")

        engine = WorkflowEngine(
            run_id="run-edge",
            config=WorkflowConfig(max_attempts=1, base_backoff_s=0.0),
            record_step=store.append_step,
            cache_get=store.cache_get,
            cache_put=boom_put,
        )
        agent = MockAgent(name=AgentName.WEB_SEARCH, latency_ms=0)

        result = await engine.run_one(agent, "q")

        # The agent already returned SUCCEEDED; losing the cache write
        # means we'll re-run on the next identical request, not that the
        # run itself failed.
        assert result.status is StepStatus.SUCCEEDED


class TestStepRecordTimings:
    async def test_terminal_step_preserves_real_started_at(self) -> None:
        """Terminal records must carry the wall-clock start captured at
        the top of `run_one` so callers can compute a real duration."""
        store = InMemoryRunStore()
        engine = _engine(store, max_attempts=1)
        # Non-trivial latency so finished_at - started_at is observably > 0.
        agent = MockAgent(name=AgentName.WEB_SEARCH, latency_ms=20)
        await engine.run_one(agent, "q")

        steps = await store.list_steps("run-edge")
        terminals = [s for s in steps if s.status is StepStatus.SUCCEEDED]
        assert terminals, "expected at least one SUCCEEDED step record"
        for step in terminals:
            assert step.finished_at is not None
            assert step.started_at <= step.finished_at
            assert (step.finished_at - step.started_at).total_seconds() > 0.0

    async def test_running_and_terminal_records_share_started_at(self) -> None:
        """RUNNING and the matching SUCCEEDED record for the same attempt
        should report the same `started_at`."""
        store = InMemoryRunStore()
        engine = _engine(store, max_attempts=1)
        agent = MockAgent(name=AgentName.WEB_SEARCH, latency_ms=5)
        await engine.run_one(agent, "q")

        steps = await store.list_steps("run-edge")
        running = next(s for s in steps if s.status is StepStatus.RUNNING)
        succeeded = next(s for s in steps if s.status is StepStatus.SUCCEEDED)
        assert running.started_at == succeeded.started_at
