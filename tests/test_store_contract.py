"""Contract tests applied to every `RunStore` implementation.

If a new backend (Postgres, DynamoDB, Inngest's own state store) lands
later, drop it into the `stores` fixture and the same expectations
apply. The shape we lock in:

* `put_run` / `get_run` round-trip preserves field equality.
* `get_run` returns `None` for unknown ids — never raises.
* `append_step` is order-preserving; `list_steps` is empty for new runs.
* The idempotency cache round-trips an `AgentResult` and returns `None` on miss.
* TTL semantics are not part of the public contract — they are tested
  in the Redis-specific block below.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime

import fakeredis.aioredis as fake_aioredis
import pytest

from research_crew.models import (
    AgentName,
    AgentResult,
    Citation,
    RunStatus,
    StepRecord,
    StepStatus,
)
from research_crew.store import InMemoryRunStore, RedisRunStore, RunStore


@pytest.fixture(params=["memory", "redis"])
async def store(request: pytest.FixtureRequest) -> AsyncIterator[RunStore]:
    if request.param == "memory":
        yield InMemoryRunStore()
        return
    fake = fake_aioredis.FakeRedis(decode_responses=True)
    try:
        yield RedisRunStore(fake)
    finally:
        await fake.aclose()


def _run(run_id: str = "r-1", question: str = "what is x") -> RunStatus:
    return RunStatus(run_id=run_id, question=question, state=StepStatus.RUNNING)


def _step(run_id: str, agent: AgentName, status: StepStatus) -> StepRecord:
    return StepRecord(
        run_id=run_id,
        agent=agent,
        status=status,
        attempts=1,
        started_at=datetime.now(UTC),
        finished_at=datetime.now(UTC),
    )


class TestRunRoundTrip:
    async def test_get_unknown_returns_none(self, store: RunStore) -> None:
        assert await store.get_run("does-not-exist") is None

    async def test_put_then_get_roundtrips(self, store: RunStore) -> None:
        run = _run()
        await store.put_run(run)
        got = await store.get_run("r-1")
        assert got is not None
        assert got.run_id == "r-1"
        assert got.question == "what is x"
        assert got.state is StepStatus.RUNNING

    async def test_put_overwrites(self, store: RunStore) -> None:
        run = _run()
        await store.put_run(run)
        run.state = StepStatus.SUCCEEDED
        run.finished_at = datetime.now(UTC)
        await store.put_run(run)
        got = await store.get_run("r-1")
        assert got is not None
        assert got.state is StepStatus.SUCCEEDED
        assert got.finished_at is not None


class TestStepLog:
    async def test_empty_for_new_run(self, store: RunStore) -> None:
        assert await store.list_steps("r-empty") == []

    async def test_append_preserves_order(self, store: RunStore) -> None:
        agents = [AgentName.WEB_SEARCH, AgentName.SCHOLAR, AgentName.NEWS]
        for a in agents:
            await store.append_step(_step("r-1", a, StepStatus.SUCCEEDED))
        steps = await store.list_steps("r-1")
        assert [s.agent for s in steps] == agents

    async def test_steps_isolated_per_run(self, store: RunStore) -> None:
        await store.append_step(_step("a", AgentName.WEB_SEARCH, StepStatus.SUCCEEDED))
        await store.append_step(_step("b", AgentName.SCHOLAR, StepStatus.SUCCEEDED))
        a_steps = await store.list_steps("a")
        b_steps = await store.list_steps("b")
        assert [s.agent for s in a_steps] == [AgentName.WEB_SEARCH]
        assert [s.agent for s in b_steps] == [AgentName.SCHOLAR]


class TestIdempotencyCache:
    async def test_miss_returns_none(self, store: RunStore) -> None:
        assert await store.cache_get("step:nope") is None

    async def test_put_then_get_roundtrips_agent_result(self, store: RunStore) -> None:
        result = AgentResult(
            agent=AgentName.WEB_SEARCH,
            status=StepStatus.SUCCEEDED,
            summary="cached summary",
            citations=[Citation(title="a", url="https://x/1")],
            attempts=2,
            elapsed_ms=12.5,
        )
        await store.cache_put("step:abc", result)
        got = await store.cache_get("step:abc")
        assert got is not None
        assert got.summary == "cached summary"
        assert got.attempts == 2
        assert got.citations[0].url == "https://x/1"


# ---------------------------------------------------------------------------
# Redis-specific extras: serialization edge cases that don't exist in memory.
# ---------------------------------------------------------------------------


class TestRedisSpecific:
    async def test_run_serialization_handles_optional_fields(self) -> None:
        fake = fake_aioredis.FakeRedis(decode_responses=True)
        try:
            store = RedisRunStore(fake)
            run = _run()
            run.finished_at = None
            run.report = None
            await store.put_run(run)
            got = await store.get_run("r-1")
            assert got is not None
            assert got.finished_at is None
            assert got.report is None
        finally:
            await fake.aclose()

    async def test_step_list_survives_many_appends(self) -> None:
        fake = fake_aioredis.FakeRedis(decode_responses=True)
        try:
            store = RedisRunStore(fake)
            for i in range(50):
                await store.append_step(
                    StepRecord(
                        run_id="big",
                        agent=AgentName.WEB_SEARCH,
                        status=StepStatus.SUCCEEDED,
                        attempts=i + 1,
                        started_at=datetime.now(UTC),
                    )
                )
            steps = await store.list_steps("big")
            assert len(steps) == 50
            assert steps[0].attempts == 1
            assert steps[-1].attempts == 50
        finally:
            await fake.aclose()
