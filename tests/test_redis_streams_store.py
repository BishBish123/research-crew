"""Tests for RedisStreamRunStore and the make_run_store() factory.

All tests use fakeredis (>=2.24) — no real Redis connection is made.

Coverage:
* XADD / XREADGROUP round-trip
* XACK removes entry from pending list
* XPENDING returns 0 after ack
* Consumer-group fan-out: 5 agents each in their own group all see
  the same input stream entry
* Orphan recovery: a step XADDed whose consumer crashed before XACK
  is re-claimable via XPENDING + XCLAIM
* Schema migration detection: if hash-store list data exists for a run
  and the streams backend is selected, StoreBackendMismatchError fires
* Round-trip: write a full run via streams store, read it back, fields match
* make_run_store() factory: env var selects correct backend
* RunStore Protocol: get_run, put_run, append_step, list_steps,
  cache_get, cache_put all work via the streams store
"""

from __future__ import annotations

import json
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
from research_crew.store import InMemoryRunStore, RedisRunStore, make_run_store
from research_crew.store.migrate import migrate_hash_to_streams
from research_crew.store.redis_streams import RedisStreamRunStore, StoreBackendMismatchError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(run_id: str = "r-1", question: str = "what is x") -> RunStatus:
    return RunStatus(run_id=run_id, question=question, state=StepStatus.RUNNING)


def _step(
    run_id: str,
    agent: AgentName = AgentName.WEB_SEARCH,
    status: StepStatus = StepStatus.SUCCEEDED,
) -> StepRecord:
    return StepRecord(
        run_id=run_id,
        agent=agent,
        status=status,
        attempts=1,
        started_at=datetime.now(UTC),
        finished_at=datetime.now(UTC),
    )


@pytest.fixture
async def fake_r() -> AsyncIterator[fake_aioredis.FakeRedis]:
    r = fake_aioredis.FakeRedis(decode_responses=True)
    try:
        yield r
    finally:
        await r.aclose()


@pytest.fixture
async def sstore(fake_r: fake_aioredis.FakeRedis) -> RedisStreamRunStore:
    return RedisStreamRunStore(fake_r, prefix="test")


# ---------------------------------------------------------------------------
# 1. XADD / XREADGROUP round-trip
# ---------------------------------------------------------------------------


class TestXAddXReadGroupRoundTrip:
    async def test_xadd_writes_step_and_xreadgroup_reads_it(
        self, sstore: RedisStreamRunStore
    ) -> None:
        step = _step("run-1")
        await sstore.append_step(step)
        await sstore.ensure_steps_group("run-1")

        entries = await sstore.read_steps_group("run-1", "consumer-A")
        assert len(entries) == 1
        msg_id, read_step = entries[0]
        assert read_step.run_id == "run-1"
        assert read_step.agent == AgentName.WEB_SEARCH
        assert msg_id != ""

    async def test_xrange_list_steps_returns_same_entry(self, sstore: RedisStreamRunStore) -> None:
        step = _step("run-2", AgentName.SCHOLAR)
        await sstore.append_step(step)

        steps = await sstore.list_steps("run-2")
        assert len(steps) == 1
        assert steps[0].agent == AgentName.SCHOLAR


# ---------------------------------------------------------------------------
# 2. XACK removes entry from pending
# ---------------------------------------------------------------------------


class TestXAck:
    async def test_xack_removes_from_pending(self, sstore: RedisStreamRunStore) -> None:
        step = _step("run-ack")
        await sstore.append_step(step)
        await sstore.ensure_steps_group("run-ack")

        entries = await sstore.read_steps_group("run-ack", "consumer-A")
        assert len(entries) == 1
        msg_id, _ = entries[0]

        # Before ack: one pending entry
        pending_before = await sstore.pending_steps("run-ack")
        assert pending_before["pending"] == 1

        acked = await sstore.ack_step("run-ack", msg_id)
        assert acked == 1

        pending_after = await sstore.pending_steps("run-ack")
        assert pending_after["pending"] == 0


# ---------------------------------------------------------------------------
# 3. XPENDING returns 0 after ack
# ---------------------------------------------------------------------------


class TestXPending:
    async def test_pending_zero_after_ack(self, sstore: RedisStreamRunStore) -> None:
        step = _step("run-pend")
        await sstore.append_step(step)

        entries = await sstore.read_steps_group("run-pend", "consumer-B")
        msg_id, _ = entries[0]
        await sstore.ack_step("run-pend", msg_id)

        summary = await sstore.pending_steps("run-pend")
        assert summary["pending"] == 0

    async def test_pending_nonzero_before_ack(self, sstore: RedisStreamRunStore) -> None:
        for i in range(3):
            await sstore.append_step(_step(f"run-multi-{i}"))

        await sstore.append_step(_step("run-three"))
        await sstore.append_step(_step("run-three", AgentName.SCHOLAR))
        await sstore.append_step(_step("run-three", AgentName.NEWS))

        await sstore.read_steps_group("run-three", "consumer-C")
        summary = await sstore.pending_steps("run-three")
        assert summary["pending"] == 3


# ---------------------------------------------------------------------------
# 4. Consumer-group fan-out: 5 agents each see the same input
# ---------------------------------------------------------------------------


class TestConsumerGroupFanOut:
    async def test_five_agents_all_see_same_input_message(
        self, sstore: RedisStreamRunStore
    ) -> None:
        run_id = "run-fanout"
        agent_names = [a.value for a in AgentName]  # 5 agents

        # Publish one input message
        msg_id = await sstore.publish_input(run_id, {"question": "what is rust"})
        assert msg_id

        # Each agent reads via its own consumer group
        for agent in agent_names:
            entries = await sstore.read_input_group(run_id, agent, consumer_name=f"worker-{agent}")
            assert len(entries) == 1, f"Agent {agent!r} did not receive the message"
            read_msg_id, fields = entries[0]
            assert read_msg_id == msg_id
            assert fields["question"] == "what is rust"

    async def test_agent_groups_are_independent(self, sstore: RedisStreamRunStore) -> None:
        """Consuming and ACKing in one group does not affect another group."""
        run_id = "run-independent"
        await sstore.publish_input(run_id, {"q": "test"})

        # web_search reads + acks
        entries_ws = await sstore.read_input_group(run_id, "web_search", "w1")
        assert len(entries_ws) == 1
        msg_id, _ = entries_ws[0]
        await sstore.ack_input(run_id, "web_search", msg_id)

        # scholar hasn't read yet — should still see it
        entries_sc = await sstore.read_input_group(run_id, "scholar", "s1")
        assert len(entries_sc) == 1


# ---------------------------------------------------------------------------
# 5. Orphan recovery via XPENDING + XCLAIM
# ---------------------------------------------------------------------------


class TestOrphanRecovery:
    async def test_crashed_consumer_step_recoverable_via_xclaim(
        self, sstore: RedisStreamRunStore
    ) -> None:
        run_id = "run-orphan"
        step = _step(run_id)
        await sstore.append_step(step)

        # consumer-A reads but "crashes" before ACKing
        await sstore.ensure_steps_group(run_id)
        entries = await sstore.read_steps_group(run_id, "consumer-A")
        assert len(entries) == 1
        orphan_id, _ = entries[0]

        # Orphan shows up in pending range
        pending_range = await sstore.pending_steps_range(run_id)
        assert len(pending_range) == 1
        assert pending_range[0]["message_id"] == orphan_id
        assert pending_range[0]["consumer"] == "consumer-A"

        # consumer-B claims the orphan (min_idle_ms=0 to claim immediately)
        claimed = await sstore.claim_step(run_id, "consumer-B", min_idle_ms=0, message_id=orphan_id)
        assert len(claimed) == 1
        claimed_id, claimed_step = claimed[0]
        assert claimed_id == orphan_id
        assert claimed_step.run_id == run_id

        # Pending now belongs to consumer-B
        pending_range2 = await sstore.pending_steps_range(run_id)
        assert len(pending_range2) == 1
        assert pending_range2[0]["consumer"] == "consumer-B"

        # ACK it from consumer-B
        await sstore.ack_step(run_id, claimed_id)
        summary = await sstore.pending_steps(run_id)
        assert summary["pending"] == 0


# ---------------------------------------------------------------------------
# 6. Schema-migration detection: hash data + streams backend → error
# ---------------------------------------------------------------------------


class TestSchemaMigrationDetection:
    async def test_error_when_hash_list_data_exists(self, fake_r: fake_aioredis.FakeRedis) -> None:
        """append_step raises StoreBackendMismatchError when the old hash-store
        list key (``{prefix}:run:{run_id}:steps``) already exists, indicating
        data that was written by the hash backend."""
        # Simulate hash-store data by writing to the list key directly.
        await fake_r.rpush("test:run:old-run:steps", '{"dummy": true}')

        sstore = RedisStreamRunStore(fake_r, prefix="test")
        with pytest.raises(StoreBackendMismatchError, match="Hash-store list data found"):
            await sstore.append_step(_step("old-run"))

    async def test_no_error_when_only_stream_data_exists(self, sstore: RedisStreamRunStore) -> None:
        """append_step succeeds when no hash-store list key is present."""
        step = _step("clean-run")
        await sstore.append_step(step)  # must not raise
        steps = await sstore.list_steps("clean-run")
        assert len(steps) == 1


# ---------------------------------------------------------------------------
# 7. Round-trip: write a full run via streams store, read it back
# ---------------------------------------------------------------------------


class TestRoundTrip:
    async def test_full_run_round_trip(self, sstore: RedisStreamRunStore) -> None:
        run = _run("rt-1", "how does asyncio work")
        await sstore.put_run(run)

        for agent in [AgentName.WEB_SEARCH, AgentName.SCHOLAR, AgentName.NEWS]:
            await sstore.append_step(_step("rt-1", agent))

        got_run = await sstore.get_run("rt-1")
        assert got_run is not None
        assert got_run.run_id == "rt-1"
        assert got_run.question == "how does asyncio work"
        assert got_run.state == StepStatus.RUNNING

        steps = await sstore.list_steps("rt-1")
        assert len(steps) == 3
        assert steps[0].agent == AgentName.WEB_SEARCH
        assert steps[1].agent == AgentName.SCHOLAR
        assert steps[2].agent == AgentName.NEWS

    async def test_cache_round_trip(self, sstore: RedisStreamRunStore) -> None:
        result = AgentResult(
            agent=AgentName.WEB_SEARCH,
            status=StepStatus.SUCCEEDED,
            summary="streams cached summary",
            citations=[Citation(title="t", url="https://example.com/1")],
            elapsed_ms=42.0,
        )
        await sstore.cache_put("step:abc123", result)
        got = await sstore.cache_get("step:abc123")
        assert got is not None
        assert got.summary == "streams cached summary"
        assert got.citations[0].url == "https://example.com/1"

    async def test_get_unknown_run_returns_none(self, sstore: RedisStreamRunStore) -> None:
        assert await sstore.get_run("does-not-exist") is None

    async def test_list_steps_empty_for_new_run(self, sstore: RedisStreamRunStore) -> None:
        assert await sstore.list_steps("brand-new") == []

    async def test_put_run_overwrite(self, sstore: RedisStreamRunStore) -> None:
        run = _run("overwrite-1")
        await sstore.put_run(run)
        run.state = StepStatus.SUCCEEDED
        run.finished_at = datetime.now(UTC)
        await sstore.put_run(run)
        got = await sstore.get_run("overwrite-1")
        assert got is not None
        assert got.state == StepStatus.SUCCEEDED
        assert got.finished_at is not None


# ---------------------------------------------------------------------------
# 8. make_run_store() factory
# ---------------------------------------------------------------------------


class TestMakeRunStoreFactory:
    def test_default_is_redis_run_store(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("RESEARCH_CREW_STORE", raising=False)
        monkeypatch.setenv("REDIS_URL", "redis://localhost:9999/0")
        # We can't connect but we can check the type returned.
        store = make_run_store()
        assert isinstance(store, RedisRunStore)

    def test_hash_env_gives_redis_run_store(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RESEARCH_CREW_STORE", "hash")
        monkeypatch.setenv("REDIS_URL", "redis://localhost:9999/0")
        store = make_run_store()
        assert isinstance(store, RedisRunStore)

    def test_streams_env_gives_streams_store(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RESEARCH_CREW_STORE", "streams")
        monkeypatch.setenv("REDIS_URL", "redis://localhost:9999/0")
        store = make_run_store()
        assert isinstance(store, RedisStreamRunStore)

    def test_memory_env_gives_in_memory_store(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RESEARCH_CREW_STORE", "memory")
        store = make_run_store()
        assert isinstance(store, InMemoryRunStore)

    def test_invalid_env_raises_value_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RESEARCH_CREW_STORE", "dynamo")
        with pytest.raises(ValueError, match="Unknown RESEARCH_CREW_STORE"):
            make_run_store()


# ---------------------------------------------------------------------------
# 9. Migration helper
# ---------------------------------------------------------------------------


class TestMigrateHashToStreams:
    async def test_copies_steps_from_list_to_stream(self, fake_r: fake_aioredis.FakeRedis) -> None:
        """migrate_hash_to_streams copies list entries → stream and renames list key."""
        run_id = "migrate-1"
        prefix = "test"
        list_key = f"{prefix}:run:{run_id}:steps"
        run_key = f"{prefix}:run:{run_id}"
        stream_key = f"{prefix}:stream:{run_id}:steps"
        archived_key = f"{prefix}:run:{run_id}:steps.migrated"

        # Write hash-store format data
        step = _step(run_id, AgentName.CODE)
        run = _run(run_id)
        await fake_r.set(run_key, run.model_dump_json())
        await fake_r.rpush(list_key, step.model_dump_json())

        copied = await migrate_hash_to_streams(fake_r, run_id, prefix=prefix)
        assert copied == 1

        # Stream must have one entry
        entries = await fake_r.xrange(stream_key, "-", "+")
        assert len(entries) == 1
        payload = json.loads(entries[0][1]["payload"])
        assert payload["agent"] == "code"

        # Old list key is gone, archived key exists
        assert not await fake_r.exists(list_key)
        assert await fake_r.exists(archived_key)

    async def test_run_state_written_as_hash_fields(self, fake_r: fake_aioredis.FakeRedis) -> None:
        run_id = "migrate-2"
        prefix = "test"
        run_key = f"{prefix}:run:{run_id}"

        run = _run(run_id, "test question")
        await fake_r.set(run_key, run.model_dump_json())

        await migrate_hash_to_streams(fake_r, run_id, prefix=prefix)

        # After migration the hash fields should exist
        hdata = await fake_r.hgetall(run_key)
        assert "run_id" in hdata
        assert "question" in hdata

        sstore = RedisStreamRunStore(fake_r, prefix=prefix)
        got = await sstore.get_run(run_id)
        assert got is not None
        assert got.run_id == run_id
        assert got.question == "test question"

    async def test_no_steps_returns_zero(self, fake_r: fake_aioredis.FakeRedis) -> None:
        copied = await migrate_hash_to_streams(fake_r, "empty-run", prefix="test")
        assert copied == 0

    async def test_after_migration_streams_store_can_append(
        self, fake_r: fake_aioredis.FakeRedis
    ) -> None:
        """After migration the StoreBackendMismatchError must not fire because
        the old list key has been renamed."""
        run_id = "migrate-3"
        prefix = "test"
        list_key = f"{prefix}:run:{run_id}:steps"

        await fake_r.rpush(list_key, _step(run_id).model_dump_json())
        await migrate_hash_to_streams(fake_r, run_id, prefix=prefix)

        sstore = RedisStreamRunStore(fake_r, prefix=prefix)
        # Should not raise after migration
        await sstore.append_step(_step(run_id, AgentName.WIKIPEDIA))
        steps = await sstore.list_steps(run_id)
        # One migrated + one new
        assert len(steps) == 2
