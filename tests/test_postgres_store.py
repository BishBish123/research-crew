"""Tests for PostgresRunStore and the related migration helpers.

All tests use ``unittest.mock.AsyncMock`` to simulate asyncpg — no real
Postgres connection is made.

Coverage:
* DDL applied on ``setup()`` — three CREATE TABLE statements executed.
* ``put_run`` / ``get_run`` round-trip — row inserted, payload JSON read back.
* ``append_step`` ordered by sequence; UNIQUE constraint idempotency via
  ON CONFLICT DO NOTHING.
* ``step_dedup`` round-trip (``cache_put`` / ``cache_get``).
* ``cache_get`` for a missing key returns ``None``.
* Migration helpers (hash → postgres, streams → postgres) — happy path
  with mocked Redis + mocked pg_pool.
* ``make_run_store()`` factory dispatches to ``PostgresRunStore`` when
  ``RESEARCH_CREW_STORE=postgres`` is set (without connecting to Postgres).
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

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
from research_crew.store import make_run_store
from research_crew.store.migrate import (
    migrate_redis_hash_to_postgres,
    migrate_redis_streams_to_postgres,
)
from research_crew.store.postgres import PostgresRunStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(run_id: str = "pg-r1", question: str = "what is postgres") -> RunStatus:
    return RunStatus(run_id=run_id, question=question, state=StepStatus.RUNNING)


def _step(
    run_id: str = "pg-r1",
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


def _agent_result(summary: str = "cached") -> AgentResult:
    return AgentResult(
        agent=AgentName.WEB_SEARCH,
        status=StepStatus.SUCCEEDED,
        summary=summary,
        citations=[Citation(title="t", url="https://example.com/1")],
        elapsed_ms=10.0,
    )


# ---------------------------------------------------------------------------
# Mock pool factory
# ---------------------------------------------------------------------------


def _make_mock_pool() -> tuple[MagicMock, AsyncMock]:
    """Return a mock asyncpg pool whose acquire() context manager yields a
    mock connection.  The connection's execute / fetchrow / fetch / fetchval
    methods are AsyncMocks so they can be awaited by the store methods.
    """
    conn = AsyncMock()
    pool = MagicMock()
    # acquire() must work as an async context manager:  async with pool.acquire() as conn
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=conn)
    cm.__aexit__ = AsyncMock(return_value=False)
    pool.acquire = MagicMock(return_value=cm)
    return pool, conn


# ---------------------------------------------------------------------------
# setup() — DDL is applied
# ---------------------------------------------------------------------------


class TestSetup:
    async def test_setup_executes_three_ddl_statements(self) -> None:
        pool, conn = _make_mock_pool()
        store = PostgresRunStore(pool)
        await store.setup()
        assert conn.execute.call_count == 3


# ---------------------------------------------------------------------------
# put_run / get_run
# ---------------------------------------------------------------------------


class TestPutGetRun:
    async def test_put_run_calls_insert_upsert(self) -> None:
        pool, conn = _make_mock_pool()
        store = PostgresRunStore(pool)
        run = _run()
        await store.put_run(run)
        # execute should have been called once with an INSERT … ON CONFLICT statement
        assert conn.execute.call_count == 1
        sql: str = conn.execute.call_args[0][0]
        assert "INSERT INTO runs" in sql
        assert "ON CONFLICT" in sql

    async def test_get_run_returns_none_when_missing(self) -> None:
        pool, conn = _make_mock_pool()
        conn.fetchrow.return_value = None
        store = PostgresRunStore(pool)
        result = await store.get_run("does-not-exist")
        assert result is None

    async def test_put_then_get_roundtrip(self) -> None:
        """Simulate a full put→get round-trip via the mock layer."""
        pool, conn = _make_mock_pool()
        store = PostgresRunStore(pool)

        run = _run("rt-1", "round-trip question")

        # Teach fetchrow what to return when get_run is called.
        payload = json.loads(run.model_dump_json())
        row_mock = MagicMock()
        row_mock.__getitem__ = MagicMock(return_value=payload)
        conn.fetchrow.return_value = row_mock

        await store.put_run(run)
        got = await store.get_run("rt-1")

        assert got is not None
        assert got.run_id == "rt-1"
        assert got.question == "round-trip question"
        assert got.state is StepStatus.RUNNING

    async def test_put_run_passes_run_id_as_first_param(self) -> None:
        pool, conn = _make_mock_pool()
        store = PostgresRunStore(pool)
        run = _run("check-id")
        await store.put_run(run)
        positional_args = conn.execute.call_args[0]
        assert positional_args[1] == "check-id"

    async def test_put_run_with_finished_at(self) -> None:
        pool, conn = _make_mock_pool()
        store = PostgresRunStore(pool)
        run = _run()
        run.state = StepStatus.SUCCEEDED
        run.finished_at = datetime.now(UTC)
        await store.put_run(run)
        positional_args = conn.execute.call_args[0]
        # ended_at ($6) should be a non-None ISO string
        assert positional_args[6] is not None


# ---------------------------------------------------------------------------
# append_step / list_steps
# ---------------------------------------------------------------------------


class TestSteps:
    async def test_append_step_inserts_with_unique_conflict(self) -> None:
        pool, conn = _make_mock_pool()
        conn.fetchval.return_value = 0  # existing_count = 0
        store = PostgresRunStore(pool)
        step = _step()
        await store.append_step(step)
        sql: str = conn.execute.call_args[0][0]
        assert "INSERT INTO steps" in sql
        assert "ON CONFLICT" in sql
        assert "DO NOTHING" in sql

    async def test_append_step_sequence_increments(self) -> None:
        """Second call should use sequence = 2 when count returns 1."""
        pool, conn = _make_mock_pool()
        conn.fetchval.return_value = 1  # existing_count = 1
        store = PostgresRunStore(pool)
        await store.append_step(_step())
        positional_args = conn.execute.call_args[0]
        assert positional_args[2] == 2  # sequence parameter ($2)

    async def test_list_steps_returns_empty_for_new_run(self) -> None:
        pool, conn = _make_mock_pool()
        conn.fetch.return_value = []
        store = PostgresRunStore(pool)
        steps = await store.list_steps("empty-run")
        assert steps == []

    async def test_list_steps_ordered_by_sequence(self) -> None:
        pool, conn = _make_mock_pool()
        store = PostgresRunStore(pool)

        step1 = _step("r1", AgentName.WEB_SEARCH)
        step2 = _step("r1", AgentName.SCHOLAR)

        def _row(s: StepRecord) -> MagicMock:
            payload = json.loads(s.model_dump_json())
            m = MagicMock()
            m.__getitem__ = MagicMock(return_value=payload)
            return m

        conn.fetch.return_value = [_row(step1), _row(step2)]
        steps = await store.list_steps("r1")
        assert len(steps) == 2
        assert steps[0].agent is AgentName.WEB_SEARCH
        assert steps[1].agent is AgentName.SCHOLAR

    async def test_list_steps_select_includes_order_by(self) -> None:
        pool, conn = _make_mock_pool()
        conn.fetch.return_value = []
        store = PostgresRunStore(pool)
        await store.list_steps("r1")
        sql: str = conn.fetch.call_args[0][0]
        assert "ORDER BY sequence" in sql


# ---------------------------------------------------------------------------
# cache_get / cache_put (step_dedup)
# ---------------------------------------------------------------------------


class TestStepDedup:
    async def test_cache_get_missing_key_returns_none(self) -> None:
        pool, conn = _make_mock_pool()
        conn.fetchrow.return_value = None
        store = PostgresRunStore(pool)
        result = await store.cache_get("step:nonexistent")
        assert result is None

    async def test_cache_put_then_get_roundtrip(self) -> None:
        pool, conn = _make_mock_pool()
        store = PostgresRunStore(pool)

        result = _agent_result("my cached summary")
        payload = json.loads(result.model_dump_json())
        row_mock = MagicMock()
        row_mock.__getitem__ = MagicMock(return_value=payload)
        conn.fetchrow.return_value = row_mock

        await store.cache_put("step:abc", result)
        got = await store.cache_get("step:abc")

        assert got is not None
        assert got.summary == "my cached summary"
        assert got.citations[0].url == "https://example.com/1"

    async def test_cache_put_uses_upsert(self) -> None:
        pool, conn = _make_mock_pool()
        store = PostgresRunStore(pool)
        await store.cache_put("step:key", _agent_result())
        sql: str = conn.execute.call_args[0][0]
        assert "INSERT INTO step_dedup" in sql
        assert "ON CONFLICT" in sql

    async def test_cache_get_queries_step_dedup_table(self) -> None:
        pool, conn = _make_mock_pool()
        conn.fetchrow.return_value = None
        store = PostgresRunStore(pool)
        await store.cache_get("step:x")
        sql: str = conn.fetchrow.call_args[0][0]
        assert "step_dedup" in sql


# ---------------------------------------------------------------------------
# Migration helpers
# ---------------------------------------------------------------------------


class TestMigrateRedisHashToPostgres:
    async def test_happy_path_hash_store(self) -> None:
        """Reads run blob from Redis string key; inserts run + steps into Postgres."""
        fake_r = fake_aioredis.FakeRedis(decode_responses=True)
        try:
            run = RunStatus(
                run_id="mig-1",
                question="test question",
                state=StepStatus.SUCCEEDED,
            )
            step = StepRecord(
                run_id="mig-1",
                agent=AgentName.NEWS,
                status=StepStatus.SUCCEEDED,
                attempts=1,
                started_at=datetime.now(UTC),
            )

            # Populate fake Redis with hash-store layout
            await fake_r.set("research:run:mig-1", run.model_dump_json())
            await fake_r.rpush("research:run:mig-1:steps", step.model_dump_json())  # type: ignore[misc]

            pool, conn = _make_mock_pool()
            conn.fetchval.return_value = 0  # existing_count for append_step

            count = await migrate_redis_hash_to_postgres(fake_r, pool, "mig-1")

            assert count == 1
            # put_run and append_step each call conn.execute once each
            assert conn.execute.call_count == 2
        finally:
            await fake_r.aclose()

    async def test_missing_run_returns_zero(self) -> None:
        fake_r = fake_aioredis.FakeRedis(decode_responses=True)
        try:
            pool, _conn = _make_mock_pool()
            count = await migrate_redis_hash_to_postgres(fake_r, pool, "no-such-run")
            assert count == 0
        finally:
            await fake_r.aclose()

    async def test_idempotent_second_call(self) -> None:
        """Second call is a no-op at the Postgres level (ON CONFLICT)."""
        fake_r = fake_aioredis.FakeRedis(decode_responses=True)
        try:
            run = RunStatus(run_id="idem-1", question="q", state=StepStatus.RUNNING)
            await fake_r.set("research:run:idem-1", run.model_dump_json())

            pool, conn = _make_mock_pool()
            conn.fetchval.return_value = 0

            count1 = await migrate_redis_hash_to_postgres(fake_r, pool, "idem-1")
            count2 = await migrate_redis_hash_to_postgres(fake_r, pool, "idem-1")
            # No steps → returns 0 both times; run upsert is harmless
            assert count1 == 0
            assert count2 == 0
        finally:
            await fake_r.aclose()


class TestMigrateRedisStreamsToPostgres:
    async def test_happy_path_streams_store(self) -> None:
        """Reads run blob from Redis hash; reads steps from stream."""
        fake_r = fake_aioredis.FakeRedis(decode_responses=True)
        try:
            run = RunStatus(
                run_id="smig-1",
                question="streams question",
                state=StepStatus.RUNNING,
            )
            step = StepRecord(
                run_id="smig-1",
                agent=AgentName.SCHOLAR,
                status=StepStatus.SUCCEEDED,
                attempts=1,
                started_at=datetime.now(UTC),
            )

            # Populate streams-store layout
            mapping = {k: json.dumps(v) for k, v in run.model_dump(mode="json").items()}
            await fake_r.hset("research:run:smig-1", mapping=mapping)  # type: ignore[misc]
            await fake_r.xadd("research:stream:smig-1:steps", {"payload": step.model_dump_json()})

            pool, conn = _make_mock_pool()
            conn.fetchval.return_value = 0

            count = await migrate_redis_streams_to_postgres(fake_r, pool, "smig-1")

            assert count == 1
            assert conn.execute.call_count == 2
        finally:
            await fake_r.aclose()

    async def test_missing_run_returns_zero(self) -> None:
        fake_r = fake_aioredis.FakeRedis(decode_responses=True)
        try:
            pool, _conn = _make_mock_pool()
            count = await migrate_redis_streams_to_postgres(fake_r, pool, "ghost-run")
            assert count == 0
        finally:
            await fake_r.aclose()

    async def test_no_steps_returns_zero(self) -> None:
        fake_r = fake_aioredis.FakeRedis(decode_responses=True)
        try:
            run = RunStatus(run_id="no-steps", question="q", state=StepStatus.RUNNING)
            mapping = {k: json.dumps(v) for k, v in run.model_dump(mode="json").items()}
            await fake_r.hset("research:run:no-steps", mapping=mapping)  # type: ignore[misc]
            # No stream entries

            pool, _conn = _make_mock_pool()
            count = await migrate_redis_streams_to_postgres(fake_r, pool, "no-steps")
            assert count == 0
        finally:
            await fake_r.aclose()


# ---------------------------------------------------------------------------
# make_run_store() factory
# ---------------------------------------------------------------------------


class TestMakeRunStoreFactory:
    def test_postgres_backend_returns_postgres_run_store(self) -> None:
        """``RESEARCH_CREW_STORE=postgres`` must return a ``PostgresRunStore``
        without actually connecting to Postgres (pool is created lazily in setup()).
        """
        with patch.dict(
            os.environ,
            {
                "RESEARCH_CREW_STORE": "postgres",
                "RESEARCH_PG_DSN": "postgresql://research:research@localhost:5432/research",
            },
        ):
            store = make_run_store()

        assert isinstance(store, PostgresRunStore)

    def test_postgres_backend_reads_dsn_env_var(self) -> None:
        """``RESEARCH_PG_DSN`` env var must be forwarded to the store."""
        custom_dsn = "postgresql://user:pass@neon.tech:5432/mydb"
        with patch.dict(
            os.environ,
            {"RESEARCH_CREW_STORE": "postgres", "RESEARCH_PG_DSN": custom_dsn},
        ):
            store = make_run_store()

        assert isinstance(store, PostgresRunStore)
        # Pool is None until setup() is called — DSN should be stored
        assert store._dsn == custom_dsn

    def test_unknown_backend_raises(self) -> None:
        with (
            patch.dict(os.environ, {"RESEARCH_CREW_STORE": "banana"}),
            pytest.raises(ValueError, match="banana"),
        ):
            make_run_store()

    def test_error_message_includes_postgres(self) -> None:
        with (
            patch.dict(os.environ, {"RESEARCH_CREW_STORE": "invalid"}),
            pytest.raises(ValueError, match="postgres"),
        ):
            make_run_store()
