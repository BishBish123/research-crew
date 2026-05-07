"""Postgres-backed RunStore using asyncpg.

Schema (applied idempotently by ``setup()``):

* ``runs``       — canonical RunStatus snapshot (one row per run)
* ``steps``      — append-only step audit log (ordered by ``sequence``)
* ``step_dedup`` — idempotency cache (dedup_key → AgentResult blob)

Activate via the ``make_run_store()`` factory:

    RESEARCH_CREW_STORE=postgres
    RESEARCH_PG_DSN=postgresql://research:research@localhost:5432/research

``asyncpg`` is a soft dependency: if it is absent from the virtualenv,
importing this module raises ``ImportError`` with a helpful message
instead of crashing at the ``import asyncpg`` line so callers that never
set ``RESEARCH_CREW_STORE=postgres`` are unaffected.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import structlog

from research_crew.models import AgentResult, RunStatus, StepRecord
from research_crew.store import migrate_run_blob

if TYPE_CHECKING:
    import asyncpg  # noqa: F401 — type-checker only; runtime import is guarded below

_log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# DDL — applied by setup(), idempotent via IF NOT EXISTS / IF NOT EXISTS
# ---------------------------------------------------------------------------

_DDL_RUNS = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    status      TEXT NOT NULL,
    query       TEXT NOT NULL,
    report_json JSONB,
    started_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    ended_at    TIMESTAMPTZ
);
"""

_DDL_STEPS = """
CREATE TABLE IF NOT EXISTS steps (
    id           BIGSERIAL PRIMARY KEY,
    run_id       TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    sequence     INT  NOT NULL,
    agent_name   TEXT,
    status       TEXT NOT NULL,
    payload_json JSONB,
    recorded_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (run_id, sequence)
);
"""

_DDL_STEP_DEDUP = """
CREATE TABLE IF NOT EXISTS step_dedup (
    dedup_key   TEXT PRIMARY KEY,
    payload_json JSONB,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


def _require_asyncpg() -> Any:
    """Import asyncpg or raise a helpful ImportError."""
    try:
        import asyncpg  # noqa: PLC0415

        return asyncpg
    except ImportError as exc:
        raise ImportError(
            "asyncpg is required for PostgresRunStore. Install it with: pip install asyncpg"
        ) from exc


class PostgresRunStore:
    """RunStore backed by a Postgres/Neon database via asyncpg.

    The constructor takes a *pool* (``asyncpg.Pool``) rather than a DSN
    so callers control connection-pool lifecycle and tests can inject a
    mock pool without a real database.

    Call ``await store.setup()`` once at application startup to apply the
    DDL idempotently.  The method is safe to call multiple times — every
    statement uses ``IF NOT EXISTS``.

    Protocol surface
    ----------------
    * ``get_run``    — SELECT FROM runs + reconstruct RunStatus
    * ``put_run``    — INSERT … ON CONFLICT DO UPDATE (upsert)
    * ``append_step``— INSERT INTO steps; UNIQUE (run_id, sequence) enforces idempotency
    * ``list_steps`` — SELECT … ORDER BY sequence
    * ``cache_get``  — SELECT FROM step_dedup
    * ``cache_put``  — INSERT … ON CONFLICT DO UPDATE (upsert)

    Constructor accepts either an already-open *pool* (for tests and
    direct usage) **or** a *dsn* string.  When a DSN is provided, the
    pool is created lazily on the first call to ``setup()``.
    """

    def __init__(self, pool: Any = None, *, dsn: str | None = None) -> None:
        if pool is None and dsn is None:
            raise ValueError("PostgresRunStore requires either a pool or a dsn")
        self._pool = pool
        self._dsn = dsn

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def setup(self) -> None:
        """Create the pool (if constructed from a DSN) and apply the DDL idempotently.

        Safe to call multiple times — every DDL statement uses ``IF NOT EXISTS``.
        """
        if self._pool is None and self._dsn is not None:
            self._pool = await create_pool(self._dsn)
        async with self._pool.acquire() as conn:
            await conn.execute(_DDL_RUNS)
            await conn.execute(_DDL_STEPS)
            await conn.execute(_DDL_STEP_DEDUP)
        _log.info("postgres_store.setup_complete")

    # ------------------------------------------------------------------
    # RunStore Protocol
    # ------------------------------------------------------------------

    async def get_run(self, run_id: str) -> RunStatus | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT payload_json FROM runs WHERE run_id = $1",
                run_id,
            )
        if row is None:
            return None
        raw: dict[str, Any] = dict(row["payload_json"])
        migrated = migrate_run_blob(raw, key=f"pg:runs:{run_id}")
        if migrated is None:
            return None
        return RunStatus.model_validate(migrated)

    async def put_run(self, run: RunStatus) -> None:
        payload = json.loads(run.model_dump_json())
        ended_at = run.finished_at.isoformat() if run.finished_at else None
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO runs (run_id, status, query, report_json, started_at, ended_at)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (run_id) DO UPDATE
                    SET status      = EXCLUDED.status,
                        query       = EXCLUDED.query,
                        report_json = EXCLUDED.report_json,
                        started_at  = EXCLUDED.started_at,
                        ended_at    = EXCLUDED.ended_at
                """,
                run.run_id,
                run.state.value,
                run.question,
                json.dumps(payload),
                run.started_at.isoformat(),
                ended_at,
            )

    async def append_step(self, step: StepRecord) -> None:
        """Insert a step row.  UNIQUE(run_id, sequence) makes this idempotent."""
        payload = json.loads(step.model_dump_json())
        # Derive sequence from the existing step count + 1.  Races are
        # harmless: the UNIQUE constraint on (run_id, sequence) turns a
        # duplicate insert into a no-op via ON CONFLICT DO NOTHING.
        async with self._pool.acquire() as conn:
            existing_count = await conn.fetchval(
                "SELECT COUNT(*) FROM steps WHERE run_id = $1",
                step.run_id,
            )
            sequence = int(existing_count) + 1
            await conn.execute(
                """
                INSERT INTO steps (run_id, sequence, agent_name, status, payload_json)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (run_id, sequence) DO NOTHING
                """,
                step.run_id,
                sequence,
                step.agent.value,
                step.status.value,
                json.dumps(payload),
            )

    async def list_steps(self, run_id: str) -> list[StepRecord]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT payload_json FROM steps WHERE run_id = $1 ORDER BY sequence",
                run_id,
            )
        steps: list[StepRecord] = []
        for row in rows:
            raw: dict[str, Any] = dict(row["payload_json"])
            migrated = migrate_run_blob(raw, key=f"pg:steps:{run_id}")
            if migrated is None:
                continue
            steps.append(StepRecord.model_validate(migrated))
        return steps

    async def cache_get(self, dedup_key: str) -> AgentResult | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT payload_json FROM step_dedup WHERE dedup_key = $1",
                dedup_key,
            )
        if row is None:
            return None
        raw: dict[str, Any] = dict(row["payload_json"])
        return AgentResult.model_validate(raw)

    async def cache_put(self, dedup_key: str, result: AgentResult) -> None:
        payload = json.loads(result.model_dump_json())
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO step_dedup (dedup_key, payload_json)
                VALUES ($1, $2)
                ON CONFLICT (dedup_key) DO UPDATE
                    SET payload_json = EXCLUDED.payload_json
                """,
                dedup_key,
                json.dumps(payload),
            )


# ---------------------------------------------------------------------------
# Pool factory (used by make_run_store; separate so tests can bypass it)
# ---------------------------------------------------------------------------


async def create_pool(dsn: str) -> Any:
    """Create an ``asyncpg`` connection pool from *dsn*."""
    asyncpg = _require_asyncpg()
    return await asyncpg.create_pool(dsn)


__all__ = [
    "PostgresRunStore",
    "create_pool",
]
