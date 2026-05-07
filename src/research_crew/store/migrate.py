"""Migration helpers for research-crew run stores.

Two helper categories are provided:

1. ``migrate_hash_to_streams`` — copy a single run's data from the
   Redis hash store to the Redis Streams store.

2. ``migrate_redis_hash_to_postgres`` / ``migrate_redis_streams_to_postgres``
   — copy a run from either Redis backend into Postgres; idempotent.

Usage (programmatic) — hash → streams
--------------------------------------
```python
import redis.asyncio as aioredis
from research_crew.store.migrate import migrate_hash_to_streams

async def main():
    r = aioredis.from_url("redis://localhost:6379/0", decode_responses=True)
    copied = await migrate_hash_to_streams(r, "my-run-id")
    print(f"Migrated {copied} step(s)")
    await r.aclose()
```

Usage (programmatic) — Redis hash → Postgres
---------------------------------------------
```python
import redis.asyncio as aioredis
import asyncpg
from research_crew.store.migrate import migrate_redis_hash_to_postgres

async def main():
    r = aioredis.from_url("redis://localhost:6379/0", decode_responses=True)
    pool = await asyncpg.create_pool("postgresql://research:research@localhost/research")
    copied = await migrate_redis_hash_to_postgres(r, pool, "my-run-id")
    print(f"Migrated run + {copied} step(s)")
    await r.aclose()
    await pool.close()
```

Usage (CLI) — hash → streams
------------------------------
```bash
REDIS_URL=redis://localhost:6379/0 \\
    python -m research_crew.store.migrate <run_id> [<run_id> ...]
```

What migrate_hash_to_streams does
----------------------------------
1. Reads the run's canonical JSON blob from the hash-store String key
   ``{prefix}:run:{run_id}`` and writes it as HSET fields into the same
   key (streams backend uses HSET for the run state).
2. Reads all step entries from the hash-store List key
   ``{prefix}:run:{run_id}:steps`` (LRANGE 0 -1) and XADDs each one to
   the streams step-audit Stream ``{prefix}:stream:{run_id}:steps``.
3. Renames the old list key to ``{prefix}:run:{run_id}:steps.migrated``
   so it is no longer treated as hash-store data by the mismatch check,
   but is recoverable in the short term.

The function is idempotent with respect to the stream: re-running it
will append duplicate entries to the stream if the list key was not
renamed on the first call — check the return value and only proceed if
``copied > 0``.

What migrate_redis_hash_to_postgres does
-----------------------------------------
1. Reads the run blob from the Redis String key (same as hash store).
2. Upserts the run row into ``runs`` via ``put_run``.
3. Reads step entries from the Redis List and inserts each into ``steps``
   (UNIQUE constraint on ``(run_id, sequence)`` makes this idempotent).

What migrate_redis_streams_to_postgres does
--------------------------------------------
Same as above but reads step entries from the Redis Stream via XRANGE
instead of LRANGE on the list key.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any

import redis.asyncio as aioredis
import structlog

from research_crew.store import DEFAULT_REDIS_PREFIX, migrate_run_blob

_log = structlog.get_logger(__name__)

_FIELD_PAYLOAD = "payload"


async def migrate_hash_to_streams(
    client: aioredis.Redis,
    run_id: str,
    *,
    prefix: str = DEFAULT_REDIS_PREFIX,
) -> int:
    """Copy one run's data from hash-store format to streams format.

    Args:
        client:  Async Redis client (decode_responses=True recommended).
        run_id:  The run identifier to migrate.
        prefix:  Key prefix (must match both stores).

    Returns:
        Number of step entries copied to the stream (0 if there was
        nothing to migrate or the list key didn't exist).

    Raises:
        ValueError: If the canonical run JSON is unparseable.
    """
    prefix = prefix.rstrip(":")
    run_string_key = f"{prefix}:run:{run_id}"
    run_hash_key = run_string_key  # same key — both stores use this name
    old_list_key = f"{prefix}:run:{run_id}:steps"
    new_stream_key = f"{prefix}:stream:{run_id}:steps"
    migrated_list_key = f"{prefix}:run:{run_id}:steps.migrated"

    # ------------------------------------------------------------------
    # 1. Migrate the run state blob: String → Hash fields
    # ------------------------------------------------------------------
    raw_run = await client.get(run_string_key)
    if raw_run is not None:
        try:
            payload = json.loads(raw_run)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Cannot parse run blob for {run_id!r}: {exc}") from exc

        migrated_run = migrate_run_blob(payload, key=run_string_key)
        if migrated_run is None:
            _log.warning(
                "migrate.run_blob_skipped",
                run_id=run_id,
                reason="unsupported schema_version",
            )
        else:
            # The hash-store wrote this key as a Redis String (SET); the
            # streams store expects a Hash (HSET).  Delete the String key
            # first so Redis does not raise WRONGTYPE.
            await client.delete(run_hash_key)
            # Write each field individually into the hash so the streams
            # store's HGETALL can reassemble it.
            mapping = {k: json.dumps(v) for k, v in migrated_run.items()}
            await client.hset(run_hash_key, mapping=mapping)  # type: ignore[misc]
            _log.info("migrate.run_state_written", run_id=run_id, key=run_hash_key)

    # ------------------------------------------------------------------
    # 2. Migrate step entries: List → Stream
    # ------------------------------------------------------------------
    step_jsons: Any = await client.lrange(old_list_key, 0, -1)  # type: ignore[misc]
    if not step_jsons:
        _log.info("migrate.no_steps_to_migrate", run_id=run_id, list_key=old_list_key)
        return 0

    copied = 0
    for step_json in step_jsons:
        await client.xadd(new_stream_key, {_FIELD_PAYLOAD: step_json})
        copied += 1

    # 3. Rename old list key so it is no longer seen as hash-store data.
    await client.rename(old_list_key, migrated_list_key)
    _log.info(
        "migrate.steps_migrated",
        run_id=run_id,
        count=copied,
        old_key=old_list_key,
        new_stream=new_stream_key,
        archived_key=migrated_list_key,
    )
    return copied


# ---------------------------------------------------------------------------
# Postgres migration helpers
# ---------------------------------------------------------------------------


async def migrate_redis_hash_to_postgres(
    redis: aioredis.Redis,
    pg_pool: Any,
    run_id: str,
    *,
    prefix: str = DEFAULT_REDIS_PREFIX,
) -> int:
    """Copy one run's data from the Redis hash store into Postgres.

    Args:
        redis:   Async Redis client (decode_responses=True recommended).
        pg_pool: asyncpg connection pool (already open).
        run_id:  The run identifier to migrate.
        prefix:  Key prefix matching the hash store.

    Returns:
        Number of step rows written to Postgres (0 if the run had no steps
        or did not exist in Redis).

    The function is idempotent: ``put_run`` is an upsert and
    ``append_step`` uses ``ON CONFLICT (run_id, sequence) DO NOTHING``.
    """
    from research_crew.models import RunStatus, StepRecord  # noqa: PLC0415
    from research_crew.store.postgres import PostgresRunStore  # noqa: PLC0415

    prefix = prefix.rstrip(":")
    run_string_key = f"{prefix}:run:{run_id}"
    list_key = f"{prefix}:run:{run_id}:steps"

    pg = PostgresRunStore(pg_pool)

    # --- run blob ---
    raw_run = await redis.get(run_string_key)
    if raw_run is None:
        _log.warning("migrate_pg.run_not_found", run_id=run_id, key=run_string_key)
        return 0

    try:
        run_payload = json.loads(raw_run)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Cannot parse run blob for {run_id!r}: {exc}") from exc

    migrated_run = migrate_run_blob(run_payload, key=run_string_key)
    if migrated_run is None:
        _log.warning("migrate_pg.run_blob_skipped", run_id=run_id)
        return 0

    run = RunStatus.model_validate(migrated_run)
    await pg.put_run(run)
    _log.info("migrate_pg.run_upserted", run_id=run_id)

    # --- steps ---
    step_jsons: Any = await redis.lrange(list_key, 0, -1)  # type: ignore[misc]
    if not step_jsons:
        _log.info("migrate_pg.no_steps", run_id=run_id)
        return 0

    copied = 0
    for step_json in step_jsons:
        try:
            step_payload = json.loads(step_json)
        except json.JSONDecodeError:
            _log.warning("migrate_pg.step_corrupt", run_id=run_id)
            continue
        migrated_step = migrate_run_blob(step_payload, key=list_key)
        if migrated_step is None:
            continue
        step = StepRecord.model_validate(migrated_step)
        await pg.append_step(step)
        copied += 1

    _log.info("migrate_pg.steps_copied", run_id=run_id, count=copied)
    return copied


async def migrate_redis_streams_to_postgres(
    redis: aioredis.Redis,
    pg_pool: Any,
    run_id: str,
    *,
    prefix: str = DEFAULT_REDIS_PREFIX,
) -> int:
    """Copy one run's data from the Redis Streams store into Postgres.

    Reads the run state from the Hash key (``{prefix}:run:{run_id}``)
    and step entries from the Stream key
    (``{prefix}:stream:{run_id}:steps``) via XRANGE.

    Args:
        redis:   Async Redis client (decode_responses=True recommended).
        pg_pool: asyncpg connection pool (already open).
        run_id:  The run identifier to migrate.
        prefix:  Key prefix matching the streams store.

    Returns:
        Number of step rows written to Postgres.

    Idempotent — same upsert/ON-CONFLICT semantics as the hash variant.
    """
    from research_crew.models import RunStatus, StepRecord  # noqa: PLC0415
    from research_crew.store.postgres import PostgresRunStore  # noqa: PLC0415

    prefix = prefix.rstrip(":")
    run_hash_key = f"{prefix}:run:{run_id}"
    stream_key = f"{prefix}:stream:{run_id}:steps"

    pg = PostgresRunStore(pg_pool)

    # --- run blob (stored as Hash by streams backend) ---
    raw_hash: Any = await redis.hgetall(run_hash_key)  # type: ignore[misc]
    if not raw_hash:
        _log.warning("migrate_pg_streams.run_not_found", run_id=run_id, key=run_hash_key)
        return 0

    try:
        run_payload: dict[str, Any] = {k: json.loads(v) for k, v in raw_hash.items()}
    except json.JSONDecodeError as exc:
        raise ValueError(f"Cannot parse streams run blob for {run_id!r}: {exc}") from exc

    migrated_run = migrate_run_blob(run_payload, key=run_hash_key)
    if migrated_run is None:
        _log.warning("migrate_pg_streams.run_blob_skipped", run_id=run_id)
        return 0

    run = RunStatus.model_validate(migrated_run)
    await pg.put_run(run)
    _log.info("migrate_pg_streams.run_upserted", run_id=run_id)

    # --- steps from XRANGE ---
    raw_entries: Any = await redis.xrange(stream_key, "-", "+")
    if not raw_entries:
        _log.info("migrate_pg_streams.no_steps", run_id=run_id)
        return 0

    copied = 0
    for _msg_id, fields in raw_entries:
        payload_str = fields.get(_FIELD_PAYLOAD)
        if payload_str is None:
            continue
        try:
            step_payload = json.loads(payload_str)
        except json.JSONDecodeError:
            _log.warning("migrate_pg_streams.step_corrupt", run_id=run_id)
            continue
        migrated_step = migrate_run_blob(step_payload, key=stream_key)
        if migrated_step is None:
            continue
        step = StepRecord.model_validate(migrated_step)
        await pg.append_step(step)
        copied += 1

    _log.info("migrate_pg_streams.steps_copied", run_id=run_id, count=copied)
    return copied


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m research_crew.store.migrate <run_id> [<run_id> ...]")
        sys.exit(1)

    run_ids = sys.argv[1:]
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    prefix = os.environ.get("RESEARCH_REDIS_PREFIX", DEFAULT_REDIS_PREFIX)

    async def _run() -> None:
        client: aioredis.Redis = aioredis.from_url(redis_url, decode_responses=True)
        try:
            for run_id in run_ids:
                count = await migrate_hash_to_streams(client, run_id, prefix=prefix)
                print(f"[{run_id}] migrated {count} step(s)")
        finally:
            await client.aclose()

    asyncio.run(_run())


if __name__ == "__main__":
    _main()
