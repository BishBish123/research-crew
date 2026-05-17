"""Schema-migration tests for StepRecord blobs read through list_steps().

The store's ``list_steps()`` implementation routes each persisted row
through ``_migrate_step_blob``; this module verifies that migration path
works end-to-end for the scenarios that matter:

1. **v1 → current** — a legacy blob written before ``schema_version`` was
   introduced (i.e. the field is absent) is treated as v1, the field is
   stamped, and the row is returned by ``list_steps()``.

2. **explicit v1** — a blob that already carries ``schema_version: 1``
   (the current value) round-trips without modification.

3. **future version** — a blob whose ``schema_version`` exceeds
   ``CURRENT_SCHEMA_VERSION`` is silently dropped from the list so an
   older reader running alongside a newer deploy never crashes.

4. **mixed list** — a steps list containing a mix of valid, legacy, and
   future blobs returns only the parseable rows in order.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import fakeredis.aioredis as fake_aioredis

from research_crew.models import (
    CURRENT_SCHEMA_VERSION,
    AgentName,
    StepRecord,
    StepStatus,
)
from research_crew.store import RedisRunStore

_RUN_ID = "migration-run-1"
_STEPS_KEY = f"research:run:{_RUN_ID}:steps"

_BASE_STEP: dict[str, object] = {
    "run_id": _RUN_ID,
    "agent": "web_search",
    "status": "succeeded",
    "attempts": 1,
    "started_at": "2026-01-01T00:00:00+00:00",
    "finished_at": "2026-01-01T00:00:01+00:00",
    "error": None,
}


async def _make_store() -> tuple[RedisRunStore, fake_aioredis.FakeRedis]:
    fake = fake_aioredis.FakeRedis(decode_responses=True)
    return RedisRunStore(fake), fake


class TestStepRecordV1ToCurrentMigration:
    async def test_legacy_blob_without_schema_version_is_returned(self) -> None:
        """A blob with no ``schema_version`` is treated as v1 and loaded."""
        store, fake = await _make_store()
        try:
            # Write a blob that has no schema_version field (pre-versioning shape).
            legacy = dict(_BASE_STEP)
            # Do NOT include schema_version.
            await fake.rpush(_STEPS_KEY, json.dumps(legacy))

            steps = await store.list_steps(_RUN_ID)
            assert len(steps) == 1, f"expected 1 step, got {steps}"
            assert steps[0].agent is AgentName.WEB_SEARCH
            assert steps[0].schema_version == 1
        finally:
            await fake.aclose()

    async def test_v1_blob_with_explicit_version_is_returned(self) -> None:
        """A blob carrying ``schema_version: 1`` round-trips unchanged."""
        store, fake = await _make_store()
        try:
            v1_blob = {**_BASE_STEP, "schema_version": 1}
            await fake.rpush(_STEPS_KEY, json.dumps(v1_blob))

            steps = await store.list_steps(_RUN_ID)
            assert len(steps) == 1
            assert steps[0].schema_version == 1
        finally:
            await fake.aclose()

    async def test_current_version_blob_is_returned(self) -> None:
        """A blob at CURRENT_SCHEMA_VERSION is returned as-is."""
        store, fake = await _make_store()
        try:
            current_blob = {**_BASE_STEP, "schema_version": CURRENT_SCHEMA_VERSION}
            await fake.rpush(_STEPS_KEY, json.dumps(current_blob))

            steps = await store.list_steps(_RUN_ID)
            assert len(steps) == 1
            assert steps[0].schema_version == CURRENT_SCHEMA_VERSION
        finally:
            await fake.aclose()

    async def test_future_version_blob_is_silently_dropped(self) -> None:
        """A blob from a newer deploy (schema_version > current) is
        silently omitted from the list so an older reader never crashes.
        """
        store, fake = await _make_store()
        try:
            future_blob = {
                **_BASE_STEP,
                "schema_version": CURRENT_SCHEMA_VERSION + 1,
                "new_field_from_future": "opaque",
            }
            await fake.rpush(_STEPS_KEY, json.dumps(future_blob))

            steps = await store.list_steps(_RUN_ID)
            assert steps == [], (
                f"future-version blob must be dropped; got {steps}"
            )
        finally:
            await fake.aclose()

    async def test_mixed_list_returns_only_parseable_rows_in_order(self) -> None:
        """A steps list containing a legacy blob, a current blob, and a
        future blob returns only the two parseable entries in insertion
        order.
        """
        store, fake = await _make_store()
        try:
            # 1 — legacy (no schema_version)
            legacy = dict(_BASE_STEP)
            legacy["agent"] = "scholar"
            await fake.rpush(_STEPS_KEY, json.dumps(legacy))

            # 2 — current
            current = {**_BASE_STEP, "agent": "news", "schema_version": CURRENT_SCHEMA_VERSION}
            await fake.rpush(_STEPS_KEY, json.dumps(current))

            # 3 — future (should be dropped)
            future = {
                **_BASE_STEP,
                "agent": "wikipedia",
                "schema_version": CURRENT_SCHEMA_VERSION + 5,
            }
            await fake.rpush(_STEPS_KEY, json.dumps(future))

            steps = await store.list_steps(_RUN_ID)
            assert len(steps) == 2, f"expected 2 parseable rows; got {steps}"
            assert steps[0].agent is AgentName.SCHOLAR
            assert steps[1].agent is AgentName.NEWS
        finally:
            await fake.aclose()

    async def test_append_step_then_list_preserves_schema_version(self) -> None:
        """Writing a StepRecord via append_step() and reading it back
        through list_steps() preserves schema_version end-to-end.
        """
        store, fake = await _make_store()
        try:
            step = StepRecord(
                run_id=_RUN_ID,
                agent=AgentName.CODE,
                status=StepStatus.SUCCEEDED,
                attempts=2,
                started_at=datetime.now(UTC),
                finished_at=datetime.now(UTC),
            )
            await store.append_step(step)
            steps = await store.list_steps(_RUN_ID)
            assert len(steps) == 1
            assert steps[0].schema_version == CURRENT_SCHEMA_VERSION
            assert steps[0].agent is AgentName.CODE
            assert steps[0].attempts == 2
        finally:
            await fake.aclose()
