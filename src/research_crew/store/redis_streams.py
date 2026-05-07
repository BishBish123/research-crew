"""Redis Streams-backed RunStore implementation.

Key / stream naming conventions
--------------------------------
* ``{prefix}:run:{run_id}``          — Hash holding the canonical RunStatus
  snapshot (state, question, report, etc.).  This is identical to the
  hash-store's run key so the same HSET/HGET operations work.
* ``{prefix}:stream:{run_id}:steps`` — Redis Stream (XADD) holding the
  append-only audit log of StepRecord entries for the run.
* ``{prefix}:stream:{run_id}:input`` — Redis Stream (XADD) used for
  inter-agent fan-out: each agent reads from this stream via its own
  consumer group, enabling at-least-once delivery without duplicates
  within a single consumer instance.
* ``{prefix}:step:{dedup_key}``      — String key for the idempotency
  cache (same as the hash store — streams don't replace the cache).

Consumer-group naming conventions
----------------------------------
* Step audit reader group: ``steps-readers``
* Per-agent input group:   ``agent:{agent_name}``

Stream patterns used
--------------------
* ``XADD``       — ``append_step`` (audit log), ``publish_input``
* ``XREADGROUP`` — ``read_steps`` (streaming consumer for the audit log)
* ``XACK``       — acknowledge a step after the consumer processes it
* ``XPENDING``   — surface stuck / abandoned steps for reconciliation
* ``XCLAIM``     — claim an orphaned pending entry from a crashed consumer

This module is opt-in; the default ``RedisRunStore`` is unchanged.  Set
``RESEARCH_CREW_STORE=streams`` to activate via ``make_run_store()``.

Schema migration detection
--------------------------
On first use of a run_id the store checks whether a hash key already
exists under the old hash-store format.  If it does AND the caller is
using the streams backend, ``StoreBackendMismatchError`` is raised so
the operator can run ``migrate_hash_to_streams`` before switching.
"""

from __future__ import annotations

import json
from typing import Any

import redis.asyncio as aioredis
import structlog

from research_crew.models import (
    AgentResult,
    ResearchReport,
    RunStatus,
    StepRecord,
    StepStatus,
)
from research_crew.store import DEFAULT_REDIS_PREFIX, migrate_run_blob

_log = structlog.get_logger(__name__)

# Consumer group name used by read_steps / the streaming audit reader.
_STEPS_READER_GROUP = "steps-readers"

# Stream field names — every XADD entry for a step carries a single
# "payload" field with the JSON-serialised StepRecord.
_FIELD_PAYLOAD = "payload"


class StoreBackendMismatchError(RuntimeError):
    """Raised when hash-store data exists for a run_id but the caller
    is using the streams backend.  Run ``migrate_hash_to_streams`` to
    copy the data across before switching backends.
    """


class RedisStreamRunStore:
    """RunStore backed by Redis Streams (XADD / XREADGROUP / XACK / XPENDING).

    **Run state** (status, report, etc.) is stored in a Redis Hash under
    ``{prefix}:run:{run_id}`` — the same key as the hash-store, using
    ``HSET`` / ``HGETALL``.  This gives atomic field-level updates without
    a full JSON round-trip on every heartbeat.

    **Step audit log** is stored as a Redis Stream under
    ``{prefix}:stream:{run_id}:steps``.  Each ``append_step`` call does an
    ``XADD``; ``list_steps`` reads all entries with ``XRANGE``; consumers
    can use ``XREADGROUP`` for at-least-once delivery.

    **Inter-agent fan-out** uses ``{prefix}:stream:{run_id}:input``.  Each
    agent subscribes via its own consumer group so all agents see the same
    messages without competing for them.

    **Idempotency cache** reuses the same ``{prefix}:step:{dedup_key}``
    String keys as the hash-store — the cache is backend-agnostic.
    """

    def __init__(
        self,
        client: aioredis.Redis,
        ttl_seconds: int = 3600,
        prefix: str = DEFAULT_REDIS_PREFIX,
    ) -> None:
        self._r = client
        self._ttl = ttl_seconds
        self._prefix = prefix.rstrip(":")

    @property
    def prefix(self) -> str:
        return self._prefix

    # ------------------------------------------------------------------
    # Internal key helpers
    # ------------------------------------------------------------------

    def _run_hash_key(self, run_id: str) -> str:
        """Hash key for the canonical RunStatus snapshot."""
        return f"{self._prefix}:run:{run_id}"

    def _steps_stream_key(self, run_id: str) -> str:
        """Stream key for the per-run step audit log."""
        return f"{self._prefix}:stream:{run_id}:steps"

    def _input_stream_key(self, run_id: str) -> str:
        """Stream key used to fan work out to per-agent consumer groups."""
        return f"{self._prefix}:stream:{run_id}:input"

    def _step_cache_key(self, dedup_key: str) -> str:
        tail = dedup_key[len("step:") :] if dedup_key.startswith("step:") else dedup_key
        return f"{self._prefix}:step:{tail}"

    # ------------------------------------------------------------------
    # Schema-migration detection helper
    # ------------------------------------------------------------------

    async def _assert_no_hash_data(self, run_id: str) -> None:
        """Raise ``StoreBackendMismatchError`` if a hash-store list key
        (``{prefix}:run:{run_id}:steps``) exists for this run.  The hash
        key itself is shared between both backends (both store the RunStatus
        there), so we use the *list* key ``…:steps`` as the discriminator.
        """
        old_list_key = f"{self._prefix}:run:{run_id}:steps"
        exists = await self._r.exists(old_list_key)
        if exists:
            raise StoreBackendMismatchError(
                f"Hash-store list data found at {old_list_key!r} for run {run_id!r}. "
                "Run `migrate_hash_to_streams(redis, run_id)` to copy the data to "
                "streams format before switching backends."
            )

    # ------------------------------------------------------------------
    # RunStore Protocol implementation
    # ------------------------------------------------------------------

    async def get_run(self, run_id: str) -> RunStatus | None:
        raw: Any = await self._r.hgetall(self._run_hash_key(run_id))  # type: ignore[misc]
        if not raw:
            return None
        # The hash holds all fields as separate string entries; reassemble.
        try:
            payload: dict[str, Any] = {k: json.loads(v) for k, v in raw.items()}
        except json.JSONDecodeError:
            _log.warning("streams_store.run_hash_corrupt", run_id=run_id)
            return None
        migrated = migrate_run_blob(payload, key=self._run_hash_key(run_id))
        if migrated is None:
            return None
        return RunStatus.model_validate(migrated)

    async def put_run(self, run: RunStatus) -> None:
        key = self._run_hash_key(run.run_id)
        # Serialise each Pydantic field individually so future HSET patches
        # can update a single field without touching the whole blob.
        mapping = {k: json.dumps(v) for k, v in run.model_dump(mode="json").items()}
        await self._r.hset(key, mapping=mapping)  # type: ignore[misc]
        await self._r.expire(key, self._ttl * 24)

    async def append_step(self, step: StepRecord) -> None:
        """Append a StepRecord to the run's step audit stream via XADD."""
        await self._assert_no_hash_data(step.run_id)
        stream_key = self._steps_stream_key(step.run_id)
        await self._r.xadd(stream_key, {_FIELD_PAYLOAD: step.model_dump_json()})
        await self._r.expire(stream_key, self._ttl * 24)

    async def list_steps(self, run_id: str) -> list[StepRecord]:
        """Read all step entries from the stream via XRANGE (non-consuming)."""
        stream_key = self._steps_stream_key(run_id)
        raw_entries: Any = await self._r.xrange(stream_key, "-", "+")
        steps: list[StepRecord] = []
        for _msg_id, fields in raw_entries:
            payload_str = fields.get(_FIELD_PAYLOAD)
            if payload_str is None:
                continue
            try:
                payload = json.loads(payload_str)
            except json.JSONDecodeError:
                _log.warning("streams_store.step_corrupt", run_id=run_id)
                continue
            migrated = migrate_run_blob(payload, key=stream_key)
            if migrated is None:
                continue
            steps.append(StepRecord.model_validate(migrated))
        return steps

    async def cache_get(self, dedup_key: str) -> AgentResult | None:
        raw = await self._r.get(self._step_cache_key(dedup_key))
        if raw is None:
            return None
        return AgentResult.model_validate(json.loads(raw))

    async def cache_put(self, dedup_key: str, result: AgentResult) -> None:
        await self._r.set(self._step_cache_key(dedup_key), result.model_dump_json(), ex=self._ttl)

    # ------------------------------------------------------------------
    # Streams-specific helpers (beyond the RunStore Protocol)
    # ------------------------------------------------------------------

    async def ensure_steps_group(self, run_id: str) -> None:
        """Create the ``steps-readers`` consumer group on the step stream.

        Idempotent: if the group already exists the call is a no-op.
        Uses ``id="0"`` so a newly joined consumer starts from the
        beginning of the stream; switch to ``"$"`` to start from the
        tail (live messages only).
        """
        stream_key = self._steps_stream_key(run_id)
        try:
            await self._r.xgroup_create(stream_key, _STEPS_READER_GROUP, id="0", mkstream=True)
        except aioredis.ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    async def read_steps_group(
        self,
        run_id: str,
        consumer_name: str,
        *,
        count: int = 100,
        block_ms: int | None = None,
    ) -> list[tuple[str, StepRecord]]:
        """XREADGROUP — consume undelivered step entries.

        Returns a list of ``(message_id, StepRecord)`` pairs.  Callers
        must ``ack_step`` each entry after processing.

        Args:
            run_id:        Run whose step stream to read.
            consumer_name: Name of this consumer within the group.
            count:         Maximum entries to return per call.
            block_ms:      If set, block up to this many milliseconds
                           waiting for new entries (streaming mode).
        """
        await self.ensure_steps_group(run_id)
        stream_key = self._steps_stream_key(run_id)
        kwargs: dict[str, Any] = {"count": count}
        if block_ms is not None:
            kwargs["block"] = block_ms
        raw: Any = await self._r.xreadgroup(
            _STEPS_READER_GROUP, consumer_name, {stream_key: ">"}, **kwargs
        )
        results: list[tuple[str, StepRecord]] = []
        if not raw:
            return results
        _stream, messages = raw[0]
        for msg_id, fields in messages:
            payload_str = fields.get(_FIELD_PAYLOAD)
            if payload_str is None:
                continue
            try:
                payload = json.loads(payload_str)
            except json.JSONDecodeError:
                _log.warning("streams_store.step_corrupt_group", run_id=run_id, msg_id=msg_id)
                continue
            migrated = migrate_run_blob(payload, key=stream_key)
            if migrated is None:
                continue
            results.append((msg_id, StepRecord.model_validate(migrated)))
        return results

    async def ack_step(self, run_id: str, message_id: str) -> int:
        """XACK — acknowledge a step entry, removing it from the PEL."""
        return int(
            await self._r.xack(self._steps_stream_key(run_id), _STEPS_READER_GROUP, message_id)
        )

    async def pending_steps(self, run_id: str) -> dict[str, Any]:
        """XPENDING summary — number of unacknowledged entries + per-consumer counts."""
        result: Any = await self._r.xpending(self._steps_stream_key(run_id), _STEPS_READER_GROUP)
        return result  # type: ignore[no-any-return]

    async def pending_steps_range(self, run_id: str, *, count: int = 100) -> list[dict[str, Any]]:
        """XPENDING range — detailed list of pending entries (for orphan recovery)."""
        result: Any = await self._r.xpending_range(
            self._steps_stream_key(run_id), _STEPS_READER_GROUP, "-", "+", count=count
        )
        return result  # type: ignore[no-any-return]

    async def claim_step(
        self, run_id: str, new_consumer: str, min_idle_ms: int, message_id: str
    ) -> list[tuple[str, StepRecord]]:
        """XCLAIM — take ownership of a stuck pending entry.

        Used by the orphan reconciler when a consumer crashes before
        ACKing.  Transfers ``message_id`` from whatever consumer last
        held it to ``new_consumer``.
        """
        stream_key = self._steps_stream_key(run_id)
        raw: Any = await self._r.xclaim(
            stream_key, _STEPS_READER_GROUP, new_consumer, min_idle_ms, [message_id]
        )
        results: list[tuple[str, StepRecord]] = []
        for msg_id, fields in raw:
            payload_str = fields.get(_FIELD_PAYLOAD)
            if payload_str is None:
                continue
            try:
                payload = json.loads(payload_str)
            except json.JSONDecodeError:
                continue
            migrated = migrate_run_blob(payload, key=stream_key)
            if migrated is None:
                continue
            results.append((msg_id, StepRecord.model_validate(migrated)))
        return results

    # ------------------------------------------------------------------
    # Inter-agent fan-out helpers
    # ------------------------------------------------------------------

    async def publish_input(self, run_id: str, payload: dict[str, str]) -> str:
        """XADD to the per-run input stream (fan-out to agent consumer groups).

        Returns the stream entry ID.
        """
        stream_key = self._input_stream_key(run_id)
        msg_id: Any = await self._r.xadd(stream_key, payload)  # type: ignore[arg-type]
        await self._r.expire(stream_key, self._ttl * 24)
        return str(msg_id)

    async def ensure_agent_group(self, run_id: str, agent_name: str) -> None:
        """Create a per-agent consumer group on the input stream.

        Group name: ``agent:{agent_name}``.  Uses ``id="0"`` so the
        agent reads from the start of the stream when it first joins.
        Idempotent — BUSYGROUP errors are swallowed.
        """
        stream_key = self._input_stream_key(run_id)
        group_name = f"agent:{agent_name}"
        try:
            await self._r.xgroup_create(stream_key, group_name, id="0", mkstream=True)
        except aioredis.ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    async def read_input_group(
        self,
        run_id: str,
        agent_name: str,
        consumer_name: str,
        *,
        count: int = 10,
        block_ms: int | None = None,
    ) -> list[tuple[str, dict[str, str]]]:
        """XREADGROUP on the input stream for a specific agent group.

        Returns ``(message_id, fields)`` pairs.  Each agent's group
        receives **all** messages in the stream independently — fan-out
        rather than competing-consumers.
        """
        await self.ensure_agent_group(run_id, agent_name)
        stream_key = self._input_stream_key(run_id)
        group_name = f"agent:{agent_name}"
        kwargs: dict[str, Any] = {"count": count}
        if block_ms is not None:
            kwargs["block"] = block_ms
        raw: Any = await self._r.xreadgroup(group_name, consumer_name, {stream_key: ">"}, **kwargs)
        if not raw:
            return []
        _stream, messages = raw[0]
        return [(msg_id, fields) for msg_id, fields in messages]

    async def ack_input(self, run_id: str, agent_name: str, message_id: str) -> int:
        """XACK on the input stream for a specific agent group."""
        group_name = f"agent:{agent_name}"
        return int(await self._r.xack(self._input_stream_key(run_id), group_name, message_id))


__all__ = [
    "RedisStreamRunStore",
    "StoreBackendMismatchError",
]


# Re-export models that callers might expect from this module directly.
__all__ += [
    "AgentResult",
    "ResearchReport",
    "RunStatus",
    "StepRecord",
    "StepStatus",
]
