"""Lifespan tests: orphan-run reconciliation on startup.

Background execution is bound to the accepting instance — a process
restart after `put_run(RUNNING)` strands the run with no worker.
The lifespan does a one-shot SCAN over `{prefix}:run:*` and flips
any RUNNING entries to FAILED with a clear "abandoned" message so
polling clients receive a terminal answer instead of looping forever.

A durable worker queue is the right long-term fix; this is the
smallest defensible interim. See README "Limitations".
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator

import fakeredis.aioredis as fake_aioredis
import pytest
from httpx import ASGITransport, AsyncClient

from research_crew.api import _execute_run, _reconcile_orphan_runs, _TerminalShadow, app
from research_crew.models import ResearchRequest, RunStatus, StepStatus
from research_crew.store import RedisRunStore


class TestReconcileOrphanRuns:
    async def test_lifespan_reconciles_orphan_running_runs(self) -> None:
        """A RUNNING record in Redis at startup gets flipped to FAILED."""
        fake = fake_aioredis.FakeRedis(decode_responses=True)
        store = RedisRunStore(fake)
        # Plant a RUNNING run as if a previous process crashed mid-flight.
        orphan = RunStatus(
            run_id="orphan-1",
            question="what is python",
            state=StepStatus.RUNNING,
        )
        await store.put_run(orphan)

        # Wire up app.state the same way the lifespan would, then run
        # the reconciliation step directly. We don't go through a real
        # lifespan here because the ASGI transport short-circuits it;
        # this is the same shape `_count_active_runs` is unit-tested
        # under.
        app.state.redis = fake
        app.state.store = store

        await _reconcile_orphan_runs(app)

        recovered = await store.get_run("orphan-1")
        assert recovered is not None
        assert recovered.state is StepStatus.FAILED
        assert recovered.error == "abandoned by previous process"
        assert recovered.finished_at is not None

        # And /runs/{id} surfaces the FAILED state to a poller.
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://t") as c:
            r = await c.get("/runs/orphan-1")
        assert r.status_code == 200
        body = r.json()
        assert body["state"] == "failed"
        assert body["error"] == "abandoned by previous process"

        await fake.aclose()

    async def test_lifespan_does_not_touch_terminal_runs(self) -> None:
        """SUCCEEDED / FAILED records must not be rewritten by the sweep."""
        fake = fake_aioredis.FakeRedis(decode_responses=True)
        store = RedisRunStore(fake)
        done = RunStatus(
            run_id="done-1",
            question="what is python",
            state=StepStatus.SUCCEEDED,
        )
        await store.put_run(done)
        app.state.redis = fake
        app.state.store = store

        await _reconcile_orphan_runs(app)

        unchanged = await store.get_run("done-1")
        assert unchanged is not None
        assert unchanged.state is StepStatus.SUCCEEDED
        assert unchanged.error is None
        await fake.aclose()

    async def test_lifespan_handles_empty_keyspace(self) -> None:
        """No runs in Redis → reconciliation is a no-op."""
        fake = fake_aioredis.FakeRedis(decode_responses=True)
        store = RedisRunStore(fake)
        app.state.redis = fake
        app.state.store = store
        # Just must not raise.
        await _reconcile_orphan_runs(app)
        await fake.aclose()


class TestExecuteRunCancellation:
    """Shutdown-time cancellation of `_execute_run` must still write a
    terminal RunStatus before the CancelledError propagates. Without
    that, the run is stuck at RUNNING until the next-process orphan
    sweep — observably "running forever" from the client's perspective.
    """

    async def test_cancelled_run_persists_terminal_state_then_reraises(self) -> None:
        fake = fake_aioredis.FakeRedis(decode_responses=True)
        store = RedisRunStore(fake)
        shadow = _TerminalShadow()

        # Plant the initial RUNNING record the way POST /research would.
        running = RunStatus(
            run_id="cancel-me",
            question="what is python",
            state=StepStatus.RUNNING,
        )
        await store.put_run(running)

        async def _go() -> None:
            await _execute_run(
                store, shadow, "cancel-me",
                ResearchRequest(question="what is python"),
            )

        task = asyncio.create_task(_go())
        # Yield once so the task starts executing the workflow body.
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # Either the store has the FAILED record, or the shadow does
        # (depending on which side of the put_run the cancel landed).
        # Both paths must surface a terminal state with the cancellation
        # message so a polling client sees a clear answer.
        observed = await store.get_run("cancel-me")
        if observed is not None and observed.state is StepStatus.FAILED:
            assert observed.error == "cancelled during shutdown"
        else:
            shadow_entry = shadow.get("cancel-me")
            assert shadow_entry is not None, (
                "cancellation must persist via store OR shadow, not vanish"
            )
            assert shadow_entry.state is StepStatus.FAILED
            assert shadow_entry.error == "cancelled during shutdown"

        await fake.aclose()


@pytest.fixture(autouse=True)
def _clear_state() -> Iterator[None]:
    """Reset `app.state.redis`/`app.state.store` after each test so a
    leaked closed client doesn't poison later fixtures."""
    yield
    if hasattr(app.state, "redis"):
        app.state.redis = None
    if hasattr(app.state, "store"):
        app.state.store = None
