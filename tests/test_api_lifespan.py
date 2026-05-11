"""Lifespan tests: orphan-run reconciliation on startup.

Background execution is bound to the accepting instance — a process
restart after `put_run(RUNNING)` strands the run with no worker.
The lifespan does a one-shot SCAN over `{prefix}:run:*` and flips
any RUNNING entries whose heartbeat has gone stale to FAILED. Runs
whose heartbeat is fresh are left alone so a peer instance running
concurrently can finish its in-flight work; only true orphans (no
heartbeat for >`RESEARCH_HEARTBEAT_STALE_S` seconds) are reconciled.

A durable worker queue is the right long-term fix; this is the
smallest defensible interim. See README "Limitations".
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import fakeredis.aioredis as fake_aioredis
import pytest
from httpx import ASGITransport, AsyncClient

from research_crew import api as api_module
from research_crew.api import (
    _execute_run,
    _heartbeat_loop,
    _is_dev_mode,
    _reconcile_orphan_runs,
    _TerminalShadow,
    app,
)
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

    async def test_lifespan_does_not_reconcile_recent_heartbeat(self) -> None:
        """A peer instance's RUNNING run with a fresh heartbeat must
        survive startup unchanged — flipping it would kill live work."""
        fake = fake_aioredis.FakeRedis(decode_responses=True)
        store = RedisRunStore(fake)
        # Plant a RUNNING run owned by a different worker, heartbeating
        # right now. This represents a peer instance mid-run.
        live = RunStatus(
            run_id="live-peer",
            question="what is python",
            state=StepStatus.RUNNING,
            owner_id="peer-worker-id",
            heartbeat_at=datetime.now(UTC),
        )
        await store.put_run(live)
        app.state.redis = fake
        app.state.store = store

        await _reconcile_orphan_runs(app)

        unchanged = await store.get_run("live-peer")
        assert unchanged is not None
        assert unchanged.state is StepStatus.RUNNING
        assert unchanged.error is None
        assert unchanged.owner_id == "peer-worker-id"
        await fake.aclose()

    async def test_lifespan_reconciles_stale_heartbeat(self) -> None:
        """A RUNNING run whose heartbeat has gone stale (older than the
        configured threshold) gets flipped to FAILED with a reason that
        names the staleness."""
        fake = fake_aioredis.FakeRedis(decode_responses=True)
        store = RedisRunStore(fake)
        # Heartbeat 10 minutes ago — well past the 120s default.
        stale_ts = datetime.now(UTC) - timedelta(minutes=10)
        stale = RunStatus(
            run_id="stale-1",
            question="what is python",
            state=StepStatus.RUNNING,
            owner_id="dead-worker",
            heartbeat_at=stale_ts,
        )
        await store.put_run(stale)
        app.state.redis = fake
        app.state.store = store

        await _reconcile_orphan_runs(app)

        recovered = await store.get_run("stale-1")
        assert recovered is not None
        assert recovered.state is StepStatus.FAILED
        assert recovered.error is not None
        assert "abandoned" in recovered.error
        assert "no heartbeat" in recovered.error
        await fake.aclose()


class TestHeartbeatLoop:
    """The heartbeat loop refreshes `RunStatus.heartbeat_at` while a run
    is RUNNING so peer instances' lifespan reconcilers can tell live
    work from a true orphan."""

    async def test_heartbeat_updates_during_run(self) -> None:
        """A loop tick must persist a newer `heartbeat_at` to the store."""
        fake = fake_aioredis.FakeRedis(decode_responses=True)
        store = RedisRunStore(fake)
        original_ts = datetime.now(UTC) - timedelta(minutes=5)
        running = RunStatus(
            run_id="hb-1",
            question="what is python",
            state=StepStatus.RUNNING,
            owner_id="me",
            heartbeat_at=original_ts,
        )
        await store.put_run(running)

        # Tiny interval so the test doesn't sleep for 30s.
        task = asyncio.create_task(_heartbeat_loop(store, "hb-1", interval_s=0.01))
        # Let at least one tick land.
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        refreshed = await store.get_run("hb-1")
        assert refreshed is not None
        assert refreshed.state is StepStatus.RUNNING
        assert refreshed.heartbeat_at is not None
        assert refreshed.heartbeat_at > original_ts
        await fake.aclose()

    async def test_heartbeat_stops_after_terminal_state(self) -> None:
        """If the run record reaches a terminal state, the loop returns
        instead of overwriting the terminal blob with a heartbeat-only
        update."""
        fake = fake_aioredis.FakeRedis(decode_responses=True)
        store = RedisRunStore(fake)
        terminal = RunStatus(
            run_id="hb-done",
            question="what is python",
            state=StepStatus.SUCCEEDED,
            heartbeat_at=datetime.now(UTC),
        )
        await store.put_run(terminal)

        # Run the loop with a tiny interval; it must observe SUCCEEDED
        # and return cleanly without raising.
        await asyncio.wait_for(
            _heartbeat_loop(store, "hb-done", interval_s=0.01), timeout=0.5
        )

        unchanged = await store.get_run("hb-done")
        assert unchanged is not None
        assert unchanged.state is StepStatus.SUCCEEDED
        await fake.aclose()


class TestExecuteRunCancellation:
    """Shutdown-time cancellation of `_execute_run` must still write a
    terminal RunStatus before the CancelledError propagates. Without
    that, the run is stuck at RUNNING until the next-process orphan
    sweep — observably "running forever" from the client's perspective.
    """

    async def test_cancelled_run_persists_terminal_state_then_reraises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
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

        # Synchronisation point: monkeypatch the workflow engine so
        # `run_parallel` signals "I'm running" via an Event, then awaits
        # forever. The test waits on that Event before cancelling, which
        # guarantees the cancel lands inside the workflow body — no
        # `asyncio.sleep`-based "hopefully it started" race.
        started: asyncio.Event = asyncio.Event()

        async def _stub_run_parallel(
            self: object, agents: object, question: object
        ) -> object:
            started.set()
            # Block until cancelled by the outer test cancellation.
            await asyncio.Event().wait()
            raise AssertionError("unreachable: must be cancelled")

        monkeypatch.setattr(
            api_module.WorkflowEngine, "run_parallel", _stub_run_parallel, raising=True
        )

        async def _go() -> None:
            await _execute_run(
                store, shadow, "cancel-me",
                ResearchRequest(question="what is python"),
            )

        task = asyncio.create_task(_go())
        # Wait deterministically until the bg task is inside the
        # workflow body — short timeout because anything > a few hundred
        # ms means the patch didn't take.
        await asyncio.wait_for(started.wait(), timeout=2.0)
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


class TestDevMode:
    """``RESEARCH_DEV_MODE`` is the local-loop opt-out for the
    auth-disabled WARNING. Truthy values demote the log to INFO; the
    helper is intentionally narrow (only the standard truthy strings)
    so a typo doesn't silently silence the warning."""

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "On"])
    def test_truthy_values(self, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
        monkeypatch.setenv("RESEARCH_DEV_MODE", value)
        assert _is_dev_mode() is True

    @pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "garbage"])
    def test_falsy_values(self, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
        monkeypatch.setenv("RESEARCH_DEV_MODE", value)
        assert _is_dev_mode() is False

    def test_unset_is_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("RESEARCH_DEV_MODE", raising=False)
        assert _is_dev_mode() is False


@pytest.fixture(autouse=True)
def _clear_state() -> Iterator[None]:
    """Reset `app.state.redis`/`app.state.store` after each test so a
    leaked closed client doesn't poison later fixtures."""
    yield
    if hasattr(app.state, "redis"):
        app.state.redis = None
    if hasattr(app.state, "store"):
        app.state.store = None
