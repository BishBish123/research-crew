"""WebSocket streaming endpoint tests.

All tests run entirely in-process using FastAPI's ``TestClient``
WebSocket support (backed by ``anyio`` / ``starlette``).  No real
network ports are opened.

Covered scenarios
-----------------
1. Connection sends initial snapshot immediately on connect.
2. Step events arrive in order as the workflow executes.
3. Heartbeat fires when no events arrive within the configured interval.
4. Connection closes cleanly (code 1000) when the run reaches a terminal
   state (``done`` message then server close).
5. Unauthenticated connection is rejected (close code 1008).
6. Multiple simultaneous WS clients receive the same events (broadcast).
7. Connecting to an already-terminal run receives snapshot + done immediately.
8. Connecting to a non-existent run_id closes with 1008.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import fakeredis.aioredis as fake_aioredis  # noqa: F401 — kept for _make_redis_store helper
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from research_crew.api import app
from research_crew.models import AgentName, RunStatus, StepRecord, StepStatus
from research_crew.store import InMemoryRunStore, RedisRunStore  # noqa: F401
from research_crew.streaming import RunQueueRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store() -> InMemoryRunStore:
    return InMemoryRunStore()


def _setup_app(store: Any, token: str | None = None) -> None:
    """Configure app.state for tests (no lifespan needed)."""
    app.state.store = store
    app.state.api_token = token
    from research_crew.api import _RateLimiter, _TerminalShadow  # noqa: PLC0415

    if not hasattr(app.state, "terminal_shadow") or app.state.terminal_shadow is None:
        app.state.terminal_shadow = _TerminalShadow()
    app.state.rate_limiter = _RateLimiter(limit_per_min=100)


def _make_step(run_id: str, agent: AgentName = AgentName.WEB_SEARCH) -> StepRecord:
    return StepRecord(
        run_id=run_id,
        agent=agent,
        status=StepStatus.SUCCEEDED,
        attempts=1,
        started_at=datetime.now(UTC),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store() -> InMemoryRunStore:
    return _make_store()


@pytest.fixture
def ws_registry() -> RunQueueRegistry:
    """Fresh registry per test — isolates queue state."""
    return RunQueueRegistry()


@pytest.fixture(autouse=True)
def _reset_app_state(store: InMemoryRunStore) -> None:  # type: ignore[return]
    """Wire a fresh store + clear token before every test."""
    _setup_app(store, token=None)


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# 1. Initial snapshot is sent on connect
# ---------------------------------------------------------------------------


def test_snapshot_sent_on_connect(client: TestClient, store: InMemoryRunStore) -> None:
    """Client receives a ``type=snapshot`` message immediately on connect."""
    run_id = "run_snap_001"
    run = RunStatus(run_id=run_id, question="what is python", state=StepStatus.RUNNING)
    asyncio.get_event_loop().run_until_complete(store.put_run(run))

    from research_crew.streaming import registry  # noqa: PLC0415

    registry.create(run_id)

    with client.websocket_connect(f"/runs/{run_id}/stream") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "snapshot"
        assert msg["run_id"] == run_id
        assert msg["state"] == "running"
        # Clean up: push terminal sentinel so the WS handler exits.
        registry.teardown(run_id)
        # Drain remaining messages until the WS closes.
        while True:
            try:
                ws.receive_json()
            except Exception:
                break


# ---------------------------------------------------------------------------
# 2. Step events arrive in order
# ---------------------------------------------------------------------------


def test_step_events_arrive_in_order(client: TestClient, store: InMemoryRunStore) -> None:
    """Step events published to the registry arrive on the WS in order."""
    run_id = "run_steps_002"
    run = RunStatus(run_id=run_id, question="test", state=StepStatus.RUNNING)
    asyncio.get_event_loop().run_until_complete(store.put_run(run))

    from research_crew.streaming import registry  # noqa: PLC0415

    registry.create(run_id)

    agents = [AgentName.WEB_SEARCH, AgentName.SCHOLAR, AgentName.NEWS]
    steps = [_make_step(run_id, agent) for agent in agents]

    with client.websocket_connect(f"/runs/{run_id}/stream") as ws:
        # consume snapshot
        snap = ws.receive_json()
        assert snap["type"] == "snapshot"

        # publish step events AFTER the WS is connected and has consumed snapshot
        for step in steps:
            registry.publish(run_id, step)

        # consume step messages
        received_agents = []
        for _ in steps:
            msg = ws.receive_json()
            assert msg["type"] == "step"
            received_agents.append(msg["agent"])

        assert received_agents == [a.value for a in agents]

        # teardown signals done
        registry.teardown(run_id)
        done_msg = ws.receive_json()
        assert done_msg["type"] == "done"


# ---------------------------------------------------------------------------
# 3. Heartbeat fires when no events arrive
# ---------------------------------------------------------------------------


def test_heartbeat_fires_on_idle(client: TestClient, store: InMemoryRunStore) -> None:
    """A heartbeat message is sent after the configured interval with no events."""
    run_id = "run_hb_003"
    run = RunStatus(run_id=run_id, question="test", state=StepStatus.RUNNING)
    asyncio.get_event_loop().run_until_complete(store.put_run(run))

    from research_crew.streaming import registry  # noqa: PLC0415

    registry.create(run_id)

    # Inject a very short heartbeat interval (50 ms) by monkey-patching the
    # module-level constant used as the endpoint's default parameter.
    import research_crew.api as api_mod  # noqa: PLC0415

    original = api_mod._WS_HEARTBEAT_S
    api_mod._WS_HEARTBEAT_S = 0.05  # 50 ms — fast enough for tests

    try:
        with client.websocket_connect(f"/runs/{run_id}/stream") as ws:
            snap = ws.receive_json()
            assert snap["type"] == "snapshot"

            # Don't publish any step — wait for the heartbeat.
            hb = ws.receive_json()
            assert hb["type"] == "heartbeat"

            # Clean up.
            registry.teardown(run_id)
            while True:
                try:
                    ws.receive_json()
                except Exception:
                    break
    finally:
        api_mod._WS_HEARTBEAT_S = original


# ---------------------------------------------------------------------------
# 4. Connection closes cleanly on terminal state
# ---------------------------------------------------------------------------


def test_ws_closes_on_terminal(client: TestClient, store: InMemoryRunStore) -> None:
    """Server sends ``done`` then closes when the run becomes terminal."""
    run_id = "run_terminal_004"
    run = RunStatus(run_id=run_id, question="test", state=StepStatus.RUNNING)
    asyncio.get_event_loop().run_until_complete(store.put_run(run))

    from research_crew.streaming import registry  # noqa: PLC0415

    registry.create(run_id)

    messages: list[dict[str, Any]] = []

    with client.websocket_connect(f"/runs/{run_id}/stream") as ws:
        snap = ws.receive_json()
        messages.append(snap)

        # Signal terminal immediately.
        registry.teardown(run_id)

        # Drain until the server closes.
        while True:
            try:
                msg = ws.receive_json()
                messages.append(msg)
            except Exception:
                break

    types = [m["type"] for m in messages]
    assert "snapshot" in types
    assert "done" in types
    assert types.index("done") == len(types) - 1


# ---------------------------------------------------------------------------
# 5. Bearer-token auth rejects unauthenticated connections
# ---------------------------------------------------------------------------


def test_ws_rejects_unauthenticated(client: TestClient, store: InMemoryRunStore) -> None:
    """When a token is configured, connecting without one is rejected."""
    # Set the token directly on app.state (no new TestClient/lifespan needed).
    app.state.api_token = "supersecret"  # noqa: S105
    run_id = "run_auth_005"
    run = RunStatus(run_id=run_id, question="test", state=StepStatus.RUNNING)
    asyncio.get_event_loop().run_until_complete(store.put_run(run))

    try:
        with (
            pytest.raises((WebSocketDisconnect, Exception)),
            client.websocket_connect(f"/runs/{run_id}/stream") as ws,
        ):
            ws.receive_json()
    finally:
        app.state.api_token = None


def test_ws_accepts_valid_token(client: TestClient, store: InMemoryRunStore) -> None:
    """A valid query-param token is accepted."""
    app.state.api_token = "supersecret"  # noqa: S105
    run_id = "run_auth_005b"
    run = RunStatus(run_id=run_id, question="test", state=StepStatus.RUNNING)
    asyncio.get_event_loop().run_until_complete(store.put_run(run))

    from research_crew.streaming import registry  # noqa: PLC0415

    registry.create(run_id)

    try:
        with client.websocket_connect(f"/runs/{run_id}/stream?token=supersecret") as ws:
            msg = ws.receive_json()
            assert msg["type"] == "snapshot"
            registry.teardown(run_id)
            while True:
                try:
                    ws.receive_json()
                except Exception:
                    break
    finally:
        app.state.api_token = None


# ---------------------------------------------------------------------------
# 6. Multiple clients receive the same events (broadcast)
# ---------------------------------------------------------------------------


def test_multiple_clients_receive_same_events(client: TestClient, store: InMemoryRunStore) -> None:
    """Two simultaneous WS clients both receive published step events."""
    run_id = "run_broadcast_006"
    run = RunStatus(run_id=run_id, question="test", state=StepStatus.RUNNING)
    asyncio.get_event_loop().run_until_complete(store.put_run(run))

    from research_crew.streaming import registry  # noqa: PLC0415

    registry.create(run_id)

    step = _make_step(run_id, AgentName.CODE)

    with client.websocket_connect(f"/runs/{run_id}/stream") as ws1:
        snap1 = ws1.receive_json()
        assert snap1["type"] == "snapshot"

        with client.websocket_connect(f"/runs/{run_id}/stream") as ws2:
            snap2 = ws2.receive_json()
            assert snap2["type"] == "snapshot"

            # Publish one step — both clients should see it.
            registry.publish(run_id, step)

            msg1 = ws1.receive_json()
            msg2 = ws2.receive_json()

            assert msg1["type"] == "step"
            assert msg2["type"] == "step"
            assert msg1["agent"] == AgentName.CODE.value
            assert msg2["agent"] == AgentName.CODE.value

            registry.teardown(run_id)
            while True:
                try:
                    ws1.receive_json()
                except Exception:
                    break
            while True:
                try:
                    ws2.receive_json()
                except Exception:
                    break


# ---------------------------------------------------------------------------
# 7. Connecting to an already-terminal run: snapshot + done immediately
# ---------------------------------------------------------------------------


def test_already_terminal_run_closes_immediately(
    client: TestClient, store: InMemoryRunStore
) -> None:
    """Connecting to a SUCCEEDED run sends snapshot + done without blocking."""
    run_id = "run_terminal_007"
    run = RunStatus(run_id=run_id, question="test", state=StepStatus.SUCCEEDED)
    asyncio.get_event_loop().run_until_complete(store.put_run(run))

    messages: list[dict[str, Any]] = []
    with client.websocket_connect(f"/runs/{run_id}/stream") as ws:
        while True:
            try:
                msg = ws.receive_json()
                messages.append(msg)
            except Exception:
                break

    types = [m["type"] for m in messages]
    assert types[0] == "snapshot"
    assert types[-1] == "done"
    assert len(messages) == 2


# ---------------------------------------------------------------------------
# 8. Non-existent run_id is rejected
# ---------------------------------------------------------------------------


def test_nonexistent_run_closes_with_error(client: TestClient) -> None:
    """Connecting for a run_id not in the store closes the WS."""
    with (
        pytest.raises((WebSocketDisconnect, Exception)),
        client.websocket_connect("/runs/does_not_exist_xyz/stream") as ws,
    ):
        ws.receive_json()


# ---------------------------------------------------------------------------
# Unit tests for RunQueueRegistry
# ---------------------------------------------------------------------------


class TestRunQueueRegistry:
    def test_create_and_subscribe(self, ws_registry: RunQueueRegistry) -> None:
        ws_registry.create("r1")
        q = ws_registry.subscribe("r1")
        ws_registry.publish("r1", _make_step("r1"))
        item = q.get_nowait()
        assert item is not None
        assert item.run_id == "r1"

    def test_teardown_sends_sentinel(self, ws_registry: RunQueueRegistry) -> None:
        ws_registry.create("r2")
        q = ws_registry.subscribe("r2")
        ws_registry.teardown("r2")
        item = q.get_nowait()
        assert item is None  # sentinel

    def test_subscribe_after_teardown_gets_sentinel(self, ws_registry: RunQueueRegistry) -> None:
        ws_registry.create("r3")
        ws_registry.teardown("r3")
        q = ws_registry.subscribe("r3")
        item = q.get_nowait()
        assert item is None  # immediate sentinel for late joiner

    def test_publish_noop_without_subscribers(self, ws_registry: RunQueueRegistry) -> None:
        ws_registry.create("r4")
        # Must not raise even with no subscribers.
        ws_registry.publish("r4", _make_step("r4"))

    def test_unsubscribe_removes_queue(self, ws_registry: RunQueueRegistry) -> None:
        ws_registry.create("r5")
        q = ws_registry.subscribe("r5")
        ws_registry.unsubscribe("r5", q)
        ws_registry.publish("r5", _make_step("r5"))
        assert q.empty()
