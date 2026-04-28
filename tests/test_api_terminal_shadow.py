"""Tests for the in-process terminal-state shadow cache.

The shadow exists for one specific failure mode: the bg task finishes
work but the store rejects the terminal-state write. Without the shadow
the run is stuck at `running` forever from the client's perspective.
With it, GET /runs/{id} can serve the terminal RunStatus from process
memory until the store recovers.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import fakeredis.aioredis as fake_aioredis
import pytest
from httpx import ASGITransport, AsyncClient

from research_crew.api import app
from research_crew.errors import StoreUnavailableError
from research_crew.models import AgentResult, RunStatus, StepRecord, StepStatus
from research_crew.store import RedisRunStore, RunStore


class _SelectiveFailStore:
    """Wraps a real store; lets a test fail specific calls on demand.

    Unlike `_ToggleFailStore` we want fine-grained control: the bg task
    must successfully read the run mid-flight, then have its *terminal*
    `put_run` fail. We also want to fail GET-side `get_run` to exercise
    the outage branch of `/runs/{id}`.
    """

    def __init__(self, inner: RunStore) -> None:
        self._inner = inner
        self.fail_put_run = False
        self.fail_put_run_after_calls: int | None = None
        self.fail_get_run = False
        self.fail_list_steps = False
        self.fail_append_step = False
        self.put_run_calls: list[RunStatus] = []

    async def get_run(self, run_id: str) -> RunStatus | None:
        if self.fail_get_run:
            raise StoreUnavailableError("simulated get outage")
        return await self._inner.get_run(run_id)

    async def put_run(self, run: RunStatus) -> None:
        self.put_run_calls.append(run)
        # Either fail every call (manual toggle), or fail every call once
        # we've crossed an N-call threshold (so the initial RUNNING write
        # from POST /research succeeds and only the bg task's terminal
        # write hits the outage).
        threshold = self.fail_put_run_after_calls
        if self.fail_put_run or (threshold is not None and len(self.put_run_calls) > threshold):
            raise StoreUnavailableError("simulated put outage")
        await self._inner.put_run(run)

    async def append_step(self, step: StepRecord) -> None:
        if self.fail_append_step:
            raise StoreUnavailableError("simulated append_step outage")
        await self._inner.append_step(step)

    async def list_steps(self, run_id: str) -> list[StepRecord]:
        if self.fail_list_steps:
            raise StoreUnavailableError("simulated list_steps outage")
        return await self._inner.list_steps(run_id)

    async def cache_get(self, dedup_key: str) -> AgentResult | None:
        return await self._inner.cache_get(dedup_key)

    async def cache_put(self, dedup_key: str, result: AgentResult) -> None:
        await self._inner.cache_put(dedup_key, result)


@pytest.fixture
async def shadow_client() -> AsyncIterator[tuple[AsyncClient, _SelectiveFailStore]]:
    fake = fake_aioredis.FakeRedis(decode_responses=True)
    selective = _SelectiveFailStore(RedisRunStore(fake))
    app.state.redis = fake
    app.state.store = selective
    # Reset shadow per-test so isolation is real, not an artefact of
    # whichever test happened to run first.
    app.state.terminal_shadow = {}
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://t") as c:
        yield c, selective
    await fake.aclose()


class TestShadowOnTerminalWriteFailure:
    async def test_bg_terminal_write_outage_is_served_from_shadow(
        self, shadow_client: tuple[AsyncClient, _SelectiveFailStore]
    ) -> None:
        """Full lifecycle: store works for submit -> bg task runs ->
        store breaks before terminal put -> GET serves the terminal
        RunStatus from the shadow instead of leaving the run stuck at
        RUNNING (or 503'ing) forever.
        """
        client, store = shadow_client

        # Initial RUNNING write (call #1) succeeds; every subsequent
        # `put_run` — i.e. the bg task's terminal write — fails. Has to
        # be set *before* POST because the ASGI transport runs the bg
        # task synchronously inside the request scope.
        store.fail_put_run_after_calls = 1

        resp = await client.post("/research", json={"question": "what is python"})
        assert resp.status_code == 202
        run_id: str = resp.json()["run_id"]

        # The bg task finished while the response was being prepared, so
        # the shadow already holds the terminal record. A single GET is
        # enough; poll only as a guard against future scheduler changes.
        body: dict[str, object] = {}
        for _ in range(20):
            r = await client.get(f"/runs/{run_id}")
            body = r.json()
            if body.get("state") in ("succeeded", "failed"):
                break
        # Even though the terminal `put_run` failed, the shadow served a
        # terminal state to the GET caller — not "running forever".
        assert body["state"] in {"succeeded", "failed"}
        assert body["run_id"] == run_id
        # Shadow should contain exactly this run.
        assert run_id in app.state.terminal_shadow

    async def test_bg_failed_path_records_failed_in_shadow(
        self, shadow_client: tuple[AsyncClient, _SelectiveFailStore]
    ) -> None:
        """Force the bg task to take the except branch by breaking the
        workflow's `append_step` mid-flight, then break `put_run` so the
        recovery write also fails. The shadow must hold a FAILED record.
        """
        client, store = shadow_client
        # First put_run (initial RUNNING blob) succeeds; later puts fail.
        store.fail_put_run_after_calls = 1
        # Workflow's first per-step record write blows up with a store
        # outage, so `engine.run_parallel` raises and `_execute_run`'s
        # except branch is what eventually writes the terminal state.
        store.fail_append_step = True

        resp = await client.post("/research", json={"question": "what is python"})
        run_id: str = resp.json()["run_id"]

        body: dict[str, object] = {}
        for _ in range(20):
            r = await client.get(f"/runs/{run_id}")
            body = r.json()
            if body.get("state") in ("succeeded", "failed"):
                break
        assert body["state"] == "failed"
        # Shadow holds the FAILED RunStatus the bg task synthesised.
        shadow_entry = app.state.terminal_shadow[run_id]
        assert shadow_entry.state is StepStatus.FAILED

    async def test_shadow_overrides_stuck_running_record(
        self, shadow_client: tuple[AsyncClient, _SelectiveFailStore]
    ) -> None:
        """Store has a stale RUNNING record but shadow has a terminal one
        for the same run_id -> GET returns the shadow.

        Simulates the recovery sequence: store accepted the RUNNING blob
        on submit, then rejected every subsequent put. The bg task wrote
        FAILED to the shadow. A later GET must NOT show RUNNING from the
        store — the in-process truth is the shadow record.
        """
        client, store = shadow_client
        run_id = "stuck-run"
        # Plant a stale RUNNING record directly into the underlying store
        # (skipping the API path so the bg task doesn't immediately drive
        # it to terminal state).
        stuck = RunStatus(
            run_id=run_id,
            question="what is python",
            state=StepStatus.RUNNING,
        )
        await store._inner.put_run(stuck)

        # Plant a terminal record into the shadow — this is what the bg
        # task would have written via `_persist_terminal` after its store
        # write got rejected.
        terminal = stuck.model_copy(update={"state": StepStatus.FAILED})
        app.state.terminal_shadow[run_id] = terminal

        r = await client.get(f"/runs/{run_id}")
        assert r.status_code == 200
        body = r.json()
        # Shadow wins over the stuck RUNNING blob.
        assert body["state"] == "failed"

    async def test_shadow_served_when_store_404s(
        self, shadow_client: tuple[AsyncClient, _SelectiveFailStore]
    ) -> None:
        """Store has no record (TTL expired, evicted) but shadow does ->
        GET returns the shadow rather than 404.
        """
        client, store = shadow_client
        run_id = "ghost-run"
        # Store has no such run.
        assert await store.get_run(run_id) is None

        terminal = RunStatus(
            run_id=run_id,
            question="what is python",
            state=StepStatus.FAILED,
        )
        app.state.terminal_shadow[run_id] = terminal

        r = await client.get(f"/runs/{run_id}")
        assert r.status_code == 200
        assert r.json()["state"] == "failed"

    async def test_shadow_served_when_store_outage_on_get(
        self, shadow_client: tuple[AsyncClient, _SelectiveFailStore]
    ) -> None:
        """`get_run` raises StoreUnavailableError but shadow has it ->
        GET returns the shadow record (200), not 503.
        """
        client, store = shadow_client
        run_id = "outage-run"
        terminal = RunStatus(
            run_id=run_id,
            question="what is python",
            state=StepStatus.FAILED,
        )
        app.state.terminal_shadow[run_id] = terminal

        store.fail_get_run = True
        r = await client.get(f"/runs/{run_id}")
        assert r.status_code == 200
        assert r.json()["state"] == "failed"

    async def test_outage_without_shadow_still_503s(
        self, shadow_client: tuple[AsyncClient, _SelectiveFailStore]
    ) -> None:
        """Store outage and the shadow does not know the id ->
        propagate 503; do not invent a fake terminal state.
        """
        client, store = shadow_client
        store.fail_get_run = True
        r = await client.get("/runs/never-heard-of-it")
        assert r.status_code == 503


class TestShadowDoesNotPolluteOnSuccess:
    async def test_happy_path_does_not_write_shadow(
        self, shadow_client: tuple[AsyncClient, _SelectiveFailStore]
    ) -> None:
        """A normal run that completes without any store failure must
        leave the terminal shadow empty for that run_id. The shadow is
        a fallback, not a write-through cache.
        """
        client, _store = shadow_client

        resp = await client.post("/research", json={"question": "what is python"})
        run_id: str = resp.json()["run_id"]

        # Drive the run to terminal state via polling.
        body: dict[str, object] = {}
        for _ in range(40):
            r = await client.get(f"/runs/{run_id}")
            body = r.json()
            if body.get("state") in ("succeeded", "failed"):
                break
        assert body["state"] == "succeeded"
        # Shadow remained empty for this run id.
        assert run_id not in app.state.terminal_shadow


class TestShadowBounded:
    """The shadow MUST cap memory growth — a long store outage with a
    steady submit rate would otherwise leak entries forever.
    """

    def test_oldest_entry_evicted_when_capacity_exceeded(self) -> None:
        from research_crew.api import _TerminalShadow

        shadow = _TerminalShadow(max_size=3)
        runs = [
            RunStatus(run_id=f"r{i}", question="q", state=StepStatus.FAILED)
            for i in range(5)
        ]
        for run in runs:
            shadow[run.run_id] = run

        # Capacity is 3, we wrote 5 → r0 and r1 evicted (FIFO).
        assert len(shadow) == 3
        assert "r0" not in shadow
        assert "r1" not in shadow
        assert "r2" in shadow
        assert "r3" in shadow
        assert "r4" in shadow

    def test_reinsert_refreshes_recency(self) -> None:
        from research_crew.api import _TerminalShadow

        shadow = _TerminalShadow(max_size=2)
        a = RunStatus(run_id="a", question="q", state=StepStatus.FAILED)
        b = RunStatus(run_id="b", question="q", state=StepStatus.FAILED)
        c = RunStatus(run_id="c", question="q", state=StepStatus.FAILED)

        shadow["a"] = a
        shadow["b"] = b
        # Re-write a → it becomes the youngest, so the next eviction drops b.
        shadow["a"] = a
        shadow["c"] = c

        assert "a" in shadow
        assert "c" in shadow
        assert "b" not in shadow

    def test_max_size_must_be_positive(self) -> None:
        from research_crew.api import _TerminalShadow

        with pytest.raises(ValueError):
            _TerminalShadow(max_size=0)
        with pytest.raises(ValueError):
            _TerminalShadow(max_size=-1)

    def test_clear_resets_to_empty(self) -> None:
        from research_crew.api import _TerminalShadow

        shadow = _TerminalShadow(max_size=10)
        shadow["a"] = RunStatus(run_id="a", question="q", state=StepStatus.FAILED)
        shadow["b"] = RunStatus(run_id="b", question="q", state=StepStatus.FAILED)
        assert len(shadow) == 2
        shadow.clear()
        assert len(shadow) == 0
        assert "a" not in shadow
