"""Tests for the in-process terminal-state shadow cache.

The shadow exists for one specific failure mode: the bg task finishes
work but the store rejects the terminal-state write. Without the shadow
the run is stuck at `running` forever from the client's perspective.
With it, GET /runs/{id} can serve the terminal RunStatus from process
memory until the store recovers.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime

import fakeredis.aioredis as fake_aioredis
import pytest
from httpx import ASGITransport, AsyncClient

from research_crew.api import _persist_terminal, _TerminalShadow, app
from research_crew.errors import StoreUnavailableError
from research_crew.models import AgentName, AgentResult, RunStatus, StepRecord, StepStatus
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
    # whichever test happened to run first. Use the production-shape
    # bounded shadow so eviction-aware code paths run in tests too.
    app.state.terminal_shadow = _TerminalShadow()
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
        last_status = -1
        for _ in range(20):
            r = await client.get(f"/runs/{run_id}")
            last_status = r.status_code
            body = r.json()
            if body.get("state") in ("succeeded", "failed"):
                break
        # Even though the terminal `put_run` failed, the shadow served a
        # terminal state to the GET caller — not "running forever".
        assert last_status == 200, f"expected 200 from shadow lookup, got {last_status}"
        assert body["state"] in {"succeeded", "failed"}
        assert body["run_id"] == run_id
        # Shadow should contain exactly this run.
        assert run_id in app.state.terminal_shadow
        # The store stayed reachable for `list_steps` (only `put_run` failed),
        # so the shadow response must include the per-agent audit rows
        # via FIX 5's hydrate-on-read path.
        steps = body.get("steps", [])
        assert isinstance(steps, list)
        assert len(steps) > 0, (
            "shadow response must hydrate steps from the store when reachable; "
            f"got empty list. body={body}"
        )

    async def test_bg_failed_path_records_failed_in_shadow(
        self, shadow_client: tuple[AsyncClient, _SelectiveFailStore]
    ) -> None:
        """Force the bg task to take the except branch by breaking the
        post-workflow `get_run` lookup, then break `put_run` so the
        recovery write also fails. The shadow must hold a FAILED record.

        Note: store-side `append_step` outages are now swallowed by the
        workflow runner (see FIX 1), so this test drives the failure
        through a path that still escapes `_execute_run`'s try block.
        """
        client, store = shadow_client
        # First put_run (initial RUNNING blob) succeeds; later puts fail.
        store.fail_put_run_after_calls = 1
        # The post-fan-out `get_run` (used to fetch the existing record
        # before the terminal write) raises -> bg task except branch -> it
        # synthesises a FAILED RunStatus, which then also can't be
        # written, and lands in the shadow.
        store.fail_get_run = True

        resp = await client.post("/research", json={"question": "what is python"})
        run_id: str = resp.json()["run_id"]

        body: dict[str, object] = {}
        last_status = -1
        for _ in range(20):
            r = await client.get(f"/runs/{run_id}")
            last_status = r.status_code
            body = r.json()
            if body.get("state") in ("succeeded", "failed"):
                break
        assert last_status == 200, f"expected 200 from shadow lookup, got {last_status}"
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
        # Plant a step row into the store too — this is what the bg task
        # would have written for the agent attempts before the store
        # outage hit on the terminal put_run. FIX 5 says the shadow
        # response should hydrate from the store when reachable.
        now = datetime.now(UTC)
        await store._inner.append_step(
            StepRecord(
                run_id=run_id,
                agent=AgentName.WEB_SEARCH,
                status=StepStatus.SUCCEEDED,
                attempts=1,
                started_at=now,
                finished_at=now,
            )
        )

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
        # Steps are hydrated from the store on the shadow path (FIX 5).
        agents_seen = {s["agent"] for s in body["steps"]}
        assert agents_seen == {"web_search"}

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

    async def test_shadow_response_hydrates_steps_when_store_returns(
        self, shadow_client: tuple[AsyncClient, _SelectiveFailStore]
    ) -> None:
        """Shadow path returns the run, but if the store is reachable
        for `list_steps` we should fold those rows into the response
        rather than returning `steps=[]`.

        The shadow is populated *before* per-attempt step rows land in
        the store, so without this hydration callers see a terminal
        state with no audit trail even when the store has the rows.
        """
        client, store = shadow_client
        run_id = "hydrate-me"
        # Plant a terminal record into the shadow only.
        terminal = RunStatus(
            run_id=run_id,
            question="what is python",
            state=StepStatus.SUCCEEDED,
        )
        app.state.terminal_shadow[run_id] = terminal
        # Pre-populate steps in the store.
        now = datetime.now(UTC)
        await store._inner.append_step(
            StepRecord(
                run_id=run_id,
                agent=AgentName.WEB_SEARCH,
                status=StepStatus.SUCCEEDED,
                attempts=1,
                started_at=now,
                finished_at=now,
            )
        )
        await store._inner.append_step(
            StepRecord(
                run_id=run_id,
                agent=AgentName.SCHOLAR,
                status=StepStatus.SUCCEEDED,
                attempts=1,
                started_at=now,
                finished_at=now,
            )
        )
        # Force the response to come from the shadow path: store outage
        # for the run-fetch, but list_steps still works.
        store.fail_get_run = True
        r = await client.get(f"/runs/{run_id}")
        assert r.status_code == 200
        body = r.json()
        agents_seen = {s["agent"] for s in body["steps"]}
        assert agents_seen == {"web_search", "scholar"}, (
            f"shadow response must hydrate steps from the store; got {agents_seen}"
        )

    async def test_shadow_response_keeps_empty_steps_when_store_unreachable(
        self, shadow_client: tuple[AsyncClient, _SelectiveFailStore]
    ) -> None:
        """If list_steps also fails, the response carries empty steps
        rather than 503 — surfacing a terminal state with no audit is
        strictly better than not surfacing it at all."""
        client, store = shadow_client
        run_id = "hydrate-fail"
        terminal = RunStatus(
            run_id=run_id,
            question="what is python",
            state=StepStatus.FAILED,
        )
        app.state.terminal_shadow[run_id] = terminal
        store.fail_get_run = True
        store.fail_list_steps = True
        r = await client.get(f"/runs/{run_id}")
        assert r.status_code == 200
        assert r.json()["steps"] == []


class TestPersistTerminalNarrowExceptions:
    """`_persist_terminal` must downgrade only *expected* outage
    exceptions to the shadow. A serialization or programmer bug should
    propagate so it gets fixed, not silently masked as if Redis were
    down.
    """

    async def test_persist_terminal_does_not_swallow_serialization_bugs(self) -> None:
        class BugStore:
            async def put_run(self, run: RunStatus) -> None:
                # Simulate a programmer bug: not a known store-outage class.
                raise TypeError("model_dump_json received an unhashable thing")

            async def get_run(self, run_id: str) -> RunStatus | None:  # pragma: no cover
                return None

            async def append_step(self, _step: object) -> None:  # pragma: no cover
                pass

            async def list_steps(self, _run_id: str) -> list[StepRecord]:  # pragma: no cover
                return []

            async def cache_get(self, _key: str) -> AgentResult | None:  # pragma: no cover
                return None

            async def cache_put(self, _key: str, _result: AgentResult) -> None:  # pragma: no cover
                pass

        shadow = _TerminalShadow()
        run = RunStatus(run_id="r-bug", question="q", state=StepStatus.SUCCEEDED)
        with pytest.raises(TypeError):
            await _persist_terminal(BugStore(), shadow, run, agent_label="test")
        # And the shadow must NOT have been used as a fallback.
        assert "r-bug" not in shadow


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
        last_status = -1
        for _ in range(40):
            r = await client.get(f"/runs/{run_id}")
            last_status = r.status_code
            body = r.json()
            if body.get("state") in ("succeeded", "failed"):
                break
        assert last_status == 200, f"expected 200 from happy path, got {last_status}"
        assert body["state"] == "succeeded"
        # Shadow remained empty for this run id.
        assert run_id not in app.state.terminal_shadow


class TestShadowBounded:
    """The shadow MUST cap memory growth — a long store outage with a
    steady submit rate would otherwise leak entries forever.
    """

    def test_oldest_entry_evicted_when_capacity_exceeded(self) -> None:


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


        with pytest.raises(ValueError):
            _TerminalShadow(max_size=0)
        with pytest.raises(ValueError):
            _TerminalShadow(max_size=-1)

    def test_clear_resets_to_empty(self) -> None:


        shadow = _TerminalShadow(max_size=10)
        shadow["a"] = RunStatus(run_id="a", question="q", state=StepStatus.FAILED)
        shadow["b"] = RunStatus(run_id="b", question="q", state=StepStatus.FAILED)
        assert len(shadow) == 2
        shadow.clear()
        assert len(shadow) == 0
        assert "a" not in shadow
