"""Tests for the Langfuse observability adapter.

Three scenarios:

1. **No env vars** — ``make_tracer()`` returns a :class:`NullTracer`; all
   methods are no-ops; running a full workflow with it raises no errors.
2. **Env vars set but SDK not installed** — ``LangfuseTracer.__init__``
   silently falls back to no-op mode (``_client is None``).
3. **Env vars set and SDK mocked** — ``LangfuseTracer`` calls the right SDK
   methods; a workflow run emits ``start_observation`` / ``update`` / ``flush``.

All tests run without any network access to Langfuse cloud.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from research_crew.agents import MockAgent
from research_crew.models import AgentName, StepRecord, StepStatus
from research_crew.observability.langfuse import (
    LangfuseTracer,
    NullTracer,
    RunHandle,
    make_tracer,
)
from research_crew.store import InMemoryRunStore
from research_crew.workflow import WorkflowConfig, WorkflowEngine

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _engine(store: InMemoryRunStore, tracer: NullTracer | LangfuseTracer) -> WorkflowEngine:
    return WorkflowEngine(
        run_id="run-obs-test",
        config=WorkflowConfig(max_attempts=1, base_backoff_s=0.0),
        record_step=store.append_step,
        cache_get=store.cache_get,
        cache_put=store.cache_put,
        tracer=tracer,
    )


def _step_record(status: StepStatus = StepStatus.SUCCEEDED) -> StepRecord:
    now = datetime.now(UTC)
    return StepRecord(
        run_id="run-obs-test",
        agent=AgentName.WEB_SEARCH,
        status=status,
        attempts=1,
        started_at=now,
        finished_at=now if status is not StepStatus.RUNNING else None,
    )


# ---------------------------------------------------------------------------
# Scenario 1: No env vars → NullTracer
# ---------------------------------------------------------------------------


class TestNoEnvVars:
    def test_make_tracer_returns_null_tracer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        tracer = make_tracer()
        assert isinstance(tracer, NullTracer)

    def test_null_tracer_start_run_returns_handle(self) -> None:
        tracer = NullTracer()
        handle = tracer.start_run("what is python")
        assert isinstance(handle, RunHandle)
        assert handle.trace_id == "null"

    def test_null_tracer_record_step_is_noop(self) -> None:
        tracer = NullTracer()
        handle = tracer.start_run("what is python")
        # Must not raise; no return value to assert.
        tracer.record_step(handle, _step_record())

    def test_null_tracer_finish_run_is_noop(self) -> None:
        tracer = NullTracer()
        handle = tracer.start_run("what is python")
        tracer.finish_run(handle, "succeeded")  # must not raise

    async def test_workflow_with_null_tracer_completes(self) -> None:
        store = InMemoryRunStore()
        tracer = NullTracer()
        engine = _engine(store, tracer)
        agent = MockAgent(name=AgentName.WEB_SEARCH, latency_ms=0)
        results = await engine.run_parallel([agent], "what is python")
        assert len(results) == 1
        assert results[0].status is StepStatus.SUCCEEDED

    async def test_workflow_without_explicit_tracer_completes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Engine with tracer=None picks up NullTracer from factory (no env vars)."""
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        store = InMemoryRunStore()
        engine = WorkflowEngine(
            run_id="run-no-tracer",
            config=WorkflowConfig(max_attempts=1, base_backoff_s=0.0),
            record_step=store.append_step,
        )
        agent = MockAgent(name=AgentName.WEB_SEARCH, latency_ms=0)
        results = await engine.run_parallel([agent], "what is python")
        assert results[0].status is StepStatus.SUCCEEDED


# ---------------------------------------------------------------------------
# Scenario 2: Env vars set but SDK not installed → LangfuseTracer falls back
# ---------------------------------------------------------------------------


class TestEnvVarsButNoSdk:
    def test_langfuse_tracer_falls_back_when_sdk_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk_test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk_test")

        # Temporarily hide the langfuse package from the import machinery.
        original = sys.modules.pop("langfuse", None)
        try:
            with patch.dict(sys.modules, {"langfuse": None}):  # type: ignore[dict-item]
                tracer = LangfuseTracer()
        finally:
            if original is not None:
                sys.modules["langfuse"] = original

        # Must be in no-op mode regardless of env vars.
        assert tracer._client is None

    def test_langfuse_tracer_noop_methods_dont_raise(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk_test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk_test")

        original = sys.modules.pop("langfuse", None)
        try:
            with patch.dict(sys.modules, {"langfuse": None}):  # type: ignore[dict-item]
                tracer = LangfuseTracer()
        finally:
            if original is not None:
                sys.modules["langfuse"] = original

        handle = tracer.start_run("query")
        assert handle.trace_id == "null"
        tracer.record_step(handle, _step_record())
        tracer.finish_run(handle, "succeeded")

    def test_make_tracer_with_env_vars_but_missing_sdk_returns_langfuse_tracer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """make_tracer() returns LangfuseTracer (in no-op mode) not NullTracer."""
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk_test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk_test")

        original = sys.modules.pop("langfuse", None)
        try:
            with patch.dict(sys.modules, {"langfuse": None}):  # type: ignore[dict-item]
                tracer = make_tracer()
        finally:
            if original is not None:
                sys.modules["langfuse"] = original

        assert isinstance(tracer, LangfuseTracer)
        assert tracer._client is None


# ---------------------------------------------------------------------------
# Scenario 3: Env vars set + SDK mocked → live calls verified
# ---------------------------------------------------------------------------


def _make_mock_langfuse_module() -> tuple[ModuleType, MagicMock]:
    """Return a fake ``langfuse`` module and the mock Langfuse client it exposes."""
    fake_trace = MagicMock()
    fake_trace.id = "trace-abc123"

    fake_client = MagicMock()
    fake_client.trace.return_value = fake_trace

    fake_mod = ModuleType("langfuse")
    # Expose Langfuse class (v2/v3 path) — v4 get_client() is NOT present so
    # the adapter takes the v2/v3 branch and we can inspect .trace() calls.
    fake_mod.Langfuse = MagicMock(return_value=fake_client)  # type: ignore[attr-defined]
    # Deliberately omit get_client to exercise the v2/v3 code path.

    return fake_mod, fake_client


class TestMockedSdk:
    def test_start_run_calls_trace(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk_test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk_test")
        monkeypatch.delenv("LANGFUSE_HOST", raising=False)

        fake_mod, fake_client = _make_mock_langfuse_module()
        with patch.dict(sys.modules, {"langfuse": fake_mod}):
            tracer = LangfuseTracer()
            handle = tracer.start_run("what is python")

        assert handle.trace_id == "trace-abc123"
        fake_client.trace.assert_called_once_with(name="research-run", input="what is python")

    def test_record_step_creates_span(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk_test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk_test")

        fake_mod, fake_client = _make_mock_langfuse_module()
        fake_trace = fake_client.trace.return_value

        with patch.dict(sys.modules, {"langfuse": fake_mod}):
            tracer = LangfuseTracer()
            handle = tracer.start_run("q")
            tracer.record_step(handle, _step_record(StepStatus.SUCCEEDED))

        fake_trace.span.assert_called_once()
        call_kwargs = fake_trace.span.call_args
        assert call_kwargs.kwargs["name"] == "agent:web_search"
        meta = call_kwargs.kwargs["metadata"]
        assert meta["agent"] == "web_search"
        assert meta["status"] == "succeeded"

    def test_finish_run_calls_update_and_flush(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk_test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk_test")

        fake_mod, fake_client = _make_mock_langfuse_module()
        fake_trace = fake_client.trace.return_value

        with patch.dict(sys.modules, {"langfuse": fake_mod}):
            tracer = LangfuseTracer()
            handle = tracer.start_run("q")
            tracer.finish_run(handle, "succeeded")

        fake_trace.update.assert_called_with(output="succeeded")
        fake_client.flush.assert_called_once()

    async def test_full_workflow_emits_trace_and_spans(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """End-to-end: run_parallel → start_run + N record_step + finish_run."""
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk_test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk_test")

        fake_mod, fake_client = _make_mock_langfuse_module()

        with patch.dict(sys.modules, {"langfuse": fake_mod}):
            tracer = LangfuseTracer()

        store = InMemoryRunStore()
        engine = _engine(store, tracer)
        agents = [
            MockAgent(name=AgentName.WEB_SEARCH, latency_ms=0),
            MockAgent(name=AgentName.NEWS, latency_ms=0),
        ]

        results = await engine.run_parallel(agents, "what is python")

        assert all(r.status is StepStatus.SUCCEEDED for r in results)
        # One trace opened.
        fake_client.trace.assert_called_once()
        # flush called at end.
        fake_client.flush.assert_called_once()
        # record_step spans: 1 SUCCEEDED span per agent (RUNNING spans are skipped).
        fake_trace = fake_client.trace.return_value
        assert fake_trace.span.call_count == 2  # one per agent

    def test_record_step_skipped_for_running_status(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """RUNNING step records must not create Langfuse spans (open without close)."""
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk_test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk_test")

        fake_mod, fake_client = _make_mock_langfuse_module()
        fake_trace = fake_client.trace.return_value

        with patch.dict(sys.modules, {"langfuse": fake_mod}):
            tracer = LangfuseTracer()
            handle = tracer.start_run("q")
            tracer.record_step(handle, _step_record(StepStatus.RUNNING))

        fake_trace.span.assert_not_called()
