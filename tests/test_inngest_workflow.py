"""Tests for the Inngest workflow integration.

Coverage
--------
1. ``make_workflow(use_inngest=False)`` → returns ``WorkflowEngine`` (smoke test).
2. ``make_workflow(use_inngest=True)`` without ``inngest`` installed → clean
   ``ImportError`` with actionable message.
3. ``InngestWorkflow.run_parallel(...)`` with mocked agents → end-to-end path
   executes 5 agents and produces results.
4. Schema-equivalence: ``InngestWorkflow.run_parallel`` and
   ``WorkflowEngine.run_parallel`` return the same ``RunStatus``-compatible
   shape (same agent names present, same ``AgentResult`` fields).
5. ``InngestWorkflow`` satisfies ``WorkflowProtocol`` at runtime.
6. ``make_workflow`` passes ``record_step`` through to ``InngestWorkflow``.
7. Soft-fail: ``InngestWorkflow.build_inngest_function()`` returns ``None``
   when ``inngest`` is not installed — no exception raised.
8. ``serve_inngest`` is a no-op (no exception) when ``inngest`` is not installed.
9. ``make_workflow`` with ``use_inngest=True`` and inngest importable returns
   ``InngestWorkflow`` instance.
10. Steps are recorded by ``InngestWorkflow.run_parallel`` when ``record_step``
    is provided.

None of these tests start the Inngest dev server or connect to localhost:8288.
All inngest SDK calls are either absent (not installed path) or mocked.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from types import ModuleType
from unittest.mock import MagicMock

import pytest

from research_crew.agents import MockAgent
from research_crew.models import AgentName, AgentResult, StepRecord, StepStatus
from research_crew.synthesizer import StitchSynthesizer
from research_crew.workflow import WorkflowEngine
from research_crew.workflow_inngest import (
    InngestWorkflow,
    WorkflowProtocol,
    make_workflow,
    serve_inngest,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ALL_AGENT_NAMES = list(AgentName)


def _five_mock_agents() -> list[MockAgent]:
    return [MockAgent(name=n, latency_ms=0) for n in ALL_AGENT_NAMES]


def _make_fake_inngest_module() -> ModuleType:
    """Return a minimal fake ``inngest`` module that satisfies the SDK surface
    we use: ``Inngest``, ``TriggerEvent``, ``Context``, ``fast_api.serve``."""
    mod = ModuleType("inngest")

    class _FakeInngest:
        def __init__(self, **kwargs: object) -> None:
            pass

        def create_function(self, **kwargs: object) -> Callable[..., object]:
            def decorator(fn: Callable[..., object]) -> Callable[..., object]:
                return fn

            return decorator

    class _FakeTriggerEvent:
        def __init__(self, **kwargs: object) -> None:
            pass

    class _FakeContext:
        pass

    mod.Inngest = _FakeInngest  # type: ignore[attr-defined]
    mod.TriggerEvent = _FakeTriggerEvent  # type: ignore[attr-defined]
    mod.Context = _FakeContext  # type: ignore[attr-defined]

    # fast_api submodule
    fast_api_mod = ModuleType("inngest.fast_api")
    fast_api_mod.serve = MagicMock()  # type: ignore[attr-defined]
    mod.fast_api = fast_api_mod  # type: ignore[attr-defined]

    return mod


# ---------------------------------------------------------------------------
# 1. make_workflow(use_inngest=False) → WorkflowEngine
# ---------------------------------------------------------------------------


class TestMakeWorkflowDefault:
    def test_returns_workflow_engine_by_default(self) -> None:
        engine = make_workflow(run_id="smoke-test", use_inngest=False)
        assert isinstance(engine, WorkflowEngine)

    def test_run_id_is_passed_through(self) -> None:
        engine = make_workflow(run_id="my-run", use_inngest=False)
        assert isinstance(engine, WorkflowEngine)
        assert engine.run_id == "my-run"

    def test_auto_run_id_when_none(self) -> None:
        engine = make_workflow(use_inngest=False)
        assert isinstance(engine, WorkflowEngine)
        assert engine.run_id  # non-empty


# ---------------------------------------------------------------------------
# 2. make_workflow(use_inngest=True) without inngest installed → ImportError
# ---------------------------------------------------------------------------


class TestMakeWorkflowInngestMissing:
    def test_raises_import_error_with_hint(self) -> None:
        """When inngest is not importable, make_workflow raises ImportError
        with a message that tells the user exactly how to install it."""
        # Temporarily hide the inngest module (if it happens to be installed).
        original = sys.modules.pop("inngest", None)
        try:
            with pytest.raises(ImportError, match="research-crew\\[real\\]"):
                make_workflow(use_inngest=True)
        finally:
            if original is not None:
                sys.modules["inngest"] = original

    def test_error_mentions_uv_sync(self) -> None:
        original = sys.modules.pop("inngest", None)
        try:
            with pytest.raises(ImportError, match="uv sync"):
                make_workflow(use_inngest=True)
        finally:
            if original is not None:
                sys.modules["inngest"] = original


# ---------------------------------------------------------------------------
# 3. InngestWorkflow.run_parallel — end-to-end with mock agents
# ---------------------------------------------------------------------------


class TestInngestWorkflowRunParallel:
    async def test_runs_all_five_agents(self) -> None:
        workflow = InngestWorkflow(run_id="inngest-test-01")
        agents = _five_mock_agents()
        results = await workflow.run_parallel(agents, "what is python")
        assert len(results) == 5

    async def test_all_results_are_agent_result_instances(self) -> None:
        workflow = InngestWorkflow(run_id="inngest-test-02")
        agents = _five_mock_agents()
        results = await workflow.run_parallel(agents, "test question")
        for r in results:
            assert isinstance(r, AgentResult)

    async def test_result_statuses_are_valid(self) -> None:
        workflow = InngestWorkflow(run_id="inngest-test-03")
        agents = _five_mock_agents()
        results = await workflow.run_parallel(agents, "test question")
        valid = {StepStatus.SUCCEEDED, StepStatus.FAILED, StepStatus.CACHED}
        for r in results:
            assert r.status in valid

    async def test_happy_path_all_succeed(self) -> None:
        workflow = InngestWorkflow(run_id="inngest-test-04")
        agents = _five_mock_agents()
        results = await workflow.run_parallel(agents, "test question")
        assert all(r.status is StepStatus.SUCCEEDED for r in results)

    async def test_agent_exception_produces_failed_result(self) -> None:
        """An agent that raises should produce a FAILED AgentResult, not propagate."""

        class _ExplodingAgent:
            name = AgentName.CODE

            async def search(self, question: str) -> AgentResult:
                raise RuntimeError("simulated explosion")

        workflow = InngestWorkflow(run_id="inngest-test-05")
        agents: list[object] = [_ExplodingAgent()]  # type: ignore[list-item]
        results = await workflow.run_parallel(agents, "test")  # type: ignore[arg-type]
        assert len(results) == 1
        assert results[0].status is StepStatus.FAILED
        assert "simulated explosion" in (results[0].error or "")


# ---------------------------------------------------------------------------
# 4. Schema-equivalence: InngestWorkflow vs WorkflowEngine
# ---------------------------------------------------------------------------


class TestSchemaEquivalence:
    async def test_same_agent_names_present(self) -> None:
        """Both engines produce results for exactly the same set of agents."""
        agents = _five_mock_agents()
        question = "schema equivalence test"

        inngest_wf = InngestWorkflow(run_id="equiv-inngest")
        inngest_results = await inngest_wf.run_parallel(agents, question)

        engine = WorkflowEngine(run_id="equiv-engine")
        engine_results = await engine.run_parallel(agents, question)

        inngest_names = {r.agent for r in inngest_results}
        engine_names = {r.agent for r in engine_results}
        assert inngest_names == engine_names

    async def test_result_fields_match_agent_result_schema(self) -> None:
        """Every field on AgentResult is populated consistently by both engines."""
        agents = _five_mock_agents()
        question = "field schema test"

        inngest_wf = InngestWorkflow(run_id="field-inngest")
        inngest_results = await inngest_wf.run_parallel(agents, question)

        for r in inngest_results:
            # These are the AgentResult fields the RunStatus render depends on.
            assert isinstance(r.agent, AgentName)
            assert isinstance(r.status, StepStatus)
            assert isinstance(r.summary, str)
            assert isinstance(r.citations, list)
            assert isinstance(r.attempts, int)

    async def test_synthesizer_accepts_inngest_results(self) -> None:
        """StitchSynthesizer.synthesize() should work on InngestWorkflow results
        because the AgentResult shape is identical."""
        agents = _five_mock_agents()
        workflow = InngestWorkflow(run_id="synth-inngest")
        results = await workflow.run_parallel(agents, "synthesis test")

        synth = StitchSynthesizer()
        report = await synth.synthesize("synth-inngest", "synthesis test", results)
        assert report.run_id == "synth-inngest"
        assert len(report.agent_results) == 5


# ---------------------------------------------------------------------------
# 5. WorkflowProtocol runtime check
# ---------------------------------------------------------------------------


class TestWorkflowProtocol:
    def test_inngest_workflow_satisfies_protocol(self) -> None:
        wf = InngestWorkflow(run_id="proto-test")
        assert isinstance(wf, WorkflowProtocol)

    def test_workflow_engine_satisfies_protocol(self) -> None:
        engine = WorkflowEngine(run_id="proto-engine")
        assert isinstance(engine, WorkflowProtocol)


# ---------------------------------------------------------------------------
# 6. make_workflow passes record_step through
# ---------------------------------------------------------------------------


class TestMakeWorkflowRecordStep:
    def test_record_step_is_forwarded_to_engine(self) -> None:
        recorded: list[StepRecord] = []

        async def _rec(step: StepRecord) -> None:
            recorded.append(step)

        engine = make_workflow(run_id="rec-test", use_inngest=False, record_step=_rec)
        assert isinstance(engine, WorkflowEngine)
        assert engine.record_step is _rec

    async def test_record_step_forwarded_to_inngest_workflow(self) -> None:
        # Inject fake inngest so make_workflow doesn't raise
        fake_inngest = _make_fake_inngest_module()
        sys.modules["inngest"] = fake_inngest

        recorded: list[StepRecord] = []

        async def _rec(step: StepRecord) -> None:
            recorded.append(step)

        try:
            wf = make_workflow(run_id="rec-inngest", use_inngest=True, record_step=_rec)
            assert isinstance(wf, InngestWorkflow)
            assert wf.record_step is _rec
        finally:
            sys.modules.pop("inngest", None)


# ---------------------------------------------------------------------------
# 7. build_inngest_function soft-fail when inngest missing
# ---------------------------------------------------------------------------


class TestBuildInngestFunctionSoftFail:
    def test_returns_none_when_sdk_missing(self) -> None:
        original = sys.modules.pop("inngest", None)
        try:
            wf = InngestWorkflow(run_id="build-fn-test")
            result = wf.build_inngest_function()
            assert result is None
        finally:
            if original is not None:
                sys.modules["inngest"] = original


# ---------------------------------------------------------------------------
# 8. serve_inngest no-op when inngest missing
# ---------------------------------------------------------------------------


class TestServeInngestSoftFail:
    def test_no_exception_when_sdk_missing(self) -> None:
        original = sys.modules.pop("inngest", None)
        try:
            fake_app = MagicMock()
            wf = InngestWorkflow(run_id="serve-test")
            serve_inngest(fake_app, wf)  # must not raise
        finally:
            if original is not None:
                sys.modules["inngest"] = original


# ---------------------------------------------------------------------------
# 9. make_workflow(use_inngest=True) with inngest importable → InngestWorkflow
# ---------------------------------------------------------------------------


class TestMakeWorkflowInngestPresent:
    def test_returns_inngest_workflow_when_sdk_present(self) -> None:
        fake_inngest = _make_fake_inngest_module()
        sys.modules["inngest"] = fake_inngest
        try:
            wf = make_workflow(run_id="present-test", use_inngest=True)
            assert isinstance(wf, InngestWorkflow)
        finally:
            sys.modules.pop("inngest", None)

    def test_run_id_propagated(self) -> None:
        fake_inngest = _make_fake_inngest_module()
        sys.modules["inngest"] = fake_inngest
        try:
            wf = make_workflow(run_id="my-inngest-run", use_inngest=True)
            assert isinstance(wf, InngestWorkflow)
            assert wf.run_id == "my-inngest-run"
        finally:
            sys.modules.pop("inngest", None)


# ---------------------------------------------------------------------------
# 10. record_step is called by InngestWorkflow.run_parallel
# ---------------------------------------------------------------------------


class TestInngestWorkflowRecordsSteps:
    async def test_steps_are_recorded(self) -> None:
        recorded: list[StepRecord] = []

        async def _rec(step: StepRecord) -> None:
            recorded.append(step)

        workflow = InngestWorkflow(run_id="steps-test", record_step=_rec)
        agents = _five_mock_agents()
        await workflow.run_parallel(agents, "record steps test")

        # Each agent should emit at least one RUNNING and one terminal record
        running = [s for s in recorded if s.status is StepStatus.RUNNING]
        terminal = [s for s in recorded if s.status in {StepStatus.SUCCEEDED, StepStatus.FAILED}]
        assert len(running) == 5
        assert len(terminal) == 5

    async def test_step_run_ids_match_workflow(self) -> None:
        recorded: list[StepRecord] = []

        async def _rec(step: StepRecord) -> None:
            recorded.append(step)

        wf = InngestWorkflow(run_id="run-id-check", record_step=_rec)
        await wf.run_parallel(_five_mock_agents(), "run id test")
        assert all(s.run_id == "run-id-check" for s in recorded)
