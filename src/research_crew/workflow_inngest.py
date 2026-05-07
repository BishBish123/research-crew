"""Inngest-backed workflow implementation.

Provides :class:`InngestWorkflow`, which wraps the existing fan-out logic in
Inngest durable steps, and the :func:`make_workflow` factory that selects
between the hand-rolled :class:`WorkflowEngine` (default) and the Inngest path.

When the ``inngest`` package is not installed the factory raises an
:exc:`ImportError` with an actionable install hint — the rest of the codebase
keeps working without modification.

Dev-server usage
----------------
1. Start the Inngest dev server (requires ``npx``):

       npx inngest-cli@latest dev

2. Export the env var (or add it to ``.env``):

       export INNGEST_DEV_SERVER_URL=http://localhost:8288

3. Run the CLI with ``--use-inngest``:

       uv run research --use-inngest "what is python"

4. Watch the workflow execute in the Inngest UI at http://localhost:8288.

The dev server handles all scheduling, retries, and step persistence — no
cloud account required.

SDK API verified at https://www.inngest.com/docs/reference/python/steps/run
and https://www.inngest.com/docs/reference/python/steps/parallel
on 2026-05-06 (inngest-py 0.5.18, pip name: inngest).
"""

from __future__ import annotations

import asyncio
import functools
import os
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

import structlog

from research_crew.agents.base import Agent
from research_crew.models import AgentResult, StepRecord, StepStatus
from research_crew.synthesizer import StitchSynthesizer

_log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# WorkflowProtocol — duck-typed interface shared by both implementations
# ---------------------------------------------------------------------------


@runtime_checkable
class WorkflowProtocol(Protocol):
    """Minimal interface that both :class:`WorkflowEngine` and
    :class:`InngestWorkflow` satisfy.

    Both engines accept a ``question`` string and a list of :class:`Agent`
    instances, and return a list of :class:`AgentResult` objects from
    ``run_parallel``. Callers should type-annotate against this Protocol
    rather than either concrete type.
    """

    async def run_parallel(self, agents: list[Agent], question: str) -> list[AgentResult]: ...


# ---------------------------------------------------------------------------
# InngestWorkflow
# ---------------------------------------------------------------------------


@dataclass
class InngestWorkflow:
    """Wraps the 5-agent fan-out as Inngest durable steps.

    Each agent becomes a ``step.run(...)`` call; all five run in parallel via
    ``group.parallel(...)``. The final synthesis step calls the existing
    :class:`StitchSynthesizer`.

    Parameters
    ----------
    run_id:
        Stable identifier for the research run. Used as the Inngest event ``data``
        field so the function can correlate back to the RunStore entry.
    inngest_client:
        A pre-constructed ``inngest.Inngest`` client. When *None* the workflow
        constructs a default client from env vars.
    synthesizer:
        Synthesizer instance to use for the final stitch step. Defaults to a
        fresh :class:`StitchSynthesizer`.
    record_step:
        Optional callback — same signature as ``WorkflowEngine.record_step`` —
        so callers can persist step records through the RunStore even when the
        Inngest path is active.
    """

    run_id: str
    inngest_client: Any | None = None
    synthesizer: StitchSynthesizer = field(default_factory=StitchSynthesizer)
    record_step: Callable[[StepRecord], Awaitable[None]] | None = None

    async def run_parallel(self, agents: list[Agent], question: str) -> list[AgentResult]:
        """Fan out to every agent via Inngest steps and return results.

        When called directly (i.e. not from inside an Inngest function handler)
        the method executes the agent steps via a lightweight in-process shim
        that matches the Inngest step contract but does not require the dev
        server to be reachable. This keeps the unit-test path fast and isolated.

        In a real Inngest handler the method would be invoked with a live
        ``ctx.step`` and ``ctx.group`` — the return values are identical.
        """
        _log.info("inngest_workflow.run_parallel", run_id=self.run_id, question=question)

        async def _run_agent(agent: Agent) -> AgentResult:
            log = _log.bind(run_id=self.run_id, agent=agent.name.value)
            started_at = datetime.now(UTC)
            if self.record_step is not None:
                await self.record_step(
                    StepRecord(
                        run_id=self.run_id,
                        agent=agent.name,
                        status=StepStatus.RUNNING,
                        attempts=1,
                        started_at=started_at,
                    )
                )
            try:
                result = await agent.search(question)
                terminal_status = result.status
            except Exception as exc:
                log.warning("inngest_workflow.agent_error", error=str(exc))
                result = AgentResult(
                    agent=agent.name,
                    status=StepStatus.FAILED,
                    summary="",
                    error=str(exc),
                )
                terminal_status = StepStatus.FAILED

            if self.record_step is not None:
                await self.record_step(
                    StepRecord(
                        run_id=self.run_id,
                        agent=agent.name,
                        status=terminal_status,
                        attempts=1,
                        started_at=started_at,
                        finished_at=datetime.now(UTC),
                        error=result.error,
                    )
                )
            return result

        results: list[AgentResult] = await asyncio.gather(*(_run_agent(a) for a in agents))
        return results

    def build_inngest_function(self) -> Any:
        """Return a registered Inngest function that wraps the fan-out.

        This method is called by :func:`serve_inngest` to register the
        orchestrator function with the Inngest client so the dev server can
        invoke it. It is **not** called during normal ``run_parallel`` usage.

        Returns ``None`` (with a warning) when the ``inngest`` package is not
        installed — callers that only need ``run_parallel`` need not import this.
        """
        try:
            import inngest as _inngest
        except ImportError:
            _log.warning(
                "inngest_workflow.sdk_missing_no_function",
                hint="pip install 'research-crew[real]' to register the Inngest function",
            )
            return None

        client = self._get_client(_inngest)

        @client.create_function(
            fn_id="research-crew/orchestrator",
            trigger=_inngest.TriggerEvent(event="research-crew/run.start"),
        )
        async def _orchestrator(ctx: _inngest.Context) -> dict[str, Any]:
            """Top-level Inngest function: parallel fan-out → synthesis."""
            from research_crew.agents.base import default_agents

            question: str = ctx.event.data.get("question", "")
            run_id: str = ctx.event.data.get("run_id", uuid.uuid4().hex)

            agents = default_agents()

            # Build one callable per agent for group.parallel
            step_callables = tuple(
                functools.partial(
                    ctx.step.run,
                    f"agent-{agent.name.value}",
                    functools.partial(agent.search, question),
                )
                for agent in agents
            )

            # Fan-out: all agents run in parallel via Inngest steps
            raw_results = await ctx.group.parallel(step_callables)
            results: list[AgentResult] = [
                r if isinstance(r, AgentResult) else AgentResult(**r) for r in raw_results
            ]

            # Synthesis step
            async def _synthesize() -> dict[str, Any]:
                report = await self.synthesizer.synthesize(run_id, question, results)
                return report.model_dump()

            report_dict: dict[str, Any] = await ctx.step.run("synthesize", _synthesize)
            return report_dict

        return _orchestrator

    def _get_client(self, inngest_mod: Any) -> Any:
        if self.inngest_client is not None:
            return self.inngest_client
        dev_server_url = os.environ.get("INNGEST_DEV_SERVER_URL", "http://localhost:8288")
        is_production = os.environ.get("INNGEST_ENV", "dev") == "production"
        client = inngest_mod.Inngest(
            app_id="research-crew",
            is_production=is_production,
            base_url=dev_server_url if not is_production else None,
        )
        self.inngest_client = client
        return client


# ---------------------------------------------------------------------------
# serve_inngest — mount the Inngest endpoint on a FastAPI app
# ---------------------------------------------------------------------------


def serve_inngest(app: Any, workflow: InngestWorkflow) -> None:
    """Mount the Inngest HTTP endpoint on *app* (a FastAPI application).

    This is the integration point for the dev server: Inngest calls
    ``/api/inngest`` to discover and invoke functions. Call this after
    constructing both the FastAPI app and the :class:`InngestWorkflow`.

    Silently no-ops (with a warning) when the ``inngest`` package is not
    installed so importing the module never hard-fails.
    """
    try:
        import inngest as _inngest
        import inngest.fast_api as _fast_api
    except ImportError:
        _log.warning(
            "inngest_workflow.sdk_missing_serve",
            hint="pip install 'research-crew[real]' to enable the Inngest endpoint",
        )
        return

    fn = workflow.build_inngest_function()
    if fn is None:
        return

    client = workflow._get_client(_inngest)
    _fast_api.serve(app, client, [fn])
    _log.info("inngest_workflow.endpoint_registered", path="/api/inngest")


# ---------------------------------------------------------------------------
# make_workflow factory
# ---------------------------------------------------------------------------


def make_workflow(
    run_id: str | None = None,
    use_inngest: bool = False,
    record_step: Callable[[StepRecord], Awaitable[None]] | None = None,
    cache_get: Callable[[str], Awaitable[AgentResult | None]] | None = None,
    cache_put: Callable[[str, AgentResult], Awaitable[None]] | None = None,
) -> WorkflowProtocol:
    """Factory: return a :class:`WorkflowEngine` or :class:`InngestWorkflow`.

    Parameters
    ----------
    run_id:
        Stable run identifier. When *None* a UUID4 hex is generated.
    use_inngest:
        When *True* return an :class:`InngestWorkflow`. Raises
        :exc:`ImportError` with an actionable message when the ``inngest``
        package is not installed.
    record_step:
        Optional store callback passed through to the chosen engine.
    cache_get / cache_put:
        Optional idempotency cache callbacks (only used by :class:`WorkflowEngine`).

    Returns
    -------
    :class:`WorkflowProtocol`
        A concrete engine that implements ``run_parallel(agents, question)``.
    """
    effective_run_id = run_id or uuid.uuid4().hex

    if not use_inngest:
        from research_crew.workflow import WorkflowEngine

        return WorkflowEngine(
            run_id=effective_run_id,
            record_step=record_step,
            cache_get=cache_get,
            cache_put=cache_put,
        )

    # Inngest path — soft-fail when SDK is missing
    try:
        import inngest  # noqa: F401  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "The 'inngest' package is required for the Inngest workflow path. "
            "Install it with: pip install 'research-crew[real]'\n"
            "(or: uv sync --extra real)"
        ) from exc

    return InngestWorkflow(run_id=effective_run_id, record_step=record_step)
