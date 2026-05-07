"""Langfuse tracing adapter for research-crew.

The adapter is **env-key-gated**: when both ``LANGFUSE_PUBLIC_KEY`` and
``LANGFUSE_SECRET_KEY`` are set the tracer attempts to import the
``langfuse`` SDK and initialise a live client. When either key is missing,
or when the SDK is not installed, all methods are no-ops and the rest of
the application continues unchanged.

Env vars
--------
LANGFUSE_PUBLIC_KEY  Required to enable tracing.
LANGFUSE_SECRET_KEY  Required to enable tracing.
LANGFUSE_HOST        Optional. Defaults to https://cloud.langfuse.com.

Typical usage
-------------
::

    from research_crew.observability.langfuse import make_tracer

    tracer = make_tracer()
    handle = tracer.start_run("what is python")
    tracer.record_step(handle, step_record)
    tracer.finish_run(handle, "succeeded")

All calls are safe when env vars are absent — no exceptions, no side-effects.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import structlog

from research_crew.models import StepRecord, StepStatus

_log = structlog.get_logger(__name__)


@dataclass
class RunHandle:
    """Opaque handle returned by :meth:`LangfuseTracer.start_run`.

    Callers pass this back to :meth:`record_step` and :meth:`finish_run`;
    the tracer stores whatever internal state it needs as fields here.
    """

    trace_id: str
    # Holds the live Langfuse trace object in live mode; None in no-op mode.
    _trace: Any = field(default=None, repr=False)


class NullTracer:
    """No-op tracer used when env vars are absent or the SDK is not installed.

    Every method returns immediately without side-effects.
    """

    def start_run(self, query: str) -> RunHandle:
        return RunHandle(trace_id="null")

    def record_step(self, handle: RunHandle, step: StepRecord) -> None:
        pass

    def finish_run(self, handle: RunHandle, outcome: str) -> None:
        pass


class LangfuseTracer:
    """Langfuse-backed tracer.

    Constructed by :func:`make_tracer` when the required env vars are present.
    Falls back to no-op behaviour on any initialisation error (e.g. the
    ``langfuse`` package is not installed in the current venv).

    In **live mode** (SDK present and env vars set):

    * :meth:`start_run` — creates a Langfuse trace scoped to one research run.
    * :meth:`record_step` — emits a child span with the agent name, status, and
      attempt count attached as metadata.
    * :meth:`finish_run` — attaches the final outcome string to the trace and
      calls ``langfuse.flush()`` so buffered events are delivered before
      the process exits.

    In **no-op mode** all methods return immediately without side-effects,
    matching the :class:`NullTracer` contract.
    """

    def __init__(self) -> None:
        public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
        secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "")
        host = os.environ.get("LANGFUSE_HOST", "")

        if not (public_key and secret_key):
            _log.debug("langfuse.tracer_noop", reason="env vars not set")
            self._client: Any = None
            return

        try:
            import langfuse as _langfuse_mod  # type: ignore[import-untyped,unused-ignore]

            kwargs: dict[str, str] = {
                "public_key": public_key,
                "secret_key": secret_key,
            }
            if host:
                kwargs["host"] = host

            # SDK v4 exposes get_client() as the preferred factory; fall
            # back to direct Langfuse() construction for SDK v2/v3 compat.
            if hasattr(_langfuse_mod, "get_client"):
                # v4: configure via env vars (already set) and fetch singleton.
                self._client = _langfuse_mod.get_client()
            else:
                # v2/v3: direct construction.
                self._client = _langfuse_mod.Langfuse(**kwargs)

            _log.info(
                "langfuse.tracer_live",
                host=host or "https://cloud.langfuse.com",
            )
        except ImportError:
            _log.warning(
                "langfuse.sdk_missing",
                hint="pip install 'research-crew[real]' to enable Langfuse tracing",
            )
            self._client = None
        except Exception as exc:
            _log.warning(
                "langfuse.init_failed",
                exc_type=type(exc).__name__,
                error=str(exc),
            )
            self._client = None

    @property
    def _live(self) -> bool:
        return self._client is not None

    def start_run(self, query: str) -> RunHandle:
        """Open a Langfuse trace for one research run.

        In no-op mode returns a :class:`RunHandle` whose ``_trace`` is None.
        """
        if not self._live:
            return RunHandle(trace_id="null")
        try:
            # SDK v4 uses start_observation(); v2/v3 uses trace().
            if hasattr(self._client, "trace"):
                trace = self._client.trace(name="research-run", input=query)
                trace_id: str = trace.id
            else:
                trace = self._client.start_observation(name="research-run", as_type="span")
                trace.update(input=query)
                trace_id = getattr(trace, "id", "unknown")
            return RunHandle(trace_id=trace_id, _trace=trace)
        except Exception as exc:
            _log.warning(
                "langfuse.start_run_failed",
                exc_type=type(exc).__name__,
                error=str(exc),
            )
            return RunHandle(trace_id="null")

    def record_step(self, handle: RunHandle, step: StepRecord) -> None:
        """Emit a Langfuse child span for one workflow step.

        Attaches agent name, status, attempt count, and optional error as
        span metadata so the Langfuse UI shows a timeline of agent attempts.
        In no-op mode or when the trace is absent returns immediately.

        RUNNING records are skipped: they have no ``finished_at`` and would
        create half-open spans that never close. Only terminal statuses
        (SUCCEEDED, FAILED, CACHED) produce a span.
        """
        if not self._live or handle._trace is None:
            return
        if step.status is StepStatus.RUNNING:
            return
        try:
            metadata: dict[str, object] = {
                "agent": step.agent.value,
                "status": step.status.value,
                "attempts": step.attempts,
            }
            if step.error:
                metadata["error"] = step.error
            elapsed_ms: float | None = None
            if step.finished_at is not None and step.started_at is not None:
                elapsed_ms = (step.finished_at - step.started_at).total_seconds() * 1000.0

            trace = handle._trace
            if hasattr(trace, "span"):
                # v2/v3 API
                span = trace.span(
                    name=f"agent:{step.agent.value}",
                    metadata=metadata,
                    start_time=step.started_at,
                    end_time=step.finished_at,
                )
                if elapsed_ms is not None:
                    span.update(output={"elapsed_ms": elapsed_ms})
                span.end()
            else:
                # v4 API
                obs = trace.start_observation(
                    name=f"agent:{step.agent.value}",
                    as_type="span",
                )
                obs.update(
                    metadata=metadata,
                    output={"status": step.status.value, "elapsed_ms": elapsed_ms},
                )
                obs.end()
        except Exception as exc:
            _log.warning(
                "langfuse.record_step_failed",
                agent=step.agent.value,
                exc_type=type(exc).__name__,
                error=str(exc),
            )

    def finish_run(self, handle: RunHandle, outcome: str) -> None:
        """Finalise the Langfuse trace and flush buffered events.

        In no-op mode or when the trace is absent returns immediately.
        """
        if not self._live or handle._trace is None:
            return
        try:
            trace = handle._trace
            if hasattr(trace, "update"):
                trace.update(output=outcome)
            if hasattr(trace, "end"):
                trace.end()
            # Flush so short-lived CLI invocations deliver buffered events.
            if hasattr(self._client, "flush"):
                self._client.flush()
        except Exception as exc:
            _log.warning(
                "langfuse.finish_run_failed",
                exc_type=type(exc).__name__,
                error=str(exc),
            )


def make_tracer() -> LangfuseTracer | NullTracer:
    """Factory: return a :class:`LangfuseTracer` when env vars are set, else a :class:`NullTracer`.

    The returned tracer satisfies the same interface in both cases so
    callers need no conditional logic.

    Usage::

        tracer = make_tracer()
        handle = tracer.start_run(query)
        ...
        tracer.finish_run(handle, "succeeded")
    """
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "")
    if not (public_key and secret_key):
        return NullTracer()
    return LangfuseTracer()
