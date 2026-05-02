"""Pydantic models shared by every layer.

Every request/response model in this module is configured with
``extra="forbid"`` and ``str_strip_whitespace=True``: unknown fields
fail validation loudly instead of being silently ignored, and string
fields are stripped before length / blank-ness checks run. The
``ResearchRequest`` validators also enforce a configurable upper bound
on the question length (``RESEARCH_MAX_QUESTION_LEN`` env var) and a
hard cap on the size of the ``agents`` list.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AgentName(StrEnum):
    WEB_SEARCH = "web_search"
    SCHOLAR = "scholar"
    CODE = "code"
    NEWS = "news"
    WIKIPEDIA = "wikipedia"


class StepStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CACHED = "cached"


# Default upper bound on the inbound question length. Operators may
# raise this for niche workloads via `RESEARCH_MAX_QUESTION_LEN`, but
# the default keeps a single misbehaving client from exhausting Redis
# value budgets or overwhelming the synthesizer.
_DEFAULT_MAX_QUESTION_LEN = 5000

# Max number of agents a single request may pin. Default fan-out is 5
# specialists; 20 is comfortably above that for future expansion while
# still bounding the worst-case fan-out from an arbitrary client.
_MAX_AGENTS_PER_REQUEST = 20


def _max_question_len() -> int:
    raw = os.environ.get("RESEARCH_MAX_QUESTION_LEN")
    if not raw:
        return _DEFAULT_MAX_QUESTION_LEN
    try:
        parsed = int(raw)
    except ValueError:
        return _DEFAULT_MAX_QUESTION_LEN
    return parsed if parsed > 0 else _DEFAULT_MAX_QUESTION_LEN


# Shared config: applied to every model in this module so the contract
# is uniform — extra fields are a 422, surrounding whitespace is
# stripped before validation, and assignment also revalidates.
_STRICT_MODEL_CONFIG = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Citation(BaseModel):
    """One source the synthesizer references in the final report."""

    model_config = _STRICT_MODEL_CONFIG

    title: str
    url: str
    snippet: str = ""


class AgentResult(BaseModel):
    """Output of one specialist agent."""

    model_config = _STRICT_MODEL_CONFIG

    agent: AgentName
    status: StepStatus
    summary: str
    citations: list[Citation] = Field(default_factory=list)
    elapsed_ms: float = 0.0
    attempts: int = 1
    error: str | None = None


class StepRecord(BaseModel):
    """Persisted record of one workflow step — what was attempted, when, how it ended."""

    model_config = _STRICT_MODEL_CONFIG

    run_id: str
    agent: AgentName
    status: StepStatus
    attempts: int
    started_at: datetime
    finished_at: datetime | None = None
    error: str | None = None


class ResearchReport(BaseModel):
    """The synthesizer's final output."""

    model_config = _STRICT_MODEL_CONFIG

    run_id: str
    question: str
    summary: str
    citations: list[Citation]
    agent_results: list[AgentResult]
    elapsed_ms: float


class ResearchRequest(BaseModel):
    model_config = _STRICT_MODEL_CONFIG

    # `min_length` here only catches the "literally empty after strip"
    # case alongside the validator below; `max_length` is set to the
    # env-configurable ceiling so the per-request 422 fires before the
    # body even hits the validator chain.
    question: str = Field(min_length=1)
    agents: list[AgentName] | None = None  # default: every agent

    @field_validator("question")
    @classmethod
    def _validate_question(cls, value: str) -> str:
        # `str_strip_whitespace` already trimmed; this catches the
        # all-whitespace input that strips to empty plus enforces the
        # configurable upper bound. Two errors with distinct messages
        # so the client can tell the cases apart.
        stripped = value.strip()
        if not stripped:
            raise ValueError("question must be non-empty after stripping whitespace")
        cap = _max_question_len()
        if len(stripped) > cap:
            raise ValueError(f"question exceeds max length of {cap} characters")
        return stripped

    @field_validator("agents")
    @classmethod
    def _validate_agents(cls, value: list[AgentName] | None) -> list[AgentName] | None:
        # `agents=[]` used to silently fall back to the default fan-out,
        # which masked client bugs. Reject it explicitly so callers either
        # name the agents they want or omit the field.
        if value is None:
            return None
        if len(value) == 0:
            raise ValueError("agents must be non-empty if provided; omit field for default fan-out")
        if len(value) > _MAX_AGENTS_PER_REQUEST:
            raise ValueError(
                f"agents list exceeds max of {_MAX_AGENTS_PER_REQUEST} entries"
            )
        return value


class RunStatus(BaseModel):
    """Runtime status snapshot for `GET /runs/{run_id}`."""

    model_config = _STRICT_MODEL_CONFIG

    run_id: str
    question: str
    state: StepStatus
    steps: list[StepRecord] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    report: ResearchReport | None = None
    # End-to-end wall-clock latency in milliseconds, populated only when
    # the run reaches a terminal state. Captured by the bg task using a
    # monotonic clock around the workflow + synthesis phase, so it's
    # robust against wall-clock skew between submit and finish.
    total_latency_ms: float | None = None
