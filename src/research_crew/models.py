"""Pydantic models shared by every layer."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


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


class Citation(BaseModel):
    """One source the synthesizer references in the final report."""

    title: str
    url: str
    snippet: str = ""


class AgentResult(BaseModel):
    """Output of one specialist agent."""

    agent: AgentName
    status: StepStatus
    summary: str
    citations: list[Citation] = Field(default_factory=list)
    elapsed_ms: float = 0.0
    attempts: int = 1
    error: str | None = None


class StepRecord(BaseModel):
    """Persisted record of one workflow step — what was attempted, when, how it ended."""

    run_id: str
    agent: AgentName
    status: StepStatus
    attempts: int
    started_at: datetime
    finished_at: datetime | None = None
    error: str | None = None


class ResearchReport(BaseModel):
    """The synthesizer's final output."""

    run_id: str
    question: str
    summary: str
    citations: list[Citation]
    agent_results: list[AgentResult]
    elapsed_ms: float


class ResearchRequest(BaseModel):
    question: str = Field(min_length=4, max_length=512)
    agents: list[AgentName] | None = None  # default: every agent

    @field_validator("agents")
    @classmethod
    def _reject_empty_agents(cls, value: list[AgentName] | None) -> list[AgentName] | None:
        # `agents=[]` used to silently fall back to the default fan-out,
        # which masked client bugs. Reject it explicitly so callers either
        # name the agents they want or omit the field.
        if value is not None and len(value) == 0:
            raise ValueError("agents must be non-empty if provided; omit field for default fan-out")
        return value


class RunStatus(BaseModel):
    """Runtime status snapshot for `GET /runs/{run_id}`."""

    run_id: str
    question: str
    state: StepStatus
    steps: list[StepRecord] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    report: ResearchReport | None = None
