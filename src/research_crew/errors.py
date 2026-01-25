"""Domain-specific error hierarchy.

The workflow runner used to swallow every `Exception` into a generic
"FAILED" string. That made it hard to distinguish "the agent itself
returned a failure" from "we hit our timeout" from "the store is
broken". Each one wants a different operator response, so they each
get their own exception type.

The hierarchy is intentionally shallow — three direct subclasses of
`ResearchCrewError` so callers can either pattern-match on a specific
type or catch the umbrella when they only care that *something* in the
research stack went wrong.
"""

from __future__ import annotations


class ResearchCrewError(Exception):
    """Base for every domain error this package raises."""


class AgentError(ResearchCrewError):
    """An agent attempt failed in some recoverable way (the runner may retry)."""


class AgentTimeoutError(AgentError):
    """An agent did not return within its per-step wall-clock budget."""

    def __init__(self, agent: str, timeout_s: float) -> None:
        super().__init__(f"agent {agent!r} timed out after {timeout_s}s")
        self.agent = agent
        self.timeout_s = timeout_s


class AgentExecutionError(AgentError):
    """An agent raised an exception during `search()`."""

    def __init__(self, agent: str, original: BaseException) -> None:
        super().__init__(f"agent {agent!r} raised {type(original).__name__}: {original}")
        self.agent = agent
        self.original = original


class RetriesExhaustedError(ResearchCrewError):
    """The retry budget was spent and no attempt succeeded."""

    def __init__(self, agent: str, attempts: int, last_error: str | None) -> None:
        super().__init__(f"agent {agent!r} exhausted {attempts} attempts; last error: {last_error}")
        self.agent = agent
        self.attempts = attempts
        self.last_error = last_error


class StoreUnavailableError(ResearchCrewError):
    """The backing RunStore is unreachable (Redis down, network partition, ...)."""


__all__ = [
    "AgentError",
    "AgentExecutionError",
    "AgentTimeoutError",
    "ResearchCrewError",
    "RetriesExhaustedError",
    "StoreUnavailableError",
]
