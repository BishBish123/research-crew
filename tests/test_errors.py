"""Tests for the domain-error hierarchy in `research_crew.errors`."""

from __future__ import annotations

import pytest

from research_crew.errors import (
    AgentError,
    AgentExecutionError,
    AgentTimeoutError,
    ResearchCrewError,
    RetriesExhaustedError,
    StoreUnavailableError,
)


class TestHierarchy:
    def test_agent_errors_are_research_crew_errors(self) -> None:
        assert issubclass(AgentError, ResearchCrewError)
        assert issubclass(AgentTimeoutError, AgentError)
        assert issubclass(AgentExecutionError, AgentError)

    def test_orthogonal_errors_are_not_agent_errors(self) -> None:
        assert not issubclass(RetriesExhaustedError, AgentError)
        assert not issubclass(StoreUnavailableError, AgentError)

    def test_all_inherit_from_exception(self) -> None:
        for cls in (
            ResearchCrewError,
            AgentError,
            AgentTimeoutError,
            AgentExecutionError,
            RetriesExhaustedError,
            StoreUnavailableError,
        ):
            assert issubclass(cls, Exception)


class TestMessageFormatting:
    def test_timeout_message_includes_agent_and_budget(self) -> None:
        err = AgentTimeoutError("scholar", 5.0)
        msg = str(err)
        assert "scholar" in msg
        assert "5" in msg
        assert err.agent == "scholar"
        assert err.timeout_s == 5.0

    def test_execution_message_wraps_original(self) -> None:
        original = ValueError("nope")
        err = AgentExecutionError("news", original)
        assert "news" in str(err)
        assert "ValueError" in str(err)
        assert err.original is original

    def test_retries_exhausted_includes_attempt_count(self) -> None:
        err = RetriesExhaustedError("web_search", 3, "boom")
        msg = str(err)
        assert "web_search" in msg
        assert "3" in msg
        assert "boom" in msg
        assert err.attempts == 3
        assert err.last_error == "boom"


class TestRaisingAndCatching:
    def test_can_catch_specific_then_general(self) -> None:
        with pytest.raises(AgentError):
            raise AgentTimeoutError("x", 1.0)

        with pytest.raises(ResearchCrewError):
            raise StoreUnavailableError("redis down")
