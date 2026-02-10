"""Tests for the `Agent` Protocol surface and the `MockAgent` deterministic stub.

The MockAgent is what every test relies on, so its determinism + failure
injection deserve direct coverage rather than implicit coverage via the
workflow tests.
"""

from __future__ import annotations

import time

from research_crew.agents import Agent, MockAgent, default_agents
from research_crew.models import AgentName, StepStatus


class TestProtocolConformance:
    def test_mock_agent_is_an_agent(self) -> None:
        agent = MockAgent(name=AgentName.WEB_SEARCH)
        assert isinstance(agent, Agent)

    def test_default_agents_yields_one_per_enum(self) -> None:
        agents = default_agents()
        assert {a.name for a in agents} == set(AgentName)
        assert len(agents) == len(AgentName)


class TestDeterminism:
    async def test_same_question_same_citations(self) -> None:
        a1 = MockAgent(name=AgentName.WEB_SEARCH, latency_ms=0)
        a2 = MockAgent(name=AgentName.WEB_SEARCH, latency_ms=0)
        r1 = await a1.search("identical question")
        r2 = await a2.search("identical question")
        assert [c.url for c in r1.citations] == [c.url for c in r2.citations]
        assert r1.summary == r2.summary

    async def test_different_questions_yield_different_urls(self) -> None:
        a = MockAgent(name=AgentName.WEB_SEARCH, latency_ms=0)
        r1 = await a.search("question one")
        r2 = await a.search("question two")
        assert {c.url for c in r1.citations} != {c.url for c in r2.citations}


class TestFailureInjection:
    async def test_full_failure_rate_always_fails(self) -> None:
        agent = MockAgent(name=AgentName.NEWS, latency_ms=0, failure_rate=1.0)
        result = await agent.search("anything")
        assert result.status is StepStatus.FAILED
        assert result.error and "simulated" in result.error

    async def test_zero_failure_rate_never_fails(self) -> None:
        agent = MockAgent(name=AgentName.SCHOLAR, latency_ms=0, failure_rate=0.0)
        for _ in range(10):
            result = await agent.search("question")
            assert result.status is StepStatus.SUCCEEDED


class TestLatencySimulation:
    async def test_latency_is_observable(self) -> None:
        agent = MockAgent(name=AgentName.WIKIPEDIA, latency_ms=20)
        t0 = time.perf_counter()
        await agent.search("q")
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        # Wide tolerance for CI noise; we just want to see latency_ms ≠ 0 actually sleeps.
        assert elapsed_ms >= 15

    async def test_zero_latency_is_immediate(self) -> None:
        agent = MockAgent(name=AgentName.WIKIPEDIA, latency_ms=0)
        t0 = time.perf_counter()
        await agent.search("q")
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        assert elapsed_ms < 50  # generous; the point is "no artificial sleep"


class TestResultShape:
    async def test_succeeded_result_carries_summary_and_citations(self) -> None:
        agent = MockAgent(name=AgentName.WEB_SEARCH, latency_ms=0)
        result = await agent.search("ok")
        assert result.status is StepStatus.SUCCEEDED
        assert result.summary
        assert result.citations
        for c in result.citations:
            assert c.url.startswith("https://")
            assert c.title
