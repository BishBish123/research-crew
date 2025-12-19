"""Synthesizer tests — dedupe, ordering, failure surfacing."""

from __future__ import annotations

from research_crew.models import AgentName, AgentResult, Citation, StepStatus
from research_crew.synthesizer import StitchSynthesizer, per_agent_citation_count


def _result(
    agent: AgentName, citations: list[Citation], status: StepStatus = StepStatus.SUCCEEDED
) -> AgentResult:
    return AgentResult(
        agent=agent,
        status=status,
        summary=f"{agent.value} summary",
        citations=citations,
    )


class TestStitchSynthesizer:
    async def test_dedupes_citations_by_url(self) -> None:
        c1 = Citation(title="a", url="https://x/a")
        c2 = Citation(title="dup", url="https://x/a")  # same url, different title
        c3 = Citation(title="b", url="https://x/b")
        results = [
            _result(AgentName.WEB_SEARCH, [c1, c3]),
            _result(AgentName.SCHOLAR, [c2]),
        ]
        report = await StitchSynthesizer().synthesize("r", "q", results)
        urls = [c.url for c in report.citations]
        assert urls == ["https://x/a", "https://x/b"]

    async def test_failures_surfaced_in_summary(self) -> None:
        results = [
            _result(AgentName.WEB_SEARCH, [Citation(title="t", url="https://x/1")]),
            AgentResult(
                agent=AgentName.NEWS,
                status=StepStatus.FAILED,
                summary="",
                error="rate limited",
            ),
        ]
        report = await StitchSynthesizer().synthesize("r", "q", results)
        assert "Failures" in report.summary
        assert "rate limited" in report.summary

    async def test_all_failed_returns_empty_summary(self) -> None:
        results = [
            AgentResult(agent=n, status=StepStatus.FAILED, summary="", error="boom")
            for n in AgentName
        ]
        report = await StitchSynthesizer().synthesize("r", "q", results)
        assert "Every agent failed" in report.summary
        assert report.citations == []

    async def test_caps_citations_per_agent(self) -> None:
        many = [Citation(title=str(i), url=f"https://x/{i}") for i in range(10)]
        results = [_result(AgentName.WEB_SEARCH, many)]
        report = await StitchSynthesizer(max_citations_per_agent=3).synthesize("r", "q", results)
        # All 10 citations still in the deduped list (we don't filter the
        # global merged set), but only 3 appear in the rendered summary.
        assert len(report.citations) == 10
        assert report.summary.count("https://x/") == 3


class TestPerAgentCitationCount:
    def test_counts_succeeded_and_cached(self) -> None:
        results = [
            _result(AgentName.WEB_SEARCH, [Citation(title="a", url="u1")] * 2),
            _result(AgentName.NEWS, [Citation(title="a", url="u2")], StepStatus.CACHED),
            AgentResult(agent=AgentName.SCHOLAR, status=StepStatus.FAILED, summary=""),
        ]
        counts = per_agent_citation_count(results)
        assert counts[AgentName.WEB_SEARCH] == 2
        assert counts[AgentName.NEWS] == 1
        assert counts[AgentName.SCHOLAR] == 0
