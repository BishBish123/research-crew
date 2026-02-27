"""Edge cases for the stitch synthesizer beyond the basic dedupe path."""

from __future__ import annotations

import pytest

from research_crew.models import AgentName, AgentResult, Citation, StepStatus
from research_crew.synthesizer import StitchSynthesizer, _normalize_url


class TestEmptyAndDegenerateInputs:
    async def test_empty_input_returns_empty_report(self) -> None:
        report = await StitchSynthesizer().synthesize("r", "q", [])
        assert report.citations == []
        assert report.agent_results == []
        # Even with no agents we still emit a header so the UI never explodes.
        assert "Research report" in report.summary

    async def test_only_cached_results_render(self) -> None:
        result = AgentResult(
            agent=AgentName.WEB_SEARCH,
            status=StepStatus.CACHED,
            summary="cached summary",
            citations=[Citation(title="t", url="https://x/1")],
        )
        report = await StitchSynthesizer().synthesize("r", "q", [result])
        assert "cached" in report.summary.lower()
        assert len(report.citations) == 1


class TestNearIdenticalUrlDedupe:
    @pytest.mark.parametrize(
        ("a", "b"),
        [
            ("https://example.com/foo", "https://example.com/foo/"),
            ("https://example.com/foo", "https://EXAMPLE.com/foo"),
            ("https://example.com/foo", "HTTPS://example.com/foo"),
            ("https://example.com/foo", "https://www.example.com/foo"),
            ("https://example.com/foo", "https://www.example.com/foo/"),
        ],
    )
    async def test_collapses_near_duplicates(self, a: str, b: str) -> None:
        results = [
            AgentResult(
                agent=AgentName.WEB_SEARCH,
                status=StepStatus.SUCCEEDED,
                summary="s",
                citations=[Citation(title="A", url=a)],
            ),
            AgentResult(
                agent=AgentName.SCHOLAR,
                status=StepStatus.SUCCEEDED,
                summary="s",
                citations=[Citation(title="B", url=b)],
            ),
        ]
        report = await StitchSynthesizer().synthesize("r", "q", results)
        # Only the first occurrence survives.
        assert len(report.citations) == 1
        assert report.citations[0].title == "A"

    async def test_keeps_distinct_paths(self) -> None:
        results = [
            AgentResult(
                agent=AgentName.WEB_SEARCH,
                status=StepStatus.SUCCEEDED,
                summary="s",
                citations=[
                    Citation(title="a", url="https://x.com/a"),
                    Citation(title="b", url="https://x.com/b"),
                ],
            ),
        ]
        report = await StitchSynthesizer().synthesize("r", "q", results)
        assert {c.url for c in report.citations} == {"https://x.com/a", "https://x.com/b"}

    async def test_query_string_not_collapsed(self) -> None:
        """`?q=1` and `?q=2` are different pages — must not collapse."""
        results = [
            AgentResult(
                agent=AgentName.WEB_SEARCH,
                status=StepStatus.SUCCEEDED,
                summary="s",
                citations=[
                    Citation(title="a", url="https://x.com/p?q=1"),
                    Citation(title="b", url="https://x.com/p?q=2"),
                ],
            ),
        ]
        report = await StitchSynthesizer().synthesize("r", "q", results)
        assert len(report.citations) == 2


class TestNormalizeUrlUnit:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("https://example.com/foo", "https://example.com/foo"),
            ("HTTPS://Example.COM/foo", "https://example.com/foo"),
            ("https://www.example.com/foo", "https://example.com/foo"),
            ("https://example.com/foo/", "https://example.com/foo"),
            ("  https://example.com/foo  ", "https://example.com/foo"),
            ("https://example.com/", "https://example.com/"),
            ("https://example.com/p?x=1", "https://example.com/p?x=1"),
        ],
    )
    def test_normalization_table(self, raw: str, expected: str) -> None:
        assert _normalize_url(raw) == expected
