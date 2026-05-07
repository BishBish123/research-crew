"""Tests for the evals harness.

Two required tests:
1. Smoke test — harness produces a non-empty REPORT.md on the full golden set.
2. Scorer unit test — citation_correctness is correct on a hand-built fixture
   (positive case: matching URL; negative case: no matching URL).

Additional tests cover keyphrase_coverage and golden-set integrity.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from evals.golden import GOLDEN_SET
from evals.harness import (
    QuestionResult,
    run_harness,
    score_citation_correctness,
    score_keyphrase_coverage,
)

from research_crew.models import Citation, ResearchReport

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_report(urls: list[str], summary: str = "test summary") -> ResearchReport:
    """Build a minimal ResearchReport with the given citation URLs."""
    citations = [Citation(title=f"title-{i}", url=u, snippet="s") for i, u in enumerate(urls)]
    return ResearchReport(
        run_id="test",
        question="q",
        summary=summary,
        citations=citations,
        agent_results=[],
        elapsed_ms=0.0,
    )


# ---------------------------------------------------------------------------
# 1. Smoke test: full harness run produces a non-empty REPORT.md
# ---------------------------------------------------------------------------


async def test_harness_produces_nonempty_report() -> None:
    """Running the harness on the full golden set writes a non-empty REPORT.md."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "REPORT.md"
        results = await run_harness(output=out)
        assert out.exists(), "REPORT.md was not created"
        content = out.read_text(encoding="utf-8")
        assert len(content) > 200, "REPORT.md is suspiciously short"
        assert "# Eval Report" in content
        assert "Aggregate Metrics" in content
        assert len(results) == len(GOLDEN_SET)


# ---------------------------------------------------------------------------
# 2. Citation-correctness scorer: positive + negative fixture
# ---------------------------------------------------------------------------


def test_citation_correctness_positive_case() -> None:
    """A URL containing an expected substring scores 1.0."""
    report = _make_report(["https://python.org/docs/3.11/"])
    score, urls = score_citation_correctness(report, ["python.org"])
    assert score == 1.0
    assert urls == ["https://python.org/docs/3.11/"]


def test_citation_correctness_negative_case() -> None:
    """A URL that matches no expected substring scores 0.0."""
    report = _make_report(["https://example.com/web_search/abc/0"])
    score, _urls = score_citation_correctness(report, ["python.org", "docs.python"])
    assert score == 0.0


def test_citation_correctness_mixed_case() -> None:
    """Only the matching URL counts; the fraction is correct."""
    report = _make_report(
        [
            "https://python.org/docs/",  # matches
            "https://example.com/other/",  # does not match
        ]
    )
    score, _ = score_citation_correctness(report, ["python.org"])
    assert score == pytest.approx(0.5)


def test_citation_correctness_empty_expected_returns_zero() -> None:
    """Empty expected_url_substrings always returns 0.0."""
    report = _make_report(["https://python.org/"])
    score, _ = score_citation_correctness(report, [])
    assert score == 0.0


def test_citation_correctness_case_insensitive() -> None:
    """Substring match is case-insensitive."""
    report = _make_report(["https://PYTHON.ORG/docs/"])
    score, _ = score_citation_correctness(report, ["python.org"])
    assert score == 1.0


# ---------------------------------------------------------------------------
# Keyphrase coverage scorer
# ---------------------------------------------------------------------------


def test_keyphrase_coverage_found() -> None:
    """Phrases present in the report score 1.0."""
    report = _make_report([], summary="Python uses the Global Interpreter Lock for thread safety.")
    score, found, missing = score_keyphrase_coverage(report, ["global interpreter lock", "python"])
    assert score == 1.0
    assert "global interpreter lock" in found
    assert missing == []


def test_keyphrase_coverage_missing() -> None:
    """Phrases absent from the report reduce the score."""
    report = _make_report([], summary="asyncio is great.")
    score, found, missing = score_keyphrase_coverage(report, ["asyncio", "threading"])
    assert score == pytest.approx(0.5)
    assert "asyncio" in found
    assert "threading" in missing


def test_keyphrase_coverage_empty_phrases_returns_zero() -> None:
    """Empty phrase list always returns 0.0."""
    report = _make_report([], summary="anything")
    score, _, _ = score_keyphrase_coverage(report, [])
    assert score == 0.0


# ---------------------------------------------------------------------------
# Golden set integrity
# ---------------------------------------------------------------------------


def test_golden_set_has_all_three_categories() -> None:
    """The golden set must cover factual, comparative, and list categories."""
    categories = {gq.category for gq in GOLDEN_SET}
    assert "factual" in categories
    assert "comparative" in categories
    assert "list" in categories


def test_golden_set_qids_are_unique() -> None:
    """Every qid in the golden set must be distinct."""
    qids = [gq.qid for gq in GOLDEN_SET]
    assert len(qids) == len(set(qids))


def test_golden_set_minimum_size() -> None:
    """The golden set must have at least 20 questions (brief mandate)."""
    assert len(GOLDEN_SET) >= 20


def test_golden_question_fields_populated() -> None:
    """Every golden question must have non-empty question, url substrings, and keyphrases."""
    for gq in GOLDEN_SET:
        assert gq.question.strip(), f"{gq.qid}: question is empty"
        assert gq.expected_url_substrings, f"{gq.qid}: expected_url_substrings is empty"
        assert gq.expected_keyphrases, f"{gq.qid}: expected_keyphrases is empty"


# ---------------------------------------------------------------------------
# QuestionResult dataclass
# ---------------------------------------------------------------------------


def test_question_result_instantiation() -> None:
    """QuestionResult can be constructed with default list fields."""
    r = QuestionResult(
        qid="test-001",
        question="what is X",
        category="factual",
        citation_correctness=0.5,
        keyphrase_coverage=1.0,
        latency_ms=10.0,
        step_count=5,
    )
    assert r.cited_urls == []
    assert r.found_keyphrases == []
    assert r.missing_keyphrases == []
