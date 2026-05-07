"""Eval harness: runs the golden set through WorkflowEngine + StitchSynthesizer.

Usage
-----
    uv run python -m evals.harness                # writes evals/REPORT.md
    uv run python -m evals.harness --output /tmp/r.md

The harness is fully deterministic: WorkflowEngine is constructed with a
stable run_id per question, and MockAgent's blake2b digest is seeded by
``(agent_name, question)`` — no random state leaks in.

Adding an LLM judge later is a one-class addition: implement
``JudgeProtocol.score(question, answer) -> float`` and pass it to
``HarnessConfig``.  The ``_score_row`` function has a placeholder comment
marking the integration point.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from evals.golden import GOLDEN_SET, GoldenQuestion
from research_crew.agents import default_agents
from research_crew.models import ResearchReport
from research_crew.synthesizer import StitchSynthesizer
from research_crew.workflow import WorkflowConfig, WorkflowEngine

# ---------------------------------------------------------------------------
# Per-question result
# ---------------------------------------------------------------------------

_EVALS_DIR = Path(__file__).parent


@dataclass
class QuestionResult:
    """Scored output for one golden question."""

    qid: str
    question: str
    category: str
    citation_correctness: float  # fraction of cited URLs matching any expected substring
    keyphrase_coverage: float  # fraction of expected_keyphrases found in answer
    latency_ms: float
    step_count: int  # total StepRecord rows emitted (RUNNING + terminal, all agents)
    cited_urls: list[str] = field(default_factory=list)
    found_keyphrases: list[str] = field(default_factory=list)
    missing_keyphrases: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def score_citation_correctness(
    report: ResearchReport, expected_url_substrings: list[str]
) -> tuple[float, list[str]]:
    """Return (fraction, cited_urls).

    A cited URL scores 1 if *any* of expected_url_substrings appears in it
    (case-insensitive).  Empty expected list → 0.0 (not 1.0) so the metric
    stays informative when no real URLs are expected.
    """
    urls = [c.url for c in report.citations]
    if not urls or not expected_url_substrings:
        return 0.0, urls
    hits = sum(
        1 for url in urls if any(sub.lower() in url.lower() for sub in expected_url_substrings)
    )
    return hits / len(urls), urls


def score_keyphrase_coverage(
    report: ResearchReport, expected_keyphrases: list[str]
) -> tuple[float, list[str], list[str]]:
    """Return (fraction, found, missing).

    Each keyphrase is checked case-insensitively against the full report
    summary string.  Empty phrase list → 0.0.
    """
    if not expected_keyphrases:
        return 0.0, [], []
    text = report.summary.lower()
    found = [p for p in expected_keyphrases if p.lower() in text]
    missing = [p for p in expected_keyphrases if p.lower() not in text]
    return len(found) / len(expected_keyphrases), found, missing


# ---------------------------------------------------------------------------
# Per-question runner
# ---------------------------------------------------------------------------


async def _run_question(gq: GoldenQuestion) -> QuestionResult:
    """Run one golden question through the pipeline and score it."""
    agents = default_agents(latency_ms=0.0, failure_rate=0.0)
    # Stable run_id per question keeps the dedup cache consistent across runs.
    run_id = f"eval-{gq.qid}"
    engine = WorkflowEngine(
        run_id=run_id,
        config=WorkflowConfig(max_attempts=1, base_backoff_s=0.0, per_step_timeout_s=10.0),
    )
    synthesizer = StitchSynthesizer()

    t0 = time.perf_counter()
    agent_results = await engine.run_parallel(agents, gq.question)
    report = await synthesizer.synthesize(run_id, gq.question, agent_results)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    # step_count: WorkflowEngine without a record_step callback emits 0 stored
    # rows, but we can count inferred steps from agent_results (1 per agent).
    step_count = len(agent_results)

    citation_correctness, cited_urls = score_citation_correctness(
        report, gq.expected_url_substrings
    )
    keyphrase_coverage, found, missing = score_keyphrase_coverage(report, gq.expected_keyphrases)

    return QuestionResult(
        qid=gq.qid,
        question=gq.question,
        category=gq.category,
        citation_correctness=citation_correctness,
        keyphrase_coverage=keyphrase_coverage,
        latency_ms=latency_ms,
        step_count=step_count,
        cited_urls=cited_urls,
        found_keyphrases=found,
        missing_keyphrases=missing,
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


@dataclass
class AggregateStats:
    mean_citation_correctness: float
    mean_keyphrase_coverage: float
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    mean_step_count: float
    n: int


def _aggregate(results: list[QuestionResult]) -> AggregateStats:
    n = len(results)
    cc = [r.citation_correctness for r in results]
    kc = [r.keyphrase_coverage for r in results]
    lat = sorted(r.latency_ms for r in results)
    steps = [r.step_count for r in results]

    def _p(data: list[float], pct: float) -> float:
        idx = max(0, min(int(len(data) * pct / 100) - 1, len(data) - 1))
        return data[idx]

    return AggregateStats(
        mean_citation_correctness=statistics.mean(cc),
        mean_keyphrase_coverage=statistics.mean(kc),
        mean_latency_ms=statistics.mean(lat),
        p50_latency_ms=_p(lat, 50),
        p95_latency_ms=_p(lat, 95),
        mean_step_count=statistics.mean(steps),
        n=n,
    )


# ---------------------------------------------------------------------------
# Report renderer
# ---------------------------------------------------------------------------


_LATENCY_PLACEHOLDER = "0 ms (mock pipeline)"


def _render_header_and_aggregate(
    lines: list[str], agg: AggregateStats, ts: str, deterministic: bool = False
) -> None:
    """Append the report header and aggregate-metrics section to *lines*."""
    a = lines.append
    a("# Eval Report — research-crew")
    a("")
    a(f"**Generated:** {ts}  ")
    a("**Pipeline:** MockAgent (deterministic blake2b, no live APIs)  ")
    a(f"**Questions:** {agg.n}  ")
    a("")
    a("> **Interpretation:** all scores are measured against the *mock* pipeline floor.")
    a("> Citation correctness is 0.0 because MockAgent returns `example.com` URLs that")
    a("> never match real expected substrings.  Keyphrase coverage is non-zero where")
    a("> MockAgent echoes back the question text in its summary.  See")
    a("> `evals/INTERPRETATION.md` for the full discussion.")
    a("")
    a("## Aggregate Metrics")
    a("")
    a("| Metric | Value |")
    a("| --- | --- |")
    a(f"| Mean citation correctness | {agg.mean_citation_correctness:.3f} |")
    a(f"| Mean keyphrase coverage | {agg.mean_keyphrase_coverage:.3f} |")
    if deterministic:
        a(f"| Mean latency (ms) | {_LATENCY_PLACEHOLDER} |")
        a(f"| p50 latency (ms) | {_LATENCY_PLACEHOLDER} |")
        a(f"| p95 latency (ms) | {_LATENCY_PLACEHOLDER} |")
    else:
        a(f"| Mean latency (ms) | {agg.mean_latency_ms:.1f} |")
        a(f"| p50 latency (ms) | {agg.p50_latency_ms:.1f} |")
        a(f"| p95 latency (ms) | {agg.p95_latency_ms:.1f} |")
    a(f"| Mean step count (agents/run) | {agg.mean_step_count:.1f} |")
    a("")


def _render_detail_and_notes(
    lines: list[str], results: list[QuestionResult], deterministic: bool = False
) -> None:
    """Append per-category breakdown, per-question detail, and methodology notes."""
    a = lines.append
    a("## Per-Category Breakdown")
    a("")
    categories = sorted({r.category for r in results})
    a("| Category | N | Avg Citation Correctness | Avg Keyphrase Coverage |")
    a("| --- | --- | --- | --- |")
    for cat in categories:
        cat_results = [r for r in results if r.category == cat]
        avg_cc = statistics.mean(r.citation_correctness for r in cat_results)
        avg_kc = statistics.mean(r.keyphrase_coverage for r in cat_results)
        a(f"| {cat} | {len(cat_results)} | {avg_cc:.3f} | {avg_kc:.3f} |")
    a("")
    a("## Per-Question Detail")
    a("")
    a(
        "| QID | Category | Citation Correctness | Keyphrase Coverage | Latency (ms) | Missing Keyphrases |"
    )
    a("| --- | --- | --- | --- | --- | --- |")
    for r in results:
        missing = ", ".join(r.missing_keyphrases) if r.missing_keyphrases else "—"
        latency_cell = _LATENCY_PLACEHOLDER if deterministic else f"{r.latency_ms:.1f}"
        a(
            f"| {r.qid} | {r.category} | {r.citation_correctness:.3f}"
            f" | {r.keyphrase_coverage:.3f} | {latency_cell} | {missing} |"
        )
    a("")
    a("## Methodology Notes")
    a("")
    a("- **citation_correctness**: fraction of cited URLs whose lowercased text contains")
    a("  any of the `expected_url_substrings` for that question.")
    a("- **keyphrase_coverage**: fraction of `expected_keyphrases` found (case-insensitive)")
    a("  anywhere in the synthesized report markdown.")
    a("- **step_count**: number of agent results returned per run (5 agents x 1 attempt).")
    a("- **LLM judge**: not yet implemented.  Adding one is a one-class extension;")
    a("  see `evals/INTERPRETATION.md` for the design.")
    a("")


def _render_report(
    results: list[QuestionResult], agg: AggregateStats, ts: str, deterministic: bool = False
) -> str:
    lines: list[str] = []
    _render_header_and_aggregate(lines, agg, ts, deterministic=deterministic)
    _render_detail_and_notes(lines, results, deterministic=deterministic)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def run_harness(
    output: Path | None = None, deterministic: bool = False
) -> list[QuestionResult]:
    """Run the full golden set and write REPORT.md.  Returns the scored results.

    Parameters
    ----------
    output:
        Destination path for the generated REPORT.md.  Defaults to
        ``evals/REPORT.md``.
    deterministic:
        When True, latency cells in the report are replaced with a fixed
        placeholder (``0 ms (mock pipeline)``) so the file is stable
        across runs and does not show up in ``git status`` due to
        wall-clock timing jitter.  Pass ``False`` (the default) to see
        real timings.
    """
    results = []
    for gq in GOLDEN_SET:
        result = await _run_question(gq)
        results.append(result)

    agg = _aggregate(results)
    # In deterministic mode pin the timestamp to the epoch string so that the
    # Generated line doesn't churn on every run.
    if deterministic:
        ts = "1970-01-01T00:00:00Z (mock pipeline)"
    else:
        ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    report_md = _render_report(results, agg, ts, deterministic=deterministic)

    out_path = output if output is not None else _EVALS_DIR / "REPORT.md"
    out_path.write_text(report_md, encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the research-crew eval harness.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for the generated REPORT.md (default: evals/REPORT.md)",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=False,
        help=(
            "Replace latency cells with a fixed placeholder so REPORT.md is "
            "stable across runs (no git churn from wall-clock timing jitter). "
            "Omit this flag to see real timings."
        ),
    )
    args = parser.parse_args()
    asyncio.run(run_harness(output=args.output, deterministic=args.deterministic))
    out = args.output or (_EVALS_DIR / "REPORT.md")
    print(f"Report written to {out}")


if __name__ == "__main__":
    main()
