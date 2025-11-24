"""Merge + dedupe + cluster + cite the per-agent results into a single report.

A `Synthesizer` is intentionally a Protocol so the no-LLM stitch path
(below) can be swapped for an LLM synthesiser without touching the
workflow runner. The stitch path is good enough to demonstrate the
fan-in shape end-to-end and carries every citation through unchanged.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from research_crew.models import (
    AgentName,
    AgentResult,
    Citation,
    ResearchReport,
    StepStatus,
)


class Synthesizer(Protocol):
    async def synthesize(
        self, run_id: str, question: str, agent_results: list[AgentResult]
    ) -> ResearchReport: ...


# ---------------------------------------------------------------------------
# Stitch synthesizer — no LLM, deterministic, suitable for CI + portfolio demo.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StitchSynthesizer:
    """Concatenate per-agent summaries, dedupe citations by URL, render markdown."""

    max_citations_per_agent: int = 5

    async def synthesize(
        self, run_id: str, question: str, agent_results: list[AgentResult]
    ) -> ResearchReport:
        t0 = time.perf_counter()
        succeeded = [r for r in agent_results if r.status is StepStatus.SUCCEEDED]
        failed = [r for r in agent_results if r.status is StepStatus.FAILED]
        cached = [r for r in agent_results if r.status is StepStatus.CACHED]

        sections = [f"# Research report — {question}", ""]
        sections.append("## Summary")
        sections.append("")
        if not succeeded and not cached:
            sections.append(
                "Every agent failed for this question — see the per-step audit "
                "below for what was tried."
            )
        else:
            sections.append(
                f"Synthesised {len(succeeded)} live + {len(cached)} cached agent "
                f"responses into a citation-grounded summary. {len(failed)} agents "
                "failed and were excluded."
            )
        sections.append("")

        for r in succeeded + cached:
            sections.append(f"## {r.agent.value.title()} ({r.status.value})")
            sections.append(r.summary)
            for c in r.citations[: self.max_citations_per_agent]:
                sections.append(f"- [{c.title}]({c.url}) — {c.snippet}")
            sections.append("")

        if failed:
            sections.append("## Failures")
            for r in failed:
                sections.append(f"- **{r.agent.value}**: {r.error}")

        merged_citations = list(_dedupe_citations(_iter_all_citations(agent_results)))
        elapsed = (time.perf_counter() - t0) * 1000.0
        return ResearchReport(
            run_id=run_id,
            question=question,
            summary="\n".join(sections),
            citations=merged_citations,
            agent_results=agent_results,
            elapsed_ms=elapsed,
        )


def _iter_all_citations(results: list[AgentResult]) -> Iterable[Citation]:
    for r in results:
        if r.status in (StepStatus.SUCCEEDED, StepStatus.CACHED):
            yield from r.citations


def _dedupe_citations(citations: Iterable[Citation]) -> Iterable[Citation]:
    """Stable-order dedupe by URL — the first occurrence wins."""
    seen: set[str] = set()
    for c in citations:
        if c.url in seen:
            continue
        seen.add(c.url)
        yield c


def per_agent_citation_count(results: list[AgentResult]) -> dict[AgentName, int]:
    """Used by the load test report to show fan-in coverage per source."""
    out: dict[AgentName, int] = {n: 0 for n in AgentName}
    for r in results:
        if r.status in (StepStatus.SUCCEEDED, StepStatus.CACHED):
            out[r.agent] = len(r.citations)
    return out
