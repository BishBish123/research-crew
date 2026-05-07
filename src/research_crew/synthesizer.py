"""Merge + dedupe + cluster + cite the per-agent results into a single report.

A `Synthesizer` is intentionally a Protocol so the no-LLM stitch path
(below) can be swapped for an LLM synthesiser without touching the
workflow runner. The stitch path is good enough to demonstrate the
fan-in shape end-to-end and carries every citation through unchanged.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol
from urllib.parse import urlsplit, urlunsplit

import structlog

from research_crew.dedup import make_semantic_dedup
from research_crew.dedup.protocol import SemanticDedup
from research_crew.models import (
    AgentName,
    AgentResult,
    Citation,
    ResearchReport,
    StepStatus,
)

_log = structlog.get_logger(__name__)


class Synthesizer(Protocol):
    async def synthesize(
        self, run_id: str, question: str, agent_results: list[AgentResult]
    ) -> ResearchReport: ...


# ---------------------------------------------------------------------------
# Stitch synthesizer — no LLM, deterministic, suitable for CI + portfolio demo.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StitchSynthesizer:
    """Concatenate per-agent summaries, dedupe citations by URL, render markdown.

    Optionally performs cross-run semantic deduplication via the injected
    *semantic_dedup* instance.  The default ``NullDedup`` keeps existing
    behaviour unchanged — URL-only dedup is the only guard.

    When a real :class:`~research_crew.dedup.pgvector.PgVectorSemanticDedup`
    is injected (activated by ``RESEARCH_PG_DSN``), the synthesizer will:

    1. Before adding a citation chunk to the report, call
       :meth:`~research_crew.dedup.protocol.SemanticDedup.is_duplicate`.
       If the chunk was seen in a prior run, log a debug line and skip it.
    2. After successful synthesis, call
       :meth:`~research_crew.dedup.protocol.SemanticDedup.add_seen` for
       each citation snippet that made it into the final report.
    """

    max_citations_per_agent: int = 5
    semantic_dedup: SemanticDedup = field(default_factory=make_semantic_dedup)

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

        # URL dedup (existing behaviour).
        url_deduped = list(_dedupe_citations(_iter_all_citations(agent_results)))

        # Semantic dedup (new, no-op with NullDedup).
        merged_citations = await _semantic_dedupe_citations(
            url_deduped, run_id, self.semantic_dedup
        )

        elapsed = (time.perf_counter() - t0) * 1000.0
        return ResearchReport(
            run_id=run_id,
            question=question,
            summary="\n".join(sections),
            citations=merged_citations,
            agent_results=agent_results,
            elapsed_ms=elapsed,
        )


async def _semantic_dedupe_citations(
    citations: list[Citation],
    run_id: str,
    dedup: SemanticDedup,
) -> list[Citation]:
    """Filter *citations* via semantic dedup, then record the survivors.

    When *dedup* is a :class:`~research_crew.dedup.null.NullDedup` this
    function returns *citations* unchanged in O(n) no-op calls.
    """
    kept: list[Citation] = []
    for c in citations:
        text = f"{c.title} {c.snippet}".strip() or c.url
        is_dup, prior_run_id = await dedup.is_duplicate(text)
        if is_dup:
            _log.debug(
                "synthesizer.semantic_dedup_skip",
                url=c.url,
                prior_run_id=prior_run_id,
            )
            continue
        kept.append(c)

    # Record survivors so future runs can compare against them.
    for c in kept:
        text = f"{c.title} {c.snippet}".strip() or c.url
        await dedup.add_seen(text, run_id)

    return kept


def _iter_all_citations(results: list[AgentResult]) -> Iterable[Citation]:
    for r in results:
        if r.status in (StepStatus.SUCCEEDED, StepStatus.CACHED):
            yield from r.citations


def _dedupe_citations(citations: Iterable[Citation]) -> Iterable[Citation]:
    """Stable-order dedupe by normalized URL — the first occurrence wins.

    Normalization collapses near-duplicates that real search backends
    routinely emit:

    * scheme + host case-folded (``HTTPS://Example.com`` ≡ ``https://example.com``)
    * trailing slash on the path stripped (``/foo/`` ≡ ``/foo``)
    * leading ``www.`` host prefix stripped
    """
    seen: set[str] = set()
    for c in citations:
        key = _normalize_url(c.url)
        if key in seen:
            continue
        seen.add(key)
        yield c


def _normalize_url(url: str) -> str:
    """Best-effort URL canonicalisation for dedupe purposes only."""
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return url
    scheme = parts.scheme.lower()
    netloc = parts.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((scheme, netloc, path, parts.query, ""))


def per_agent_citation_count(results: list[AgentResult]) -> dict[AgentName, int]:
    """Used by the load test report to show fan-in coverage per source."""
    out: dict[AgentName, int] = {n: 0 for n in AgentName}
    for r in results:
        if r.status in (StepStatus.SUCCEEDED, StepStatus.CACHED):
            out[r.agent] = len(r.citations)
    return out
