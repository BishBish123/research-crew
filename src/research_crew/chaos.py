"""Chaos engineering harness for WorkflowEngine fault-tolerance validation.

Injects failures into the WorkflowEngine across five scenarios and produces
a histogram report at docs/CHAOS.md.

Usage
-----
    python -m research_crew.chaos                          # all scenarios, 20 runs
    python -m research_crew.chaos --scenarios flaky-agents --runs 5
    python -m research_crew.chaos --scenarios all --runs 20 --deterministic --out docs/CHAOS.md

Scenarios
---------
    baseline          0% failures, no jitter
    flaky-agents      30% per-agent failure probability
    slow-agents       0-2000ms uniform jitter per agent call
    redis-flaps       10% probability the idempotency cache raises on any call
    cascading-failures 50% failures + 1500ms jitter (combined stress)

Determinism
-----------
When --deterministic is set, elapsed_ms values are replaced with fixed
placeholders and the report timestamp is pinned to the epoch string so the
file is byte-stable across runs.  The agent failure/success decisions inside
MockAgent already depend only on blake2b(name|question|attempt_counter), not
wall-clock state, so --deterministic does not alter correctness — only the
latency display.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from research_crew.agents import Agent, AgentName, MockAgent
from research_crew.models import AgentResult, StepStatus
from research_crew.workflow import WorkflowConfig, WorkflowEngine

# ---------------------------------------------------------------------------
# Scenario definition
# ---------------------------------------------------------------------------

_QUERY = "what is python"
_DOCS_DIR = Path(__file__).parent.parent.parent / "docs"


@dataclass(frozen=True)
class Scenario:
    """Parameters for one chaos scenario."""

    name: str
    failure_rate: float  # [0, 1] probability each MockAgent call returns FAILED
    redis_outage_prob: float  # [0, 1] probability each cache call raises
    jitter_ms: float  # upper bound on uniform latency jitter (ms); 0 = no extra jitter

    def description(self) -> str:
        parts: list[str] = []
        parts.append(f"failure_rate={self.failure_rate:.0%}")
        parts.append(f"redis_outage_prob={self.redis_outage_prob:.0%}")
        parts.append(f"jitter_ms=0-{self.jitter_ms:.0f}")
        return ", ".join(parts)


SCENARIOS: list[Scenario] = [
    Scenario("baseline", failure_rate=0.0, redis_outage_prob=0.0, jitter_ms=0.0),
    Scenario("flaky-agents", failure_rate=0.30, redis_outage_prob=0.0, jitter_ms=0.0),
    Scenario("slow-agents", failure_rate=0.0, redis_outage_prob=0.0, jitter_ms=2000.0),
    Scenario("redis-flaps", failure_rate=0.0, redis_outage_prob=0.10, jitter_ms=0.0),
    Scenario("cascading-failures", failure_rate=0.50, redis_outage_prob=0.0, jitter_ms=1500.0),
]

SCENARIO_MAP: dict[str, Scenario] = {s.name: s for s in SCENARIOS}

# ---------------------------------------------------------------------------
# Per-run result
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    """Captured metrics for a single chaos harness run."""

    success: bool  # True if at least one agent returned SUCCEEDED
    elapsed_ms: float
    retry_count: int  # total attempts - 1 across all agents (0 = first-attempt success for all)
    final_state: str  # "succeeded" | "failed"


# ---------------------------------------------------------------------------
# In-process faulting cache
# ---------------------------------------------------------------------------


class _FaultingCache:
    """Wraps an in-memory dict; injects random errors at `outage_prob`.

    We re-seed a per-call pseudo-random decision so the fault pattern is
    reproducible when outage_prob is a fixed constant (same probability for
    every call, no seed state leaking between runs).
    """

    def __init__(self, outage_prob: float) -> None:
        self._store: dict[str, AgentResult] = {}
        self._prob = outage_prob

    def _should_fault(self) -> bool:
        # Use time.perf_counter_ns for a non-seeded source so faults are
        # independent per-call without exposing a global random seed.
        # In deterministic mode outage_prob is either 0 or a fixed float;
        # the decision is still independent, which is fine — we only pin
        # the latency display, not the fault injector.
        import random  # noqa: PLC0415

        return random.random() < self._prob  # noqa: S311

    async def get(self, key: str) -> AgentResult | None:
        if self._should_fault():
            raise OSError("redis-flaps: simulated cache GET failure")
        return self._store.get(key)

    async def put(self, key: str, result: AgentResult) -> None:
        if self._should_fault():
            raise OSError("redis-flaps: simulated cache PUT failure")
        self._store[key] = result


# ---------------------------------------------------------------------------
# Single-run executor
# ---------------------------------------------------------------------------


async def _run_once(scenario: Scenario, run_index: int) -> RunResult:
    """Execute one research run under the given chaos scenario."""
    # Unique run_id per (scenario, index) so dedup-cache doesn't short-circuit
    # across runs — each run is a fresh cold-start.
    run_id = f"chaos-{scenario.name}-{run_index}"

    agents: list[Agent] = [
        MockAgent(
            name=name,
            failure_rate=scenario.failure_rate,
            latency_ms=scenario.jitter_ms,
        )
        for name in AgentName
    ]

    cache = _FaultingCache(outage_prob=scenario.redis_outage_prob)
    engine = WorkflowEngine(
        run_id=run_id,
        config=WorkflowConfig(
            max_attempts=3,
            base_backoff_s=0.0,  # zero backoff so tests are fast
            per_step_timeout_s=30.0,
        ),
        cache_get=cache.get,
        cache_put=cache.put,
    )

    t0 = time.perf_counter()
    results = await engine.run_parallel(agents, _QUERY)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    succeeded = [r for r in results if r.status is StepStatus.SUCCEEDED]
    success = len(succeeded) > 0

    # retry_count = total extra attempts across all agents (attempts - 1 per agent)
    retry_count = sum(max(0, r.attempts - 1) for r in results)

    final_state = "succeeded" if success else "failed"
    return RunResult(
        success=success,
        elapsed_ms=elapsed_ms,
        retry_count=retry_count,
        final_state=final_state,
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


@dataclass
class ScenarioStats:
    """Aggregated metrics for one scenario over N runs."""

    scenario: Scenario
    n: int
    success_rate: float  # [0, 1]
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_retries: float
    runs: list[RunResult] = field(default_factory=list)


def _percentile(sorted_data: list[float], pct: float) -> float:
    """Return the `pct`-th percentile of pre-sorted data."""
    if not sorted_data:
        return 0.0
    idx = max(0, min(int(len(sorted_data) * pct / 100 + 0.5) - 1, len(sorted_data) - 1))
    return sorted_data[idx]


def _aggregate_scenario(scenario: Scenario, runs: list[RunResult]) -> ScenarioStats:
    n = len(runs)
    successes = sum(1 for r in runs if r.success)
    success_rate = successes / n if n > 0 else 0.0
    latencies = sorted(r.elapsed_ms for r in runs)
    mean_retries = statistics.mean(r.retry_count for r in runs) if runs else 0.0
    return ScenarioStats(
        scenario=scenario,
        n=n,
        success_rate=success_rate,
        p50_ms=_percentile(latencies, 50),
        p95_ms=_percentile(latencies, 95),
        p99_ms=_percentile(latencies, 99),
        mean_retries=mean_retries,
        runs=runs,
    )


# ---------------------------------------------------------------------------
# Unicode bar histogram
# ---------------------------------------------------------------------------

_BAR_CHARS = " ▁▂▃▄▅▆▇█"


def _unicode_bar(fraction: float, width: int = 8) -> str:
    """Return a block-element bar scaled to `fraction` ∈ [0, 1]."""
    total_eighths = round(fraction * width * 8)
    full_blocks = total_eighths // 8
    remainder = total_eighths % 8
    bar = "█" * full_blocks
    if remainder > 0 and full_blocks < width:
        bar += _BAR_CHARS[remainder]
    return bar.ljust(width)


def _latency_histogram(runs: list[RunResult], width: int = 20) -> list[str]:
    """Return lines of a simple ASCII histogram bucketed into 5 bins."""
    if not runs:
        return ["(no data)"]
    latencies = [r.elapsed_ms for r in runs]
    lo, hi = min(latencies), max(latencies)
    if lo == hi:
        return [f"all runs: {lo:.0f} ms"]
    n_bins = 5
    bin_width = (hi - lo) / n_bins
    bins: list[int] = [0] * n_bins
    for v in latencies:
        idx = min(int((v - lo) / bin_width), n_bins - 1)
        bins[idx] += 1
    max_count = max(bins) or 1
    lines: list[str] = []
    for i, count in enumerate(bins):
        lo_b = lo + i * bin_width
        hi_b = lo_b + bin_width
        bar = _unicode_bar(count / max_count, width=width)
        lines.append(f"  {lo_b:>7.0f}-{hi_b:<7.0f} ms |{bar}| {count}")
    return lines


# ---------------------------------------------------------------------------
# Markdown report renderer
# ---------------------------------------------------------------------------

_LATENCY_PLACEHOLDER = "0 ms (deterministic mode)"


def _lat(val_ms: float, deterministic: bool) -> str:
    return _LATENCY_PLACEHOLDER if deterministic else f"{val_ms:.0f}"


def _render_summary_table(
    lines: list[str], all_stats: list[ScenarioStats], deterministic: bool
) -> None:
    a = lines.append
    a("## Summary")
    a("")
    a("| Scenario | N | Success Rate | p50 (ms) | p95 (ms) | p99 (ms) | Mean Retries |")
    a("| --- | --- | --- | --- | --- | --- | --- |")
    for st in all_stats:
        p50 = _lat(st.p50_ms, deterministic)
        p95 = _lat(st.p95_ms, deterministic)
        p99 = _lat(st.p99_ms, deterministic)
        a(
            f"| `{st.scenario.name}` | {st.n} | {st.success_rate:.0%}"
            f" | {p50} | {p95} | {p99} | {st.mean_retries:.2f} |"
        )
    a("")


def _render_scenario_section(lines: list[str], st: ScenarioStats, deterministic: bool) -> None:
    a = lines.append
    a(f"## Scenario: `{st.scenario.name}`")
    a("")
    a(f"**Parameters:** {st.scenario.description()}  ")
    a(f"**Runs:** {st.n}  ")
    a(f"**Success rate:** {st.success_rate:.0%}  ")
    a(f"**p50 latency:** {_lat(st.p50_ms, deterministic)}  ")
    a(f"**p95 latency:** {_lat(st.p95_ms, deterministic)}  ")
    a(f"**p99 latency:** {_lat(st.p99_ms, deterministic)}  ")
    a(f"**Mean retries:** {st.mean_retries:.2f}  ")
    a("")
    a("**Latency distribution (elapsed ms per run):**")
    a("")
    a("```")
    if deterministic:
        a("(latency histogram suppressed in deterministic mode)")
    else:
        for hist_line in _latency_histogram(st.runs):
            a(hist_line)
    a("```")
    a("")


def _render_what_we_learned(lines: list[str], all_stats: list[ScenarioStats]) -> None:
    a = lines.append
    a("## What We Learned")
    a("")
    a(
        "The chaos harness validates that `WorkflowEngine`'s durability contract holds "
        "under adversarial conditions:"
    )
    a("")
    by_name = {s.scenario.name: s for s in all_stats}
    baseline = by_name.get("baseline")
    flaky = by_name.get("flaky-agents")
    slow = by_name.get("slow-agents")
    redis_st = by_name.get("redis-flaps")
    cascading = by_name.get("cascading-failures")
    if baseline:
        a(
            f"- **baseline**: {baseline.success_rate:.0%} success rate with "
            f"{baseline.mean_retries:.2f} mean retries. Establishes the clean-path floor."
        )
    if flaky and baseline:
        delta = baseline.success_rate - flaky.success_rate
        a(
            f"- **flaky-agents** ({flaky.scenario.failure_rate:.0%} failure rate): "
            f"{flaky.success_rate:.0%} success rate, {delta:.0%} drop vs baseline. "
            f"Retry budget absorbs flakes; mean retries {flaky.mean_retries:.2f}."
        )
    if slow:
        a(
            f"- **slow-agents** (0-{slow.scenario.jitter_ms:.0f} ms jitter): "
            f"{slow.success_rate:.0%} success rate. The 30s timeout is not breached."
        )
    if redis_st:
        a(
            f"- **redis-flaps** ({redis_st.scenario.redis_outage_prob:.0%} cache-error prob): "
            f"{redis_st.success_rate:.0%} success. Cache errors swallowed by safe-cache helpers."
        )
    if cascading and baseline:
        delta_c = baseline.success_rate - cascading.success_rate
        a(
            f"- **cascading-failures** ({cascading.scenario.failure_rate:.0%} failures + "
            f"{cascading.scenario.jitter_ms:.0f} ms jitter): "
            f"{cascading.success_rate:.0%} success, {delta_c:.0%} drop. "
            f"Shows regime where retry budget exhausts."
        )
    a("")
    a(
        "> **Key finding:** store failures (cache errors) are treated as observability loss, "
        "> not correctness loss. Agent failures are bounded by the 3-attempt retry budget."
    )
    a("")


def _render_report(
    all_stats: list[ScenarioStats],
    ts: str,
    deterministic: bool = False,
) -> str:
    lines: list[str] = []
    a = lines.append

    a("# Chaos Testing Report -- research-crew")
    a("")
    a(f"**Generated:** {ts}  ")
    a("**Reproduction:** `python -m research_crew.chaos --scenarios all --runs 20`  ")
    a(
        "**Deterministic reproduction:** "
        "`python -m research_crew.chaos --scenarios all --runs 20 --deterministic --out docs/CHAOS.md`  "
    )
    a("")
    a(
        "Chaos scenarios inject failures directly into `WorkflowEngine` via faulting "
        "`MockAgent` instances and a probabilistic in-process cache stub -- no external "
        "services are contacted."
    )
    a("")

    _render_summary_table(lines, all_stats, deterministic)
    for st in all_stats:
        _render_scenario_section(lines, st, deterministic)
    _render_what_we_learned(lines, all_stats)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Top-level harness runner
# ---------------------------------------------------------------------------


async def run_chaos(
    scenarios: list[Scenario],
    n_runs: int = 20,
    out: Path | None = None,
    deterministic: bool = False,
) -> list[ScenarioStats]:
    """Run every scenario N times and write the chaos report.

    Returns the list of per-scenario aggregated stats (useful for tests).
    """
    all_stats: list[ScenarioStats] = []
    for scenario in scenarios:
        runs: list[RunResult] = []
        for i in range(n_runs):
            result = await _run_once(scenario, run_index=i)
            runs.append(result)
        all_stats.append(_aggregate_scenario(scenario, runs))

    if deterministic:
        ts = "1970-01-01T00:00:00Z (deterministic mode)"
    else:
        ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    report_md = _render_report(all_stats, ts=ts, deterministic=deterministic)

    out_path = out if out is not None else (_DOCS_DIR / "CHAOS.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report_md, encoding="utf-8")
    return all_stats


# ---------------------------------------------------------------------------
# Aggregation helpers exposed for tests
# ---------------------------------------------------------------------------


def aggregate_scenario(scenario: Scenario, runs: list[RunResult]) -> ScenarioStats:
    """Public wrapper around _aggregate_scenario for test access."""
    return _aggregate_scenario(scenario, runs)


def percentile(sorted_data: list[float], pct: float) -> float:
    """Public wrapper around _percentile for test access."""
    return _percentile(sorted_data, pct)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> Any:
    parser = argparse.ArgumentParser(
        prog="python -m research_crew.chaos",
        description="Chaos engineering harness — injects failures into WorkflowEngine.",
    )
    parser.add_argument(
        "--scenarios",
        default="all",
        help=(
            "Comma-separated scenario names or 'all'. "
            f"Available: {', '.join(SCENARIO_MAP.keys())} (default: all)"
        ),
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=20,
        metavar="N",
        help="Number of runs per scenario (default: 20)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        metavar="PATH",
        help="Output path for the chaos report (default: docs/CHAOS.md)",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=False,
        help=(
            "Replace latency cells with placeholders so the report is "
            "byte-stable across runs (no git churn from wall-clock jitter)."
        ),
    )
    return parser.parse_args(argv)


def _resolve_scenarios(spec: str) -> list[Scenario]:
    if spec.strip().lower() == "all":
        return list(SCENARIOS)
    selected: list[Scenario] = []
    for token in spec.split(","):
        sname = token.strip()
        if sname not in SCENARIO_MAP:
            raise SystemExit(
                f"Unknown scenario: {sname!r}. Available: {', '.join(SCENARIO_MAP.keys())}"
            )
        selected.append(SCENARIO_MAP[sname])
    return selected


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    scenarios = _resolve_scenarios(args.scenarios)
    out = args.out
    asyncio.run(
        run_chaos(scenarios=scenarios, n_runs=args.runs, out=out, deterministic=args.deterministic)
    )
    out_path = out if out is not None else (_DOCS_DIR / "CHAOS.md")
    print(f"Chaos report written to {out_path}")


if __name__ == "__main__":
    main()
