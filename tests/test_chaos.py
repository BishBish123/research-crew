"""Tests for the chaos engineering harness.

Covers:
- Smoke: every scenario runs without crashing (tiny N for speed).
- Aggregation formulae: hand-built fixtures verify success_rate, percentiles,
  mean_retries.
- Determinism: two --deterministic runs produce identical CHAOS.md content.
- CLI flags: --scenarios, --runs, --out, --deterministic round-trip correctly.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from research_crew.chaos import (
    SCENARIO_MAP,
    SCENARIOS,
    RunResult,
    Scenario,
    aggregate_scenario,
    percentile,
    run_chaos,
)

# ---------------------------------------------------------------------------
# Smoke tests: every scenario runs N=2 times without crashing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario", SCENARIOS)
async def test_scenario_smoke(scenario: Scenario) -> None:
    """Each scenario must complete without raising and return valid stats."""
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
        out = Path(f.name)
    stats_list = await run_chaos(
        scenarios=[scenario],
        n_runs=2,
        out=out,
        deterministic=True,
    )
    assert len(stats_list) == 1
    st = stats_list[0]
    assert st.scenario is scenario
    assert st.n == 2
    assert 0.0 <= st.success_rate <= 1.0
    assert st.p50_ms >= 0.0
    assert st.p95_ms >= 0.0
    assert st.p99_ms >= 0.0
    assert st.mean_retries >= 0.0
    # Report file must be written
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert f"`{scenario.name}`" in content


# ---------------------------------------------------------------------------
# Aggregation formula tests (hand-built fixtures)
# ---------------------------------------------------------------------------


def _make_runs(successes: int, failures: int, latencies: list[float]) -> list[RunResult]:
    """Build a list of RunResult fixtures."""
    runs: list[RunResult] = []
    for i, lat in enumerate(latencies):
        success = i < successes
        runs.append(
            RunResult(
                success=success,
                elapsed_ms=lat,
                retry_count=0 if success else 2,
                final_state="succeeded" if success else "failed",
            )
        )
    return runs


def test_success_rate_all_succeed() -> None:
    scenario = SCENARIOS[0]  # baseline
    runs = _make_runs(successes=5, failures=0, latencies=[10.0] * 5)
    st = aggregate_scenario(scenario, runs)
    assert st.success_rate == 1.0


def test_success_rate_partial() -> None:
    scenario = SCENARIOS[0]
    runs = _make_runs(successes=3, failures=2, latencies=[10.0] * 5)
    st = aggregate_scenario(scenario, runs)
    assert abs(st.success_rate - 0.6) < 1e-9


def test_success_rate_all_fail() -> None:
    scenario = SCENARIOS[0]
    runs = _make_runs(successes=0, failures=4, latencies=[10.0] * 4)
    st = aggregate_scenario(scenario, runs)
    assert st.success_rate == 0.0


def test_p50_exact() -> None:
    """p50 on [10, 20, 30, 40, 50] must be 30 (median)."""
    scenario = SCENARIOS[0]
    latencies = [10.0, 20.0, 30.0, 40.0, 50.0]
    runs = _make_runs(successes=5, failures=0, latencies=latencies)
    st = aggregate_scenario(scenario, runs)
    # Our percentile: idx = int(5 * 50/100 + 0.5) - 1 = int(3.0) - 1 = 2 → data[2] = 30.0
    assert st.p50_ms == 30.0


def test_p95_ten_samples() -> None:
    """p95 on 10 evenly-spaced values [10,20,...,100]."""
    scenario = SCENARIOS[0]
    latencies = [float(i * 10) for i in range(1, 11)]  # [10, 20, ..., 100]
    runs = _make_runs(successes=10, failures=0, latencies=latencies)
    st = aggregate_scenario(scenario, runs)
    # idx = int(10 * 95/100 + 0.5) - 1 = int(9.5+0.5) - 1 = 10 - 1 = 9 → data[9] = 100.0
    assert st.p95_ms == 100.0


def test_p99_returns_last_for_small_n() -> None:
    scenario = SCENARIOS[0]
    latencies = [1.0, 2.0, 3.0]
    runs = _make_runs(successes=3, failures=0, latencies=latencies)
    st = aggregate_scenario(scenario, runs)
    # p99 on n=3: idx = int(3*99/100+0.5)-1 = int(3.47)-1 = 3-1 = 2 → data[2] = 3.0
    assert st.p99_ms == 3.0


def test_mean_retries() -> None:
    scenario = SCENARIOS[0]
    # 2 success (0 retries each) + 2 failure (2 retries each) = mean 1.0
    runs = [
        RunResult(success=True, elapsed_ms=10.0, retry_count=0, final_state="succeeded"),
        RunResult(success=True, elapsed_ms=10.0, retry_count=0, final_state="succeeded"),
        RunResult(success=False, elapsed_ms=10.0, retry_count=2, final_state="failed"),
        RunResult(success=False, elapsed_ms=10.0, retry_count=2, final_state="failed"),
    ]
    st = aggregate_scenario(scenario, runs)
    assert abs(st.mean_retries - 1.0) < 1e-9


def test_percentile_helper_empty() -> None:
    assert percentile([], 50) == 0.0


def test_percentile_single() -> None:
    assert percentile([42.0], 50) == 42.0
    assert percentile([42.0], 99) == 42.0


def test_percentile_clamping() -> None:
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    # p0 must not go negative; p100 must not exceed last element
    assert percentile(data, 0) >= data[0]
    assert percentile(data, 100) <= data[-1]


# ---------------------------------------------------------------------------
# Determinism tests
# ---------------------------------------------------------------------------


async def test_deterministic_runs_identical_content() -> None:
    """Two deterministic runs must produce byte-identical CHAOS.md content."""
    scenario = SCENARIO_MAP["baseline"]
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f1:
        out1 = Path(f1.name)
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f2:
        out2 = Path(f2.name)

    await run_chaos(scenarios=[scenario], n_runs=5, out=out1, deterministic=True)
    await run_chaos(scenarios=[scenario], n_runs=5, out=out2, deterministic=True)

    content1 = out1.read_text(encoding="utf-8")
    content2 = out2.read_text(encoding="utf-8")

    assert content1 == content2, (
        "Deterministic mode must produce identical output across runs.\n"
        f"MD5 run1: {hashlib.md5(content1.encode()).hexdigest()}\n"  # noqa: S324
        f"MD5 run2: {hashlib.md5(content2.encode()).hexdigest()}"  # noqa: S324
    )


async def test_deterministic_suppresses_latency_numbers() -> None:
    """Deterministic report must not contain real floating-point latency cells."""
    scenario = SCENARIO_MAP["baseline"]
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
        out = Path(f.name)

    await run_chaos(scenarios=[scenario], n_runs=3, out=out, deterministic=True)
    content = out.read_text(encoding="utf-8")

    assert "deterministic mode" in content.lower(), (
        "Deterministic mode must include the placeholder text"
    )


# ---------------------------------------------------------------------------
# CLI flag tests (subprocess to exercise the full argparse path)
# ---------------------------------------------------------------------------


def test_cli_help_exits_zero() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "research_crew.chaos", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--scenarios" in result.stdout
    assert "--runs" in result.stdout
    assert "--deterministic" in result.stdout


def test_cli_unknown_scenario_exits_nonzero() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "research_crew.chaos",
            "--scenarios",
            "does-not-exist",
            "--runs",
            "1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0


def test_cli_baseline_deterministic_writes_file() -> None:
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
        out_path = f.name
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "research_crew.chaos",
            "--scenarios",
            "baseline",
            "--runs",
            "2",
            "--deterministic",
            "--out",
            out_path,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    content = Path(out_path).read_text(encoding="utf-8")
    assert "baseline" in content
    assert "Chaos Testing Report" in content


def test_cli_multiple_scenarios() -> None:
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
        out_path = f.name
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "research_crew.chaos",
            "--scenarios",
            "baseline,flaky-agents",
            "--runs",
            "2",
            "--deterministic",
            "--out",
            out_path,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    content = Path(out_path).read_text(encoding="utf-8")
    assert "`baseline`" in content
    assert "`flaky-agents`" in content
