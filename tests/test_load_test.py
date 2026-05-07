"""Tests for load/parse_results.py and the load test harness.

Unit tests (default, no markers):
    - parse_results.parse_stats_csv turns fixture CSV into correct rows
    - parse_results.build_markdown emits the expected markdown table
    - parse_results.parse_stats_csv handles missing columns gracefully
    - parse_results.build_markdown handles an empty row list
    - scripts/run_load_test.sh exists and passes bash -n syntax check

Integration tests (marked @pytest.mark.integration):
    - Full smoke: API up + Locust + non-empty results dir + parse
"""

from __future__ import annotations

import csv
import io
import os
import subprocess
from pathlib import Path

import pytest
from load.parse_results import build_markdown, parse_stats_csv, update_results_file

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent


def _make_csv(rows: list[dict[str, str]]) -> str:
    """Build a CSV string from a list of row dicts."""
    if not rows:
        return ""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


# Minimal fixture data matching Locust's _stats.csv column names.
FIXTURE_ROWS: list[dict[str, str]] = [
    {
        "Type": "POST",
        "Name": "POST /research",
        "Request Count": "120",
        "Failure Count": "2",
        "Median Response Time": "12",
        "Average Response Time": "15",
        "Min Response Time": "5",
        "Max Response Time": "120",
        "Average Content Size": "100",
        "Requests/s": "4.5",
        "Failures/s": "0.07",
        "50%": "12",
        "66%": "14",
        "75%": "16",
        "80%": "17",
        "90%": "20",
        "95%": "25",
        "98%": "35",
        "99%": "45",
        "99.9%": "90",
        "99.99%": "110",
        "100%": "120",
    },
    {
        "Type": "GET",
        "Name": "GET /runs/{id}",
        "Request Count": "120",
        "Failure Count": "0",
        "Median Response Time": "5",
        "Average Response Time": "6",
        "Min Response Time": "2",
        "Max Response Time": "30",
        "Average Content Size": "800",
        "Requests/s": "4.5",
        "Failures/s": "0",
        "50%": "5",
        "66%": "6",
        "75%": "7",
        "80%": "8",
        "90%": "12",
        "95%": "15",
        "98%": "22",
        "99%": "30",
        "99.9%": "30",
        "99.99%": "30",
        "100%": "30",
    },
    {
        "Type": "GET",
        "Name": "GET /health",
        "Request Count": "40",
        "Failure Count": "0",
        "Median Response Time": "3",
        "Average Response Time": "4",
        "Min Response Time": "1",
        "Max Response Time": "15",
        "Average Content Size": "60",
        "Requests/s": "1.5",
        "Failures/s": "0",
        "50%": "3",
        "66%": "4",
        "75%": "5",
        "80%": "6",
        "90%": "8",
        "95%": "10",
        "98%": "12",
        "99%": "14",
        "99.9%": "15",
        "99.99%": "15",
        "100%": "15",
    },
    {
        "Type": "",
        "Name": "Aggregated",
        "Request Count": "280",
        "Failure Count": "2",
        "Median Response Time": "7",
        "Average Response Time": "10",
        "Min Response Time": "1",
        "Max Response Time": "120",
        "Average Content Size": "300",
        "Requests/s": "10.5",
        "Failures/s": "0.07",
        "50%": "7",
        "66%": "10",
        "75%": "13",
        "80%": "15",
        "90%": "18",
        "95%": "22",
        "98%": "32",
        "99%": "42",
        "99.9%": "85",
        "99.99%": "105",
        "100%": "120",
    },
]


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestParseStatsCsv:
    """parse_stats_csv: CSV → list-of-dicts."""

    def test_parses_fixture_rows(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "locust_stats.csv"
        csv_file.write_text(_make_csv(FIXTURE_ROWS), encoding="utf-8")
        rows = parse_stats_csv(csv_file)
        assert len(rows) == 4
        names = {r["Name"] for r in rows}
        assert "POST /research" in names
        assert "Aggregated" in names

    def test_request_counts_are_strings(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "locust_stats.csv"
        csv_file.write_text(_make_csv(FIXTURE_ROWS), encoding="utf-8")
        rows = parse_stats_csv(csv_file)
        agg = next(r for r in rows if r["Name"] == "Aggregated")
        assert agg["Request Count"] == "280"

    def test_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            parse_stats_csv(tmp_path / "nonexistent_stats.csv")

    def test_missing_required_columns_raises_value_error(self, tmp_path: Path) -> None:
        # Omit "95%" and "99%" columns.
        minimal_rows = [
            {
                "Name": "POST /research",
                "Request Count": "10",
                "Failure Count": "0",
                "50%": "12",
                "Requests/s": "1.0",
            }
        ]
        csv_file = tmp_path / "locust_stats.csv"
        csv_file.write_text(_make_csv(minimal_rows), encoding="utf-8")
        with pytest.raises(ValueError, match="missing required columns"):
            parse_stats_csv(csv_file)

    def test_empty_csv_returns_empty_list(self, tmp_path: Path) -> None:
        # Write header only (no data rows).
        csv_file = tmp_path / "locust_stats.csv"
        csv_file.write_text(
            "Name,Request Count,Failure Count,50%,95%,99%,Requests/s\n",
            encoding="utf-8",
        )
        rows = parse_stats_csv(csv_file)
        assert rows == []

    def test_extra_columns_are_passed_through(self, tmp_path: Path) -> None:
        """Extra columns beyond the required set are silently passed through."""
        rows = [
            {
                "Name": "GET /health",
                "Request Count": "5",
                "Failure Count": "0",
                "50%": "3",
                "95%": "8",
                "99%": "10",
                "Requests/s": "0.5",
                "Extra Column": "ignored",
            }
        ]
        csv_file = tmp_path / "locust_stats.csv"
        csv_file.write_text(_make_csv(rows), encoding="utf-8")
        result = parse_stats_csv(csv_file)
        assert len(result) == 1
        assert result[0]["Extra Column"] == "ignored"


class TestBuildMarkdown:
    """build_markdown: list-of-dicts → markdown string."""

    def test_contains_header_row(self) -> None:
        md = build_markdown(FIXTURE_ROWS)
        assert "| Endpoint |" in md
        assert "p50 (ms)" in md
        assert "p95 (ms)" in md
        assert "p99 (ms)" in md
        assert "RPS" in md
        assert "Failure rate" in md

    def test_aggregated_row_appears_first(self) -> None:
        md = build_markdown(FIXTURE_ROWS)
        lines = md.splitlines()
        data_lines = [
            ln for ln in lines if ln.startswith("|") and "---" not in ln and "Endpoint" not in ln
        ]
        assert data_lines, "Expected data rows"
        assert "Aggregated" in data_lines[0], (
            f"First data row should be Aggregated, got: {data_lines[0]}"
        )

    def test_failure_rate_computed_correctly(self) -> None:
        md = build_markdown(FIXTURE_ROWS)
        # POST /research: 2/120 = 1.7%
        assert "1.7%" in md

    def test_zero_failure_rate(self) -> None:
        md = build_markdown(FIXTURE_ROWS)
        # GET /health: 0/40 = 0.0%
        assert "0.0%" in md

    def test_run_meta_appears_as_blockquote(self) -> None:
        md = build_markdown(FIXTURE_ROWS, run_meta="10 users, 30s, memory store")
        assert "> 10 users, 30s, memory store" in md

    def test_empty_rows_returns_na_message(self) -> None:
        md = build_markdown([])
        assert "No data" in md

    def test_p50_p95_p99_values_present(self) -> None:
        md = build_markdown(FIXTURE_ROWS)
        assert "| `Aggregated` |" in md
        # p50=7, p95=22, p99=42
        assert "7" in md
        assert "22" in md
        assert "42" in md

    def test_synthesises_aggregate_when_missing(self) -> None:
        rows_no_agg = [r for r in FIXTURE_ROWS if r["Name"] != "Aggregated"]
        md = build_markdown(rows_no_agg)
        assert "Aggregated" in md
        assert "synthesised" in md

    def test_endpoint_names_in_table(self) -> None:
        md = build_markdown(FIXTURE_ROWS)
        assert "POST /research" in md
        assert "GET /runs/{id}" in md
        assert "GET /health" in md


class TestUpdateResultsFile:
    """update_results_file: inserts or replaces section in markdown file."""

    def test_creates_file_if_not_exists(self, tmp_path: Path) -> None:
        out = tmp_path / "results.md"
        update_results_file(out, "| a | b |\n")
        assert out.exists()
        assert "| a | b |" in out.read_text()

    def test_replaces_existing_section(self, tmp_path: Path) -> None:
        out = tmp_path / "results.md"
        out.write_text(
            "# Heading\n\n## Real measured numbers\n\nold table\n\n## Other\n\nother content\n"
        )
        update_results_file(out, "new table\n")
        text = out.read_text()
        assert "new table" in text
        assert "old table" not in text
        assert "Other" in text  # next section preserved

    def test_appends_when_section_absent(self, tmp_path: Path) -> None:
        out = tmp_path / "results.md"
        out.write_text("# Existing content\n\nSome text.\n")
        update_results_file(out, "fresh table\n")
        text = out.read_text()
        assert "Existing content" in text
        assert "fresh table" in text


class TestScriptExists:
    """Shell script existence and syntax checks."""

    def test_script_exists(self) -> None:
        script = REPO_ROOT / "scripts" / "run_load_test.sh"
        assert script.exists(), f"Expected {script} to exist"

    def test_script_is_executable(self) -> None:
        script = REPO_ROOT / "scripts" / "run_load_test.sh"
        assert os.access(script, os.X_OK), f"Expected {script} to be executable"

    def test_script_bash_n_clean(self) -> None:
        """bash -n (parse-only check) exits 0."""
        script = REPO_ROOT / "scripts" / "run_load_test.sh"
        result = subprocess.run(
            ["bash", "-n", str(script)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, (
            f"bash -n failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )

    def test_script_help_flag(self) -> None:
        """--help prints usage and exits 0."""
        script = REPO_ROOT / "scripts" / "run_load_test.sh"
        result = subprocess.run(
            ["bash", str(script), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, f"--help exited {result.returncode}: {result.stderr}"
        combined = result.stdout + result.stderr
        assert "users" in combined.lower() or "usage" in combined.lower()


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestSmokeLoadTest:
    """Full smoke: bring up API (memory store) + Locust + parse.

    Requires: locust installed in the venv (uv sync --extra load).
    Does NOT require Docker / Redis — uses RESEARCH_CREW_STORE=memory.
    """

    def test_full_smoke_run(self, tmp_path: Path) -> None:
        """Run 3 users for 15s; assert non-empty results dir + parsed markdown."""
        script = REPO_ROOT / "scripts" / "run_load_test.sh"
        out_md = tmp_path / "results.md"
        result = subprocess.run(
            [
                "bash",
                str(script),
                "--users",
                "3",
                "--duration",
                "15s",
                "--out",
                str(out_md),
            ],
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, "RESEARCH_CREW_STORE": "memory"},
            cwd=str(REPO_ROOT),
            check=False,
        )
        assert result.returncode == 0, (
            f"Smoke run failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        results_dir = REPO_ROOT / "load" / "results"
        csv_files = list(results_dir.glob("locust_stats*.csv"))
        assert csv_files, (
            f"Expected locust_stats*.csv in {results_dir}; found: {list(results_dir.iterdir())}"
        )
        assert out_md.exists(), "Expected markdown output file to be written"
        md_text = out_md.read_text()
        assert "Aggregated" in md_text or "N/A" in md_text, (
            f"Expected table content in markdown:\n{md_text}"
        )
