"""Parse Locust CSV output and emit a markdown table with real numbers.

Locust --csv=<prefix> produces two files:
  <prefix>_stats.csv        — per-endpoint aggregates (p50/p95/p99/rps/failures)
  <prefix>_stats_history.csv — time-series (not used here)

Usage:
    python -m load.parse_results --csv load/results/locust --out docs/load-test-results.md
    python -m load.parse_results --csv load/results/locust --stdout

Exit 0 on success, 1 on missing/malformed CSV.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Required columns in the Locust stats CSV.
_REQUIRED_COLS = {
    "Name",
    "Request Count",
    "Failure Count",
    "50%",
    "95%",
    "99%",
    "Requests/s",
}

# Fallback sentinel when a metric is unavailable.
_NA = "N/A"


def _pct(value: str, *, scale: int = 1) -> str:
    """Format a numeric string as a percentage; pass through _NA."""
    if value in ("", "N/A", None):
        return _NA
    try:
        return f"{float(value) * scale:.1f}%"
    except ValueError:
        return _NA


def _ms(value: str) -> str:
    """Format a numeric string as integer milliseconds; pass through _NA."""
    if value in ("", "N/A", None):
        return _NA
    try:
        return f"{round(float(value))}"
    except ValueError:
        return _NA


def _rps(value: str) -> str:
    """Format requests-per-second to one decimal place."""
    if value in ("", "N/A", None):
        return _NA
    try:
        return f"{float(value):.1f}"
    except ValueError:
        return _NA


def _failure_rate(req: str, fail: str) -> str:
    """Compute failure rate as a percentage string."""
    try:
        r = float(req)
        f = float(fail)
        if r == 0:
            return "0.0%"
        return f"{f / r * 100:.1f}%"
    except (ValueError, ZeroDivisionError):
        return _NA


def parse_stats_csv(path: Path) -> list[dict[str, str]]:
    """Read Locust stats CSV and return list of row dicts.

    Raises FileNotFoundError if *path* doesn't exist.
    Raises ValueError if required columns are missing.
    Returns an empty list if the CSV is empty (header only).
    """
    if not path.exists():
        raise FileNotFoundError(f"Locust stats CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            return []
        present = set(reader.fieldnames)
        missing = _REQUIRED_COLS - present
        if missing:
            raise ValueError(
                f"CSV missing required columns: {sorted(missing)}. Available: {sorted(present)}"
            )
        rows = list(reader)
    return rows


def build_markdown(rows: list[dict[str, str]], *, run_meta: str = "") -> str:
    """Turn parsed CSV rows into a markdown table string.

    The *headline* row (Name == "Aggregated") is placed first; individual
    endpoint rows follow sorted by Name.  If no Aggregated row exists one
    is synthesised from the endpoint rows.

    *run_meta* is an optional one-line string prepended as a blockquote.
    """
    if not rows:
        return "_No data — Locust produced an empty stats CSV._\n"

    # Separate aggregate from per-endpoint rows.
    aggregate: dict[str, str] | None = None
    endpoints: list[dict[str, str]] = []
    for row in rows:
        name = row.get("Name", "").strip()
        if name.lower() == "aggregated":
            aggregate = row
        elif name:
            endpoints.append(row)

    def _row_md(r: dict[str, str]) -> str:
        name = r.get("Name", "").strip()
        p50 = _ms(r.get("50%", ""))
        p95 = _ms(r.get("95%", ""))
        p99 = _ms(r.get("99%", ""))
        rps_val = _rps(r.get("Requests/s", ""))
        fail = _failure_rate(r.get("Request Count", "0"), r.get("Failure Count", "0"))
        reqs = r.get("Request Count", _NA)
        return f"| `{name}` | {p50} | {p95} | {p99} | {rps_val} | {fail} | {reqs} |"

    header = "| Endpoint | p50 (ms) | p95 (ms) | p99 (ms) | RPS | Failure rate | Total reqs |"
    separator = "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"

    lines: list[str] = []
    if run_meta:
        lines.append(f"> {run_meta}\n")
    lines.append(header)
    lines.append(separator)

    # Headline row first.
    if aggregate:
        lines.append(_row_md(aggregate) + "  <!-- headline -->")
    else:
        # Synthesise aggregate from endpoint totals.
        total_reqs = sum(int(r.get("Request Count", 0) or 0) for r in endpoints)
        total_fail = sum(int(r.get("Failure Count", 0) or 0) for r in endpoints)
        synth: dict[str, str] = {
            "Name": "Aggregated",
            "Request Count": str(total_reqs),
            "Failure Count": str(total_fail),
            "50%": _NA,
            "95%": _NA,
            "99%": _NA,
            "Requests/s": _NA,
        }
        lines.append(_row_md(synth) + "  <!-- headline (synthesised) -->")

    for ep in sorted(endpoints, key=lambda r: r.get("Name", "")):
        lines.append(_row_md(ep))

    return "\n".join(lines) + "\n"


def update_results_file(
    doc_path: Path,
    table_md: str,
    *,
    section_header: str = "## Real measured numbers",
) -> None:
    """Insert/replace a results section in *doc_path*.

    The section starts at *section_header* and extends to the next ``## ``
    heading or end-of-file.  If the section doesn't exist it is appended.
    The file is created if it doesn't exist.
    """
    marker_start = f"\n{section_header}\n"
    new_section = f"{marker_start}\n{table_md}"

    if doc_path.exists():
        original = doc_path.read_text(encoding="utf-8")
    else:
        doc_path.parent.mkdir(parents=True, exist_ok=True)
        original = ""

    if section_header in original:
        # Replace from the section header to the next ## heading or EOF.
        start_idx = original.index(section_header)
        # Look for next ## heading *after* the section header.
        next_section = original.find("\n## ", start_idx + len(section_header))
        if next_section == -1:
            updated = original[:start_idx] + new_section
        else:
            updated = original[:start_idx] + new_section + "\n" + original[next_section + 1 :]
    else:
        # Append.
        updated = original.rstrip() + "\n" + new_section

    doc_path.write_text(updated, encoding="utf-8")


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Parse Locust CSV output and emit a markdown table."
    )
    parser.add_argument(
        "--csv",
        required=True,
        metavar="PREFIX",
        help="Locust --csv prefix (e.g. load/results/locust); expects <prefix>_stats.csv to exist.",
    )
    parser.add_argument(
        "--out",
        default="",
        metavar="FILE",
        help="Markdown file to update. If omitted, --stdout is implied.",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print the markdown table to stdout instead of (or in addition to) --out.",
    )
    parser.add_argument(
        "--meta",
        default="",
        metavar="TEXT",
        help="One-line run metadata to prepend as a blockquote (e.g. '10 users, 30s, memory store').",
    )
    args = parser.parse_args(argv)

    stats_path = Path(f"{args.csv}_stats.csv")
    try:
        rows = parse_stats_csv(stats_path)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    table = build_markdown(rows, run_meta=args.meta)

    if args.stdout or not args.out:
        print(table, end="")

    if args.out:
        out_path = Path(args.out)
        update_results_file(out_path, table)
        print(f"Updated {out_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(_main())
