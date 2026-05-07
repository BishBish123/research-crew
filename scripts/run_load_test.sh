#!/usr/bin/env bash
# scripts/run_load_test.sh — bring up the API, run Locust headlessly, parse CSVs.
#
# Usage:
#   scripts/run_load_test.sh [OPTIONS]
#
# Options:
#   -u, --users N       Number of concurrent Locust users (default: 10)
#   -d, --duration DUR  Run duration (default: 30s; locust accepts 30s, 1m, 2m, etc.)
#   -p, --port PORT     API port to bind (default: 8765 to avoid clashing with dev server)
#   -o, --out FILE      Markdown results file to update (default: docs/load-test-results.md)
#       --no-update     Print table to stdout but do not update the markdown file
#   -h, --help          Show this help
#
# Environment variables honoured:
#   RESEARCH_CREW_STORE  — Store backend; defaults to "memory" (no Redis needed)
#   RESEARCH_RATE_LIMIT_PER_MIN — Rate-limit cap; defaults to 10000 to avoid throttling
#
# The script tears down the API process on exit (including Ctrl-C).
set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
USERS=10
DURATION=30s
API_PORT=8765
OUT_FILE="docs/load-test-results.md"
NO_UPDATE=0

# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------
usage() {
    grep '^#' "$0" | sed 's/^# \{0,2\}//' | grep -v '^!'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -u|--users)     USERS="$2";     shift 2 ;;
        -d|--duration)  DURATION="$2";  shift 2 ;;
        -p|--port)      API_PORT="$2";  shift 2 ;;
        -o|--out)       OUT_FILE="$2";  shift 2 ;;
        --no-update)    NO_UPDATE=1;    shift   ;;
        -h|--help)      usage           ;;
        *)  echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Resolve repo root (script lives at <root>/scripts/run_load_test.sh)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

CSV_PREFIX="load/results/locust"
mkdir -p load/results

# ---------------------------------------------------------------------------
# Toolchain check
# ---------------------------------------------------------------------------
UV="${UV:-uv}"
if ! command -v "$UV" > /dev/null 2>&1; then
    echo "ERROR: 'uv' not found. Install via: curl -Ls https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

if ! "$UV" run locust --version > /dev/null 2>&1; then
    echo "ERROR: locust not in the venv. Run: uv sync --extra load" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# API startup
# ---------------------------------------------------------------------------
API_LOG="load/results/api.log"
API_PID=""

cleanup() {
    if [[ -n "$API_PID" ]] && kill -0 "$API_PID" 2>/dev/null; then
        echo "[load-test] Stopping API (pid $API_PID)..." >&2
        kill "$API_PID" 2>/dev/null || true
        # Give uvicorn up to 3 seconds to drain gracefully before SIGKILL.
        local i=0
        while kill -0 "$API_PID" 2>/dev/null && [[ $i -lt 30 ]]; do
            sleep 0.1
            i=$((i + 1))
        done
        kill -9 "$API_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

echo "[load-test] Starting API on port $API_PORT (store=${RESEARCH_CREW_STORE:-memory})..." >&2

RESEARCH_CREW_STORE="${RESEARCH_CREW_STORE:-memory}" \
RESEARCH_RATE_LIMIT_PER_MIN="${RESEARCH_RATE_LIMIT_PER_MIN:-10000}" \
RESEARCH_DEV_MODE=1 \
    "$UV" run research-api --host 127.0.0.1 --port "$API_PORT" \
    > "$API_LOG" 2>&1 &
API_PID=$!

# ---------------------------------------------------------------------------
# Health poll (30 s timeout)
# ---------------------------------------------------------------------------
echo "[load-test] Waiting for /health on port $API_PORT..." >&2
READY=0
for i in $(seq 1 150); do
    if curl -fsS "http://127.0.0.1:${API_PORT}/health" > /dev/null 2>&1; then
        READY=1
        break
    fi
    # Check API process hasn't died already.
    if ! kill -0 "$API_PID" 2>/dev/null; then
        echo "ERROR: API process died before /health responded. Log:" >&2
        cat "$API_LOG" >&2
        exit 1
    fi
    sleep 0.2
done

if [[ "$READY" -eq 0 ]]; then
    echo "ERROR: API did not become ready within 30s. Log:" >&2
    cat "$API_LOG" >&2
    exit 1
fi
echo "[load-test] API ready." >&2

# ---------------------------------------------------------------------------
# Locust headless run
# ---------------------------------------------------------------------------
SPAWN_RATE=$(( USERS < 5 ? USERS : 5 ))
LOCUST_LOG="load/results/locust.log"
META="${USERS} users, ${DURATION}, store=${RESEARCH_CREW_STORE:-memory}"

echo "[load-test] Running Locust: users=$USERS spawn-rate=$SPAWN_RATE duration=$DURATION" >&2

"$UV" run locust \
    -f load/locustfile.py \
    --headless \
    --users "$USERS" \
    --spawn-rate "$SPAWN_RATE" \
    --run-time "$DURATION" \
    --host "http://127.0.0.1:${API_PORT}" \
    --csv "$CSV_PREFIX" \
    --csv-full-history \
    --only-summary \
    2>&1 | tee "$LOCUST_LOG"

echo "[load-test] Locust finished." >&2

# ---------------------------------------------------------------------------
# Parse results → markdown
# ---------------------------------------------------------------------------
if [[ -f "${CSV_PREFIX}_stats.csv" ]]; then
    echo "[load-test] Parsing CSV → markdown..." >&2
    if [[ "$NO_UPDATE" -eq 1 ]]; then
        "$UV" run python -m load.parse_results \
            --csv "$CSV_PREFIX" \
            --meta "$META" \
            --stdout
    else
        "$UV" run python -m load.parse_results \
            --csv "$CSV_PREFIX" \
            --meta "$META" \
            --out "$OUT_FILE"
        echo "[load-test] Results written to $OUT_FILE" >&2
    fi
else
    echo "WARNING: Locust stats CSV not found at ${CSV_PREFIX}_stats.csv — skipping parse." >&2
    exit 1
fi

echo "[load-test] Done." >&2
