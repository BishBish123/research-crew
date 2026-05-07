#!/usr/bin/env bash
# scripts/deploy.sh — Deploy research-crew to Fly.io
#
# Usage:
#   scripts/deploy.sh [--app <app-name>]
#
# Options:
#   --app NAME   Fly app name (overrides FLY_APP_NAME env var and fly.toml)
#   -h, --help   Show this help
#
# Required env vars (pushed as Fly secrets):
#   REDIS_URL             Redis connection URL (e.g. redis://...)
#   RESEARCH_PG_DSN       Postgres DSN (optional if using Redis-only mode)
#   TAVILY_API_KEY        Tavily search API key
#   BRAVE_API_KEY         Brave search API key
#   EXA_API_KEY           Exa search API key
#   LANGFUSE_PUBLIC_KEY   Langfuse public key (optional tracing)
#   LANGFUSE_SECRET_KEY   Langfuse secret key (optional tracing)
#
# Non-empty vars from .env are pushed; missing/empty vars are skipped with a warning.
set -euo pipefail

# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------
APP_FLAG=""

usage() {
    grep '^#' "$0" | sed 's/^# \{0,2\}//' | grep -v '^!'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --app)      APP_FLAG="$2"; shift 2 ;;
        -h|--help)  usage ;;
        *)          echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Resolve repo root
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Toolchain check
# ---------------------------------------------------------------------------
if ! command -v fly > /dev/null 2>&1; then
    echo "ERROR: 'fly' CLI not found on PATH." >&2
    echo "       Install via: https://fly.io/docs/hands-on/install-flyctl/" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Auth check
# ---------------------------------------------------------------------------
echo "[deploy] Checking Fly.io authentication..." >&2
if ! fly auth whoami > /dev/null 2>&1; then
    echo "ERROR: Not authenticated with Fly.io. Run: fly auth login" >&2
    exit 1
fi
WHOAMI="$(fly auth whoami 2>/dev/null)"
echo "[deploy] Authenticated as: $WHOAMI" >&2

# ---------------------------------------------------------------------------
# Resolve app name: --app flag > FLY_APP_NAME env > fly.toml
# ---------------------------------------------------------------------------
if [[ -n "$APP_FLAG" ]]; then
    APP_NAME="$APP_FLAG"
elif [[ -n "${FLY_APP_NAME:-}" ]]; then
    APP_NAME="$FLY_APP_NAME"
else
    # Extract from fly.toml
    if [[ ! -f "$REPO_ROOT/fly.toml" ]]; then
        echo "ERROR: fly.toml not found and --app / FLY_APP_NAME not set." >&2
        exit 1
    fi
    APP_NAME="$(grep '^app' "$REPO_ROOT/fly.toml" | head -1 | sed 's/app *= *"\(.*\)"/\1/')"
fi

# Guard against the placeholder sentinel
if [[ "$APP_NAME" == *"REPLACE-ME"* ]]; then
    echo "ERROR: App name contains the placeholder 'REPLACE-ME'." >&2
    echo "       Set a real app name via --app, FLY_APP_NAME, or edit fly.toml." >&2
    exit 1
fi

echo "[deploy] Target app: $APP_NAME" >&2

# ---------------------------------------------------------------------------
# Push secrets from .env (if present)
# ---------------------------------------------------------------------------
SECRET_VARS=(
    REDIS_URL
    RESEARCH_PG_DSN
    TAVILY_API_KEY
    BRAVE_API_KEY
    EXA_API_KEY
    LANGFUSE_PUBLIC_KEY
    LANGFUSE_SECRET_KEY
)

if [[ -f "$REPO_ROOT/.env" ]]; then
    echo "[deploy] Loading secrets from .env..." >&2
    # source .env to pick up values, skipping comment and blank lines
    set -o allexport
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.env" || true
    set +o allexport
fi

SECRET_ARGS=()
for var in "${SECRET_VARS[@]}"; do
    val="${!var:-}"
    if [[ -n "$val" ]]; then
        SECRET_ARGS+=("${var}=${val}")
    else
        echo "[deploy] WARN: $var is not set — skipping." >&2
    fi
done

if [[ ${#SECRET_ARGS[@]} -gt 0 ]]; then
    echo "[deploy] Pushing ${#SECRET_ARGS[@]} secret(s) to Fly..." >&2
    fly secrets set --app "$APP_NAME" "${SECRET_ARGS[@]}"
else
    echo "[deploy] No secrets to push." >&2
fi

# ---------------------------------------------------------------------------
# Deploy
# ---------------------------------------------------------------------------
echo "[deploy] Running: fly deploy --remote-only --app $APP_NAME" >&2
fly deploy --remote-only --app "$APP_NAME"

echo "[deploy] Done. App URL: https://${APP_NAME}.fly.dev" >&2
