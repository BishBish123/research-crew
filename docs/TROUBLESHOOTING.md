# Troubleshooting

Real ops gotchas surfaced during development. Each entry: what you'll see, what's actually happening, how to fix it.

## Port conflicts

### Symptom: `make up` fails immediately with a compose error
**Cause:** Port 6379 is already bound (host redis-server, another container, or a previous run that didn't `make down`).
**Fix:** `REDIS_PORT=6390 make up` — the Makefile prints a next-port hint on failure: `REDIS_PORT=$((6379+1)) make up`.
**Source:** `Makefile:up` target, port-conflict hint block

### Symptom: `make pg-up` fails with address already in use
**Cause:** Port 5432 is taken (local Postgres or another container).
**Fix:** `PG_PORT=5433 make pg-up` — the Makefile prints the same hint pattern on failure.
**Source:** `Makefile:pg-up` target

### Symptom: `make api` connects but runs return Redis errors immediately
**Cause:** `REDIS_PORT` was overridden for `make up` but not for `make api`, so the API dials the default `redis://localhost:6379/0`.
**Fix:** Use the same override for both: `REDIS_PORT=6390 make up && REDIS_PORT=6390 make api`. An explicit `REDIS_URL` always wins over derived port.
**Source:** `Makefile:api` target comment

## Store backend selection (`RESEARCH_CREW_STORE`)

### Symptom: `make api` starts but runs don't persist between restarts
**Cause:** Default store is `memory` (in-process); it vanishes with the process.
**Fix:** Set `RESEARCH_CREW_STORE=hash` (or `streams` or `postgres`) and ensure the backing service is up.
**Source:** `src/research_crew/store/` factory, README Redis Streams section

### Symptom: `RESEARCH_CREW_STORE=streams` and steps show as undelivered
**Cause:** `fakeredis` < 2.35.1 has incomplete Streams support (XPENDING / XCLAIM may be stubs).
**Fix:** `uv sync` with the locked deps — `fakeredis>=2.35.1` is pinned; upgrading fixes the test and production paths simultaneously.
**Source:** loop_history.md Round 11; `pyproject.toml` fakeredis pin

### Symptom: `RESEARCH_CREW_STORE=postgres` raises `ImportError` at startup
**Cause:** `asyncpg` is a soft dependency — absent unless the `[real]` extras group was installed.
**Fix:** `uv sync --extra real` — the import only fails at pool creation, not at module import, so the error appears late.
**Source:** README Postgres run history section

## Authentication / WebSocket

### Symptom: `GET /runs/{id}` returns `401 Unauthorized`
**Cause:** `RESEARCH_API_TOKEN` is set in the server environment but the curl / client is not sending `Authorization: Bearer <token>`.
**Fix:** Pass `-H "Authorization: Bearer $RESEARCH_API_TOKEN"` or set the token query param `?token=<value>` for WebSocket connections.
**Source:** README Auth + rate limiting section

### Symptom: WebSocket `/runs/{run_id}/stream` closes immediately with code 1008
**Cause:** Bearer token mismatch — the WebSocket auth is checked on the initial HTTP upgrade request. Browser `WebSocket` API can't send custom headers; use `?token=<value>` instead.
**Fix:** Connect as `ws://localhost:8000/runs/<id>/stream?token=<value>`.
**Source:** README WebSocket protocol section

## Missing optional extras (`[real]`)

### Symptom: `uv run research --use-inngest "..."` raises `ModuleNotFoundError: inngest`
**Cause:** The `inngest` SDK is in the `[real]` optional extras group, not in the base install.
**Fix:** `uv sync --extra real` before running the Inngest path.
**Source:** README Inngest section

### Symptom: Langfuse tracing silently does nothing even with keys set
**Cause:** `langfuse` package missing — the adapter no-ops instead of raising.
**Fix:** `uv sync --extra real`; verify with `python -c "import langfuse"`.
**Source:** README Tracing / No-op fallback section

## Eval / determinism

### Symptom: `evals/REPORT.md` changes on every `make eval` (latency cells differ)
**Cause:** `make eval` without `--deterministic` uses real wall-clock timestamps.
**Fix:** `make eval` already passes `--deterministic` by default (freezes latency to placeholder). Use `make eval-timings` when you want real numbers.
**Source:** `Makefile:eval` target comment

### Symptom: `citation_correctness = 0.000` in REPORT.md
**Cause:** MockAgent returns `example.com` URLs; the eval expects real domain URLs. This is the mock-pipeline floor, not a bug.
**Fix:** Wire real `TavilyAgent` / `BraveAgent` / `ExaAgent` via API keys (see `.env.example`). The number becomes meaningful once real adapters are active.
**Source:** README Evals section; evals/INTERPRETATION.md
