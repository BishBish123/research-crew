# research-crew

> Concurrent multi-agent research service. 5 specialist agents fan out in parallel via a Redis-backed durable workflow with idempotent step semantics + bounded exponential-backoff retries; results merge through a Synthesizer into a single citation-grounded report.

<!-- Hidden until repo is public -->
<!-- [![ci](https://github.com/BishBish123/research-crew/actions/workflows/ci.yml/badge.svg)](https://github.com/BishBish123/research-crew/actions/workflows/ci.yml) -->
[![python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](pyproject.toml)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## What it actually does

```
                 HTTP POST /research { "question": "..." }
                                  │
                                  ▼
                         ┌────────────────┐
                         │  FastAPI       │
                         └───────┬────────┘
                                 │ enqueue background task
                                 ▼
                  ┌───────────────────────────────┐
                  │     WorkflowEngine.run_parallel│
                  │   ┌─────────────────────────┐ │
                  │   │  asyncio.gather over 5  │ │
                  │   │  agents:                │ │
                  │   │    web_search           │ │
                  │   │    scholar              │ │
                  │   │    code                 │ │
                  │   │    news                 │ │
                  │   │    wikipedia            │ │
                  │   └────────────┬────────────┘ │
                  │   each step:                  │
                  │     - retry × 3 (exp backoff) │
                  │     - dedup by H(run+agent+q) │
                  │     - per-step wall timeout   │
                  └────────────┬──────────────────┘
                               │
                               ▼
                  ┌─────────────────────────┐
                  │ Synthesizer (stitch)    │
                  │   merge → dedupe by url │
                  │   → cite → markdown     │
                  └────────────┬────────────┘
                               ▼
                       ┌──────────────┐
                       │   Redis      │  run + steps + idempotency cache
                       └──────────────┘
```

Every run produces a `ResearchReport` with the per-agent summaries, deduped citations across all sources, and a step-by-step audit (`GET /runs/{id}`).

## Why these design choices

The brief asked for Inngest + Redis Streams + LangGraph. The repo ships both the *hand-rolled contracts* those engines provide **and** a real Inngest integration you can run locally today — no cloud account required.

| Brief | Shipped here | Production swap |
| --- | --- | --- |
| Inngest durable workflow | `WorkflowEngine` (default) **+ `InngestWorkflow`** via `--use-inngest` | Already real Inngest. Point `INNGEST_DEV_SERVER_URL` at your hosted instance |
| Redis Streams consumer groups | `RedisRunStore` with persistent run/step keys | Already real Redis. Swap `RedisRunStore` for `RedisStreamRunStore` for at-least-once fan-in |
| LangGraph synthesizer | `StitchSynthesizer` deterministic + plug-point Protocol | Drop in a `LangGraphSynthesizer` implementing `Synthesizer.synthesize(...)` |
| Tavily / Brave / Exa search | `MockAgent` with deterministic blake2b results | Implement `Agent.search(...)` with a `httpx` call to the real API |
| Locust load test | `load/locustfile.py` ready to run | `make up && make api && make load` |

The architecture is the artifact. Live API keys are runtime config — not what a portfolio review should grade.

## Inngest

The repo ships a real `inngest-py` integration alongside the default hand-rolled engine. Both implement the same `WorkflowProtocol` and produce identical `AgentResult` / `RunStatus` shapes.

### Two workflow modes

| Mode | Default | When to use |
| --- | --- | --- |
| `WorkflowEngine` | Yes | Offline, CI, no external dependencies |
| `InngestWorkflow` | `--use-inngest` | Local dev with Inngest dev server; production Inngest cloud |

### Running the Inngest path locally

No cloud account needed — the Inngest dev server runs entirely locally via `npx`.

```bash
# Terminal 1: start the Inngest dev server
make inngest-dev
# (or directly: npx inngest-cli@latest dev)
# Dev UI: http://localhost:8288

# Terminal 2: install the inngest SDK extra, then run with --use-inngest
uv sync --extra real
export INNGEST_DEV_SERVER_URL=http://localhost:8288
uv run research --use-inngest "what is python"
```

### Env vars

| Var | Default | Purpose |
| --- | --- | --- |
| `INNGEST_DEV_SERVER_URL` | `http://localhost:8288` | URL of the local Inngest dev server |
| `INNGEST_ENV` | `dev` | Set to `production` when deploying to Inngest cloud |

### What changes operationally

- With `--use-inngest`: steps are scheduled, retried, and observed through the Inngest dev server. The Inngest UI at `localhost:8288` shows each step, its inputs/outputs, and retry history.
- Without `--use-inngest` (default): the `WorkflowEngine` handles all durability semantics in-process. No external services needed.
- The `RunStore` (Redis or in-memory) is still used for run/step persistence in both modes when `record_step` is wired in.

## Prerequisites

- **Python 3.11 or 3.12** — pinned in `pyproject.toml`; 3.13 is not yet exercised in CI.
- **[`uv`](https://docs.astral.sh/uv/)** — the project's package manager. `make install` shells out to `uv sync`; the `make api` / `make test` / `make load` targets all wrap `uv run <cmd>`, so anywhere you see `uv run research`, `uv run pytest`, etc., that's `uv` resolving the locked virtualenv before invoking the underlying tool — no global `pip install` step required.
- **Docker** — only needed for `make up` (brings up the Redis 7.4 container in `docker-compose.yml`). The CLI path (`uv run research ...`) and the unit tests run against an in-memory store, so Docker is optional unless you're exercising the API or the integration suite.

## Quick start

```bash
git clone https://github.com/BishBish123/research-crew.git
cd research-crew
make install

# Bring Redis up
make up
# If port 6379 is already taken: REDIS_PORT=6390 make up

# Run a single research job from the CLI (memory store, no API needed)
uv run research "what is python"

# Same with retry-path exercise (forces re-attempts)
uv run research "test retries" --failure-rate 0.5

# Run the API. `make api` derives REDIS_URL from REDIS_PORT, so the
# same override flows through: REDIS_PORT=6390 make api & — an
# explicit REDIS_URL still wins.
make api &

# Wait for the service to be ready before curling. `make api &`
# returns before uvicorn finishes startup; the loop polls /health.
until curl -fsS http://localhost:8000/health > /dev/null; do sleep 0.2; done

curl -sS -X POST -H 'content-type: application/json' \
    -d '{"question":"how does Inngest handle step retries"}' \
    http://localhost:8000/research

# Subset of agents (the `agents` field is optional; default = all 5).
curl -sS -X POST -H 'content-type: application/json' \
    -d '{"question":"how does Inngest handle step retries","agents":["web_search","scholar"]}' \
    http://localhost:8000/research

# Get the run status. NOTE: with RESEARCH_API_TOKEN set this returns
# 401 unless you also pass `-H "Authorization: Bearer $RESEARCH_API_TOKEN"`;
# the Quick Start above runs the API unauthenticated.
curl -sS http://localhost:8000/runs/<run_id> | jq .

# Interactive API explorer + machine-readable schema:
#   http://localhost:8000/docs       (Swagger UI)
#   http://localhost:8000/redoc      (ReDoc)
#   http://localhost:8000/openapi.json
```

## Browser frontend

The API ships a static single-page UI served at `http://localhost:8000/`.

### Starting the UI

```bash
# Start Redis (if not already running)
make up

# Start the API + frontend
uv run research-api
# or: make api
```

Then open `http://localhost:8000/` in your browser.

### UI elements

| Element | Description |
| --- | --- |
| Question input + Submit | Type a research question and press **Research**. Triggers `POST /research` and receives a `run_id`. |
| Agent fan-out grid | 5 cards (Web Search, Scholar, Code, News, Wikipedia), one per agent. Each card shows the current status badge (`pending → running → succeeded / failed`), step count, and per-agent latency. Cards update in real time via the WebSocket `/runs/{run_id}/stream` endpoint. |
| Cancel button | Closes the WebSocket and sends `POST /runs/{id}/cancel`. |
| Synthesized Report | When the run completes successfully, the markdown summary and citations from the `ResearchReport` are rendered in a `<pre>` block below the cards. |
| Run error | When a run fails, the error string from `RunStatus.error` is displayed. |

### UX states

| State | Description |
| --- | --- |
| **idle** | Initial state; form enabled, no run in flight. |
| **submitting** | `POST /research` in progress; submit button disabled. |
| **streaming** | WebSocket open; agent cards updating live; Cancel button visible. |
| **complete** | Terminal state reached; report rendered; form re-enabled. |
| **error** | Network or API error; message shown in-form; form re-enabled. |

### Mock mode (no API keys)

When `OPENAI_API_KEY` / search API keys are absent the agents fall back to
their stub implementations (the same path the test suite uses). Runs still
complete and a synthetic report is returned — useful for checking the UI
locally without real credentials.

### WebSocket protocol

`app.js` connects with a native `WebSocket` to `ws://host/runs/{run_id}/stream`.
Messages are JSON objects with a `type` field:

| `type` | Description |
| --- | --- |
| `snapshot` | Sent immediately on connect; contains the current `RunStatus`. |
| `step` | Sent for each `StepRecord` event as agents execute. |
| `heartbeat` | Sent every 15 s to keep the connection alive. |
| `done` | Sent when the run reaches a terminal state; `app.js` then fetches `GET /runs/{id}` for the final report. |

## API

| Verb | Path | Body | Returns |
| --- | --- | --- | --- |
| `GET` | `/health` | — | `{ "status": "ok", "redis": "up", "active_runs": 0, "shadow_size": 0 }`; `503 { "detail": "redis unavailable: ..." }` if Redis is unreachable |
| `POST` | `/research` | `{ "question": "...", "agents": ["web_search", ...] }` | `202 { "run_id": "...", "status_url": "http://host/runs/..." }` |
| `GET` | `/runs/{id}` | — | `RunStatus` (state, per-step audit, embedded `ResearchReport`) |
| `GET` | `/docs` | — | FastAPI's Swagger UI (interactive API explorer). |
| `GET` | `/redoc` | — | ReDoc rendering of the same OpenAPI spec. |
| `GET` | `/openapi.json` | — | Raw OpenAPI 3 schema. |

The `steps` array on `RunStatus` is an append-only audit log: every
agent attempt writes one `running` row when it starts and one terminal
row (`succeeded` / `failed`) when it ends, so a retried step shows up
as multiple `running` + terminal pairs rather than mutating a single
record.

`/health` includes a workload snapshot beyond the basic Redis ping:

* `active_runs` — count of run records currently in the `RUNNING`
  state. Computed via a SCAN over `{prefix}:run:*` keys; returns
  `null` if the store isn't a `RedisRunStore` (e.g. tests using the
  in-memory backend) or if the SCAN failed (the probe still 200s).
* `shadow_size` — number of terminal RunStatus entries the in-process
  shadow is currently holding. Non-zero means the bg task has been
  recovering from a Redis outage; operators want to alert on this.

### Auth + rate limiting

> The Quick Start above runs unauthenticated. Set `RESEARCH_API_TOKEN`
> only when you're ready to enable auth — the Quick Start curl examples
> will start returning 401 without an `Authorization` header once it's set.

`/research` and `/runs/{id}` are gated behind a bearer token when
`RESEARCH_API_TOKEN` is set; `/health` stays open so a load balancer can
probe without a credential.

```bash
export RESEARCH_API_TOKEN=$(openssl rand -hex 32)
make api &
curl -sS -H "Authorization: Bearer $RESEARCH_API_TOKEN" \
     -H 'content-type: application/json' \
     -d '{"question":"how does Inngest handle step retries"}' \
     http://localhost:8000/research
```

If `RESEARCH_API_TOKEN` is unset the service runs unauthenticated and
the lifespan logs a loud `api.auth_disabled` warning — that's the dev
path; in production the operator MUST set the env var.

POST `/research` is additionally rate-limited per client IP
(sliding-window counter, 60s window). Default `10 req/min/IP`;
override via `RESEARCH_RATE_LIMIT_PER_MIN`. Exhausted callers get
`429 Too Many Requests` plus a `Retry-After` header in seconds.

When the service runs behind a reverse proxy (ALB, nginx, Cloudflare),
set `RESEARCH_TRUSTED_PROXIES` to a CSV of the proxy IPs. The limiter
will then key on the first non-trusted IP from the
`X-Forwarded-For` chain instead of the proxy's own IP. With this
unset (the default) the header is ignored, so a direct caller cannot
spoof XFF to get a fresh bucket.

## API documentation

Machine-readable artifacts live in `docs/` and are regenerated with `make openapi`.

| Artifact | Path | Use |
| --- | --- | --- |
| OpenAPI 3.1 spec | [`docs/openapi.json`](docs/openapi.json) | Import into any OpenAPI tool (Insomnia, Stoplight, etc.) |
| Postman collection | [`docs/postman_collection.json`](docs/postman_collection.json) | Import directly into Postman (File → Import) |
| Swagger UI | `http://localhost:8000/docs` | Interactive browser explorer (served by FastAPI) |
| ReDoc | `http://localhost:8000/redoc` | Read-only rendered spec |

> **CDN dependency**: FastAPI's built-in Swagger UI loads its JS and CSS from
> `https://cdn.jsdelivr.net`. An internet connection is required to use `/docs`.
> For fully offline documentation, use the OpenAPI JSON with a locally-installed
> tool such as [Redocly CLI](https://redocly.com/docs/cli/) or import the
> Postman collection into Postman Desktop.

To refresh the tracked artifacts after schema changes:

```bash
make openapi
```

## Postgres run history

`PostgresRunStore` is a durable long-term archive backend backed by Postgres or Neon. Use it when runs need to survive beyond Redis's working-set TTL or when you want SQL-queryable history.

### Schema

| Table | Purpose |
| --- | --- |
| `runs` | One row per run — run_id (PK), status, query, report_json (JSONB), timestamps |
| `steps` | Append-only step audit log — UNIQUE(run_id, sequence) enforces idempotency |
| `step_dedup` | Idempotency cache — dedup_key (PK) → AgentResult JSONB |

The DDL is applied idempotently by `await store.setup()` (safe to call multiple times).

### Activating

```bash
# Start Postgres container
make pg-up  # override port: PG_PORT=5433 make pg-up

# Point the service at it
export RESEARCH_CREW_STORE=postgres
export RESEARCH_PG_DSN=postgresql://research:research@localhost:5432/research
make api
```

### Factory contract

| `RESEARCH_CREW_STORE` | `RESEARCH_PG_DSN` | Result |
| --- | --- | --- |
| `postgres` | set | `PostgresRunStore(dsn=…)` — pool created lazily on `setup()` |
| `postgres` | unset | `PostgresRunStore(dsn="postgresql://research:research@localhost:5432/research")` |

`asyncpg` is a soft dependency: if it is absent the import fails at pool creation time (not at `import research_crew`) so callers that never set `RESEARCH_CREW_STORE=postgres` are unaffected.

### Migrating existing data into Postgres

```python
import redis.asyncio as aioredis
import asyncpg
from research_crew.store.migrate import (
    migrate_redis_hash_to_postgres,
    migrate_redis_streams_to_postgres,
)

async def run():
    r = aioredis.from_url("redis://localhost:6379/0", decode_responses=True)
    pool = await asyncpg.create_pool("postgresql://research:research@localhost/research")

    # From hash store
    count = await migrate_redis_hash_to_postgres(r, pool, "my-run-id")
    # From streams store
    count = await migrate_redis_streams_to_postgres(r, pool, "my-run-id")

    print(f"Migrated {count} step(s)")
    await r.aclose()
    await pool.close()
```

Both helpers are idempotent (upsert on `runs`, `ON CONFLICT DO NOTHING` on `steps`).

## Cross-run semantic dedup

The synthesizer performs URL-based deduplication within a single run.  `SemanticDedup` extends this with **cross-run content deduplication**: when two runs return different URLs that carry paraphrased versions of the same content, the synthesizer detects and drops the repeat.

### How it works

1. Before adding a citation to the final report the synthesizer calls `is_duplicate(text)`.
2. The implementation embeds the citation snippet via a 384-dim sentence embedding (or a deterministic hash-based fake if `sentence-transformers` is not installed) and runs an ANN nearest-neighbour search in Postgres using `pgvector`'s `ivfflat` index.
3. If cosine similarity ≥ `threshold` (default `0.85`) the citation is skipped and a `synthesizer.semantic_dedup_skip` debug line is logged.
4. After synthesis, surviving citations are recorded via `add_seen` so future runs can compare against them.

The default `NullDedup` is a no-op: all citations pass through and **existing behaviour is unchanged** when `RESEARCH_PG_DSN` is unset.

### Schema

```sql
CREATE TABLE IF NOT EXISTS seen_content (
    id         SERIAL PRIMARY KEY,
    run_id     TEXT        NOT NULL,
    text_hash  TEXT        NOT NULL,
    embedding  vector(384),
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS seen_content_embedding_idx
    ON seen_content USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
```

### Activating

```bash
export RESEARCH_PG_DSN=postgresql://research:research@localhost:5432/research
make api
```

### Factory contract

| `RESEARCH_PG_DSN` | Result |
| --- | --- |
| set | `PgVectorSemanticDedup` — ANN search via pgvector |
| unset (default) | `NullDedup` — no-op, zero deps |

### Optional embedding dependency

Real embeddings use `sentence-transformers` (`all-MiniLM-L6-v2`, 384 dims).  If the package is absent the module soft-fails to a deterministic hash-based fake encoder.  To enable real embeddings:

```bash
pip install sentence-transformers
# or add to your venv extras — see [embed] note in pyproject.toml
```

The `NullDedup` default and the `RESEARCH_PG_DSN`-gate mean neither `asyncpg` nor `sentence-transformers` are required for the standard CI/test path.

## Redis Streams

`RedisStreamRunStore` is an opt-in alternative to the default hash-based store. It uses Redis Streams (XADD / XREADGROUP / XACK / XPENDING) for the per-run step audit log and a per-agent consumer-group fan-out pattern for distributing work.

### When to use streams vs hash

| | `hash` (default) | `streams` |
| --- | --- | --- |
| Step audit log | Redis List (`RPUSH`) | Redis Stream (`XADD`) |
| Inter-agent fan-out | in-process `asyncio.gather` | per-agent consumer groups |
| At-least-once delivery | not provided | via XREADGROUP + XACK |
| Orphan recovery | lifespan SCAN reconciler | XPENDING + XCLAIM |
| Use when | simple deployment, default | multi-worker, at-least-once |

### Activating the streams store

Set `RESEARCH_CREW_STORE=streams` before starting the API or CLI:

```bash
RESEARCH_CREW_STORE=streams make api
```

The factory (`make_run_store()`) reads this env var:

| `RESEARCH_CREW_STORE` | Backend |
| --- | --- |
| `hash` (default / unset) | `RedisRunStore` — original hash + list |
| `streams` | `RedisStreamRunStore` — Redis Streams |
| `memory` | `InMemoryRunStore` — in-process, tests only |

### Sample workflow

```bash
make up                           # start Redis container
RESEARCH_CREW_STORE=streams make api &
until curl -fsS http://localhost:8000/health > /dev/null; do sleep 0.2; done
curl -sS -X POST -H 'content-type: application/json' \
    -d '{"question":"how does Redis Streams work"}' \
    http://localhost:8000/research
```

### Migrating existing hash data

If you have runs written by the hash store and want to switch to streams:

```bash
# Programmatic (Python)
from research_crew.store.migrate import migrate_hash_to_streams
import redis.asyncio as aioredis, asyncio

async def run():
    r = aioredis.from_url("redis://localhost:6379/0", decode_responses=True)
    count = await migrate_hash_to_streams(r, "my-run-id")
    print(f"Migrated {count} step(s)")
    await r.aclose()

asyncio.run(run())

# CLI
REDIS_URL=redis://localhost:6379/0 python -m research_crew.store.migrate <run_id>
```

The helper copies the JSON blob to HSET format, XADDs each list entry to the stream, then renames the old list key to `…:steps.migrated` so the mismatch check no longer fires.

### Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `RESEARCH_API_TOKEN` | *(unset)* | Bearer token enforced on `/research` and `/runs/{id}`. Unset = unauthenticated dev mode (loud warning logged at startup). |
| `RESEARCH_DEV_MODE` | *(unset)* | When truthy (`1`, `true`, `yes`, `on`), demotes the unauthenticated-mode `api.auth_disabled` log from WARNING to INFO. Local dev opt-out only — does not affect auth enforcement. |
| `RESEARCH_RATE_LIMIT_PER_MIN` | `10` | Per-IP sliding-window cap on `POST /research`. Exhausted callers get `429` + `Retry-After`. |
| `RESEARCH_TRUSTED_PROXIES` | *(empty)* | CSV of proxy IPs. When the immediate peer is in this set, the limiter keys on the first non-trusted IP from `X-Forwarded-For`. Empty = ignore XFF (safe default for direct exposure). |
| `REDIS_URL` | `redis://localhost:6379/0` | Connection URL for the run store. Read by both the API service AND the CLI when the latter is pointed at a Redis-backed store. `make api` derives this from `REDIS_PORT` if unset, so `REDIS_PORT=6390 make up && REDIS_PORT=6390 make api` works without setting both. |
| `REDIS_PORT` | `6379` | Host port that `make up` binds the Redis container to (override when 6379 is taken on the host). `make api` reads it to derive a matching `REDIS_URL` when none is set. CLI / hand-rolled `uvicorn` invocations do **not** read it — use `REDIS_URL` directly there. |
| `RESEARCH_REDIS_PREFIX` | `research` | Key prefix for run / step / cache keys. Set per environment to share one Redis without cross-talk. |
| `RESEARCH_HEARTBEAT_STALE_S` | `120` | Heartbeat-staleness threshold for the lifespan orphan reconciler. A peer's `RUNNING` run is left alone when its heartbeat age is less than or equal to this value; the run is considered abandoned only when the age is strictly greater than (older than) the threshold. Bumps every 30s while live. |
| `RESEARCH_SHADOW_MAX` | `10000` | Maximum entries in the in-process terminal-state shadow cache (only populated while the run store is unreachable). Oldest entry evicted on overflow. |
| `RESEARCH_MAX_QUESTION_LEN` | `5000` | Hard cap on the `POST /research` `question` field. Requests beyond this length get `422`. |

Bind host / port are CLI flags on the entrypoint, not env vars:

```bash
uv run research-api --host 0.0.0.0 --port 8000

# Behind a reverse proxy: tell uvicorn which IPs to trust X-Forwarded-* from
# (otherwise the absolute `status_url` in API responses uses the internal scheme/host).
uv run research-api --host 0.0.0.0 --port 8000 \
  --forwarded-allow-ips '127.0.0.1,10.0.0.0/8'
```

The `--forwarded-allow-ips` flag is wired through to `uvicorn.run(..., proxy_headers=True)`,
so any peer in the trusted list has its `X-Forwarded-Proto` / `X-Forwarded-Host` honoured by
`request.url_for(...)`. Default is loopback only — sufficient for local dev, **not** for a
reverse-proxied production deployment.

`--host 0.0.0.0` is what you want inside a container or to expose the
service on the LAN; the default `127.0.0.1` is loopback-only.

## Single-process, horizontally-scalable design

The service runs as one FastAPI process — a research call fans out
across asyncio inside that single process, not across machines. What's
shipped is concurrent fan-out plus a durability contract that lets
multiple instances share *state* safely, with one important caveat
about *execution* spelled out in Limitations below.

* **Idempotency cache lives in Redis** (`step:{dedup_key}`), so two
  instances behind a load balancer never double-execute the same
  `(run_id, agent, question)` triple. The second arrival short-circuits
  on the cache and returns the result the first one wrote.
* **Run records and step audits** live in shared Redis, so any
  instance can serve `GET /runs/{id}` regardless of which instance
  accepted the original POST. Cross-instance reads are stateless.
* **Background execution is bound to the accepting instance.** When a
  POST `/research` is accepted, the run is executed by an in-process
  FastAPI BackgroundTask on that same instance. `GET /runs/{id}` is
  cross-instance, but the worker is not.

Promoting this to a true distributed worker pool is a swap of the
in-process BackgroundTask for an Inngest function or a Redis Streams
consumer group; the store contract is already shaped for it. See
`ARCHITECTURE.md` § "Future work".

## Limitations

* **Execution is single-instance, state is multi-instance.** POST
  `/research` and GET `/runs/{id}` are stateless across instances
  (everything lives in Redis). The actual *workflow execution*,
  however, runs as an in-process FastAPI BackgroundTask on the
  instance that accepted the submit. This is documented above; the
  implication is that a process restart mid-run strands that run with
  no worker behind it.
* **Best-effort orphan reconciliation on startup.** At lifespan
  startup the API SCANs `{prefix}:run:*` and flips any RUNNING record
  to FAILED with `error: "abandoned by previous process"` so polling
  clients see a terminal answer instead of looping forever. This is
  *not* a durable retry — the run isn't re-executed, it's marked
  failed. A proper durable worker queue is on the roadmap (Inngest /
  Redis Streams consumer group).
* **Best-effort terminal-state shadow.** If the bg task finishes work
  but the store rejects the terminal write (Redis outage), the
  RunStatus is stashed in an in-process bounded shadow so a
  subsequent GET still surfaces a terminal state. The shadow is
  per-process and does not survive a restart.
* **In-process rate limiter.** `RESEARCH_RATE_LIMIT_PER_MIN` is a
  per-process counter; multi-instance deployments will see N times the
  per-IP cap until the limiter is moved into Redis.

## Load testing

`load/locustfile.py` posts a question, polls the run-status endpoint, and exercises `/health` for noise. The agent layer is mocked deterministically so the load test isolates the *workflow plumbing* — what you'd run before signing up for paid search APIs to find out whether the orchestration scales.

### Interactive UI

```bash
make api &
make load   # opens Locust on http://localhost:8089
```

### Headless deterministic run (no Redis required)

```bash
make load-real
# or with custom pressure:
bash scripts/run_load_test.sh --users 50 --duration 1m
```

`scripts/run_load_test.sh` brings up the API with `RESEARCH_CREW_STORE=memory` (no Redis needed), runs Locust headlessly, and updates [`docs/load-test-results.md`](docs/load-test-results.md) with **real measured numbers** via `load/parse_results.py`.

See [`docs/load-test-results.md`](docs/load-test-results.md) for the latest run (10 users, 30s, memory store):

| Endpoint | p50 (ms) | p95 (ms) | p99 (ms) | RPS | Failure rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `POST /research` | 9 | 24 | 48 | 7.3 | 0.0% |
| `GET /runs/{id}` | 17 | 43 | 210 | 7.3 | 0.0% |
| `GET /health` | 310 | 1100 | 1300 | 1.6 | 0.0% |

> `/health` p95 is high because it scans active runs in Redis (here, in-memory); under real Redis it is sub-millisecond.

### Script options

```
-u, --users N       Concurrent users (default: 10)
-d, --duration DUR  Run duration (default: 30s)
-p, --port PORT     API bind port (default: 8765)
-o, --out FILE      Markdown output file (default: docs/load-test-results.md)
    --no-update     Print table to stdout only
```

## Tests

```bash
make test                # unit tests (workflow, synthesizer, store, API, agents, properties)
make test-integration    # placeholder for future integration tests; currently 0 collected
make check               # ruff check + ruff format --check + mypy --strict (matches CI)
```

What's covered today:

- **Workflow durability** — idempotency cache hit, retry-then-succeed, timeout-then-retry, exhausted retries, per-step timeout, parallel partial-failure isolation, cancellation cleanup.
- **Idempotency-key properties** — Hypothesis property tests assert determinism + per-component sensitivity across 200 random examples per property.
- **Store contract** — every method round-trips on both `InMemoryRunStore` and `RedisRunStore` (via fakeredis), parametrized over both.
- **Synthesizer correctness** — citation dedupe by URL (incl. case / trailing-slash / `www.` collapse), failure surfacing, all-failed empty path, per-agent caps, query-string preservation.
- **API contracts** — submit → background → poll → report renders; 404 on unknown run; 422 on bad payloads (missing, too long, wrong type, unknown agent); 503 on uninitialised store; `/health` shape contract.
- **Agent base** — Protocol conformance, determinism, latency simulation, failure injection.

## Evals

A deterministic eval harness in `evals/` measures pipeline quality against a **20-question golden set**
covering six categories: **factual**, **comparative** ("X vs Y"), **list**, **trend** (2024–2026
recency questions), **oos** (out-of-scope / failure-likely), and **refusal** (ill-defined questions
where the right answer is a clarifying refusal).

Two metrics per question:
- **citation_correctness** — fraction of cited URLs from an expected domain (e.g., `python.org`).
- **keyphrase_coverage** — fraction of expected concept phrases found in the synthesized report.

```bash
make eval            # runs harness, regenerates evals/REPORT.md (latency cells fixed → no git churn)
make eval-timings    # same run but with real wall-clock latency numbers
```

`make eval` passes `--deterministic` by default so latency cells in `evals/REPORT.md` are replaced
with a fixed placeholder (`0 ms (mock pipeline)`), keeping the file stable across runs and preventing
spurious `git status` noise.  Use `make eval-timings` to see real timings.

Current scores reflect the **mock-pipeline floor**: `citation_correctness = 0.000` (MockAgent
returns `example.com` URLs that never match real domains) and `keyphrase_coverage ≈ 0.78` (MockAgent
echoes the question back, so keyphrases that appear in the question itself score 1.0).  These numbers
become meaningful quality signals once real search adapters and an LLM synthesizer are wired in.
See [`evals/INTERPRETATION.md`](evals/INTERPRETATION.md) for the full discussion.

## Chaos testing

A chaos engineering harness in `src/research_crew/chaos.py` injects faults directly into
`WorkflowEngine` — no external services are contacted. Five scenarios are tested:

| Scenario | What it injects |
| --- | --- |
| `baseline` | 0% failures, no jitter — clean-path floor |
| `flaky-agents` | 30% per-agent failure probability |
| `slow-agents` | 0–2000ms uniform latency jitter |
| `redis-flaps` | 10% probability the idempotency cache raises on every call |
| `cascading-failures` | 50% failures + 1500ms jitter (combined stress) |

Each scenario is run N times; the harness aggregates **success rate**, **p50/p95/p99 latency**,
and **mean retry count** per scenario, then writes a full histogram report to
[`docs/CHAOS.md`](docs/CHAOS.md).

```bash
make chaos              # runs all 5 scenarios × 20 runs, writes docs/CHAOS.md
```

Or drive the harness directly:

```bash
uv run python -m research_crew.chaos --scenarios all --runs 20 --deterministic --out docs/CHAOS.md
uv run python -m research_crew.chaos --scenarios flaky-agents --runs 5
uv run python -m research_crew.chaos --help
```

See [`docs/CHAOS.md`](docs/CHAOS.md) for the latest report with per-scenario numbers and the
"What We Learned" analysis comparing baseline vs each fault mode.

## Tracing

Langfuse distributed tracing is supported via the env-key-gated adapter in
`src/research_crew/observability/langfuse.py`. When the required keys are set,
every research run emits:

* **One trace** per `POST /research` call, named `research-run`, carrying
  the question as input and `"succeeded"` / `"failed"` as output.
* **One span** per agent fan-out step for each terminal attempt
  (`succeeded`, `failed`, or `cached`). `RUNNING` records are intentionally
  omitted — they are intermediate state, not closed events.

When the keys are absent the adapter is a no-op; the default workflow path
is unchanged.

### Required env vars

| Variable | Purpose |
| --- | --- |
| `LANGFUSE_PUBLIC_KEY` | Public key from your Langfuse project. |
| `LANGFUSE_SECRET_KEY` | Secret key from your Langfuse project. |
| `LANGFUSE_HOST` | *(optional)* Override for self-hosted deployments; defaults to `https://cloud.langfuse.com`. |

### Getting a Langfuse account

Sign up at **<https://cloud.langfuse.com>** (EU or US region) or self-host
the open-source stack — <https://langfuse.com/self-hosting>.

### Enabling tracing

```bash
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
# Optional: LANGFUSE_HOST=https://your-selfhosted-langfuse.example.com

uv run research "what is python"
# or
make api
```

### Installing the SDK

The `langfuse` package is **not** in base dependencies — it lives under
the `real` optional extras group so the default install stays lean:

```bash
uv sync --extra real
# or: pip install 'research-crew[real]'
```

### No-op fallback

If neither key is set, or if the `langfuse` package is missing from the
venv, all tracer methods return immediately without error. The workflow
continues normally; no exception is raised. This means you can add
`LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` to your environment at any
time without code changes.

## Design docs

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — layered shape, design rationale, future work.
- [`docs/adrs/ADR-001-redis-store.md`](docs/adrs/ADR-001-redis-store.md) — three-keyspace Redis design.
- [`docs/adrs/ADR-002-idempotency-key.md`](docs/adrs/ADR-002-idempotency-key.md) — `blake2b` digest derivation + properties.
- [`docs/adrs/ADR-003-retry-policy.md`](docs/adrs/ADR-003-retry-policy.md) — bounded exponential backoff.
- [`docs/load-test-results.md`](docs/load-test-results.md) — Locust scenario shape + reference numbers.

## Layout

```
src/research_crew/
  models.py        Pydantic types (Citation, AgentResult, ResearchReport, RunStatus)
  errors.py        Domain-specific exception hierarchy
  agents/          Agent protocol + MockAgent + default crew of 5
  workflow.py      WorkflowEngine: parallel + retries + idempotency + timeouts
  synthesizer.py   StitchSynthesizer (no LLM); plug-point Protocol for LLM swap
  store.py         InMemoryRunStore + RedisRunStore (RunStore Protocol)
  api.py           FastAPI: POST /research, GET /runs/{id}, /health
  cli.py           `research <question>` — single-run, no HTTP

load/locustfile.py Locust scenario for the orchestration layer
docker-compose.yml Redis 7.4
docs/              ADRs + load-test write-up
tests/             unit tests (incl. Hypothesis property tests + store contract)
```

## Real search adapters

Three real HTTP adapters are implemented and wired into the pipeline. They are
activated automatically when the corresponding API key is present in the
environment — no code change required.

| Slot (`AgentName`) | Adapter | Endpoint / filter | Key |
| --- | --- | --- | --- |
| `web_search` | `TavilyAgent` | Generic web search (`/search`) | `TAVILY_API_KEY` |
| `news` | `BraveAgent(endpoint="news")` | Brave news endpoint (`/res/v1/news/search`) — real news index, not generic web | `BRAVE_API_KEY` |
| `scholar` | `ExaAgent(category="research paper")` | Exa filtered to academic / peer-reviewed sources (`category="research paper"`) | `EXA_API_KEY` |
| `code` | `MockAgent` | *(no adapter yet)* | — |
| `wikipedia` | `MockAgent` | *(no adapter yet)* | — |

Each adapter targets the endpoint that matches its slot label: `BraveAgent` queries
Brave's dedicated news index (not the generic web path), and `ExaAgent` passes
`category="research paper"` to restrict results to academic sources. `TavilyAgent`
provides broad web coverage for the `WEB_SEARCH` slot unchanged.

When a key is absent the slot falls back to `MockAgent`, so the fan-out
always returns exactly 5 results and the eval harness shape is stable.

Copy `.env.example` to `.env` and populate the keys you have:

```bash
cp .env.example .env
# edit .env with your keys
source .env   # or use direnv / dotenv-cli
make api
```

See `.env.example` for the full list of supported variables including
Langfuse tracing keys.

## Deploy

Deploy to [Fly.io](https://fly.io) using the included `Dockerfile`, `fly.toml`, and `scripts/deploy.sh`.

### Quickstart

```bash
# 1. Install flyctl
curl -L https://fly.io/install.sh | sh

# 2. Authenticate
fly auth login

# 3. Create the app (one-time)
fly apps create <your-app-name>

# 4. Edit fly.toml — replace REPLACE-ME with your app name
sed -i '' 's/research-crew-REPLACE-ME/<your-app-name>/' fly.toml

# 5. Provision a Redis instance (Upstash, Redis Cloud, or fly.io Redis)
# Then set the URL:
export REDIS_URL=redis://...

# 6. Populate .env with your secrets (see .env.example), then deploy:
./scripts/deploy.sh --app <your-app-name>

# Or with FLY_APP_NAME set:
FLY_APP_NAME=<your-app-name> ./scripts/deploy.sh
```

### Required secrets

| Secret | Purpose |
| --- | --- |
| `REDIS_URL` | Run store — required unless using `RESEARCH_CREW_STORE=memory` |
| `RESEARCH_PG_DSN` | Postgres run history (optional) |
| `TAVILY_API_KEY` | Web search agent (optional — falls back to MockAgent) |
| `BRAVE_API_KEY` | News agent (optional) |
| `EXA_API_KEY` | Scholar agent (optional) |
| `LANGFUSE_PUBLIC_KEY` | Distributed tracing (optional) |
| `LANGFUSE_SECRET_KEY` | Distributed tracing (optional) |

`scripts/deploy.sh` reads these from your local `.env` and pushes them as Fly secrets automatically.

### Build locally

```bash
make docker-build   # builds research-crew:latest
make docker-run     # runs on http://localhost:8000 with in-memory store (no Redis)
```

### Machine size and cost estimate

The default `fly.toml` targets Fly's **shared-cpu-1x, 1 GB RAM** tier with `auto_stop_machines = "stop"`. At ~$0.0001/s when running, a lightly used instance that sleeps most of the day costs roughly **$3–5/month**.

- `auto_stop_machines = "stop"` — machine shuts down after all connections close.
- `auto_start_machines = true` — machine wakes on the next request (cold-start ~1–2 s).
- `min_machines_running = 1` — keeps one machine alive to avoid cold starts (increases cost to ~$5–8/month; remove to stay at the low end).

### CI validation

`.github/workflows/docker.yml` builds the image and polls `/health` on every push to `main` — no registry push, no Fly account needed.

## Real-time streaming

`GET /runs/{id}` is a polling endpoint. For live progress you can connect
a WebSocket to `WS /runs/{run_id}/stream` and watch step events arrive as
the agent fan-out executes.

### Protocol

All messages are JSON objects with a `type` field:

| `type`      | When sent                                               |
|-------------|----------------------------------------------------------|
| `snapshot`  | Immediately on connect — current `RunStatus` snapshot.  |
| `step`      | Each time a workflow step fires — `StepRecord` fields.  |
| `heartbeat` | Every 15 s when no step event arrives (keepalive).      |
| `done`      | Run reached terminal state; server closes after this.   |

### Authentication

Send the bearer token as:
- `Authorization: Bearer <token>` request header, **or**
- `?token=<value>` query parameter (required for browser `WebSocket` API).

### Try it with `websocat`

```sh
# 1. Start the API (in another terminal)
make api

# 2. Submit a run
RUN_ID=$(curl -s -X POST http://localhost:8000/research \
  -H 'Content-Type: application/json' \
  -d '{"question":"what is python"}' | jq -r .run_id)

# 3. Stream events
websocat "ws://localhost:8000/runs/$RUN_ID/stream"
```

### CLI watch flag

```sh
uv run research watch "what is python"
# Requires a running API: make api
```

### Python client

```python
from research_crew.client import stream_run

async for msg in stream_run("http://localhost:8000", run_id, token="..."):
    print(msg["type"], msg.get("agent", ""))
    if msg["type"] == "done":
        break
```

## Real-mode verification

The repo ships a local mock-API server stack that closes the "no real HTTP path exercised" gap without requiring real cloud credentials.

### What it proves vs. unit-mocked tests

Unit tests in `tests/test_real_agents.py` patch `httpx.AsyncClient` — no real sockets, no TCP, no wire serialisation. The mock-server integration tests exercise the **full production code path**:

| Layer | Unit tests | Mock-server integration tests |
|---|---|---|
| httpx client lifecycle | Patched mock | Real `AsyncClient` with real TCP connection |
| JSON serialisation | Bypassed | Request body serialised by httpx, deserialised by FastAPI |
| HTTP response parsing | Bypassed | Real status codes / headers / body over the wire |
| Retry logic | Patched side-effect | Real second request to a server that returns `500` then `200` |
| Auth header delivery | Not verified | Server returns `401` if the header is absent — proves the adapter sends it |

### Mock server endpoints

The `tests/mock_servers/` FastAPI app mounts these endpoints:

| Endpoint | Method | Returns |
|---|---|---|
| `/tavily/search` | POST | `tavily_search_response.json` |
| `/brave/web/search` | GET | `brave_web_response.json` |
| `/brave/news/search` | GET | `brave_news_response.json` |
| `/exa/search` | POST | `exa_search_response.json` (or `exa_research_paper_response.json` when `category="research paper"`) |
| `/langfuse/api/public/ingestion` | POST | `{"successes": [...], "errors": []}` |

Each handler validates that the provider's auth header is present. Add `?force=500` to any endpoint to receive a 500 on the first call and 200 on the next (exercises the one-retry loop).

### URL-override mechanism

Each adapter accepts a `base_url` constructor parameter (and a corresponding env var) that redirects it to a local server:

| Adapter | Constructor param | Env var |
|---|---|---|
| `TavilyAgent` | `base_url` | `TAVILY_API_BASE_URL` |
| `BraveAgent` | `base_url` | `BRAVE_API_BASE_URL` |
| `ExaAgent` | `base_url` | `EXA_API_BASE_URL` |

Default production URLs are unchanged when neither is set.

### How to run

```bash
# Run only the mock-server integration tests (no cloud services needed):
make test-integration-mock

# Expected output (17 tests, ~5 s):
# tests/test_real_adapter_integration.py::TestTavilyAdapterIntegration::test_search_returns_expected_citations PASSED
# tests/test_real_adapter_integration.py::TestTavilyAdapterIntegration::test_search_sends_auth_header PASSED
# tests/test_real_adapter_integration.py::TestTavilyAdapterIntegration::test_retry_on_5xx_succeeds PASSED
# ... (14 more)
# 17 passed in ~5s

# Run unit tests (unaffected):
make test      # 510 passed

# Run all integration suites:
make test-integration-all
```

## Honest limitations

- No real search adapters — that's the point of the `Agent` Protocol; pick your search vendor and implement `search()`.
- No LLM synthesizer wired in. The trace shape is identical, so plugging one in is one new class implementing `Synthesizer`.
- No Langfuse — but the `WorkflowEngine`'s `record_step` callback is exactly the right shape to back with a Langfuse span emitter.
- The deduplication is by URL only; semantic dedup (embed citations, cluster by cosine) is the obvious next commit.
- This is a single FastAPI process with `asyncio.gather` fan-out, not a distributed system. The "Single-process, horizontally-scalable design" section above explains the deployment story; for true distribution swap `WorkflowEngine` for an Inngest function, or use Redis Streams consumer groups for fan-in across worker processes.

## License

MIT. See [LICENSE](LICENSE).
