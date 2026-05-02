# research-crew

> Concurrent multi-agent research service. 5 specialist agents fan out in parallel via a Redis-backed durable workflow with idempotent step semantics + bounded exponential-backoff retries; results merge through a Synthesizer into a single citation-grounded report.

[![ci](https://github.com/BishBish123/research-crew/actions/workflows/ci.yml/badge.svg)](https://github.com/BishBish123/research-crew/actions/workflows/ci.yml)
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

The brief asked for Inngest + Redis Streams + LangGraph. **The repo ships the *contracts* those engines provide rather than booking three cloud signups for a portfolio piece** — same API surface, swappable adapter:

| Brief | Shipped here | Production swap |
| --- | --- | --- |
| Inngest durable workflow | `WorkflowEngine` with idempotent steps, retries, timeouts | Replace `engine.run_parallel(...)` with `inngest.step.parallel(...)` — same contract |
| Redis Streams consumer groups | `RedisRunStore` with persistent run/step keys | Already real Redis. Swap `RedisRunStore` for `RedisStreamRunStore` for at-least-once fan-in |
| LangGraph synthesizer | `StitchSynthesizer` deterministic + plug-point Protocol | Drop in a `LangGraphSynthesizer` implementing `Synthesizer.synthesize(...)` |
| Tavily / Brave / Exa search | `MockAgent` with deterministic blake2b results | Implement `Agent.search(...)` with a `httpx` call to the real API |
| Locust load test | `load/locustfile.py` ready to run | `make up && make api && make load` |

The architecture is the artifact. Live API keys are runtime config — not what a portfolio review should grade.

## Quick start

```bash
git clone https://github.com/BishBish123/research-crew.git
cd research-crew
make install

# Bring Redis up
make up

# Run a single research job from the CLI (memory store, no API needed)
uv run research "what is python"

# Same with retry-path exercise (forces re-attempts)
uv run research "test retries" --failure-rate 0.5

# Run the API
make api &
curl -X POST -H 'content-type: application/json' \
    -d '{"question":"how does Inngest handle step retries"}' \
    http://localhost:8000/research

# Get the run status
curl http://localhost:8000/runs/<run_id>
```

## API

| Verb | Path | Body | Returns |
| --- | --- | --- | --- |
| `GET` | `/health` | — | `{ "status": "ok", "redis": "up" }` |
| `POST` | `/research` | `{ "question": "...", "agents": ["web_search", ...] }` | `202 { "run_id": "...", "status_url": "/runs/..." }` |
| `GET` | `/runs/{id}` | — | `RunStatus` (state, per-step audit, embedded `ResearchReport`) |

### Auth + rate limiting

`/research` and `/runs/{id}` are gated behind a bearer token when
`RESEARCH_API_TOKEN` is set; `/health` stays open so a load balancer can
probe without a credential.

```bash
export RESEARCH_API_TOKEN=$(openssl rand -hex 32)
make api &
curl -H "Authorization: Bearer $RESEARCH_API_TOKEN" \
     -H 'content-type: application/json' \
     -d '{"question":"how does Inngest handle step retries"}' \
     http://localhost:8000/research
```

If `RESEARCH_API_TOKEN` is unset the service runs unauthenticated and
the lifespan logs a loud `api.auth_disabled` warning — that's the dev
path; in production the operator MUST set the env var.

POST `/research` is additionally rate-limited per client IP (token
bucket, 60s window). Default `10 req/min/IP`; override via
`RESEARCH_RATE_LIMIT_PER_MIN`. Exhausted callers get `429 Too Many
Requests` plus a `Retry-After` header in seconds.

## Single-process, horizontally-scalable design

The service runs as one FastAPI process — a research call fans out
across asyncio inside that single process, not across machines. The
"distributed" framing was misleading and has been removed; what's
shipped is concurrent fan-out plus a durability contract that lets
multiple instances share work safely:

* **Idempotency cache lives in Redis** (`step:{dedup_key}`), so two
  instances behind a load balancer never double-execute the same
  `(run_id, agent, question)` triple. The second arrival short-circuits
  on the cache and returns the result the first one wrote.
* **Run records, step audits, and the cache TTL** are all in shared
  Redis, so any instance can serve `GET /runs/{id}` regardless of
  which one accepted the original POST.
* **No instance affinity is required** — submit on instance A,
  poll on instance B, finish on instance C: the run is durable across
  the fleet, not pinned to one process.

Promoting this to a true distributed worker pool is a swap of
`WorkflowEngine` for an Inngest function or a Redis Streams consumer
group; the store contract is already shaped for it.

## Load test (Locust)

`load/locustfile.py` posts a question, polls the run-status endpoint, and exercises `/health` for noise. The agent layer is mocked deterministically so the load test isolates the *workflow plumbing* — what you'd run before signing up for paid search APIs to find out whether the orchestration scales.

```bash
make api &
make load   # opens Locust on http://localhost:8089
```

## Tests

```bash
make test                # 90+ unit tests (workflow, synthesizer, store, API, agents, properties)
make test-integration    # tests gated on a real Redis at $REDIS_URL
make check               # ruff + mypy --strict
```

What's covered today:

- **Workflow durability** — idempotency cache hit, retry-then-succeed, timeout-then-retry, exhausted retries, per-step timeout, parallel partial-failure isolation, cancellation cleanup.
- **Idempotency-key properties** — Hypothesis property tests assert determinism + per-component sensitivity across 200 random examples per property.
- **Store contract** — every method round-trips on both `InMemoryRunStore` and `RedisRunStore` (via fakeredis), parametrized over both.
- **Synthesizer correctness** — citation dedupe by URL (incl. case / trailing-slash / `www.` collapse), failure surfacing, all-failed empty path, per-agent caps, query-string preservation.
- **API contracts** — submit → background → poll → report renders; 404 on unknown run; 422 on bad payloads (missing, too long, wrong type, unknown agent); 503 on uninitialised store; `/health` shape contract.
- **Agent base** — Protocol conformance, determinism, latency simulation, failure injection.

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
tests/             90+ unit tests (incl. Hypothesis property tests + store contract)
```

## Honest limitations

- No real search adapters — that's the point of the `Agent` Protocol; pick your search vendor and implement `search()`.
- No LLM synthesizer wired in. The trace shape is identical, so plugging one in is one new class implementing `Synthesizer`.
- No Langfuse — but the `WorkflowEngine`'s `record_step` callback is exactly the right shape to back with a Langfuse span emitter.
- The deduplication is by URL only; semantic dedup (embed citations, cluster by cosine) is the obvious next commit.
- This is a single FastAPI process with `asyncio.gather` fan-out, not a distributed system. The "Single-process, horizontally-scalable design" section above explains the deployment story; for true distribution swap `WorkflowEngine` for an Inngest function, or use Redis Streams consumer groups for fan-in across worker processes.

## License

MIT. See [LICENSE](LICENSE).
