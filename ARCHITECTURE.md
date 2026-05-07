# Architecture

This document is the design write-up for `research-crew`. The README is
the elevator pitch; this is what you'd hand a senior reviewer who asks
"why did you build it this way".

## 1. Goal

Take one user question, fan it out to N specialist research sources in
parallel, and stitch the results into a single citation-grounded
report — durably, idempotently, and with bounded blast radius when any
single source misbehaves.

## 2. Layered shape

```
            ┌────────────┐
            │  FastAPI   │   POST /research → run_id
            └────┬───────┘   GET  /runs/{id}
                 │
                 ▼
         ┌─────────────────┐
         │ WorkflowEngine  │   per-run orchestrator
         │  (run_parallel) │
         └────┬────────────┘
              │   asyncio.gather
              ▼
   ┌──────────────────────────────┐
   │ run_one (per agent):         │
   │   1. dedup_key = H(run|a|q)  │
   │   2. cache_get → CACHED?     │
   │   3. for attempt in 1..N:    │
   │        wait_for(agent.search,│
   │                 timeout=T)   │
   │        on success: cache_put │
   │        on fail/timeout: log, │
   │                 record, back │
   │                 off          │
   │   4. return AgentResult      │
   └─────────────┬────────────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ Synthesizer     │   merge per-agent → dedupe → markdown
        └────┬────────────┘
             ▼
        ┌─────────────┐
        │  RunStore   │  put_run + append_step + cache_get/put
        └─────────────┘   (RedisRunStore | RedisStreamRunStore | InMemoryRunStore)
```

Each layer is a Protocol so it can be swapped without touching the
others. The most likely swaps:

* `Agent` ⟶ real Tavily / Brave / Semantic Scholar adapters
* `Synthesizer` ⟶ LangGraph / LLM-grounded synthesis
* `RunStore` ⟶ Inngest's own state store, or a Postgres-backed
  table-driven runbook

## 3. Workflow engine — two implementations

The repo ships two workflow engines behind a shared `WorkflowProtocol`:

```
WorkflowProtocol
├── WorkflowEngine      (default, no external deps)
└── InngestWorkflow     (--use-inngest, requires inngest-py SDK)
```

Both satisfy `isinstance(engine, WorkflowProtocol)` at runtime, expose
the same `run_parallel(agents, question) -> list[AgentResult]` method,
and return structurally identical results that `StitchSynthesizer` and
`RunStore` accept unchanged.

### WorkflowEngine (default)

The hand-rolled engine in `src/research_crew/workflow.py` owns every
durability contract in-process:

- Idempotent steps via `H(run_id|agent|question)` dedup keys
- Bounded retries with exponential backoff + jitter
- Per-step wall-clock timeout via `asyncio.wait_for`
- Parallel fan-out via `asyncio.gather`
- Step records persisted through the injected `RunStore`

This path has zero external dependencies. `make test` exercises every
durability property without any running services.

### InngestWorkflow (opt-in)

`src/research_crew/workflow_inngest.py` wraps the same fan-out logic as
Inngest durable steps:

- `group.parallel(...)` — all 5 agents run concurrently as Inngest steps
- `step.run(...)` — each agent invocation is a named, retriable step
- `step.run("synthesize", ...)` — the final synthesis is its own step

The Inngest dev server (`npx inngest-cli@latest dev`, no cloud account)
handles step scheduling, retries, and the observability UI. The
`--use-inngest` CLI flag activates this path.

```
make inngest-dev          # starts npx inngest-cli@latest dev
uv run research --use-inngest "what is python"
```

### Migration path

Switching from the hand-rolled engine to Inngest in production is:

1. `pip install 'research-crew[real]'` (adds `inngest>=0.4`)
2. Set `INNGEST_DEV_SERVER_URL` (or configure Inngest cloud)
3. Add `--use-inngest` (CLI) or call `make_workflow(use_inngest=True)` directly

No other code changes — the `WorkflowProtocol` contract is identical.

## 4. Idempotency

### Why
A retry should not double-charge a paid search API, double-write a
result, or produce a non-deterministic report when run twice with the
same input.

### How
Every step computes a stable cache key:

```python
dedup_key = "step:" + blake2b(f"{run_id}|{agent}|{question}", digest_size=12).hexdigest()
```

Properties that fall out of using a hash:

* **Deterministic** — same triple ⇒ same key forever.
* **Collision-resistant** — 96 bits of digest is overkill for the
  number of in-flight runs we'd ever have, but keys go through Redis
  so we want zero ambiguity in shared state.
* **Namespaced** — `step:` prefix keeps the cache out of the way of
  `run:`/`run:{id}:steps` keys.

Property tests in `tests/test_dedup_property.py` enforce determinism +
sensitivity-to-each-component via Hypothesis with 200 examples per
property.

## 5. Retries + backoff

* `max_attempts` (default 3) is the hard ceiling.
* Per-attempt sleep is `base_backoff_s * 2 ** (attempt - 1)`.
* `base_backoff_s = 0.05` so a 3-attempt run is at most ~0.15s of
  sleep — fast enough for tests, slow enough that real adapters get
  the jitter benefit.
* On the *last* attempt we don't sleep — the budget is already spent.

What happens on each failure mode:

| Failure | Recovery |
| --- | --- |
| `agent.search()` raises | wrapped in `AgentExecutionError` (the wrapper's name + the original exception text both land in the StepRecord and the `workflow.agent_error` log line, so callers can match on the typed handle), recorded, retried |
| `agent.search()` returns `FAILED` | recorded, retried (its own choice — distinct from a raised exception) |
| `asyncio.wait_for` fires | `AgentTimeoutError`, recorded, retried |
| `asyncio.CancelledError` | recorded once, then re-raised — cancellation is *not* a retry signal |

## 6. Timeout

Each step has a wall-clock budget enforced via `asyncio.wait_for`. The
budget covers a single attempt — retries reset the clock. This is the
right choice because a timeout doesn't necessarily mean the agent is
broken; the next attempt may succeed quickly.

## 7. Parallel fan-out

`run_parallel` is a thin `asyncio.gather` over `run_one(agent)` for
each agent. There is no shared state between agents, so a hard failure
in one agent never starves the others. A test in
`tests/test_workflow_edges.py::TestParallelFanoutIsolation` asserts
this isolation directly.

## 8. Observability

### Structured logging

Every workflow event emits a structlog line bound with `run_id`,
`agent`, `dedup_key`. Operationally that's the join key you want when
correlating a slow agent across a fleet of runs:

```
workflow.timeout       run_id=… agent=scholar dedup_key=step:… attempt=1 timeout_s=30.0
workflow.success       run_id=… agent=scholar dedup_key=step:… attempt=2 elapsed_ms=412.6
```

### Langfuse distributed tracing

`src/research_crew/observability/langfuse.py` ships the Langfuse adapter.
It is **env-key-gated**: when `LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY`
are set and the `langfuse` SDK is installed (via `pip install 'research-crew[real]'`),
the adapter sends traces to Langfuse. Otherwise it is a strict no-op —
no exceptions, no side-effects on the default path.

```
WorkflowEngine.run_parallel(agents, question)
    │
    ├── tracer.start_run(question)         → one Langfuse trace
    │       (RunHandle holds trace ID)
    │
    ├── run_one(agent) × N  (asyncio.gather)
    │       └── _record(agent, SUCCEEDED/FAILED)
    │               └── tracer.record_step(handle, step)  → child span
    │
    └── tracer.finish_run(handle, "succeeded"|"failed")
            └── langfuse.flush()
```

**What gets emitted per run:**

| Event | Langfuse type | Fields |
| --- | --- | --- |
| Research run start | Trace (input = question) | `run_id`, `query` |
| Agent terminal attempt | Span | `agent`, `status`, `attempts`, `elapsed_ms`, `error?` |
| Research run end | Trace update (output = outcome) | `"succeeded"` or `"failed"` |

Only terminal step statuses (`SUCCEEDED`, `FAILED`, `CACHED`) produce spans.
`RUNNING` records are intentionally skipped — they are intermediate state
without a closed time range and would create half-open spans.

**Adapter classes:**

* `NullTracer` — returned by `make_tracer()` when env vars are absent.
  All methods are no-ops.
* `LangfuseTracer` — returned when env vars are present; falls back to
  no-op mode if the SDK import fails (soft-fail).
* `RunHandle` — small dataclass holding the active trace ID and the live
  SDK trace object (or `None` in no-op mode).
* `make_tracer()` — factory function; always safe to call regardless of
  configuration.

## 9. Failure surfaces, not failure crashes

A run with three of five agents failing still succeeds — the
synthesizer surfaces the failures in a `## Failures` section so the
reviewer can see what was attempted. That preserves user trust in
edge cases that the alternative ("the run failed, try again") doesn't.

The all-failed case is still reported successfully (HTTP 200 +
`state: succeeded` is the wrong answer here): `state` flips to
`failed` only when *every* agent ultimately failed.

## 10. Auth + rate limiting

The service is intentionally simple here:

* `RESEARCH_API_TOKEN` env var, when set, gates `/research` and
  `/runs/{id}` behind a `Authorization: Bearer …` header. `/health`
  stays unauthenticated so external probes don't need provisioning.
* When the token is unset the lifespan logs a loud
  `api.auth_disabled` warning — that's the dev path, and the warning
  is the contract that says "don't ship this to prod".
* `RESEARCH_RATE_LIMIT_PER_MIN` (default 10) is a per-client-IP
  sliding-window counter on POST `/research`: the limiter holds a
  deque of request timestamps newer than `now - 60s` and rejects when
  the deque length already equals the cap. Overflow returns `429`
  with a `Retry-After` header derived from the oldest in-window
  timestamp. The limiter is in-process; multi-instance deployments
  need a Redis-backed limiter before the per-IP cap is shared across
  the fleet.

These live in `api.py` next to the lifespan rather than as a separate
middleware so the test fixtures can poke `app.state.api_token` /
`app.state.rate_limiter` directly without touching imports.

## 11. Run store implementations

The `RunStore` Protocol is satisfied by four backends:

| Backend | Class | Activated by | Storage |
| --- | --- | --- | --- |
| Hash (default) | `RedisRunStore` | `RESEARCH_CREW_STORE=hash` or unset | Redis String + List |
| Streams | `RedisStreamRunStore` | `RESEARCH_CREW_STORE=streams` | Redis Hash + Stream |
| In-memory | `InMemoryRunStore` | `RESEARCH_CREW_STORE=memory` | Python dicts |
| Postgres | `PostgresRunStore` | `RESEARCH_CREW_STORE=postgres` | Postgres / Neon tables |

The `make_run_store()` factory in `src/research_crew/store/__init__.py`
reads the env var and constructs the correct backend.

### RedisRunStore (default)

Uses three keyspaces (all under `{prefix}:`):

* `run:{run_id}` — String key, JSON `RunStatus` blob.
* `run:{run_id}:steps` — Redis List (`RPUSH`/`LRANGE`), ordered step audit log.
* `step:{dedup_key}` — String key, JSON `AgentResult` idempotency cache.

### RedisStreamRunStore (opt-in)

Defined in `src/research_crew/store/redis_streams.py`.

| Keyspace | Redis type | Purpose |
| --- | --- | --- |
| `{prefix}:run:{run_id}` | Hash (`HSET`/`HGETALL`) | Canonical run state |
| `{prefix}:stream:{run_id}:steps` | Stream (`XADD`) | Append-only step audit log |
| `{prefix}:stream:{run_id}:input` | Stream (`XADD`) | Per-agent fan-out input |
| `{prefix}:step:{dedup_key}` | String | Idempotency cache (same as hash store) |

**Consumer groups:**

* `steps-readers` on the steps stream — streaming audit log consumers.
* `agent:{agent_name}` on the input stream — one group per agent for fan-out;
  each group sees all messages independently (fan-out, not competing).

**Stream patterns:**

* `XADD` — `append_step`, `publish_input`.
* `XREADGROUP` — `read_steps_group`, `read_input_group` (at-least-once delivery).
* `XACK` — `ack_step`, `ack_input` (remove from PEL on successful processing).
* `XPENDING` — `pending_steps` summary; `pending_steps_range` for orphan list.
* `XCLAIM` — `claim_step` (orphan recovery: take ownership from a crashed consumer).

### PostgresRunStore (opt-in)

Defined in `src/research_crew/store/postgres.py`. Backed by Postgres or Neon.
Suitable for long-term archival where runs must survive beyond Redis TTLs.

**Schema (three tables, applied idempotently by `setup()`):**

| Table | Key columns | Purpose |
| --- | --- | --- |
| `runs` | `run_id TEXT PK` | Canonical RunStatus snapshot as JSONB |
| `steps` | `UNIQUE(run_id, sequence)` | Append-only step audit log; sequence enforces ordering + idempotency |
| `step_dedup` | `dedup_key TEXT PK` | Idempotency cache (same role as `step:{dedup_key}` in Redis stores) |

**Why three tables:**
* `runs` and `steps` are separate so `GET /runs/{id}` can fetch both with one JOIN or two small selects without re-parsing a giant JSONB blob on every step append.
* `step_dedup` is isolated to give it its own key-value contract without polluting the run/step tables with cache rows.

**Pool lifecycle:** `PostgresRunStore(dsn=...)` stores the DSN only; the asyncpg connection pool is created lazily on the first call to `setup()`. This keeps the factory synchronous and lets the API lifespan `await` the setup in an async context.

**Environment variables:**

| Variable | Default | Purpose |
| --- | --- | --- |
| `RESEARCH_PG_DSN` | `postgresql://research:research@localhost:5432/research` | Postgres connection string |

**Docker compose:** `docker-compose.postgres.yml` — start with `make pg-up`, stop with `make pg-down`.

### Migration matrix

`src/research_crew/store/migrate.py` provides helpers for all cross-backend migrations:

| Source → Destination | Helper | Notes |
| --- | --- | --- |
| hash → streams | `migrate_hash_to_streams(redis, run_id)` | Converts String→Hash, List→Stream; renames list key to `…:steps.migrated` |
| hash → postgres | `migrate_redis_hash_to_postgres(redis, pg_pool, run_id)` | Reads String key + List; upserts into `runs` + `steps` |
| streams → postgres | `migrate_redis_streams_to_postgres(redis, pg_pool, run_id)` | Reads Hash key + XRANGE; upserts into `runs` + `steps` |
| memory → any | No helper needed — in-memory is ephemeral (test-only) | — |

All three helpers are idempotent: `put_run` upserts, `append_step` uses `ON CONFLICT (run_id, sequence) DO NOTHING`.

The streams store detects unconverted list data via `StoreBackendMismatchError`
and refuses to `append_step` until migration runs, preventing mixed-format state.

## 12. Cross-run semantic deduplication

### ADR: Why URL dedup alone is not enough

URL normalisation collapses `https://example.com/foo/` and
`https://EXAMPLE.com/foo` into the same key. It does not help when:

* Two different URLs serve identical or near-identical content (e.g. a
  news aggregator mirroring the same AP wire story across 10 domain
  names, or a Wikipedia article and a "simple Wikipedia" paraphrase).
* The same underlying fact appears as a quoted excerpt in a primary
  source and as a paraphrase in a secondary source with an unrelated URL.

In both cases the synthesizer's final `citations` list carries redundant
content that increases noise for the reader without adding new information.

### Decision

Add an embedding-based ANN nearest-neighbour check as a second
deduplication pass **after** URL dedup. Each citation's `"${title}
${snippet}"` string is embedded (384-dim cosine space) and compared
against all previously seen content stored in a `seen_content` Postgres
table. If cosine similarity ≥ `threshold` (default 0.85), the citation
is skipped and logged.

### Why pgvector + ivfflat

* **Postgres** is already in the stack for `PostgresRunStore`. Reusing
  the same connection pool (same `RESEARCH_PG_DSN`) keeps the operational
  footprint minimal.
* **pgvector** is the most widely deployed Postgres vector extension;
  available on Neon, Supabase, AWS RDS, and self-hosted.
* **ivfflat** (inverted file index, approximate nearest neighbours) over
  `vector_cosine_ops` gives sub-millisecond search at the citation-count
  scales we operate at (thousands of rows, not billions). An exact index
  (`USING hnsw`) would give higher recall but is not needed here — false
  negatives (missed dups) are acceptable; false positives (over-dedup)
  are gated by the tunable threshold.

### Why 384 dimensions

`all-MiniLM-L6-v2` (384 dims) from `sentence-transformers` is small
enough to be fast on CPU and large enough to distinguish paraphrases
from genuinely distinct content. The table schema is fixed at 384 dims
(`vector(384)`) — changing the model would require a schema migration.

### Embedding soft-fail

The module detects whether `sentence-transformers` is installed at
construction time (`_load_sentence_transformer()`). When absent it falls
back to `_fake_embed`: a deterministic SHA-256-derived unit vector. This
fake encoder is **not semantically meaningful** — it will not detect
paraphrases — but it lets the plumbing be exercised in CI without
downloading models, and it makes the `PgVectorSemanticDedup` path
testable offline.

### Zero-overhead default

`NullDedup` is the default when `RESEARCH_PG_DSN` is unset. Every method
is a coroutine that returns immediately — no DB connection, no model load,
no allocation beyond the coroutine frame. Existing synthesizer behaviour
is unchanged.

### Module layout

```
src/research_crew/dedup/
├── __init__.py     make_semantic_dedup() factory + re-exports
├── protocol.py     SemanticDedup Protocol (is_duplicate, add_seen)
├── null.py         NullDedup — no-op default
└── pgvector.py     PgVectorSemanticDedup — ANN via asyncpg + pgvector
```

`StitchSynthesizer` accepts `semantic_dedup: SemanticDedup` as a
dataclass field (default `make_semantic_dedup()`). The synthesizer
invokes the protocol after URL dedup; on `NullDedup` this is two
immediate coroutine awaits per citation with no side-effects.

## 13. Real-time WebSocket streaming

### In-process queue design

`src/research_crew/streaming.py` implements a `RunQueueRegistry` —
a module-level singleton that holds a `dict[run_id, list[asyncio.Queue]]`.

Lifecycle:

1. **POST /research** — `registry.create(run_id)` initialises the slot.
2. **WorkflowEngine** — `record_step` is wrapped in `api.py` to call
   `registry.publish(run_id, step)` after every `store.append_step`.
3. **WS /runs/{run_id}/stream** — each connection calls
   `registry.subscribe(run_id)` to receive its own queue.  The handler
   loops on `asyncio.wait_for(q.get(), timeout=15)`, sending step events
   as JSON and a `{"type":"heartbeat"}` on timeout.
4. **Terminal** — `registry.teardown(run_id)` pushes a `None` sentinel
   onto every subscriber queue; the handler sends `{"type":"done"}` and
   closes with code 1000.

All mutations are on the event loop; no lock needed.

### Broadcast pattern

`subscribe` returns a _per-connection_ `asyncio.Queue`.  `publish`
iterates the subscriber list and calls `put_nowait` on each — O(n
connections) but unbounded since queues are in-memory.  Late joiners
(connect after `teardown`) get the sentinel immediately via the
`_terminal` set check in `subscribe`.

### Authentication

The WS endpoint accepts the bearer token via:

- `Authorization: Bearer <token>` upgrade header, or
- `?token=<value>` query parameter (required for browser `WebSocket` API).

Both paths use `secrets.compare_digest` for constant-time comparison.

### Future: Redis pub/sub upgrade path

The in-process queue works for a single API instance.  For horizontal
scaling, replace `publish` with a Redis `PUBLISH research:ws:{run_id}`
call and have each WS handler subscribe via `redis.subscribe(channel)`.
The handler loop is unchanged — swap `q.get()` for an asyncio iterator
over the Redis channel.  No changes needed to the workflow or store layer.

```
Single instance (current):       Multi-instance (future):
  WorkflowEngine                   WorkflowEngine
      | record_step                    | record_step
      v                               v
  RunQueueRegistry              Redis PUBLISH
      | put_nowait                     | SUBSCRIBE
      v                               v
  asyncio.Queue x N subs        asyncio iterator x N subs
```

## 14. Future work

* **Real search adapters** — `httpx` calls behind the existing `Agent`
  protocol; the workflow layer needs no changes.
* **Distributed worker pool** — `WorkflowEngine` runs in-process today.
  Wire the `run_one(agent)` tasks up to `RedisStreamRunStore.publish_input` /
  `read_input_group` to fan work out across multiple worker processes — each
  process runs its own consumer in the `agent:{name}` group.
* **LLM synthesizer** — `StitchSynthesizer` is the deterministic
  fallback. Wiring up a `LangGraphSynthesizer` is one new class
  implementing `Synthesizer.synthesize(...)`.
* **WS multi-instance** — Replace `RunQueueRegistry` publish with Redis
  pub/sub as described in section 13; no handler changes required.
