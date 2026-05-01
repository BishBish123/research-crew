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
        └─────────────┘   (RedisRunStore | InMemoryRunStore)
```

Each layer is a Protocol so it can be swapped without touching the
others. The most likely swaps:

* `Agent` ⟶ real Tavily / Brave / Semantic Scholar adapters
* `Synthesizer` ⟶ LangGraph / LLM-grounded synthesis
* `RunStore` ⟶ Inngest's own state store, or a Postgres-backed
  table-driven runbook

## 3. Why a hand-rolled durable workflow

The brief asked for Inngest. The portfolio version ships the *contract*
Inngest provides — idempotent steps, retries, timeouts, parallel
fan-out, persisted step records — without booking a third-party signup
that no reviewer can run locally. The shape of the code matches an
Inngest function 1:1, so swapping `engine.run_parallel(agents, q)` for
`inngest.step.parallel(agents, q)` is a one-liner.

The cost is that we own the orchestration semantics ourselves; the
benefit is that a reviewer can `git clone && make test` and see every
durability property exercised.

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
| `agent.search()` raises | wrapped via `AgentExecutionError`, recorded, retried |
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

Every workflow event emits a structlog line bound with `run_id`,
`agent`, `dedup_key`. Operationally that's the join key you want when
correlating a slow agent across a fleet of runs:

```
workflow.timeout       run_id=… agent=scholar dedup_key=step:… attempt=1 timeout_s=30.0
workflow.success       run_id=… agent=scholar dedup_key=step:… attempt=2 elapsed_ms=412.6
```

The same `record_step` callback that drives `GET /runs/{id}` is the
exact shape that would back a Langfuse span emitter — point a future
commit at the Langfuse SDK and the trace UI lights up without changing
any caller.

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
* `RESEARCH_RATE_LIMIT_PER_MIN` (default 10) is a per-client-IP token
  bucket on POST `/research`. The window is 60 seconds and overflow
  returns `429` with a `Retry-After` header. The limiter is
  in-process; multi-instance deployments need a Redis-backed limiter
  before the per-IP cap is shared across the fleet.

These live in `api.py` next to the lifespan rather than as a separate
middleware so the test fixtures can poke `app.state.api_token` /
`app.state.rate_limiter` directly without touching imports.

## 11. Future work

* **Real search adapters** — `httpx` calls behind the existing `Agent`
  protocol; the workflow layer needs no changes.
* **Semantic dedupe** — current dedupe is by URL canonicalisation.
  Embedding citations and clustering by cosine would catch the case
  where two different URLs serve the same content.
* **Distributed worker pool** — `WorkflowEngine` runs in-process
  today. The natural next step is to write step results to a Redis
  Stream consumer group, fanning the `run_one(agent)` work out across
  multiple worker processes. The store contract already supports this.
* **LLM synthesizer** — `StitchSynthesizer` is the deterministic
  fallback. Wiring up a `LangGraphSynthesizer` is one new class
  implementing `Synthesizer.synthesize(...)`.
