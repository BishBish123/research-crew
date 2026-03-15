# Load test results

This document captures the shape of the Locust load test against the
local API. The goal is to validate the *orchestration plumbing*
(workflow runner, parallel fan-out, Redis writes) under sustained
concurrent load — not to benchmark any external search vendor.

## How to reproduce

```bash
make up          # bring Redis up
make api &       # uvicorn on :8000
make load        # opens Locust UI on http://localhost:8089
```

In the Locust UI, set:

* **Number of users:** 50–200 depending on what you want to stress
* **Spawn rate:** 10/s

The agent layer is `MockAgent` with `latency_ms=50`, so each `run_one`
call adds a deterministic 50ms of artificial latency. With
`asyncio.gather` over 5 agents in parallel, the per-run minimum is
~50ms (latency-bound, not CPU-bound).

## Reference shape (50 users, 60s, fakeredis-equivalent)

The scenario alternates POST `/research` (weight 3) with GET
`/health` (weight 1), polling each new run once.

| Endpoint        | Median (ms) | p95 (ms) | p99 (ms) | RPS      | Failures |
| --------------- | ----------- | -------- | -------- | -------- | -------- |
| POST /research  | ~12         | ~25      | ~45      | ~50/s    | 0%       |
| GET /runs/{id}  | ~5          | ~15      | ~30      | ~50/s    | 0%       |
| GET /health     | ~3          | ~8       | ~12      | ~17/s    | 0%       |

Numbers are illustrative — they will vary by hardware and Redis
backend. Real numbers from your run get exported by Locust to CSV
(`--csv=…`) for pasting in a PR description.

## What this exercises

* **Workflow plumbing throughput** — POST returns 202 immediately and
  schedules a background task; the median latency on POST measures
  pure orchestration overhead, not agent latency.
* **Parallel fan-out scaling** — every accepted POST kicks off
  `asyncio.gather` over 5 agents. The CPU cost is dominated by
  Pydantic validation + Redis serialisation, not the agent code.
* **Redis hot path** — `put_run`, `append_step`, and `cache_get`
  fire ~6× per accepted run. The `RedisRunStore` keeps these as
  single-key ops, so they remain O(1) even with thousands of runs in
  flight.

## What this does *not* exercise

* Real search APIs (Tavily/Brave/etc.) — they sit behind `Agent`
  adapters, which `MockAgent` replaces. Real network latency would
  swamp the orchestration overhead measured here.
* LLM synthesis cost — `StitchSynthesizer` is deterministic and
  near-free.
* Cross-process distribution — `WorkflowEngine` runs in a single
  uvicorn worker. For multi-worker fan-in, the natural next step is
  Redis Streams + consumer groups (see ARCHITECTURE.md §10).

## Failure modes seen so far

| Scenario | Symptom | Mitigation |
| --- | --- | --- |
| Redis dropped mid-run | 503 from `/health`, run state stuck `RUNNING` | run TTL expires the orphan in 24h |
| Agent injected `failure_rate=0.5` | extra `attempts` recorded; final state still `SUCCEEDED` for a majority | retry budget catches it (ADR-003) |
| Repeated identical question on same run | second call returns CACHED, agent never invoked | idempotency cache (ADR-002) |
