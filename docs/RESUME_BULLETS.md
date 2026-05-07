# Research-Crew — resume bullets

## Defensible (use these)

- **Built a parallelized research-orchestration API** (FastAPI + asyncio.gather) that fans one question out to N specialist agents (Tavily, Brave, Exa), deduplicates results semantically with Hypothesis-property-tested invariants, and stitches them into a citation-grounded markdown report; 510 unit tests + 17 mock-server integration tests (527 total).

  *Evidence:* `tests/REPORT.md` documents 510 unit + 17 integration tests, verified pre/post counts. `src/research_crew/workflow.py` (`run_parallel` via `asyncio.gather`). `tests/test_dedup_property.py` (Hypothesis property tests). `tests/test_real_adapter_integration.py` (17 mock-server tests over real TCP sockets).

- **Wired three production search adapters** (TavilyAgent, BraveAgent, ExaAgent) with real-TCP integration tests: each test opens an actual socket to a local FastAPI mock server (no httpx patching), validates auth-header delivery, and exercises the retry loop against real 500 responses — proving the adapter sends the correct auth header on every call and retries correctly.

  *Evidence:* `tests/test_real_adapter_integration.py` and `tests/mock_servers/app.py`; `tests/REPORT.md` mock-server section documents exactly which headers are validated per adapter.

- **Implemented idempotent durable-workflow semantics** without an external scheduler: step deduplication via `H(run_id|agent|question)` keys, per-attempt exponential backoff, run-level timeout, and a `RunStore` Protocol with three interchangeable backends (in-memory, fakeredis Streams, Postgres); tested against a Redis Streams backend with fakeredis.

  *Evidence:* `src/research_crew/workflow.py` (dedup key construction); `tests/test_redis_streams_store.py`; `tests/test_postgres_store.py`; `ARCHITECTURE.md` workflow-engine section.

- **Designed a 20-question golden eval harness** for the mock pipeline, measuring keyphrase coverage (mean 0.667) across 6 question categories (factual, comparative, list, trend, oos, refusal) — establishing a reproducible regression floor before any real search adapter is wired.

  *Evidence:* `evals/REPORT.md` (20 questions, per-category breakdown); `evals/golden.py` (question definitions); `evals/INTERPRETATION.md` explains why citation correctness is 0.0 at the mock floor.

---

## Stretch / claim with caveat (use cautiously)

- **"Langfuse observability with SDK v2/v3/v4 compatibility"** — `tests/test_observability_langfuse.py` tests null + live modes across SDK versions. What is not defensible: a live Langfuse dashboard URL or real traces visible to a reviewer.

  *Pushback:* "Can I see the dashboard?" — honest answer: "No live deploy; the tracer is wired and tested, but the Langfuse project was never pushed to a hosted instance."

- **"Inngest durable workflow integration"** — `tests/test_inngest_workflow.py` exists and passes. The Inngest path is behind a `--use-inngest` flag and tested against mocked SDK calls, not a live Inngest server.

  *Pushback:* "How was it tested?" — mocked SDK, no live Inngest event bus. The real-adapter integration tests are TCP-level but against local mock servers only.

- **"WebSocket streaming endpoint"** — `tests/test_websocket_streaming.py` covers the WS path. Not defensible: a deployed public endpoint.

---

## DO NOT claim

- **"Real measured latencies for Tavily/Brave/Exa"** — the eval REPORT.md shows "0 ms (mock pipeline)" for all latencies because the eval runs against MockAgent. No real adapter latency has been measured.

  *Alternate:* "Built a harness that measures per-agent latency; all published numbers are from the deterministic mock pipeline."

- **"LLM judge with coherent citation correctness scores"** — citation correctness is 0.000 for every question in REPORT.md; MockAgent returns `example.com` URLs that never match expected domains. The harness is wired for an LLM judge (`JudgeProtocol` placeholder in `evals/harness.py`) but it is not implemented.

  *Alternate:* "Designed a pluggable `JudgeProtocol` for citation scoring; mock floor is 0.000 (expected), real numbers require wiring a live search adapter."

- **"50-conversation golden eval"** — the golden set has 20 questions, not 50.

  *Alternate:* "20-question golden eval set across 6 question categories."

- **"Deployed to Fly.io"** — `fly.toml` is present but no live URL is verifiable.

- **"Load test results showing N req/s"** — `tests/test_load_test.py` exists and `docs/load-test-results.md` is present, but claiming specific production-grade numbers requires running Locust against a live server. Check `docs/load-test-results.md` for what was actually measured before citing any number.

---

## How to defend each bullet in an interview

**Bullet 1 — 510 unit + 17 integration tests, parallel fan-out:**
> "The 510/17 split is in `tests/REPORT.md`. The workflow fan-out is `asyncio.gather` in `workflow.py`. The integration tests (`test_real_adapter_integration.py`) open a real TCP socket — no httpx patching — to a local FastAPI mock server. The mock server validates auth headers and returns real 500s to exercise the retry loop. `tests/REPORT.md` documents exactly which headers each adapter is tested against."

**Bullet 2 — real TCP integration tests:**
> "The unit tests in `test_real_agents.py` patch `httpx.AsyncClient`. The 17 integration tests in `test_real_adapter_integration.py` do not — httpx connects to `127.0.0.1:<random port>` for real. The mock server in `tests/mock_servers/app.py` returns 401 if the auth header is absent, which proves the adapter sends it on every call. The retry loop is exercised by the `force=500` endpoint that cycles: first request gets 500, second gets 200."

**Bullet 3 — idempotent workflow, 3 store backends:**
> "The dedup key is `blake2b(run_id + agent_name + question)` — same inputs always hash to the same key. `RunStore` is a Protocol with `put_run`, `append_step`, `cache_get`, `cache_put`. The three implementations are `InMemoryRunStore` (tested in `test_store_contract.py`), `RedisStreamsRunStore` (tested with fakeredis in `test_redis_streams_store.py`), and `PostgresRunStore` (tested in `test_postgres_store.py`)."

**Bullet 4 — 20-question golden eval:**
> "`evals/golden.py` has 20 questions across 6 categories. `evals/harness.py` runs them against the mock pipeline and scores keyphrase coverage and citation correctness. `evals/REPORT.md` shows keyphrase coverage of 0.667 mean — that's the mock floor, not a real retrieval number. The interpretation doc explains why: MockAgent echoes the question back, so keyphrases in the question body score 1.0, but domain-specific expected phrases like 'jitter' or '429' score 0."
