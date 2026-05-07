# Test Report

## Suite overview

| Suite | Run command | Count | Description |
|-------|-------------|-------|-------------|
| Unit (default) | `make test` | 510 | All tests not marked `integration` |
| Mock-server integration | `make test-integration-mock` | 17 | Real-mode adapters against local FastAPI mock servers |
| All integration | `make test-integration-all` | 17 | Aggregates all integration suites |

## Unit tests (`make test` — 510 tests, `not integration` marker)

All existing unit tests in the suite. Covers:

- `test_agents_base.py` — MockAgent, base agent contracts
- `test_api*.py` — FastAPI routes, auth, rate limits, lifespan, terminal state, websocket streaming
- `test_chaos.py` — chaos harness scenarios
- `test_concurrency.py` — concurrent fan-out
- `test_dedup_property.py` — Hypothesis property tests for dedup
- `test_deploy.py` — deployment config validation
- `test_errors.py` — error propagation
- `test_evals.py` — golden eval set
- `test_frontend.py` — frontend contract
- `test_inngest_workflow.py` — Inngest durable-workflow path
- `test_load_test.py` — Locust load-test harness
- `test_observability_langfuse.py` — Langfuse tracer (null + live modes, SDK v2/v3/v4)
- `test_openapi.py` — OpenAPI schema contract
- `test_postgres_store.py` — Postgres RunStore backend
- `test_real_agents.py` — TavilyAgent, BraveAgent, ExaAgent (mock-mode and httpx-patched real-mode)
- `test_redis_streams_store.py` — Redis Streams RunStore (fakeredis)
- `test_semantic_dedup.py` — semantic dedup pipeline
- `test_step_record_schema_migration.py` — StepRecord schema migration
- `test_store_contract.py` — RunStore contract (memory + fakeredis)
- `test_synthesizer*.py` — Synthesizer + edge cases
- `test_websocket_streaming.py` — WebSocket streaming
- `test_workflow*.py` — WorkflowEngine + edge cases

## Mock-server integration tests (`make test-integration-mock` — 17 tests)

Introduced in the real-mode verification pass. Tests in `tests/test_real_adapter_integration.py`
exercise the production HTTP path against a real TCP socket (no httpx patching).

### What is verified

| Test class | Tests | What it exercises |
|---|---|---|
| `TestTavilyAdapterIntegration` | 3 | Real POST over TCP, fixture parsing, Authorization header presence, 5xx retry loop |
| `TestBraveAdapterIntegration` | 4 | Real GET over TCP for web + news endpoints, X-Subscription-Token header, 5xx retry |
| `TestExaAdapterIntegration` | 4 | Real POST over TCP, research-paper category routing, x-api-key header, 5xx retry |
| `TestBaseUrlEnvOverride` | 3 | TAVILY/BRAVE/EXA_API_BASE_URL env-var redirect wires adapter to local server |
| `TestMockServerSmoke` | 3 | 401 on missing auth, Exa category-to-fixture routing, force=500 cycling |

### Mock server endpoints

| Endpoint | Method | Adapter | Auth header validated | Fixture served |
|---|---|---|---|---|
| `/tavily/search` | POST | TavilyAgent | `Authorization` | `tavily_search_response.json` |
| `/brave/web/search` | GET | BraveAgent (endpoint=web) | `X-Subscription-Token` | `brave_web_response.json` |
| `/brave/news/search` | GET | BraveAgent (endpoint=news) | `X-Subscription-Token` | `brave_news_response.json` |
| `/exa/search` | POST | ExaAgent | `x-api-key` | `exa_search_response.json` or `exa_research_paper_response.json` |
| `/langfuse/api/public/ingestion` | POST | (Langfuse SDK) | `X-Langfuse-Public-Key` or `Authorization` | `{"successes": [...], "errors": []}` |

### What mock-server tests prove vs. unit-mocked tests

Unit tests (`test_real_agents.py`) patch `httpx.AsyncClient` — no real socket I/O.
Mock-server integration tests exercise:

- **Real TCP sockets**: httpx opens an actual connection to `127.0.0.1:<random port>`.
- **Real JSON serialisation**: request body is serialised by httpx and deserialised by FastAPI.
- **Real HTTP parsing**: response status codes, headers, and body traverse the real httpx response pipeline.
- **Real retry logic**: `_post_with_retry` / `_get_with_retry` issue a real second request after receiving a real 500 response — not a mock side-effect.
- **Real auth header delivery**: the mock server returns 401 if the header is absent, proving the adapter sends it on every call.

## Test count

| Point in time | Unit tests | Integration tests | Total |
|---|---|---|---|
| Before this pass | 510 | 0 | 510 |
| After this pass | 510 | 17 | 527 |
