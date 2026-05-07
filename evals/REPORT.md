# Eval Report — research-crew

**Generated:** 1970-01-01T00:00:00Z (mock pipeline)  
**Pipeline:** MockAgent (deterministic blake2b, no live APIs)  
**Questions:** 20  

> **Interpretation:** all scores are measured against the *mock* pipeline floor.
> Citation correctness is 0.0 because MockAgent returns `example.com` URLs that
> never match real expected substrings.  Keyphrase coverage is non-zero where
> MockAgent echoes back the question text in its summary.  See
> `evals/INTERPRETATION.md` for the full discussion.

## Aggregate Metrics

| Metric | Value |
| --- | --- |
| Mean citation correctness | 0.000 |
| Mean keyphrase coverage | 0.667 |
| Mean latency (ms) | 0 ms (mock pipeline) |
| p50 latency (ms) | 0 ms (mock pipeline) |
| p95 latency (ms) | 0 ms (mock pipeline) |
| Mean step count (agents/run) | 5.0 |

## Per-Category Breakdown

| Category | N | Avg Citation Correctness | Avg Keyphrase Coverage |
| --- | --- | --- | --- |
| comparative | 5 | 0.000 | 0.933 |
| factual | 5 | 0.000 | 0.667 |
| list | 5 | 0.000 | 0.533 |
| oos | 2 | 0.000 | 0.667 |
| refusal | 1 | 0.000 | 0.000 |
| trend | 2 | 0.000 | 0.667 |

## Per-Question Detail

| QID | Category | Citation Correctness | Keyphrase Coverage | Latency (ms) | Missing Keyphrases |
| --- | --- | --- | --- | --- | --- |
| factual-001 | factual | 0.000 | 0.667 | 0 ms (mock pipeline) | threading |
| factual-002 | factual | 0.000 | 1.000 | 0 ms (mock pipeline) | — |
| factual-003 | factual | 0.000 | 0.333 | 0 ms (mock pipeline) | retry, jitter |
| comp-001 | comparative | 0.000 | 1.000 | 0 ms (mock pipeline) | — |
| comp-002 | comparative | 0.000 | 1.000 | 0 ms (mock pipeline) | — |
| comp-003 | comparative | 0.000 | 1.000 | 0 ms (mock pipeline) | — |
| list-001 | list | 0.000 | 0.333 | 0 ms (mock pipeline) | idempotency, retry |
| list-002 | list | 0.000 | 1.000 | 0 ms (mock pipeline) | — |
| list-003 | list | 0.000 | 0.667 | 0 ms (mock pipeline) | 429 |
| factual-004 | factual | 0.000 | 0.667 | 0 ms (mock pipeline) | virtual nodes |
| factual-005 | factual | 0.000 | 0.667 | 0 ms (mock pipeline) | selector |
| comp-004 | comparative | 0.000 | 1.000 | 0 ms (mock pipeline) | — |
| comp-005 | comparative | 0.000 | 0.667 | 0 ms (mock pipeline) | protobuf |
| list-004 | list | 0.000 | 0.333 | 0 ms (mock pipeline) | consistency, partition tolerance |
| list-005 | list | 0.000 | 0.333 | 0 ms (mock pipeline) | single responsibility, open/closed |
| trend-001 | trend | 0.000 | 0.667 | 0 ms (mock pipeline) | task group |
| trend-002 | trend | 0.000 | 0.667 | 0 ms (mock pipeline) | gateway api |
| oos-001 | oos | 0.000 | 0.667 | 0 ms (mock pipeline) | real-time price |
| oos-002 | oos | 0.000 | 0.667 | 0 ms (mock pipeline) | request method |
| refusal-001 | refusal | 0.000 | 0.000 | 0 ms (mock pipeline) | depends, use case, trade-offs |

## Methodology Notes

- **citation_correctness**: fraction of cited URLs whose lowercased text contains
  any of the `expected_url_substrings` for that question.
- **keyphrase_coverage**: fraction of `expected_keyphrases` found (case-insensitive)
  anywhere in the synthesized report markdown.
- **step_count**: number of agent results returned per run (5 agents x 1 attempt).
- **LLM judge**: not yet implemented.  Adding one is a one-class extension;
  see `evals/INTERPRETATION.md` for the design.
