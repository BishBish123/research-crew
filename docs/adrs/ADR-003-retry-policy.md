# ADR-003: Retry policy and backoff

* **Status:** accepted
* **Date:** 2026-01-19

## Context

Specialist agents are remote calls. They can fail for transient
reasons (rate limit, brief network partition, search-vendor 5xx) or
for permanent reasons (bad query format, removed API surface). The
runner needs a policy that:

* recovers from transients without operator intervention,
* gives up on permanents quickly enough that users aren't waiting
  on a doomed attempt,
* never re-fires forever.

## Options considered

* **No retries.** Simple, but every transient failure becomes an
  end-user-visible error.
* **Linear backoff (constant sleep).** Adds delay but doesn't reduce
  load on a struggling backend.
* **Unbounded exponential backoff.** Fixes "thundering herd" but
  blows the latency budget for an interactive endpoint.
* **Bounded exponential backoff.** Caps total delay, gives a
  struggling backend room to recover, and the ceiling means a
  request that's never going to succeed bails predictably.

## Decision

* `max_attempts = 3` (configurable via `WorkflowConfig`).
* Sleep between attempts: `base_backoff_s * 2 ** (attempt - 1)`,
  multiplied by a uniform `[0.75, 1.25]` jitter factor so parallel
  retries don't synchronise on the same wall-clock instant. With five
  fan-out agents all hitting the same upstream during a transient
  outage, deterministic backoff would re-fire every attempt at the
  same second; the jitter spreads them across a 50%-wide window per
  attempt, breaking the thundering-herd loop without changing the
  expected latency budget.
* `base_backoff_s = 0.05` so worst-case sleep before jitter is
  0.05 + 0.10 = 0.15s spread across two retries; with jitter that
  becomes [0.1125, 0.1875]s.
* Per-step wall-clock timeout (`per_step_timeout_s`, default 30s)
  protects against an attempt that simply *hangs* without raising.
* `asyncio.CancelledError` is recorded once and re-raised — we never
  retry through cancellation; the surrounding task tree is unwinding.
* The last attempt does *not* sleep — once the budget is spent, sleep
  is wasted.

## Consequences

* **Good:** maximum latency for a run with 5 agents that all retry to
  the limit is bounded by `5 × (3 × 30s + 0.15s) ≈ 7m30s`, which is
  the right ceiling for a user-visible research request. In practice
  agents either succeed on the first attempt (~50ms) or fail all
  three (~90ms total sleep).
* **Good:** retries are transparent to the user — they see one
  `AgentResult` with `attempts=3`. The per-step audit
  (`GET /runs/{id}`) shows every individual attempt for debugging.
* **Trade-off:** a permanent failure burns three attempts before
  giving up. The alternative (classify errors as retryable vs.
  non-retryable) requires adapter cooperation we don't have today.
  Adapters that learn the difference can short-circuit by raising
  `AgentError` subclasses we don't retry — that's a future commit.
