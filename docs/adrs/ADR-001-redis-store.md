# ADR-001: Redis as the run / step / cache store

* **Status:** accepted
* **Date:** 2025-12-09

## Context

The runner needs to persist three things while a research run is in
flight:

1. The `RunStatus` snapshot a `GET /runs/{id}` returns.
2. An ordered, append-only log of every step attempt (the per-step audit).
3. An idempotency cache so repeat `run_one(...)` calls short-circuit
   without re-hitting paid search APIs.

The brief mentions Redis Streams and Inngest. We need a backing store
that is realistic for production but trivial to bring up locally.

## Options considered

* **In-memory dict** — fast, but a process restart wipes every run.
  Already used as a test fallback.
* **Postgres + a workflow table** — durable, but adds a migration tool
  and a connection-pool dependency for what is fundamentally
  K/V-shaped data.
* **Redis** — already in the stack via `docker-compose`, async client
  is mature, supports the three keyspaces with one connection.
* **Inngest's hosted state store** — externalises everything but
  requires a third-party signup that defeats "clone and try".

## Decision

Use Redis for all three keyspaces, encapsulated behind a `RunStore`
Protocol so the test suite can drop in `InMemoryRunStore` without
running Redis at all.

Three keyspaces, every one namespaced under a `prefix` (default
`research`, override via `RESEARCH_REDIS_PREFIX`):

| Key                                   | Type    | Purpose                  | TTL                |
| ------------------------------------- | ------- | ------------------------ | ------------------ |
| `{prefix}:run:{run_id}`               | string  | latest `RunStatus` JSON  | `ttl_seconds * 24` |
| `{prefix}:run:{run_id}:steps`         | list    | append-only step log     | `ttl_seconds * 24` |
| `{prefix}:step:{dedup_key}`           | string  | cached `AgentResult`     | `ttl_seconds`      |

The prefix knob lets multiple environments (dev / staging / prod) or
logical tenants share one Redis without collisions; the constructor
also accepts an explicit `prefix=` for tests that pin keys without
env-var manipulation.

The 24× factor on run/step TTL means a finished run remains queryable
for a full day even after the cache that produced it has expired —
useful for postmortems.

## Consequences

* **Good:** the API stays trivial, `make up` is one command, fakeredis
  gives end-to-end tests in milliseconds.
* **Bad:** Redis is not a queue. If we want at-least-once fan-in
  across worker processes (the obvious next-step distribution model),
  we need to graduate this to Redis Streams + consumer groups. That's
  documented as future work in `ARCHITECTURE.md`.
* **Good:** the `prefix` parameter (default `research`, env override
  `RESEARCH_REDIS_PREFIX`) lets multiple deployments or tenants share
  one Redis without collisions. Per-tenant routing on top of this is
  still future work — picking the right prefix per request is an API
  concern, not a store one.
