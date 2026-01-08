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

Three keyspaces:

| Key                       | Type    | Purpose                  | TTL                |
| ------------------------- | ------- | ------------------------ | ------------------ |
| `run:{run_id}`            | string  | latest `RunStatus` JSON  | `ttl_seconds * 24` |
| `run:{run_id}:steps`      | list    | append-only step log     | `ttl_seconds * 24` |
| `step:{dedup_key}`        | string  | cached `AgentResult`     | `ttl_seconds`      |

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
* **Bad:** there is no per-tenant isolation in the keyspace today.
  Adding a tenant prefix when multi-tenancy lands is a one-line change
  in `RedisRunStore`, but ADR-worthy when it happens.
