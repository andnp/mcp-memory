# Architecture Decision Record: Queue-Backed Search Repair

**Status:** Proposed direction
**Date:** 2026-03-24

This is a proposed architectural direction, not the canonical source of current runtime behavior.
Prefer active specs, runbooks, and the root `README` for implemented behavior.

## 1. Context

`mcp-memory` currently repairs missing or stale memory embeddings inside the search request path.

That approach preserves semantic freshness, but it creates an undesirable coupling:

- a user search can trigger large repair work before any response is returned
- daemon transport timeouts become coupled to semantic index maintenance
- one query can accidentally become surprise backlog catch-up work
- search latency becomes more sensitive to stale embedding volume than to the actual query itself

Recent debugging made the failure mode explicit:

- the daemon request path was timing out during memory search
- the transport timeout problem was real, but the deeper latency driver was hot-path embedding repair

The product goal for search should be:

> return the best answer available within a bounded latency budget, while still ensuring missing repair work is durably completed afterward.

## 2. Decision

Move memory-embedding repair toward a **queue-backed, best-effort search model**.

The preferred behavior is:

1. detect missing or stale embeddings during search
2. enqueue the needed repair work durably
3. wait only within a bounded request budget
4. run search with whatever repaired state is available by the deadline
5. keep draining the remaining repair backlog in the background

This changes the semantic contract from:

> search must finish repair before it may answer

to:

> search should opportunistically benefit from repair progress, but repair completion is not a prerequisite for answering.

## 3. Critical Refinement

The wait condition should **not** be “drain the whole repair queue.”

That would couple one request to unrelated backlog and create head-of-line blocking.

Instead, the search path should wait for **its own requested repair set**—or, more precisely, the subset of durable repair work relevant to that request.

The correct shape is:

- enqueue a request-relevant repair set durably
- wait up to a fixed budget for that set to make progress or complete
- then continue with best-effort search

## 4. Alignment with Existing Architecture

This direction matches the repo’s current durable-work trajectory:

- one global daemon/runtime/store authority
- background workers already driven by durable SQLite tasks
- an existing `work_items` abstraction with:
  - execution lanes
  - lease ownership
  - heartbeat/defer/release semantics

That makes queue-backed repair a natural extension rather than a foreign subsystem.

## 5. Preferred Runtime Shape

### 5.1 Request Path

During search:

- compute keyword candidates immediately
- detect missing/stale embeddings for semantic candidates
- enqueue durable repair units
- ensure a deterministic repair worker is scheduled
- wait up to a fixed latency budget
- execute semantic search against the embeddings that are available at deadline
- merge semantic + keyword results using the existing ranking logic

### 5.2 Background Repair Path

In the background:

- claim repair work in deterministic batches
- generate embeddings in batch
- upsert repaired vectors
- mark repair units complete
- continue draining until the batch/run budget is exhausted or the queue is empty

## 6. Durable Repair Unit Semantics

The durable repair identity must be version-aware.

At minimum, it should deduplicate by:

- `memory_id`
- `model_name`
- `memory_updated_at` (or equivalent source version)

That preserves the invariant that:

- repeated searches do not endlessly enqueue duplicate repair work for the same version
- repairs queued for an older memory version can be safely skipped when the memory changes

## 7. Search Contract

Search should become explicitly **best-effort semantic** rather than “all-or-nothing semantic freshness.”

That means:

- keyword retrieval remains immediately available
- semantic retrieval may be partial when repair backlog exists
- graph/reranking logic should continue to operate over the results that are actually available
- the request should not fail just because the full repair set did not complete in time

This is consistent with the project’s existing resilience principle that semantic-path issues should degrade safely rather than fail the whole request.

## 8. Worker and Queue Isolation

Repair work should stay on the **deterministic lane**.

It should not compete directly with agentic maintenance sessions for execution semantics.

Important consequences:

- repair backlog should not wait behind unrelated agentic-provider work if the request is depending on quick deterministic progress
- queue selection should avoid coupling one search request to unrelated agentic backlog
- repair workers should use batch-oriented deterministic execution, not provider loops

## 9. First Implementation Slice

The first slice does **not** need the perfect final backlog model.

It can use the current architecture incrementally by:

- representing repair units through the existing durable `work_items` mechanism
- adding a deterministic `embedding-repair` task handler to drain those items in batches
- enabling bounded request waiting only in the daemon/runtime path where a background worker actually exists

This slice is intentionally pragmatic:

- it moves repair out of the search hot path
- it proves the request-budget behavior
- it reuses the existing worker/control-plane machinery

The likely later refinement is to move from one work item per memory toward a more specialized repair-backlog model once behavior and observability are stable.

## 10. Invariants

Any implementation should preserve the following:

1. **No hard dependency on full repair completion**
   - search must still answer within budget

2. **Durable repair completion**
   - once queued, repair should continue draining in the background

3. **Version-aware dedupe**
   - avoid duplicate repair storms

4. **Safe staleness skipping**
   - queued work for obsolete memory versions must not overwrite newer state

5. **Deterministic execution lane**
   - repair is local, batchable, and provider-free

6. **Request-scoped waiting semantics**
   - do not make one search wait on unrelated queue backlog

## 11. Observability Expectations

The architecture should eventually expose:

- pending repair count
- oldest repair age
- repairs queued by searches
- repairs completed within request budget vs background only
- partial-semantic search rate

The first slice does not need full dashboard support, but it should preserve room for these metrics.

## 12. Recommended Migration Order

### Phase 1 — Durable off-hot-path repair

- queue repair units durably
- add deterministic repair task handler
- wait within bounded request budget
- continue with best-effort search

### Phase 2 — Improve observability

- track partial semantic coverage and repair backlog state
- surface request-budget outcomes in health or debug output

### Phase 3 — Refine repair storage model

- evaluate whether generic work items remain sufficient
- move to a specialized repair-backlog table if work-item granularity becomes too noisy

## 13. Decision Summary

The preferred direction is:

- **queue repair durably**
- **wait on the request’s repair set, not the whole queue**
- **bound request latency**
- **search with whatever semantic state is available at deadline**
- **finish the rest in the background**

In short:

> repair should help search, not hold search hostage.
