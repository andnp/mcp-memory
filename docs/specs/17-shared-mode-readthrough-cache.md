# Architecture Decision Record: Shared-Mode Readthrough Cache and Narrow Record-Thought Writeback

**Status:** Active, partially implemented

## 1. Decision

`mcp-memory` now supports an optional local SQLite sidecar cache for **shared Postgres mode**.

This cache improves bounded-latency and degraded-read UX for:

- `search_memory_records`
- `read_memory_record`

The cache is **not** authoritative.
Postgres remains authoritative at all times.

### Current implementation

1. `storage.cache.mode = "readonly"` provides readthrough and degraded cached behavior for `search_memory_records` and `read_memory_record`.
2. Exact hot search-response cache hits are active with conservative short-lived freshness.
3. Cached memory-record payloads and cached search projections use authoritative validation tokens.
4. Cached search projections can power degraded local search over recently seen records when authoritative search fails.
5. `storage.cache.mode = "writeback"` includes all readonly behavior and adds a durable local outbox for `record_thought` only.
6. Successful authoritative `record_thought` writes opportunistically flush older queued outbox entries.
7. The daemon owns a periodic background flusher for queued `record_thought` outbox entries.

### Still deferred

- generalized writeback/replay beyond `record_thought`
- offline mutation for maintenance or other public tools
- cache-backed internal maintenance reads by default
- full offline parity with authoritative search
- bidirectional SQLite/Postgres sync or dual-write

## 2. Why this decision exists

Search latency on the current LAN path is already much better than the earlier timeout-heavy baseline.
For roaming/shared-mode use, the remaining user-visible cost is often the network round trip rather than only server-side search time.

That shifts the optimization target.
For shared mode, a small local sidecar cache provides more value than continuing to squeeze only server-side milliseconds, as long as the cache remains explicitly non-authoritative.

The storage architecture is no longer merely reserving this direction:

- SQLite is still the default local backend
- Postgres is the authoritative shared backend
- the shared-mode local cache is now active
- writeback now exists, but only in a narrow `record_thought` slice

This ADR documents the shipped shape and keeps the boundary crisp between active behavior and still-deferred broader offline mutation.

## 3. Scope

### 3.1 In scope

- shared Postgres mode only
- local SQLite sidecar cache
- exact-key search-response cache
- cached read payloads for memory records
- cached search projections for recently seen memories
- degraded cached reads when authoritative access fails
- validation-based read/projection freshness checks
- narrow `record_thought` writeback outbox and flush behavior in `storage.cache.mode = "writeback"`

### 3.2 Out of scope

- default local SQLite mode
- generalized queued mutations or replay
- cache-aware maintenance agents by default
- full offline search parity
- active-active replication between SQLite and Postgres
- broader writeback for non-`record_thought` mutations

## 4. Authority rules

The following invariants apply:

1. **Postgres remains authoritative in shared mode.**
2. **The local cache is derivative and disposable.**
3. **The cache must never become a hidden second source of truth.**
4. **Only `record_thought` may currently degrade into the local outbox, and only in `writeback` mode.**
5. **Internal maintenance and background mutation flows remain authoritative-only.**

## 5. Cache model

The cache lives in a **separate local SQLite sidecar database**.
It does not reuse the authoritative SQLite-mode database.
It is safe to delete and rebuild.

The current default sidecar path is under the runtime memory root:

- `.../memories/cache/shared_read_cache.sqlite3`

### 5.1 Cached search results

Current implementation uses `cached_search_results`, keyed by an exact normalized request.

The stored payload is a full cached search response plus normalized request parameters and cache timestamp.

### 5.2 Cached memory records

Current implementation uses `cached_memory_records`, keyed by `memory_id`.

Each row stores the cached read payload, an authoritative validation token, and cache timestamp.

### 5.3 Cached search projections

Current implementation uses `cached_memory_projections`, keyed by `memory_id`.

These rows store compact record projections plus validation tokens for degraded local search over recently seen records.

### 5.4 Record-thought writeback outbox

When `storage.cache.mode = "writeback"`, the same sidecar also stores `record_thought_outbox` rows.

Each queued row stores:

- raw thought content
- workspace association
- original thought timestamp
- local cache timestamp

This outbox is intentionally narrow.
It is not a general mutation queue.

## 6. Freshness and validation policy

The current cache uses two different strategies.

### 6.1 Object validation for records and projections

Cached memory records and cached projections rely on authoritative validation tokens.

If the authoritative token still matches, the cached object can be used directly.
If validation fails or the token no longer matches, the runtime falls back to the authoritative path or discards stale projection rows.

### 6.2 Conservative freshness for exact search-result caches

Exact search-result caches remain intentionally conservative.

A cached query result can become wrong even when the cached objects are unchanged, because:

- a new matching memory may appear
- a previously lower-ranked memory may overtake the cached set
- supersession or link changes may affect visibility
- ranking inputs may change outside simple content timestamps

Therefore, exact search-response caches use short-lived freshness and do not rely only on per-object validation.

## 7. Online and degraded behavior

### 7.1 Online and healthy

- authoritative search/read remains the default path
- successful authoritative responses warm the local cache
- external non-debug requests may hit the fresh exact search cache directly
- validated cached reads may short-circuit a fresh authoritative fetch when the validation token still matches
- in `writeback` mode, successful authoritative `record_thought` writes can opportunistically flush older queued outbox entries

### 7.2 Online but slow or failing

If the authoritative request fails or times out:

- cached stale search responses may be served
- degraded projection-backed search may be served
- cached stale read responses may be served
- responses are marked as degraded/cached
- in `writeback` mode, `record_thought` may queue into the local outbox when the authoritative write fails with a timeout-ish/connectivity failure

### 7.3 Offline or disconnected

- cached reads may be served if present
- degraded search over cached projections may be served if present
- uncached reads/searches fail normally
- only `record_thought` can currently queue locally, and only in `writeback` mode
- broader writes remain unavailable

## 8. Integration boundary

The shipped cache implementation is intentionally narrow and lives above the repository layer.

Current integration points:

- `src/mcp_memory/storage/postgres.py`
  - creates the sidecar only for `storage.backend = "postgres"`
- `src/mcp_memory/mcp/services.py`
  - handles fresh search hits, validated read hits, cache warming, and degraded cached fallbacks
- `src/mcp_memory/core/journal_operations.py`
  - owns narrow `record_thought` outbox queueing and opportunistic foreground flush
- `src/mcp_memory/daemon_app.py`
  - owns the periodic background flusher for queued `record_thought` outbox entries
- `src/mcp_memory/management/service.py`
  - exposes cache mode/state/path and cache metrics in operator health surfaces

This keeps the current implementation scoped to user-facing read/search flows plus the narrow `record_thought` durability path.

## 9. Current slice status

### 9.1 Shipped today

- local sidecar cache in shared Postgres mode
- warm-on-read/search cache population
- short-lived fresh exact search hits for external non-debug requests
- validated cached read hits
- degraded cached read/search fallbacks
- operator-visible cache mode/state/path and metrics
- narrow `record_thought` writeback outbox with foreground and daemon-owned background flush paths

### 9.2 Still deferred

- broader writeback for other mutation families
- maintenance-tool reads from cache by default
- richer query-level search validation beyond short-lived exact cache hits
- full offline parity or conflict-resolution workflows

## 10. Risks to avoid

1. **Accidental second source of truth**
2. **Bad cache keys that ignore workspace/filter/version context**
3. **Promising full offline parity**
4. **Over-claiming `writeback` as a general offline mutation mode**
5. **Blocking fast cache paths on non-critical telemetry work**
6. **Extending cache use to internal maintenance flows too early**

## 11. Consequences

This decision gives shared Postgres mode a practical local-read resilience story and a narrow `record_thought` durability fallback without taking on general distributed writeback semantics.

The main benefits are:

- lower perceived WAN read latency for hot data
- useful degraded-mode reads during connectivity problems
- bounded `record_thought` durability during transient authoritative failures

The main trade-off is that offline capability remains intentionally partial.
Exact query-result caches stay conservative, and broader writeback remains deliberately deferred.

## 12. Follow-up work

1. keep the operator/runbook docs aligned with the active cache/writeback slice
2. improve operator visibility for outbox depth/age and replay health
3. decide whether exact search caching needs a stronger query-level validator later
4. decide deliberately whether any mutation family beyond `record_thought` is safe enough for future writeback support
5. preserve the invariant that cache/writeback remains derivative even if the feature surface widens later
