# Architecture Decision Record: Shared-Mode Readthrough Cache and Validation-Based Local Read Cache

**Status:** Proposed direction

## 1. Decision

`mcp-memory` should support an optional **readonly local readthrough cache** for **shared Postgres mode**.

This cache is intended to improve WAN and intermittent-connectivity read UX for:

- `search_memory_records`
- `read_memory_record`

The cache is **not** authoritative.
Postgres remains authoritative at all times.

The first cache implementation should support:

1. **exact hot search-response caching** with conservative freshness rules
2. **cached memory-record payloads** with validation-based freshness extension
3. **cached search projections** for degraded local search over recently seen records

The first cache implementation must **not** support:

- local writes
- writeback or replay
- bidirectional SQLite sync
- transparent offline parity with authoritative search
- internal maintenance-tool reads from cache

## 2. Why this decision exists

Search latency on the current LAN path is now mostly acceptable in warm steady state.
The remaining search costs are in the tens of milliseconds, not seconds.

For roaming use, however, end-to-end latency is no longer dominated only by server-side search time.
It also includes network round trips:

$$
T_{total} \approx T_{network\_roundtrip} + T_{server\_search} + T_{serialization/client}
$$

That changes the optimization target.
For WAN use, a local read cache can provide more product value than continued small server-side latency wins.

The storage architecture already recognizes this direction:

- SQLite is the default local backend
- Postgres is the authoritative shared backend
- a future shared-mode local cache is desirable
- writeback must remain deferred until the primary shared-mode path is stable

This ADR narrows that future cache work to a first useful, low-risk shape.

## 3. Scope

### 3.1 In scope

- shared Postgres mode only
- readonly local SQLite sidecar cache
- exact-key search-response cache
- cached read payloads for memory records
- cached search projections for recently seen memories
- degraded cached reads when authoritative access fails
- validation-based freshness extension for object caches

### 3.2 Out of scope

- default local SQLite mode
- writeback outbox
- queued mutations or replay
- cache-aware maintenance agents
- full offline search parity
- active-active replication between SQLite and Postgres

## 4. Authority rules

The following invariants apply:

1. **Postgres remains authoritative in shared mode.**
2. **The local cache is derivative and disposable.**
3. **The cache must never become a hidden second source of truth.**
4. **Writes remain online-only in the first cache implementation.**
5. **Internal maintenance and background mutation flows remain authoritative-only.**

## 5. Cache model

The cache should live in a **separate local SQLite sidecar database**.
It should not reuse the authoritative SQLite mode database.
It should be safe to delete and rebuild.

### 5.1 Cached search results

Use a `cached_search_results` table keyed by an exact normalized request.

Suggested cache key inputs:

- query text
- effective workspace id
- `limit`
- `memory_type`
- `status`
- `include_superseded`
- cache contract version
- ranking-config hash

Suggested stored fields:

- `cache_key`
- normalized params JSON
- ordered result IDs JSON
- cached result payload JSON
- `created_at`
- `last_hit_at`
- `fresh_until`
- `stale_until`

### 5.2 Cached memory records

Use a `cached_memory_records` table keyed by `memory_id`.

Suggested stored fields:

- `memory_id`
- `record_json`
- `relationships_json`
- `superseded_json`
- `authoritative_updated_at`
- `cached_at`
- `last_hit_at`
- `fresh_until`
- `stale_until`

### 5.3 Cached search projections

Use a `cached_memory_projections` table keyed by `memory_id`.

This table is intended for degraded local search over recently seen records, not for authoritative parity.

Suggested stored fields:

- `memory_id`
- `title`
- `summary`
- `memory_type`
- `status`
- `tags_json`
- `workspace_ids_json`
- `authoritative_updated_at`
- `cached_at`
- `last_hit_at`

## 6. Freshness and validation policy

The cache should use **two different strategies**.

### 6.1 Object-cache validation for records and projections

For cached memory records and cached projections, the system should support **validation-based freshness extension**.

The authoritative backend should be able to answer a cheap version query for a batch of memory IDs, returning fields such as:

- `memory_id`
- `cache_version` or equivalent version token
- `updated_at`
- deletion / visibility state if needed

If the authoritative version is unchanged, the local cache entry can have its freshness extended without fetching the full payload again.

This allows longer soft TTLs for object caches while keeping bandwidth low.

### 6.2 Conservative freshness for exact search-result caches

Exact search-result caches should be treated more conservatively.

A cached query result can become wrong even when the currently cached result objects are unchanged, because:

- a new matching memory may appear
- a previously lower-ranked memory may overtake the cached set
- supersession or link changes may affect visibility
- ranking inputs may change outside simple content timestamps

Therefore, per-object `updated_at` validation is **not sufficient** to prove that a full cached search-response payload is still valid.

For the first version:

- exact search-response caches should use **short TTLs**
- exact search-response caches should not rely on per-object timestamp validation alone
- longer-lived exact query caches require a future **query-level validator** if needed

## 7. Online and offline behavior

### 7.1 Online and healthy

- authoritative search/read remains the default path
- successful authoritative responses warm the local cache
- exact search-response cache hits may be enabled later for short-TTL non-debug requests

### 7.2 Online but slow or failing

If the authoritative request fails or times out:

- return a cached stale response if available
- mark the response as degraded/cached
- fail normally if no usable cache entry exists

### 7.3 Offline

- cached reads may be served
- cached degraded search over local projections may be served
- results must be marked stale/degraded
- uncached requests fail normally
- writes remain unavailable

## 8. Integration boundary

The first cache implementation should be integrated at the **MCP service layer**, not by decorating low-level repositories directly.

Recommended integration points:

- `src/mcp_memory/mcp/services.py`
  - wrap `search_memory_records_service`
  - wrap `read_memory_record_service`
- `src/mcp_memory/mcp/runtime.py`
  - create the cache capability during runtime assembly
- `src/mcp_memory/storage/factory.py`
  - enable cache creation only for `storage.backend = "postgres"`
- `src/mcp_memory/context.py`
  - add the cache capability to `ApplicationContext`

This keeps the first implementation narrowly scoped to user-facing read/search flows and preserves current repository contracts.

## 9. Rollout order

### 9.1 Slice 1 — warm-on-read/search fallback cache

- add the local sidecar cache
- warm cache on successful authoritative reads/searches
- use cache only as fallback when authoritative access fails
- enable only in shared Postgres mode

### 9.2 Slice 2 — exact hot search cache hits

- allow short-TTL exact search-response hits for external non-debug calls
- keep authoritative refresh on miss
- continue warming projections and records

### 9.3 Slice 3 — object validation path

- add batched authoritative version checks for cached records/projections
- extend object freshness without full payload fetch when versions match
- keep exact query caches on conservative TTL rules

### 9.4 Slice 4 — degraded local projection search and operator visibility

- allow degraded local search over cached projections
- surface cache hit/miss and stale-serve telemetry
- expose cache mode/state in operator health surfaces

## 10. Risks to avoid

1. **Accidental second source of truth**
2. **Bad cache keys that ignore workspace/filter/version context**
3. **Promising full offline parity**
4. **Blocking fast cache paths on non-critical telemetry work**
5. **Using per-object timestamps to justify long-lived exact query caches**
6. **Extending cache use to internal maintenance flows too early**

## 11. Consequences

This decision gives shared Postgres mode a practical local-read resilience story without taking on writeback or distributed conflict semantics.

The main benefits are:

- lower perceived WAN read latency for hot data
- useful degraded-mode reads during connectivity problems
- a clean path to validation-based freshness for cached records/projections

The main trade-off is that exact query-result caches remain intentionally conservative until the system has a proper query-level validation mechanism.

## 12. Follow-up work

1. define the local cache SQLite schema and lifecycle
2. add a cache capability to runtime assembly for shared Postgres mode only
3. implement Slice 1 warm-on-read/search fallback behavior
4. define a backend version-check endpoint or repository API for batched object validation
5. decide whether additive cached/degraded metadata should be exposed in MCP payloads immediately or only in debug/operator surfaces
