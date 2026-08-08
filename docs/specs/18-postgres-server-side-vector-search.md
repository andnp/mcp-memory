# Architecture Decision Record: Server-Side Semantic Search for Hosted Postgres

**Status:** Active, partially implemented

## 1. Decision

`mcp-memory` now supports **capability-gated server-side semantic ranking** for shared-mode Postgres instead of treating that direction as future-only.

The current implementation uses `pgvector` when it is available and the schema exposes the expected vector column, while preserving a safe fallback to the Python fetch/decode/score path when those capabilities are missing.

The rollout remains phased, but the first slices are already in code:

1. capability detection
2. diagnostics that disclose the active search mode
3. server-side ranking when the backend is ready

Further work is now about backfill/tuning/operator visibility, not about inventing the feature from scratch.

## 2. Why this decision exists

Recent investigation of slow `search_memory_records` requests showed that local MCP transport is not the dominant bottleneck.
The remaining slow path comes from semantic selection against a **hosted Postgres** backend.

Today, the Postgres vector store does the following:

1. fetch candidate embeddings from Postgres
2. decode the embedding payloads in Python
3. compute cosine similarity in Python
4. sort the scored results in Python

That means end-to-end semantic selection cost grows with both:

- the number of embedding rows fetched
- the network cost of pulling those rows across the WAN

For broad semantic fallback queries, the current path may fetch hundreds or thousands of embeddings before ranking them locally.
When Postgres is internet-hosted, the system pays both database work and transfer cost for rows that should ideally be filtered and ranked server-side.

## 3. Current implementation shape

The current shared-mode semantic path has two active modes.

### 3.1 Fallback mode

Fallback mode stores embeddings in `embeddings.embedding_json` and performs ranking outside the database.
The client-side scan is deliberately bounded by `fallback_row_cap` to protect
memory and network costs; when the cap is reached, diagnostics set
`fallback_row_cap_applied = true` and recall may be lower than server-side
vector ranking.

In simplified form, the current flow is:

$$
T_{semantic} \approx T_{db\_fetch\_rows} + T_{network\_transfer} + T_{client\_decode} + T_{client\_score} + T_{client\_sort}
$$

This remains the correct fallback shape when `pgvector` support is unavailable.

### 3.2 Capability-gated server-side mode

When both of the following are true:

- the `vector` extension is installed
- the `embeddings` table exposes `embedding_vector`

the runtime uses server-side ranking in `PostgresVectorStore.search()` and reports:

- `search_mode = "server_side_pgvector"`
- `pgvector_extension_installed = true`
- `embedding_vector_column_present = true`
- `server_side_vector_search_available = true`

When those conditions are not met, the runtime reports:

- `search_mode = "client_python_fallback"`

This means the product already ships a safe dual path rather than a purely aspirational plan.

## 4. Goals

1. Reduce semantic-search latency for shared-mode Postgres, especially under WAN conditions.
2. Reduce row transfer volume by ranking and limiting in the database.
3. Preserve current behavior when `pgvector` is unavailable.
4. Keep rollout low-risk and observable.
5. Avoid forcing all Postgres installations to upgrade in one step.

## 5. Non-goals

The first rollout is **not** trying to deliver:

- approximate-nearest-neighbor tuning on day one
- a hard dependency on `pgvector`
- perfect parity between all ranking implementations before instrumentation exists
- a broad search-stack rewrite
- changes to SQLite search behavior

## 6. Target architecture

### 6.1 Preferred path: `pgvector`

The currently shipped preferred path is:

1. store embeddings in a native vector column on Postgres
2. issue similarity-ranking queries directly in SQL
3. apply workspace and candidate filters in SQL before ranking when possible
4. return only the top $k$ scored candidates to Python

At a high level, the target query shape is:

$$
\text{score}(m) = 1 - d(\text{embedding}_m, \text{query})
$$

where $d$ is a vector distance supported by `pgvector`.

The important architectural change is not the exact distance function.
It is that ranking moves from the client to the database when the backend advertises the required capability.

### 6.2 Fallback path

If `pgvector` is not installed, or the schema does not yet contain the required vector column, the system continues using the existing client-side JSON fetch/decode/score/sort path.

That fallback must remain correct and fully supported throughout rollout.

## 7. Rollout status and remaining plan

### 7.1 Slice 1 — capability detection and diagnostics

**Status:** shipped

The Postgres vector store now includes a cached capability probe that answers:

- is the `vector` extension installed?
- does the `embeddings` table expose the expected vector column?
- which search mode is currently active?

This slice is no longer behavior-preserving in isolation because later slices now build on it in the shipped runtime.

### 7.2 Slice 2 — additive schema support

**Status:** partially shipped

Add a migration that introduces a nullable vector column for embeddings, along with any needed indexes.

This remains additive.
The runtime still preserves JSON embedding storage as the portable fallback representation.

### 7.3 Slice 3 — dual-write / repair path

**Status:** partially shipped

When vector storage is available, write both:

- the existing JSON embedding payload
- the new vector column

The current code already dual-writes `embedding_json` plus `embedding_vector` when server-side vector search is available. Broader backfill/repair remains the follow-up concern.

### 7.4 Slice 4 — server-side ranking path

**Status:** shipped behind capability checks

`PostgresVectorStore.search()` now uses server-side similarity ranking when the capability probe says the backend is ready.

Current behavior:

- SQL filtering remains authoritative
- SQL ranking returns only the top limited rows
- Python fallback remains available if the capability probe fails or the backend is not ready

### 7.5 Slice 5 — operator visibility and tuning

**Status:** partially shipped

Add diagnostics and observability that make it easy to answer:

- which mode is active?
- how often does the system fall back?
- how many candidate rows are being ranked?
- where is time spent?

Search diagnostics already expose the active semantic mode and capability state. Optional later work can evaluate ANN indexes, richer operator controls, and comparative telemetry.

## 8. Operational constraints

1. Hosted Postgres remains part of the latency budget even when local daemon IPC is healthy.
2. Capability detection must be cheap and cached.
3. Fallback behavior must not regress correctness.
4. The migration path must tolerate mixed environments during rollout.

## 9. Risks to avoid

1. Treating capability detection as a behavior change.
2. Requiring `pgvector` before operator environments are ready.
3. Dropping the JSON fallback too early.
4. Hiding which search path is active.
5. Introducing repeated catalog probes on every search.

## 10. Consequences

This decision now gives the product an active path—not just a future direction—for reducing WAN-sensitive semantic-search latency without forcing an all-or-nothing storage migration.

The trade-off is a dual-path system:

- one path optimized for portability and compatibility
- one path optimized for hosted Postgres performance

That complexity is acceptable because the current performance problem is real and the phased rollout keeps risk low.

## 11. Current baseline

The current shipped baseline already does the following:

1. add cached capability detection to `PostgresVectorStore`
2. surface the capability state in search diagnostics
3. explicitly report whether the current mode is `client_python_fallback` or `server_side_pgvector`
4. use server-side ranking when the backend is ready
5. preserve `embedding_json` fallback storage even when vector writes are enabled

This means the remaining work is about migration hygiene, backfill/repair, and performance/observability refinement.

## 12. Follow-up work

1. document the current additive migration/vector-column requirement more explicitly in operator docs
2. integrate background repair/backfill for older rows
3. expand telemetry to compare fallback and server-side paths under real workloads
4. evaluate ANN/index tuning once real hosted workloads justify it
5. decide when the fallback path can be considered secondary rather than primary compatibility behavior
