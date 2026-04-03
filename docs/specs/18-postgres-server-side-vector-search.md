# Architecture Decision Record: Server-Side Semantic Search for Hosted Postgres

**Status:** Proposed direction

## 1. Decision

`mcp-memory` should move shared-mode Postgres semantic search toward **server-side ranking** instead of the current client-side fetch-and-score path.

The target implementation should use `pgvector` when it is available, while preserving a safe fallback to the current Python scoring path when the extension or schema support is missing.

The rollout should be phased.
The first implementation slice should add **capability detection and diagnostics only**, without changing search behavior.

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

The current shared-mode semantic path stores embeddings in `embeddings.embedding_json` and performs ranking outside the database.

In simplified form, the current flow is:

$$
T_{semantic} \approx T_{db\_fetch\_rows} + T_{network\_transfer} + T_{client\_decode} + T_{client\_score} + T_{client\_sort}
$$

This is safe and portable, but it is the wrong shape for hosted Postgres once semantic fallback becomes broad.

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

The preferred end state is:

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
It is that ranking moves from the client to the database.

### 6.2 Fallback path

If `pgvector` is not installed, or the schema does not yet contain the required vector column, the system should continue using the existing client-side JSON fetch/decode/score/sort path.

That fallback must remain correct and fully supported throughout rollout.

## 7. Rollout plan

### 7.1 Slice 1 — capability detection and diagnostics

Add a small, cached capability probe in the Postgres vector store that answers:

- is the `vector` extension installed?
- does the `embeddings` table expose the expected vector column?
- which search mode is currently active?

This slice is intentionally behavior-preserving.
It exists to make the next rollout steps measurable and safer.

### 7.2 Slice 2 — additive schema support

Add a migration that introduces a nullable vector column for embeddings, along with any needed indexes.

This migration should be additive.
Existing JSON embedding storage should remain available during transition.

### 7.3 Slice 3 — dual-write / repair path

When vector storage is available, write both:

- the existing JSON embedding payload
- the new vector column

Backfill or repair older rows incrementally using the existing operational repair mechanisms rather than a risky one-shot migration.

### 7.4 Slice 4 — server-side ranking path

Teach `PostgresVectorStore.search()` to use server-side similarity ranking when the capability probe says the backend is ready.

Expected behavior:

- SQL filtering remains authoritative
- SQL ranking returns only the top limited rows
- Python fallback remains available if the capability probe fails or is disabled

### 7.5 Slice 5 — operator visibility and tuning

Add diagnostics and observability that make it easy to answer:

- which mode is active?
- how often does the system fall back?
- how many candidate rows are being ranked?
- where is time spent?

Optional later work can evaluate ANN indexes and operator controls.

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

This decision gives the product a clear path to reducing WAN-sensitive semantic-search latency without forcing an all-or-nothing storage migration.

The trade-off is a temporary dual-path system:

- one path optimized for portability and compatibility
- one path optimized for hosted Postgres performance

That complexity is acceptable because the current performance problem is real and the phased rollout keeps risk low.

## 11. First implementation slice

The first code change associated with this ADR should do only the following:

1. add cached capability detection to `PostgresVectorStore`
2. surface the capability state in search diagnostics
3. explicitly report that the current mode is still `client_python_fallback`

This provides the minimum scaffolding needed before changing behavior, migrations, or write paths.

## 12. Follow-up work

1. add additive vector-column migration support
2. define dual-write behavior for newly written embeddings
3. integrate background repair/backfill for older rows
4. switch `search()` to server-side ranking when capability checks pass
5. expand telemetry to compare fallback and server-side paths under real workloads