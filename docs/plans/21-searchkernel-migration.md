# Searchkernel Migration and Primitive Extraction Plan

**Status:** Shadow/cutover work complete; remaining items are follow-up stabilization
**Date:** 2026-07-30
**Scope:** `mcp-memory` and `../andnp-searchkernel`

## Summary

Make `searchkernel` the reusable retrieval and indexing toolkit while keeping
`mcp-memory` authoritative for memory semantics, relational storage, lifecycle,
curation, and runtime orchestration.

This is not a package rename or a direct replacement of the memory repository.
The migration should use searchkernel ports, stages, and adapters around the
existing memory stores. Memory remains a domain-owned source that can also be
exposed through searchkernel federation.

The standalone extraction core is already substantially implemented. The
mcp-memory retrieval cutover is complete: searchkernel is now the sole
production retrieval path, and native comparison is retired after accepted
parity and latency evidence. Remaining items are live ingest-to-search and
multi-source verification, public package/API stabilization, and removal of
unrelated duplicated memory-side primitives.

## Current capability inventory

### Searchkernel

The standalone library currently provides:

- Source-agnostic `Record`, `Chunk`, search-result, provenance, and status types.
- `ContentSource` and `SearchableSource` ports.
- Declarative query pipelines with:
  - vector and keyword retrieval
  - RRF fusion and score calibration
  - recency and project/source filtering
  - graph expansion
  - tag expansion
  - exact, n-gram, and semantic deduplication
  - cross-encoder reranking
  - parent expansion, hydration, and provenance
- Vector, keyword, graph, and cache ports.
- FAISS/SQLite and Postgres/pgvector index adapters.
- Incremental indexing, manifests, reconciliation, chunking, content hashes,
  embedding planning, and index epochs.
- Query-embedding caching, result caching, federation fan-out, runtime tracing,
  and source registries.
- Hugging Face embedding/reranker adapters, Copilot LLM integration, and
  evaluation helpers.

### mcp-memory

The current memory search and indexing path owns:

- SQLite FTS5 and Postgres weighted full-text search.
- SQLite JSON-vector and Postgres/pgvector semantic search.
- Candidate-bounded vector fallback and Postgres server-side filtering.
- Technical-token keyword fast paths and semantic-only abstention.
- RRF, score calibration, adaptive result limits, and detailed diagnostics.
- Memory-specific ranking for:
  - workspace membership
  - type-aware recency
  - access decay
  - authority links
  - graph support
  - stale/degraded status
  - supersession
- Inline and durable embedding repair, integrity scans, stale detection, and
  fallback-embedding write policies.
- Deterministic fact deduplication, observation absorption, canonical selection,
  and agentic merge workflows.
- Shared-mode read-through caching, in-flight search coalescing, validated
  projections, and narrow writeback behavior.

## Target architecture

### Searchkernel owns

- Generic domain contracts and ports.
- Reusable retrieval, fusion, ranking, graph, deduplication, embedding, caching,
  and evaluation primitives.
- Source-agnostic pipeline execution and tracing.
- Optional storage adapters that do not impose a source's domain schema.

### mcp-memory owns

- The `memories`, links, workspaces, embeddings, repair queues, and mutation
  history schemas.
- Memory lifecycle and supersession rules.
- The `degraded` status and memory-specific status policy.
- Workspace, access, authority, recency, and curation policy.
- Canonical fact selection, observation absorption, and archive/merge behavior.
- MCP, daemon, task, maintenance, and management surfaces.

### Integration boundary

Build a memory adapter layer over the existing repository and vector stores:

- `MemoryRecordAdapter`
- `MemoryKeywordStore`
- `MemoryVectorStore`
- `MemoryGraphStore`
- `MemoryHydrator`
- `MemoryEmbeddingSink`
- `MemoryRepairQueue`
- `MemorySearchPolicy`

The adapter layer should preserve authoritative memory behavior while allowing
searchkernel's generic stages to execute the reusable portions of the query.

## Blocking compatibility issues

1. `mcp-memory` still declares and locks `mcp-markdown-ragdocs`, not
   `andnp-searchkernel`.
2. `mcp-memory` imports `searchkernel.ingestion`, but the checked-out standalone
   source does not currently contain that module. The installed environment
   still exposes the older implementation.
3. Searchkernel's current `SearchOrchestrator` is chunk/file-oriented, while
   memory search is record/relationship-oriented.
4. Searchkernel's Postgres adapter owns a generic `records` schema; it cannot
   replace mcp-memory's relational `memories` and link schema directly.
5. Searchkernel models `active`, `stale`, and `archived`; memory also needs
   `degraded`.
6. The `search_anything` implementation currently performs one late reranker
   pass over merged candidates, while its documentation describes RRF
   federation. The contract must be made explicit before adoption.
7. Cache policies differ between the two repositories. Memory-specific TTL,
   capacity, and invalidation behavior must be preserved during migration.

## Phased implementation plan

### Phase 0: Establish package and contract parity

- Reconcile the standalone public API with every `searchkernel` import currently
  used by mcp-memory.
- Add or replace the ingestion surface:
  - `EmbeddingInput`
  - `embed_in_batches`
  - `embed_and_upsert`
- Define supported public exports and mark internal modules as private.
- Add import and API compatibility tests in both repositories.
- Use an editable local dependency while the API is changing.
- Replace the legacy git dependency only after the standalone package passes the
  compatibility suite.
- Decide whether heavyweight providers and database drivers belong in optional
  extras rather than the base package.

### Phase 1: Extract cohesive generic primitives

Extract by contract and cohesion, not by usage count.

#### Embedding and indexing

- Batch embedding inputs and deterministic ordering.
- Content-hash embedding identity and cache planning.
- Model/version/dimension namespaces.
- Version-safe, stale-aware embedding upserts.
- Integrity scanning and generic repair planning.
- Progress and diagnostic reporting.

#### Retrieval and ranking

- Candidate-aware vector search.
- Keyword/semantic candidate fusion.
- RRF and score calibration.
- Ranking signals and provenance.
- Adaptive result limits.
- Query routing for artifact-like and conceptual queries.
- Configurable retrieval budgets.

#### Graph and deduplication

- Bounded graph expansion with typed edge weights.
- Expansion provenance and seed tracking.
- Exact-content, n-gram, and semantic deduplication.
- Similarity clustering and reusable tag-overlap helpers.

#### Runtime and cache

- Single-flight request coalescing.
- TTL/LRU query embedding cache.
- Epoch-based result invalidation.
- Stale-while-revalidate helpers.
- Reusable stage timing and trace records.

#### Evaluation

- Golden-query harness for source adapters.
- Ranking metrics and latency thresholds.
- Retrieval regression reports that can compare native and kernel paths.

### Phase 2: Build mcp-memory adapters

- Adapt `MemoryRecord` into a searchkernel record view without changing the
  authoritative memory schema.
- Implement keyword and vector ports over the existing SQLite and Postgres
  implementations.
- Add candidate-ID filtering as a formal port capability rather than an
  adapter-specific attribute.
- Map typed memory links into the graph port while enforcing memory lifecycle
  and supersession rules.
- Provide a hydrator that returns authoritative memory records and relationships.
- Implement embedding sinks that preserve memory version checks and fallback
  persistence policy.
- Keep repair queue persistence and task execution in mcp-memory.
- Inject memory policy stages for workspace, access, authority, degradation,
  status, and supersession behavior.

### Phase 3: Run native and kernel paths in shadow mode — complete

Create a memory-specific golden query set and compare both paths on:

- Recall@k and NDCG@k.
- Irrelevant-result rate.
- p50 and p95 latency.
- Semantic fallback and abstention rates.
- Candidate counts and graph expansion contribution.
- Embedding repair backlog and partial-search rates.
- Cache hit and request-coalescing rates.
- Score explanations and provenance completeness.

The comparison evidence was accepted for parity and latency. Native
comparison and its shadow controls are retired; searchkernel is authoritative.

### Phase 4: Cut over and remove duplicate code — retrieval cutover complete

- Route production memory search through the kernel pipeline.
- Replace local ingestion and similarity helpers with searchkernel APIs.
- Replace the custom query-embedding cache with the kernel cache while retaining
  memory-specific configuration.
- Replace generic graph expansion and deduplication code.
- Preserve memory-specific merge, archive, and canonicalization code.
- Remove the legacy package dependency and compatibility shims.
- Move library-owned tests to searchkernel and keep memory-policy tests local.

### Phase 5: Release and federation

- Publish a stable `andnp-searchkernel` release.
- Add explicit versioned contracts for source adapters and pipeline stages.
- Use `MemorySearchableSource` for cross-source federation with Git, Jira,
  notes, analytics, and other adapters.
- Clarify whether federation is RRF, late reranking, or a configurable
  combination, and test that behavior directly.
- Add live multi-source and ingest-to-search smoke tests.

### SearchKernel cutover verification

Run these checks from the same environment that will launch the daemon:

```bash
uv lock --check
uv run python -c "from importlib.metadata import version; assert version('andnp-searchkernel') == '0.18.0'"
uv run python -c "import mcp_memory.daemon_runtime; import searchkernel"
uv run pytest -q tests/medium/test_searchkernel_ingest_search_parity.py
```

The version assertion verifies the locally resolved package used by the live
runtime; it does not claim or require an upstream release announcement.

## Candidate extraction matrix

| Primitive | Destination | Keep domain policy in mcp-memory? |
| --- | --- | --- |
| Batch embedding and content-hash planning | `searchkernel.indexing` | Yes |
| Version-safe embedding sink contract | `searchkernel.ports` | Yes |
| Integrity scan and repair planning | `searchkernel.indexing` | Yes |
| RRF, calibration, and ranking signals | `searchkernel.search` | Weights and policy |
| Technical-token query routing | `searchkernel.search` | Thresholds and memory exceptions |
| Candidate-bounded vector search | `searchkernel.ports` and adapters | Storage implementation |
| Bounded graph expansion | `searchkernel.search` | Link eligibility and lifecycle |
| Exact/ngram/semantic deduplication | `searchkernel.search` | Merge/archive decisions |
| Similarity and tag-overlap utilities | `searchkernel.utils` | Memory taxonomy |
| Query/result caches and single-flight | `searchkernel.runtime` | Cache authority and writeback |
| Retrieval diagnostics and provenance | `searchkernel.search` | Product presentation |
| Fact canonicalization and observation absorption | Stay in mcp-memory | Yes |
| Supersession, degraded status, and access telemetry | Stay in mcp-memory | Yes |
| Curation, task queues, MCP, daemon, and management | Stay in mcp-memory | Yes |

## Improvements to prioritize in mcp-memory

1. Use one content-hash embedding planner across ingest, deduplication, repair,
   and rebuild tasks.
2. Add retrieval budgets so artifact-like queries avoid unnecessary semantic
   work and ambiguous queries can opt into semantic or graph enrichment.
3. Add optional local cross-encoder reranking only after native candidate
   selection, guarded by latency and quality budgets.
4. Expose stage-level provenance in MCP debug payloads, CLI output, and the
   dashboard.
5. Tie memory mutation epochs directly to result-cache invalidation.
6. Make graph expansion status-aware so archived and superseded records cannot
   re-enter through neighbors.
7. Use stale-while-revalidate embedding repair so search does not wait on the
   entire repair set.
8. Build a shared retrieval evaluation suite covering relevance, latency,
   fallback behavior, repair health, and cache behavior.
9. Use searchkernel federation to search memory alongside other source-owned
   systems without copying them into the memory database.
10. Refactor the large relational search service into a kernel pipeline plus
    injected memory policy stages.

## Non-goals

- Do not replace the memory relational repository with searchkernel's current
  generic Postgres schema.
- Do not move MCP, daemon, task, curation, mutation-history, or management
  concerns into the library.
- Do not flatten memory-specific lifecycle states into generic active/stale
  behavior without an explicit policy contract.
- Do not enable expensive reranking or LLM-based query expansion by default
  before latency and relevance evaluation.
- Native retrieval and native comparison are retired after accepted parity and
  latency evidence. Curation shadow infrastructure remains outside this
  migration and is not removed here.

## Acceptance gates

### Package gate

- `mcp-memory` imports only the standalone package.
- No runtime import depends on `mcp-markdown-ragdocs`.
- All required public APIs are present in the released package.

### Behavior gate

- Native and kernel search agree on lifecycle filtering, supersession, and
  workspace behavior.
- SQLite and Postgres backends preserve their existing authority and fallback
  semantics.
- Embedding repair remains idempotent and version-safe.

### Quality gate

- Golden-query relevance does not regress.
- p95 search latency does not regress without an intentional feature change.
- Search diagnostics identify the retrieval and ranking path taken.
- Searchkernel import boundaries remain dependency-safe.

### Operations gate

- Cache invalidation is correct after memory mutation.
- Repair backlog and partial semantic-search health remain observable.
- Kernel-path failures have an explicit fallback or surfaced error contract.
- Multi-source and ingest-to-search smoke tests pass before legacy removal.

## First implementation work items

1. Reconcile and publish the standalone ingestion API.
2. Replace the legacy dependency with a local `andnp-searchkernel` path
   dependency.
3. Add the memory adapter ports and candidate-aware vector capability.
4. Extract generic embedding repair and retrieval diagnostics.
5. Build the first kernel-backed memory search path behind a feature flag.
6. Create the golden-query and shadow-comparison harness. **Complete; native
   comparison is now retired after accepted parity and latency evidence.**
7. Extract generic deduplication and graph-expansion helpers.
8. Remove duplicated helpers after parity is established.
