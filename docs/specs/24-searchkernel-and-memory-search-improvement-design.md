# Design: SearchKernel and mcp-memory Search Improvement Exchange

**Status:** Draft design
**Date:** 2026-08-10

This design defines the next improvements that should flow between
`mcp-memory` and `andnp-searchkernel`. It builds on the completed SearchKernel
0.22.0 integration and the evidence-led roadmap in
[23-searchkernel-post-upgrade-quality-roadmap.md](23-searchkernel-post-upgrade-quality-roadmap.md).

The design has two directions:

1. use more of SearchKernel's generic retrieval, evaluation, caching, and
   federation capabilities in mcp-memory; and
2. extract provider-neutral contracts from mcp-memory that improve SearchKernel
   without moving memory-specific authority upstream.

The proposal is deliberately incremental. The current SearchKernel 0.22.0
policy and the passing deterministic benchmark remain the rollback baseline.

## 1. Current state

`mcp-memory` already uses SearchKernel's canonical record pipeline for keyword,
vector, graph, fusion, query embeddings, hydration, and optional policy hooks.
The integration is assembled in
`src/mcp_memory/integrations/searchkernel_record_pipeline.py` and its
authoritative-store adapters are in `searchkernel_adapters.py`.

The application still owns:

- relational memory records and workspace scope;
- vector and graph persistence;
- lifecycle, supersession, and authorization rules;
- authoritative hydration and payload redaction;
- embedding repair and integrity handling;
- daemon, MCP, and management API behavior;
- cache epochs, freshness tokens, and stale-response policy; and
- memory-specific ranking adjustments such as authority, access, recency, and
  degradation penalties.

SearchKernel owns generic retrieval composition, candidate routing, fusion,
calibration, optional reranking, query embedding support, and generic outcome
contracts.

The current versioned local benchmark has 12 labeled queries and reports
Hit@1 = 1.0 and Hit@5 = 1.0. This is a quality baseline, not evidence that an
advanced ranking policy should become the default.

## 2. Goals

1. Make every search execution path describe the same observable behavior.
2. Make lane routing, candidate budgets, degradation, abstention, and duplicate
   semantics explicit and machine-readable.
3. Evaluate calibrated fusion, query expansion, and reranking safely before
   enabling any of them by default.
4. Avoid duplicate cache ownership while preserving mcp-memory's authoritative
   freshness and stale-fallback semantics.
5. Make SearchKernel's reusable evaluation and diagnostics contracts stronger.
6. Add federation only where independent systems own their own indexes or
   authorization boundaries.
7. Preserve storage authority, workspace behavior, lifecycle filtering, and
   reversible rollback.

## 3. Non-goals

This design does not:

- replace mcp-memory's relational, vector, graph, or cache stores with
  SearchKernel-native storage;
- move MCP, daemon, maintenance, provider-admission, or mutation-history
  concerns into SearchKernel;
- flatten memory lifecycle or supersession semantics into generic statuses;
- enable expensive reranking or provider-backed expansion by default;
- make live provider calls part of normal CI; or
- introduce a broad search rewrite before a labeled benchmark demonstrates a
  concrete failure.

## 4. Target architecture

The target local path remains:

```text
MCP query
  -> mcp-memory request and scope policy
  -> SearchKernel query plan
  -> keyword/vector/graph candidate lanes
  -> SearchKernel fusion, calibration, and optional reranking
  -> mcp-memory authoritative hydration and redaction
  -> compact result plus structured diagnostics
```

An optional federated path is separate:

```text
MCP or management federated query
  -> bounded SearchKernel federation executor
  -> memory source + independently owned source adapters
  -> local-rank fusion and identity/URI deduplication
  -> partial-result diagnostics
```

Federation must not become an alternate local memory pipeline. The existing
`MemoryFederationSource` remains the adapter around the canonical retrieval
facade.

## 5. Proposed workstreams

### A. Diagnostic parity and structured execution evidence

#### Problem

SearchKernel returns failures, missing IDs, candidate counts, stage timings,
cache diagnostics, and optional traces. The synchronous mcp-memory diagnostic
path exposes most of this, but the asynchronous MCP path exposes a smaller
shape. The two paths can therefore provide different evidence for the same
query.

The application also currently conflates multiple concepts: lane overlap,
multi-lane provenance, and final duplicate results are not the same condition.

#### Design

Define one application-facing diagnostic serializer over
`RecordSearchOutcome` plus a normalized execution context. The context carries
the planner decision and memory-owned signal observations that are not present
in the SearchKernel 0.22.0 outcome. Both synchronous and asynchronous search
services must use it. The serialized shape should include:

```json
{
  "degraded": false,
  "failures": [],
  "missing_record_ids": [],
  "candidate_counts": {
    "keyword": 0,
    "vector": 0,
    "graph": 0
  },
  "lane_decisions": {
    "enabled": ["keyword", "vector"],
    "budgets": {"keyword": 50, "vector": 50},
    "skipped": ["graph:awaiting_seed_confidence"]
  },
  "overlap": {
    "raw_lane_overlap_count": 0,
    "multi_lane_result_count": 0,
    "final_duplicate_count": 0
  },
  "semantic": {
    "candidate_count": 0,
    "semantic_only_candidate_count": 0,
    "abstention_count": 0,
    "abstention_rate": null
  },
  "cache_diagnostics": [],
  "timing_ms": {},
  "trace": null,
  "scope": {}
}
```

The exact wire shape may evolve, but the semantic distinctions are required.
`raw_lane_overlap_count` is nullable during the local phase because
SearchKernel 0.22.0 exposes final candidate counts and result provenance, but
not the pre-fusion identity sets needed to calculate raw overlap. The local
serializer must not infer raw overlap from final results. Once F1 is released,
the field becomes required for executions that ran more than one lane.

The semantic distinctions are:

- `raw_lane_overlap_count`: an identity appeared in multiple pre-fusion lanes;
  `null` means the installed SearchKernel contract cannot observe this;
- `multi_lane_result_count`: a returned result has multiple contributing
  strategies;
- `final_duplicate_count`: duplicate identities survived final result shaping;
- `missing_record_ids`: candidates could not be hydrated; and
- `skipped`: a planner decision, not an execution failure.

The normal compact MCP response remains unchanged. The diagnostic shape is
returned only for `debug=true`, benchmark observations, or sampled operator
telemetry.

#### Acceptance

- sync and async search produce equivalent diagnostic fields for equivalent
  outcomes;
- local tests cover zero, positive, and degraded values for the overlap fields
  the installed SearchKernel contract can observe, while asserting that raw
  overlap is not falsely reported;
- a forced keyword-only fixture proves that vector suppression is observable;
- a forced hybrid fixture proves multi-lane provenance is not reported as final
  duplication;
- after F1, a forced hybrid fixture proves positive raw lane overlap; and
- diagnostic serialization failure cannot fail the underlying search request.

### B. Cache ownership and freshness

#### Problem

The memory query adapter uses SearchKernel's process-wide query-embedding cache,
while each SearchKernel pipeline also owns a query-embedding cache. The
application additionally owns a shared response/projection cache with
freshness-token validation, policy fingerprints, stale fallback, and in-flight
coalescing.

Multiple cache owners make metrics and invalidation difficult to reason about.

#### Design

Use explicit cache layers with one owner per responsibility:

| Layer | Owner | Scope |
| --- | --- | --- |
| Query embedding | SearchKernel pipeline or an explicitly injected shared instance | Process/runtime |
| Candidate results | SearchKernel | Derivative candidates, invalidated by search epochs and policy identity |
| Hydrated records | SearchKernel, only when supplied a memory hydration-version provider | Derivative hydrated values |
| Search response/projection | mcp-memory | Cross-process response cache, freshness validation, stale fallback |

The implementation must choose one query-embedding cache strategy and measure
it before changing behavior. The current adapter also delegates to
SearchKernel's module-level `get_or_compute_query_embedding`, while each
pipeline owns another cache. The preferred direction is explicit injection of
one SearchKernel cache from the retrieval-facade/service lifetime into all
mcp-memory pipeline variants, including the standard and adaptive pipelines.
The factory must receive that owner rather than instantiate a cache per
variant. Reconcile or remove the module-level and adapter-level duplicate
paths only after parity tests pass.

Search-response cache keys must include:

- request filters and scope;
- the active SearchKernel policy version;
- the enabled-feature fingerprint; and
- any relevant authoritative search-epoch or freshness token.

The current response-cache implementation validates freshness tokens only for
records already present in a cached response. That protects updates and
deletions of returned records, but it cannot notice a newly inserted matching
record. Before relying on the fresh-hit TTL, add an authoritative epoch
snapshot to the request key or validate the snapshot alongside the payload.
Insertion, deletion, update, and embedding-repair cases must all be covered;
purging only the records named by a cached response is insufficient.

Freshness must be separated from the response-cache lookup key. The response
cache should retain a stable request key for stale fallback, store the
authoritative epoch snapshot with the payload, and validate that snapshot for
fresh hits. A stale-fallback lookup may intentionally return the older payload
after an authoritative outage, marked degraded. SearchKernel derivative-cache
keys may use epochs when that cache's invalidation semantics require it, but
that must not be conflated with the mcp-memory response-cache key.

SearchKernel should receive a provider-neutral freshness/version contract in a
later upstream change. It must not receive mcp-memory table or database
details.

#### Acceptance

- repeated queries report one cache hit/miss model rather than two opaque
  layers;
- policy changes cannot return a response produced by another policy;
- mutation and embedding-repair epoch changes invalidate affected derivative
  results;
- fresh-hit validation notices insertion, deletion, update, and
  embedding-repair epoch changes without removing the stale-fallback path;
- stale fallback remains explicitly degraded; and
- cache failures remain best effort and do not fail authoritative search;
- the global runtime query-embedding cache and pipeline-level cache have one
  documented owner and one observable hit/miss model.

### C. Controlled SearchKernel policy experiments

#### Design

Use the existing deterministic policy-comparison harness. Each policy builder
must create an isolated retrieval pipeline over the same corpus and execution
order:

1. baseline SearchKernel policy;
2. calibrated fusion;
3. query expansion; and
4. reranking, only when a bounded reranker is supplied.

Each experiment records:

- corpus and policy fingerprints;
- query-class metrics for exact, broad, historical, global, and paraphrase
  cases;
- latency percentiles and stage timings;
- degraded-case rate;
- diagnostics completeness;
- duplicate-result cases; and
- semantic-abstention rate where semantic retrieval actually ran.

The planner branch under test must be forced by the fixture or asserted in the
diagnostic output, with `require_diagnostics=true` when the harness supports
it. Assertions should name the observed planner decision (for example,
`vector:artifact_keyword_confident`) rather than relying on callback
invocation. A callback invocation alone is not evidence that the target branch
was reached if the planner can legitimately skip that lane.

#### Acceptance

An experimental policy is accepted only when its selected gates pass:

- exact Hit@1 is not worse than baseline;
- broad and historical Hit@5 meet their configured deltas;
- degradation does not increase beyond its threshold;
- p95 latency stays within its budget;
- diagnostics are complete when required;
- final duplicate cases stay within their threshold; and
- semantic abstention stays within its threshold when measurable.

No experiment changes production configuration or persistent data.

### D. MCP search-tool and operator improvements

#### Design

Keep `search_memory_records` summary-first for normal use. Add opt-in
capabilities only where they help an agent or operator make a better next
decision:

- `debug=true`: structured planner and execution diagnostics;
- optional explanation fields: contributing lanes, normalized score, and
  bounded score adjustments;
- optional excerpts: bounded SearchKernel chunk matches when available; and
- an explicit retrieval mode for troubleshooting keyword, semantic, or hybrid
  behavior if the product decides that exposing it is useful.

Expose sampled aggregate diagnostics in the management retrieval surface rather
than persisting every full query payload. Never persist query text or record
content solely for diagnostics unless an existing privacy contract explicitly
allows it.

#### Acceptance

- normal search responses remain summary-first and backward compatible;
- debug output is bounded, redacted, and clearly marked as diagnostic data;
- explanation and excerpt fields are opt-in, deterministic, and covered by
  privacy and size-limit tests; and
- sampled operator metrics report diagnostic serialization failures without
  storing query text or record content.

### E. Optional federation

#### Design

Use the existing SearchKernel federation source and executor for systems with
independent indexes or authorization boundaries, such as GitHub, Jira, or
external documentation.

The federation surface must preserve:

- source-owned authorization and filtering;
- bounded concurrency and per-source deadlines;
- source capability checks;
- local-rank fusion rather than incomparable native-score fusion;
- canonical source identity; and
- explicit partial/degraded results.

This should be a separate MCP or management capability. It is not part of the
default local memory search path until at least one real external source needs
it.

#### Acceptance

- each source enforces its own authorization and deadline;
- one source timing out does not discard successful source results;
- duplicate identities are removed only when canonical source identity proves
  they are the same resource; and
- federation remains disabled for the default local-memory tool until a real
  source and its contract tests exist.

### F. Upstream SearchKernel extraction

Extract only provider-neutral contracts that have demonstrated reuse.

#### F1. Structured search diagnostics

Add typed outcome diagnostics for lane decisions, overlap, missing hydration,
degradation, and stage timing. Keep source-specific metadata in opaque details
or application adapters.

Acceptance requires backward-compatible outcome construction, provider-neutral
tests for skipped and degraded lanes, and an explicit capability marker for
raw lane identity evidence. A consumer must be able to distinguish unavailable
evidence from a measured zero.

#### F2. Evaluation evidence and gates

Generalize the reusable parts of the mcp-memory benchmark:

- structured per-query observations;
- query-class or slice aggregation;
- diagnostics completeness;
- degradation-rate gates;
- duplicate-result gates; and
- abstention stability gates.

Memory-specific evaluation labels and corpus lifecycle remain in mcp-memory.

Acceptance requires SearchKernel tests for deterministic aggregation and gate
evaluation, while mcp-memory retains the compatibility test that maps its
query classes and labels into the generic observations.

#### F3. Freshness/version contract

Propose a small SearchKernel port for derivative-cache freshness. It should
describe version tokens, policy identity, invalidation, and stale-read behavior
without knowing whether the authority is SQLite, Postgres, or another store.

Acceptance requires a fake version provider, fresh-hit validation, stale-read
behavior, policy-identity isolation, and a proof that the contract does not
require database schema knowledge.

#### F4. Generic failure envelope

Extend SearchKernel's existing failure model only where it improves safe
partial execution for malformed candidate data or unavailable providers. Vector
row validation and embedding repair remain application-owned.

Acceptance requires safe partial results for provider failures and malformed
generic candidate data, while proving that application-owned repair is not
silently moved into SearchKernel.

## 6. Dependency order

The work must proceed in this order:

1. Freeze the SearchKernel 0.22.0 and 12-query benchmark baseline.
2. Implement diagnostic parity and the locally observable overlap semantics;
   explicitly leave raw lane overlap nullable until the upstream contract exists.
3. Measure and simplify cache ownership locally.
4. Run controlled policy comparisons.
5. Add an opt-in MCP/operator explanation surface if the evidence shows it is
   useful.
6. Add federation only for a concrete independently owned source.
7. Propose upstream contracts with provider-neutral tests, including the raw
   lane-identity evidence that cannot be derived locally today.
8. Upgrade mcp-memory to an upstream release only after the contract is
   released and the compatibility tests pass; then complete raw-overlap
   diagnostics.

Dependency upgrades and optional capability adoption remain separate changes.
An upstream storage adapter is not an acceptable substitute for an upstream
policy or diagnostic contract.

## 7. Verification plan

### Local fast checks

For each mcp-memory implementation slice:

```bash
uv run ruff check .
uv run pyright
uv run pytest tests/small/
```

Never use `ruff format`.

### Focused behavioral checks

- sync/async diagnostic parity;
- forced keyword-only, vector-only, and hybrid planner cases;
- lane overlap versus final duplicate semantics;
- malformed embedding-row fallback;
- cache policy fingerprint changes;
- cache invalidation after mutation epochs, including insertion of a new
  matching record;
- concurrent identical-query coalescing;
- stale fallback degradation; and
- isolated policy builder construction and corpus order.

### Wider checks

Run medium tests when changing MCP wiring, storage integration, management
surfaces, federation, or daemon behavior. Run large tests when changing
transport, lifecycle, or end-to-end flows.

### SearchKernel checks

Any upstream extraction must add provider-neutral unit tests in SearchKernel,
retain mcp-memory adapters as compatibility proof, and pass SearchKernel's
existing safe suite plus the new contract tests.

## 8. Rollback and failure behavior

Rollback is configuration- and policy-based:

- disable experimental SearchKernel features;
- select the baseline policy fingerprint;
- purge or bypass derivative responses from another policy; and
- leave authoritative memory data untouched.

If a semantic or reranking lane fails, return available keyword/graph results,
mark the outcome degraded, preserve the failure reason, and avoid unbounded
repair work inside the request.

If diagnostics fail, return the search result without diagnostics rather than
turning observability into a retrieval outage. Emit a structured warning or
metric, and make benchmark mode fail its diagnostics-completeness gate when
`require_diagnostics=true`; ordinary production search remains available.

If federation partially fails, preserve successful source results and report
the failed or timed-out sources explicitly.

If an upstream proposal changes storage ownership, lifecycle semantics, or
authorization behavior, reject it until those invariants are separately
specified and tested.

## 9. Open decisions

1. Should structured diagnostics be returned only for debug and benchmarks, or
   should a sampled subset enter operator telemetry by default?
2. Should SearchKernel expose a first-class typed `QueryPlan` in the public
   outcome, or is a stable diagnostic projection sufficient?
3. Which hydration-version contract can mcp-memory provide without coupling
   SearchKernel to memory-specific record tokens?
4. Is a federated MCP tool needed now, or should federation remain an internal
   management/API capability until an external source is available?
5. Which policy experiment, calibrated fusion or query expansion, produces the
   first meaningful hypothesis after diagnostics parity is complete?

## 10. Decision summary

The first implementation target is diagnostic parity and semantic correctness
of the integration boundary. The second is cache ownership and freshness. Only
then should advanced SearchKernel ranking features be evaluated for adoption.

The best upstream contributions are structured diagnostics, evaluation gates,
and a minimal freshness contract. Memory-specific storage, lifecycle, graph,
authorization, repair, and ranking policy remain local.

The governing rule is:

> Improve retrieval with evidence, preserve authority, and keep every change
> reversible.

## 11. Legacy surface inventory

The legacy-removal program must distinguish obsolete implementation paths from
compatibility that still protects existing installations, callers, or data.
The inventory below is the starting removal ledger. A surface is not eligible
for deletion merely because a newer implementation exists; it must also have
no supported callers, a migration path, and a rollback story.

### 11.1 Search execution compatibility

| Surface | Current role | Target replacement | Removal risk |
| --- | --- | --- | --- |
| `MemoryRetrievalFacade` | Stable async/sync retrieval port, pipeline lifetime, and safe sync bridge | One canonical application retrieval service with a thin boundary adapter | High: CLI, management, maintenance, and compatibility callers still use sync methods |
| `RelationalMemorySearchService` | Older relational search service exposed by storage composition and management code | Canonical application retrieval service plus explicit storage/query ports | High: constructed by SQLite and Postgres resources; widely covered by tests |
| `SearchMemoryRecordsOperation` | Compatibility use-case wrapper around synchronous retrieval | Typed application search request/response service | Medium: preserves operation-level diagnostics and legacy call signatures |
| `search_memory_records_service` | Synchronous MCP/CLI adapter | Async MCP service plus a deliberately narrow sync boundary | Medium: direct CLI, tests, and internal compatibility callers remain |
| `searchkernel_source.py` compatibility factory | Backward-compatible name for the memory federation adapter | Canonical memory federation source export | Low to medium: public import compatibility must be measured before removal |
| Separate async/sync diagnostic projections | Async MCP returns a reduced shape while sync/debug retrieval is richer | One serializer over one normalized outcome context | High: behavioral drift can hide failures or abstention evidence |

The facade and relational service are migration surfaces, not dead code. Their
removal depends on moving every supported caller to the canonical service and
proving equivalent scope, lifecycle, cache, failure, and diagnostic behavior.

### 11.2 Runtime and API compatibility

| Surface | Current role | Target replacement | Removal condition |
| --- | --- | --- | --- |
| `MemoryReadDependencies.memory_retrieval: Any` | Optional compatibility injection for old retrieval implementations | Typed retrieval capability or canonical service dependency | All runtime composition and test doubles satisfy the typed port |
| Sync methods on `MemoryRetrievalPort` | Synchronous callers avoid owning an event loop | Boundary-only `run_sync` adapter outside the core retrieval port | No core service depends on sync methods |
| Legacy result/payload constructors | Preserve direct construction and older serialized fields | Canonical application payload builders | Consumer inventory and one deprecation window complete |
| Compatibility aliases and re-exports | Preserve imports across internal refactors | Canonical module paths | Repository and downstream import scans are empty |
| Fallback telemetry fields | Read old telemetry or preserve older payloads | Versioned telemetry schema with explicit migration | Old readers are retired and schema version is advanced |

These surfaces must be tracked by caller, not removed by a broad search-and-
replace. A compatibility alias with an external consumer is an API contract;
an unused internal re-export is a safe cleanup candidate.

### 11.3 Operational compatibility

The repository also contains legacy task names, configuration keys, endpoint
aliases, and compatibility dispatch. These are separate from search and must
be inventoried before deletion:

- legacy maintenance task-name aliases and migration dispatch;
- old configuration switches such as legacy provider-routing inputs;
- deprecated management or administrative endpoint aliases;
- compatibility groups and work-item claim aliases;
- old ingress or curation stage names accepted during replay; and
- fallback provider, embedding, or storage paths that may still protect
  availability rather than represent obsolete behavior.

The removal program must classify each item as one of:

1. **Dead bridge** — no supported caller and no persisted-data dependency;
2. **Active compatibility** — a caller or external artifact still depends on
   it;
3. **Migration-only** — required to read or transform older persisted data;
   or
4. **Resilience fallback** — intentionally retained because the primary path
   can be unavailable or degraded.

Only dead bridges are immediate deletion candidates. Active compatibility and
migration-only surfaces require an announced support window. Resilience
fallbacks require operational evidence before they can be reclassified as
legacy.

### 11.4 Persistence compatibility

`apply_legacy_additive_migrations` and historical schema migration tests are
not equivalent to dead application code. Existing databases may require them
to start successfully. The removal plan must therefore preserve immutable
migration history while separately deciding whether old migration execution
paths can be retired in a future storage-format release.

Before removing a migration path, record:

- the oldest supported database/schema version;
- an idempotent upgrade or export path for older installations;
- evidence that supported deployments have crossed the migration window;
- backup and rollback instructions; and
- a major-version or explicit release note for the breaking storage change.

The target is zero obsolete runtime branches, not deletion of historical facts
needed to reconstruct or upgrade a supported database.

## 12. Canonical retrieval target

Legacy removal should converge on one application-owned retrieval service. The
service owns request normalization, scope policy, SearchKernel pipeline
composition, authoritative hydration, payload shaping, and structured
diagnostics. SearchKernel remains responsible for provider-neutral retrieval
orchestration; mcp-memory remains responsible for memory policy and authority.

### 12.1 Target request path

```text
MCP / CLI / management request
  -> typed MemorySearchRequest
  -> canonical async application retrieval service
  -> memory policy and scope filters
  -> SearchKernel RecordSearchPipeline
  -> authoritative hydration and redaction
  -> one SearchExecutionDiagnostics serializer
  -> caller-specific compact or debug payload
```

The canonical service must be usable by public MCP tools, internal
maintenance tools, management views, CLI commands, and background workers.
Those callers may choose different authorization, caching, or payload policies,
but they must not construct independent retrieval pipelines or interpret
SearchKernel outcomes through separate diagnostic grammars.

### 12.2 Async core and sync boundary

The retrieval core is asynchronous. It may await embedding, vector, graph,
hydration, or provider work without blocking the daemon event loop. A single
small `run_sync` adapter may remain at process boundaries that are inherently
synchronous, such as a legacy CLI entry point or a synchronous management
interface. That adapter:

- accepts the same typed request as the async service;
- invokes the canonical async service;
- preserves the same result and diagnostic semantics;
- runs through `asyncio.run` when no loop exists; and
- uses a bounded worker bridge when called from an active loop.

The sync adapter is a boundary mechanism, not a second retrieval
implementation. It must not own a separate pipeline factory, cache policy,
filter interpretation, or result serializer.

### 12.3 Compatibility transition

During migration, `MemoryRetrievalFacade` may remain as a thin adapter that
implements the old port by delegating to the canonical service. It must stop
owning behavior that belongs in the canonical service. In particular, the
following must have one implementation:

- request normalization and filter precedence;
- standard and adaptive pipeline selection;
- workspace, lifecycle, supersession, and ranking scope;
- query embedding and derivative-cache ownership;
- failure, missing-hydration, and degraded-result handling;
- semantic-abstention calculation; and
- debug and sampled diagnostic serialization.

`RelationalMemorySearchService` should become a compatibility adapter over the
same service before it is removed. Storage composition must expose the typed
repository and canonical retrieval capability directly rather than constructing
an older relational search object as the primary runtime service.

### 12.4 One diagnostic contract

Every caller that requests debug or operator evidence receives the same
application-facing diagnostic semantics. The serializer must preserve at least:

- candidate counts by lane;
- enabled, budgeted, and skipped lanes;
- failures, missing records, and degraded state;
- semantic candidates, semantic-only candidates, abstention count, and
  `semantic_abstained` where measurable;
- raw lane overlap, multi-lane provenance, and final duplicates as separate
  concepts;
- cache and phase timings; and
- hard workspace scope versus ranking workspace context.

Caller-specific payloads may omit fields for compactness, but omission must be
an explicit projection of the canonical contract rather than a second
calculation. Diagnostic projection failure must never turn a successful search
into a failed request.

### 12.5 Target ownership boundary

The cutover is complete when:

1. SearchKernel is the only retrieval orchestrator;
2. mcp-memory has one application retrieval service and one diagnostic
   serializer;
3. storage composition exposes typed repositories and capabilities rather
   than `RelationalMemorySearchService`;
4. sync callers use only the narrow boundary adapter;
5. no caller depends on `MemoryRetrievalFacade` implementation details; and
6. lifecycle, workspace, authorization, hydration, repair, cache freshness,
   and payload redaction remain demonstrably application-owned.

## 13. Legacy removal sequence

Legacy removal is a sequence of independently verifiable cutovers. Each phase
must leave the repository runnable and must either remove a complete dead
surface or make the next surface easier to remove. No phase may delete a
compatibility path merely because its replacement passes unit tests; callers,
serialized data, deployed databases, and rollback behavior must be accounted
for first.

### Phase 0 — Baseline and ownership ledger

Freeze the current release, dependency versions, search-quality corpus,
transport behavior, and supported storage versions. Generate a removal ledger
with one row per compatibility surface:

| Field | Required evidence |
| --- | --- |
| Surface and owner | Source path, runtime owner, and responsible team |
| Current callers | Static references plus runtime/telemetry observations |
| Replacement | Target module, port, or serialized contract |
| Compatibility kind | Dead bridge, active compatibility, migration-only, or resilience fallback |
| External contract | Public import, CLI/API, persisted field, or internal-only |
| Removal gate | Exact zero-caller, release, migration, or benchmark condition |
| Rollback | Configuration, package, or data rollback path |

Static search is necessary but insufficient. Dynamic callers, plugin imports,
old databases, serialized payload readers, and external scripts must be
included in the review. Unknown consumers are treated as active compatibility
until disproven.

### Phase 1 — Establish one behavior contract

Before moving callers, make the canonical async service behavior complete and
observable:

1. normalize all request forms into `MemorySearchRequest`;
2. centralize workspace, lifecycle, supersession, memory-type, tag, and
   ranking-scope interpretation;
3. centralize standard/adaptive pipeline selection;
4. centralize response-cache, epoch, freshness, and stale-fallback policy;
5. serialize one diagnostic model for sync and async callers;
6. expose semantic abstention, degraded execution, failures, missing records,
   and lane provenance consistently; and
7. prove that diagnostics are best effort and cannot fail retrieval.

The existing facade and relational service remain in place during this phase,
but their behavior delegates to the canonical implementation. This phase ends
only when equivalent requests through public MCP, internal MCP, CLI,
management, and direct application entry points have equivalent result and
diagnostic semantics.

### Phase 2 — Migrate callers to the canonical service

Migrate callers in dependency order:

1. public `search_memory_records` MCP service;
2. internal maintenance search service;
3. CLI search and administrative commands;
4. management service and dashboard read models;
5. curator and background task support;
6. storage/runtime composition; and
7. tests, fixtures, examples, and integration helpers.

Each migration must remove one caller's dependency on facade or relational
service implementation details. The caller may retain a local synchronous
boundary when its public API is synchronous, but it must delegate to the same
typed request and canonical async service. No caller may reconstruct a second
SearchKernel pipeline or reimplement filter interpretation.

During this phase, preserve compatibility adapters at the outermost boundary.
They should emit bounded usage telemetry or a deprecation signal so the ledger
can prove whether they still receive traffic. Do not emit query text or memory
content solely to count compatibility use.

### Phase 3 — Remove duplicate search seams

Once callers have moved, remove duplication in this order:

1. move `SearchMemoryRecordsOperation` behavior into the canonical application
   request service while retaining a temporary type alias if needed;
2. reduce `MemoryRetrievalFacade` to a thin adapter with no pipeline or
   diagnostic logic;
3. reduce `RelationalMemorySearchService` to a thin adapter over the same
   service;
4. remove direct `search_sync` and `search_sync_with_diagnostics` dependencies
   from core application code;
5. remove duplicate async MCP diagnostic projection; and
6. delete the adapters once the zero-caller gate passes.

The sync boundary may survive as a small process-edge utility. It is not a
legacy retrieval system if it only converts a synchronous call into the
canonical async request and returns the canonical response.

### Phase 4 — Retire search compatibility names

After a release window in which compatibility usage is zero or explicitly
accepted by the product owner:

- remove `MemoryRetrievalFacade` exports and implementation;
- remove `RelationalMemorySearchService` from storage composition and package
  exports;
- remove the old search operation wrapper and direct service aliases;
- remove `searchkernel_source.py` compatibility names after import consumers
  have migrated;
- remove old result constructors and payload aliases; and
- delete tests that only protect removed names, retaining contract tests for
  the canonical service.

Every deletion must be accompanied by an import scan and a release note. If a
downstream consumer is discovered after the window, restore the smallest
boundary adapter rather than reintroducing a second search implementation.

### Phase 5 — Retire runtime and protocol shims

The same ledger-driven process applies to non-search compatibility:

1. migrate legacy task-name aliases to canonical task names;
2. migrate old configuration keys and remove implicit legacy defaults;
3. migrate deprecated management and administrative endpoints;
4. migrate ingress, curation, and work-item stage aliases;
5. replace `Any` compatibility slots with typed capabilities;
6. remove unused provider-policy and telemetry fallbacks; and
7. remove dead re-exports and compatibility import paths.

Aliases that are required to replay persisted work or read existing records
remain until the relevant migration-only gate passes. A fallback that handles
provider outage, unavailable embeddings, or bounded storage degradation is not
removed as legacy without evidence that its replacement preserves availability
and diagnostics.

### Phase 6 — Retire migration-only runtime branches

Migration-only code is removed last because it protects existing data rather
than callers. For each database or serialized format:

1. declare the oldest supported version;
2. ship an idempotent upgrade, export, or backup path;
3. measure adoption of versions requiring the old branch;
4. announce the removal version and support-window end;
5. verify restore and rollback procedures; and
6. remove only the obsolete runtime branch while preserving immutable history
   needed to understand prior migrations.

Historical migration functions and tests may remain permanently if they are
needed to build or audit supported upgrades. Their presence is not, by itself,
evidence of active legacy behavior.

### Phase 7 — Final deletion and release cutover

The final legacy-removal release may delete a surface only after its gates pass
and the complete verification suite is green. The release notes must identify
removed imports, endpoints, task aliases, configuration keys, and storage
support. The rollback package or configuration must be available before the
release is published.

The final target is not a repository with no word named `legacy`. It is a
repository with one intentional canonical path per behavior, no unowned
compatibility callers, no obsolete runtime branches, and explicit historical
support where deletion would endanger existing data.

## 14. Legacy removal gates

No legacy surface is deleted until its specific gate is satisfied and recorded
in the removal ledger. The gates are intentionally stronger than a green unit
test run because compatibility can exist in imports, deployed data, serialized
payloads, and external automation.

### 14.1 Caller and import gates

For each surface scheduled for deletion:

- `rg` finds no production caller outside the replacement or migration test;
- import-linter and package import checks show no dependency on the old module;
- CLI, MCP, management, and internal tool registries expose only the canonical
  name;
- runtime composition constructs only the canonical service;
- dynamic compatibility telemetry is zero for the declared observation
  window; and
- examples, scripts, documentation, and downstream integration fixtures have
  been updated.

An unknown external consumer blocks deletion. If the consumer cannot be
identified, retain a narrow adapter and document the unresolved ownership.

### 14.2 Behavioral parity gates

Before removing an adapter, run the same request corpus through old and new
entry points and compare:

- result identity and ordering within the supported contract;
- workspace and lifecycle filtering;
- supersession and status behavior;
- ranking workspace versus hard workspace scope;
- semantic-only abstention and keyword-backed preservation;
- failure, missing-hydration, and degraded-result behavior;
- cache hits, invalidation, coalescing, and stale fallback; and
- debug diagnostics, including unavailable versus measured-zero evidence.

Differences must be classified as intentional contract changes or fixed before
deletion. A compatibility test that only checks that a method still exists is
not sufficient.

### 14.3 Type and architecture gates

The canonical path must pass structural checks before old types are removed:

- `MemoryReadDependencies` uses typed retrieval capabilities;
- no production retrieval caller requires `Any` to cross the application
  boundary;
- SearchKernel remains the only generic retrieval orchestrator;
- mcp-memory remains the authority for lifecycle, authorization, hydration,
  repair, freshness, and payload redaction;
- the sync bridge is confined to explicit process edges; and
- import-linter rules prevent the old relational or facade layer from being
  reintroduced.

The final architecture should be explainable from the dependency graph, not
only from convention or code review memory.

### 14.4 API and release gates

Before removing public or semi-public compatibility:

1. identify supported import, CLI, MCP, HTTP, and serialized contracts;
2. publish the replacement contract and migration example;
3. mark the old contract deprecated for at least one supported release;
4. emit bounded usage telemetry without recording query or memory content;
5. verify usage is zero or obtain an explicit exception; and
6. publish release notes describing the breaking removal and rollback path.

Internal-only names may use a shorter window when the caller/import gate is
proven, but the decision belongs in the removal ledger.

### 14.5 Persistence and migration gates

Database and serialized-data compatibility is removable only when:

- the oldest supported format is explicitly documented;
- all supported installations can be upgraded or exported without the old
  branch;
- backups have been tested before and after the upgrade;
- restore has been tested against the new release;
- the support-window end has been announced; and
- the release is allowed to make the corresponding storage break.

Historical migration definitions may remain permanently. The gate concerns
runtime support for obsolete formats, not deletion of audit history.

### 14.6 Required verification ladder

Every migration and deletion commit must pass the narrowest applicable checks,
then the full repository gates:

```bash
uv run ruff check .
uv run pyright
uv run pytest tests/small/
```

Changes to MCP wiring, daemon lifecycle, management, storage, or search
execution additionally require the relevant medium tests. Changes to
transport, startup, shutdown, or end-to-end compatibility require the large
tests. The final legacy-removal release must also pass:

- import-linter and static zero-caller scans;
- fresh SQLite bootstrap and supported legacy-database upgrade tests;
- Postgres and vector/search parity tests where configured;
- installed-tool verification from a clean environment;
- daemon restart and health checks; and
- the labeled search-quality corpus with no unexplained regression.

### 14.7 Final deletion checklist

The deletion owner must attach this evidence to the final change:

- removal-ledger row with owner, replacement, and gate results;
- `git grep`/`rg` caller and import scan;
- compatibility telemetry window and decision;
- API/config/task/endpoint migration notes;
- database and serialized-format support decision;
- parity and rollback test results;
- focused, small, medium, and large QA results as applicable; and
- confirmation that unrelated resilience fallbacks were not removed.

If any item is missing, the change is a migration step, not a deletion step.

### 14.8 Rollback

Rollback must restore the previous adapter or package version without changing
authoritative memory data. A failed cutover may disable the canonical feature
flag, re-enable the thin compatibility adapter, or redeploy the previous
package. It must not require reconstructing deleted records, rewriting
historical migrations, or guessing which policy produced a cached response.
