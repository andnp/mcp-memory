# SearchKernel Legacy-Removal Commit Plan

**Status:** Documentation-only execution plan  
**Date:** 2026-08-10  
**Scope:** mcp-memory retrieval compatibility and the SearchKernel boundary

This plan turns the legacy-removal sequence in
[24-searchkernel-and-memory-search-improvement-design.md](24-searchkernel-and-memory-search-improvement-design.md)
into 83 small, dependency-ordered commits. The commits are numbered for
review and execution; the numbers are not release versions. Every commit must
remain runnable, preserve unrelated resilience fallbacks, and carry the
narrowest relevant verification evidence.

The plan incorporates five verified findings:

- typed capabilities are extracted from `RelationalMemorySearchService`;
- asynchronous callers migrate before `MemoryRetrievalFacade` delegates;
- internal prompts and tool registrations are treated as contracts;
- cache schema/version changes have explicit rollback gates; and
- async/sync parity is proved with medium runtime verification before removal.

## Commit ledger

### Phase 0 — Baseline and ownership (1–8)

1. `docs: freeze search quality baseline` — record the labeled corpus, policy, and current metrics.
2. `docs: inventory retrieval compatibility surfaces` — list owners, callers, replacements, and removal risks.
3. `docs: classify legacy runtime branches` — separate dead bridges, active compatibility, migration-only code, and resilience fallbacks.
4. `docs: record supported persistence versions` — define the oldest database and serialized formats still supported.
5. `test: capture async retrieval baseline` — preserve canonical async result and diagnostic behavior before migration.
6. `test: capture sync retrieval baseline` — preserve synchronous boundary behavior and error semantics.
7. `test: add medium runtime baseline` — verify SQLite and configured Postgres/vector runtime composition before changes.
8. `docs: define rollback ownership` — identify policy, package, cache, and data rollback paths for each surface.

### Phase 1 — Typed capability extraction (9–20)

9. `refactor: define memory search request` — establish one typed request shape for all callers.
10. `refactor: define memory search response` — establish one typed result and diagnostic envelope.
11. `refactor: define retrieval capabilities` — replace implicit service reach-through with explicit application capabilities.
12. `refactor: define hydration capability` — keep authoritative payload loading and redaction application-owned.
13. `refactor: define freshness capability` — expose version tokens without leaking database schema details.
14. `refactor: define diagnostic capability` — make serialization and evidence ownership explicit.
15. `refactor: extract capabilities from relational service` — adapt `RelationalMemorySearchService` to the typed boundary.
16. `refactor: type memory read dependencies` — remove the retrieval-shaped `Any` slot where callers can satisfy the port.
17. `test: verify capability composition` — prove SQLite and Postgres resources provide the same typed capabilities.
18. `test: verify capability ownership` — prove lifecycle, authorization, hydration, repair, and freshness remain local.
19. `test: reject untyped retrieval injection` — prevent new callers from bypassing the typed boundary.
20. `docs: record typed extraction gate` — attach type, import, and runtime evidence to the ledger.

### Phase 2 — Internal contracts (21–29)

21. `docs: inventory internal search callers` — include maintenance, curator, workers, CLI, management, and tests.
22. `refactor: define internal search request contract` — normalize non-public callers onto the typed request.
23. `refactor: define internal search response contract` — preserve bounded diagnostics and degraded semantics.
24. `docs: inventory internal prompts` — identify prompt versions, job families, and completion expectations.
25. `docs: inventory internal tool registrations` — identify allowed tools, names, and registration ownership.
26. `test: lock internal prompt contracts` — verify prompt version and required instructions for each job family.
27. `test: lock internal tool contracts` — verify registrations, allowed-tool sets, and completion tools.
28. `test: verify internal search compatibility` — ensure prompt/tool callers receive the canonical response contract.
29. `docs: record internal contract gate` — make prompt and tool compatibility a prerequisite for caller migration.

### Phase 3 — Async caller migration (30–43)

30. `refactor: migrate public search service` — route the public MCP operation to the canonical async service.
31. `test: verify public async search parity` — compare identities, ordering, filters, failures, and diagnostics.
32. `refactor: migrate maintenance search` — remove maintenance dependence on relational-service implementation details.
33. `test: verify maintenance async parity` — cover degraded and missing-hydration outcomes.
34. `refactor: migrate management search` — use the typed async request in management read paths.
35. `test: verify management async parity` — compare compact and debug projections from the same outcome.
36. `refactor: migrate curator search` — preserve internal prompt/tool lifecycle while changing retrieval ownership.
37. `test: verify curator search contract` — prove completion and failure behavior remains stable.
38. `refactor: migrate worker search` — make background callers use the async service explicitly.
39. `test: verify worker event-loop safety` — prevent nested-loop and blocked-daemon regressions.
40. `refactor: migrate CLI async entry` — use the canonical request at the CLI process boundary.
41. `refactor: migrate administrative callers` — remove direct relational search construction from administrative paths.
42. `test: verify migrated async callers` — run the caller corpus through one canonical service.
43. `docs: record async migration gate` — require zero migrated-caller implementation dependencies before facade delegation.

### Phase 4 — Cache ownership and rollback (44–54)

44. `docs: define cache ownership matrix` — assign query, candidate, hydration, and response caches to one owner each.
45. `refactor: define cache schema version` — make payload and key schema versions explicit.
46. `refactor: include policy version in cache keys` — isolate responses across SearchKernel policy changes.
47. `refactor: include feature fingerprint in cache keys` — isolate optional lane and ranking behavior.
48. `refactor: store freshness snapshots` — validate insertion, deletion, update, and embedding-repair epochs.
49. `test: verify cache invalidation matrix` — cover every mutation and repair case, including new matching records.
50. `test: verify stale fallback semantics` — preserve degraded fallback under authoritative outages.
51. `test: verify cache schema migration` — prove new readers and old readers handle version changes safely.
52. `test: verify cache rollback bypass` — prove the previous release can ignore or bypass new derivative entries.
53. `refactor: consolidate query embedding ownership` — remove duplicate module, adapter, or pipeline cache ownership.
54. `docs: record cache rollback gate` — block caller cutover until schema, version, and rollback evidence is complete.

### Phase 5 — Async/sync parity (55–63)

55. `refactor: define sync boundary adapter` — keep synchronous execution at explicit process edges only.
56. `refactor: route sync requests to async service` — eliminate independent sync retrieval behavior.
57. `test: compare sync and async identities` — require equivalent returned identities and ordering.
58. `test: compare sync and async filters` — require equivalent scope, lifecycle, tag, and supersession behavior.
59. `test: compare sync and async failures` — require equivalent unavailable-provider and malformed-row handling.
60. `test: compare sync and async degradation` — require equivalent partial-result and abstention semantics.
61. `test: compare sync and async diagnostics` — require one normalized diagnostic meaning with projection-only differences.
62. `test: run medium parity matrix` — verify SQLite and configured Postgres/server-side/fallback paths.
63. `docs: record parity gate` — prohibit facade delegation or removal until medium parity is green.

### Phase 6 — Facade delegation and seam reduction (64–71)

64. `refactor: delegate search operation` — move operation behavior to the canonical request service.
65. `refactor: thin retrieval facade` — retain only the compatibility adapter and sync boundary.
66. `refactor: delegate relational service` — make `RelationalMemorySearchService` a typed compatibility adapter.
67. `test: verify facade delegation` — prove no second pipeline, cache policy, filter grammar, or serializer remains.
68. `test: verify relational adapter delegation` — prove storage composition exposes the typed canonical capability.
69. `refactor: unify diagnostic projection` — remove duplicate async and sync diagnostic calculations.
70. `test: verify adapter rollback` — prove the previous boundary can be restored without data changes.
71. `docs: record delegation gate` — document zero implementation ownership in compatibility adapters.

### Phase 7 — Compatibility-name retirement (72–77)

72. `chore: measure facade usage` — collect bounded compatibility usage without query or memory content.
73. `chore: measure relational service usage` — prove the declared zero-use observation window.
74. `refactor: migrate compatibility imports` — update repository, examples, and integration fixtures to canonical names.
75. `test: verify clean import graph` — require static and import-linter scans to exclude retired implementation paths.
76. `refactor: remove retrieval compatibility names` — delete only adapters whose caller and release gates pass.
77. `docs: publish retrieval removal notes` — document migration examples, breaking imports, and rollback package.

### Phase 8 — Runtime and protocol shims (78–80)

78. `refactor: migrate task and configuration aliases` — replace legacy internal names while preserving replay and outage fallbacks.
79. `refactor: remove dead protocol reexports` — delete unreferenced aliases after import and dynamic-caller evidence.
80. `test: verify prompt tool and runtime registry` — ensure canonical registrations remain complete after shim removal.

### Phase 9 — Migration-only cleanup and release (81–83)

81. `test: verify supported database rollback` — exercise upgrade, backup, restore, and previous-release startup paths.
82. `refactor: remove obsolete runtime migration branches` — preserve immutable migration history and supported upgrade paths.
83. `release: cut over legacy removal` — run full QA, publish support and rollback notes, and remove only gated surfaces.

## Required evidence for every commit

Each implementation commit must pass the narrowest relevant checks and leave a
bisect-safe tree. Retrieval, cache, caller, or runtime changes must include
the applicable small tests; changes to MCP wiring, storage composition,
daemon lifecycle, or runtime behavior must include medium tests. The final
cutover additionally requires Ruff, Pyright, small tests, import-linter,
SQLite bootstrap and upgrade coverage, configured Postgres/vector parity,
daemon restart/health checks, and the labeled search-quality corpus.

No commit in this plan may delete authoritative memory data, historical
migrations required by supported upgrades, or resilience fallbacks solely
because a newer path exists.
