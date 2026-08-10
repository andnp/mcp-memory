# Design: Searchkernel Post-Upgrade Quality Roadmap

**Status:** Draft
**Date:** 2026-08-09

This document is a follow-up design for the `andnp-searchkernel` 0.21 upgrade.
It describes the next search-quality, latency, diagnostics, and upstream-
extraction work. It is not the canonical description of current runtime
behavior; active specs, runbooks, and the root `README` remain authoritative.

## 1. Executive summary

The upgrade to searchkernel 0.21 is complete and the daemon is healthy. Live
dogfooding showed that exact technical and degraded-path queries return useful
hybrid results with correct workspace scope and no semantic degradation. It
also exposed three follow-up problems:

1. broad or historical queries can miss an expected record or return noisy
   curator-health memories;
2. search latency varies from sub-second to roughly two seconds, while daemon
   logs still contain slow-transport warnings; and
3. the `duplicate_candidate` diagnostic name does not make clear that it
   reports post-fusion multi-lane provenance rather than duplicate final
   records.

The recommended sequence is:

1. establish a small, durable search-evaluation corpus;
2. instrument phase-level latency and clarify diagnostics;
3. improve recall and ranking only where the corpus demonstrates a gap;
4. evaluate advanced searchkernel scoring behind a feature flag; and
5. extract provider-neutral capabilities from mcp-memory into searchkernel
   only after their contracts and benchmarks are stable.

The design deliberately keeps mcp-memory's relational, vector, and graph
stores authoritative. Searchkernel remains the search policy and orchestration
layer; it should not become a second storage authority.

## 2. Current evidence

### 2.1 Upgrade and verification baseline

The dependency now requires searchkernel 0.21 and the lockfile resolves
`andnp-searchkernel==0.21.0`. The integration uses searchkernel's query-aware
policy context for vector candidate selection, vector ordering, and score
adjustment.

Verification completed with the upgrade:

| Check | Result |
| --- | --- |
| Ruff | passed |
| Pyright | passed |
| Focused searchkernel tests | 39 passed |
| Relevant medium parity tests | 7 passed, 1 skipped |
| Small suite | 1,285 passed; 2 unrelated schema-version assertions failed |
| Search health after restart | semantic enabled, not degraded, no fallback/rebuild/failure/missing-ID diagnostics |

The two small-suite failures expect schema version 34 while the current
schema is 35. They are tracked separately from this search design so that
search changes are not coupled to unrelated migration-test drift.

### 2.2 Live search observations

The live daemon was restarted after the upgrade. Search results were strongest
for queries with distinctive technical terms, exact identifiers, or explicit
degraded-path language. These searches showed keyword/vector provenance and
returned relevant top results.

Broader queries such as daemon/search-quality history and the expected
"searchkernel core migration" record were noisier or failed to surface the
expected historical memory. This is evidence of a recall/ranking gap, not
evidence that the semantic backend is unhealthy.

Observed request times ranged from approximately 0.3 to 2.0 seconds during
the dogfood session. Daemon logs also contained slow-transport warnings. The
first action should therefore be measurement: identify whether time is spent
in embedding, keyword retrieval, vector retrieval, graph expansion, ranking,
hydration, transport, or queueing.

## 3. Goals

1. Make search quality measurable against representative user queries.
2. Improve recall for broad and historical queries without regressing exact
   technical retrieval.
3. Bound and explain search latency, especially in shared Postgres mode.
4. Make search diagnostics accurately describe what happened.
5. Evaluate searchkernel's advanced scoring features safely and reversibly.
6. Identify reusable, provider-neutral capabilities that belong upstream in
   searchkernel.
7. Preserve the existing authority, workspace, degradation, and storage
   contracts.

## 4. Non-goals

This roadmap does not propose:

- replacing mcp-memory's authoritative storage adapters with searchkernel's
  native SQLite, FAISS, Postgres, or graph stores;
- changing workspace identity into a storage-tenancy boundary;
- introducing a broad search-stack rewrite before evaluation evidence exists;
- requiring an embedding provider for the default test suite;
- making live provider calls part of normal CI; or
- fixing unrelated schema-version test drift in the search commits.

## 5. Design principles

### 5.1 Measure before tuning

Every ranking or retrieval change should have a before/after query set and a
clear failure mode. A synthetic test must force the planner branch it intends
to exercise; a test that passes without invoking that branch is not evidence
for the branch's contract.

### 5.2 Keep authority explicit

mcp-memory owns memory durability, workspace filtering, embeddings, graph
relationships, hydration, and lifecycle. Searchkernel owns generic search
composition, candidate routing, fusion, calibration, and optional reranking.
An upstream extraction is justified only when it removes a domain-specific
dependency without duplicating authority.

### 5.3 Prefer additive, reversible rollout

New ranking policies and diagnostics should be feature-gated or shadowed
before they become the default. A rollback must be a configuration or policy
selection, not a data migration.

### 5.4 Treat degraded behavior as a contract

Partial semantic coverage, provider unavailability, malformed embedding rows,
and backend fallback must continue to produce an explicit, useful result.
Health must distinguish "the service answered" from "the intended quality
projection was available."

## 6. Target architecture

The target flow retains the current layered ownership:

```text
query
  -> mcp-memory scope/filter policy
  -> searchkernel query planner
  -> keyword/vector/graph candidate lanes
  -> searchkernel fusion/calibration/rerank policy
  -> mcp-memory authoritative hydration
  -> result plus provenance and diagnostics
```

The planner may skip a lane when another lane already satisfies the requested
limit, but only when the upstream candidate count reflects the downstream
eligibility contract. Workspace, status, tag, supersession, and application
candidate filters must already be applied, or the planner must use a safety
multiplier. Tests and diagnostics must expose that decision so lane-specific
budgets are not mistaken for execution guarantees. In particular, the
artifact-keyword shortcut must prove that its confident keyword candidates
survive mcp-memory's later policy filters before it suppresses vector search.

### 6.1 Authority boundary

The following remain mcp-memory-owned:

- relational records and workspace scope;
- vector and graph storage;
- embedding repair and integrity checks;
- authoritative hydration and redaction;
- cache epochs and invalidation;
- daemon lifecycle and transport;
- provider admission and health; and
- product-specific score adjustments.

The following are candidates for searchkernel when generalized:

- query-aware candidate-lane contracts;
- normalized/calibrated hybrid score composition;
- explicit lane-overlap provenance;
- evaluation evidence schemas and metric helpers; and
- backend-neutral reranking interfaces.

## 7. Workstream A: evaluation corpus

### 7.1 Corpus shape

Create a versioned, workspace-local corpus of approximately 30–50 queries.
Each entry should include:

```yaml
id: search-history-001
query: searchkernel core migration
workspace: /home/andy/Projects/personal/mcp-memory
intent: historical design retrieval
evaluation_label: searchkernel-core-migration
expected_memory_refs:
  - mem-<stable-reference>
acceptable_top_k: 5
query_class: broad | exact | historical | degraded | relational
```

The initial corpus should cover:

- exact memory references and distinctive identifiers;
- broad architecture concepts;
- historical migration and decision queries;
- malformed or degraded embedding scenarios;
- workspace-scoped searches;
- graph/relationship-oriented queries; and
- queries that intentionally exercise keyword-only, vector-only, and hybrid
  planner decisions.

The corpus must not encode private content outside the workspace or depend on
unstable generated summaries. Stable evaluation labels are the primary oracle;
memory references are resolved when the corpus is built and must record whether
the expected memory was superseded, deleted, or replaced. This keeps the
evaluation useful without making every memory consolidation a test edit.

### 7.2 Metrics

Track at least:

- hit@1 and hit@5 for expected records;
- reciprocal rank of the first expected record;
- query-class recall, not only aggregate recall;
- false-positive rate for exact identifiers;
- semantic-abstention rate and precision of the low-confidence semantic tail;
- semantic and keyword lane participation;
- final-result duplicate count versus lane-overlap count;
- degraded/fallback/error rate;
- fallback-row-cap application rate;
- p50, p95, and maximum end-to-end latency.

The evaluation should report regressions by query class. An aggregate score
must not hide a loss of exact-identifier precision behind gains on broad
semantic queries.

### 7.3 Test layers

- Small tests: deterministic ranking math, provenance normalization, and
  diagnostics naming.
- Medium tests: real search pipeline with deterministic stores and planner
  routing, including fixtures that force keyword-only, vector-only, hybrid,
  and lane-skipping branches.
- Large tests: daemon restart, transport behavior, and end-to-end thought
  capture followed by search/read.
- Backend parity tests: run the corpus against local SQLite and shared Postgres
  server-side `pgvector` and Python-fallback modes when those fixtures exist;
  compare ordering, abstention, degradation, and latency separately.
- Live dogfood: a small opt-in command that runs the corpus against the live
  daemon and stores only metrics, not query content or result payloads.

## 8. Workstream B: latency and transport diagnosis

### 8.1 Instrumentation

Add phase timing to the existing search diagnostics, with bounded fields for:

- daemon-internal request queue wait;
- cache lookup and cache fallback;
- embedding/provider time;
- keyword retrieval;
- vector retrieval, split between server-side `pgvector` query time and
  client-side Python fallback decode/score time;
- graph expansion;
- fusion/calibration/rerank;
- request-scoped embedding-repair wait;
- hydration;
- serialization; and
- daemon-internal transport handling.

Each phase should report elapsed time and a small outcome code. Diagnostics
must remain safe to return when a phase fails or is skipped.

Daemon diagnostics cannot measure the complete client-observed IPC/MCP round
trip on their own. The live corpus runner should add a client-side start/end
timestamp and correlate it with the daemon request ID, so transport overhead
can be calculated rather than inferred from internal timings.

### 8.2 Investigation order

1. Compare daemon transport time with in-process search time.
2. Compare keyword-only, vector-only, and hybrid corpus queries.
3. Compare SQLite with shared Postgres server-side `pgvector` and Python
   fallback modes where available.
4. Check cache hits/fallbacks, embedding repair/backfill activity, and provider
   admission state.
5. Check hydration size and graph expansion fanout.
6. Only then tune candidate limits, caching, or ranking policy.

This order avoids treating transport warnings as proof that transport is the
root cause. Existing search-repair design also requires that embedding repair
not hold a user search hostage; any repair wait must be request-scoped and
bounded.

### 8.3 Initial latency targets

These are rollout targets, not current guarantees:

| Query mode | Target p95 | Action if exceeded |
| --- | ---: | --- |
| Keyword-only | 250 ms | inspect storage/filter path |
| Hybrid, warm | 750 ms | inspect lane and hydration timings |
| Hybrid, cold/provider-backed | 1,500 ms | bound provider wait and degrade explicitly |
| Daemon transport overhead | 100 ms | inspect queueing/serialization/socket health |

Targets should be revisited after one representative corpus run.

## 9. Workstream C: recall and ranking

### 9.1 First hypotheses to test

The broad-query miss should be investigated in this order:

1. query tokenization and synonym coverage;
2. keyword candidate limit and filtering;
3. semantic candidate limit and planner lane selection;
4. score calibration between keyword and vector lanes;
5. exact-match or historical-recency adjustments; and
6. hydration or post-ranking filtering that removes otherwise good candidates.

Do not add manual synonyms or score boosts until the corpus identifies a
repeatable class of misses and the relevant phase is known.

### 9.2 Advanced searchkernel evaluation

Evaluate the following independently behind a feature flag:

- calibrated hybrid scoring;
- normalized lane scores;
- filtered exact/approximate vector retrieval;
- backend-neutral reranking; and
- evaluation evidence support.

Each experiment must record the policy version, candidate limits, active lanes,
latency, and corpus metrics. The default policy remains the rollback target.

### 9.3 Acceptance gates

An advanced policy may become the default only if it:

- improves broad/historical hit@5 without reducing exact-identifier hit@1;
- does not increase degraded or missing-ID results;
- stays within the applicable latency target; and
- has a documented rollback switch and focused regression coverage.

## 10. Workstream D: diagnostics ergonomics

Clarify or split the current `duplicate_candidate` output. Today it is computed
after fusion by checking whether a result has more than one provenance
strategy; it does not mean that duplicate records remain in the final result
list. The future diagnostic shape should distinguish:

- `raw_lane_overlap_count`: a record appeared in multiple pre-fusion
  retrieval lanes;
- `multi_lane_result_count`: a post-fusion result has more than one strategy;
- `final_duplicate_count`: duplicate IDs remained after fusion; and
- `missing_record_ids`: candidates could not be hydrated.

The existing hybrid provenance should remain available, but diagnostics should
not imply that final results contain duplicates when they only share candidates
between lanes. Add one small test per field for zero, positive, and degraded
cases, plus a regression test proving the current multi-lane result behavior.

## 11. Workstream E: upstream extraction

### 11.1 Candidate extraction: query-aware policy contracts

The mcp-memory upgrade exposed value in callbacks receiving a typed query
context rather than ambient filters or module state. Searchkernel should keep
this contract provider-neutral and document the planner's permission to skip a
lane when the requested limit is already satisfied.

### 11.2 Candidate extraction: cache epochs

mcp-memory's derivative cache freshness tokens and invalidation rules may be
useful as a generic searchkernel cache contract, but only if the abstraction
describes freshness/version tokens rather than SQLite or Postgres details.
First define the required lifecycle and stale-read semantics locally; then
propose the smallest upstream interface. Cache contents remain disposable and
Postgres remains authoritative.

Any search-response cache used during policy experiments must include the
active search policy version and feature-flag fingerprint in its key, or be
explicitly purged when policy configuration changes. This applies to both
fresh hits and degraded stale fallbacks.

### 11.3 Candidate extraction: embedding integrity diagnostics

mcp-memory has concrete diagnostics for malformed embedding rows and safe
fallback behavior. Embedding storage validation and repair remain strictly in
mcp-memory's storage/application adapters. A possible searchkernel contribution
is limited to a generic candidate/search outcome error envelope, not storage
table validation or provider-specific integrity checks.

### 11.4 What should not move upstream

Workspace policy, authoritative hydration, graph semantics, provider
admission, daemon ownership, and mcp-memory-specific score adjustments should
remain local. Upstreaming these would make searchkernel opinionated about the
product domain it is meant to serve.

## 12. Rollout plan

### Phase 0 — Baseline and hygiene

- preserve the searchkernel 0.21 upgrade;
- add the versioned evaluation corpus;
- reconcile the two schema-version tests in a separate migration-test commit;
- capture current metrics and daemon health.

### Phase 1 — Observability

- add phase timings;
- clarify lane-overlap diagnostics;
- expose planner lane decisions and candidate budgets;
- run the corpus in-process and through the live daemon.

### Phase 2 — Recall experiments

- reproduce the historical-query miss;
- test one hypothesis at a time;
- add regression cases for confirmed failures;
- keep the default policy unchanged until gates pass.

### Phase 3 — Advanced policy trial

- enable one searchkernel policy feature at a time;
- compare against the baseline corpus;
- verify cache bypass or invalidation when the policy version or feature
  fingerprint changes;
- run medium and large runtime tests;
- canary the policy for live dogfooding;
- revert to the baseline policy on any gate failure.

### Phase 4 — Upstream proposals

- extract only contracts with evidence of reuse;
- add provider-neutral tests in searchkernel;
- retain mcp-memory adapters as the compatibility proof;
- upgrade mcp-memory only after an upstream release is available.

## 13. Rollback and failure behavior

Rollback must be possible by selecting the previous search policy and disabling
experimental diagnostics or features. No search-quality experiment should
require destructive data changes.

If a semantic lane fails, the system should:

- return keyword or other available results;
- mark the search as degraded with a concrete reason;
- preserve missing IDs and backend failure diagnostics; and
- avoid retrying unbounded repair work inside the request.

If phase instrumentation itself fails, it must not fail the search request.
If a new upstream adapter changes storage ownership or lifecycle semantics, it
must be rejected until those invariants are explicitly preserved.

Policy rollback must not serve a response produced by a different policy from
the readthrough cache. The cache key/version or an explicit purge is part of
the rollback contract, including when the authoritative Postgres backend is
temporarily unavailable and a stale derivative response is returned.

## 14. Open decisions

1. Which durable evaluation-label lifecycle should represent superseded or
   replaced memories without turning corpus maintenance into manual relinking?
2. What is the minimum corpus size that gives stable conclusions across
   SQLite, Postgres fallback, and Postgres server-side vector modes?
3. Should phase timing be returned by default in debug output only, or also be
   sampled into operator telemetry?
4. Which searchkernel experimental policy should be evaluated first: calibrated
   fusion or reranking?
5. What freshness guarantees should a generic cache-epoch contract promise?

## 15. Decision summary

The next investment is measurement and observability, followed by targeted
recall work. Advanced searchkernel features should earn adoption through a
versioned corpus and latency gates. mcp-memory should contribute generic
contracts for query-aware planning, freshness, and embedding-integrity
diagnostics only after proving that those contracts are reusable.

The governing rule is:

> improve search quality with evidence, preserve storage authority, and keep
> every experiment reversible.
