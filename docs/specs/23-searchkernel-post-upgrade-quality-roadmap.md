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
3. the `duplicate_candidate` diagnostic appears to describe keyword/vector
   lane overlap, not duplicate final results.

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
limit. Tests and diagnostics must expose that decision so lane-specific
budgets are not mistaken for execution guarantees.

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
unstable generated summaries. Stable memory references and intent labels are
the minimum useful oracle.

### 7.2 Metrics

Track at least:

- hit@1 and hit@5 for expected records;
- reciprocal rank of the first expected record;
- query-class recall, not only aggregate recall;
- false-positive rate for exact identifiers;
- semantic and keyword lane participation;
- final-result duplicate count versus lane-overlap count;
- degraded/fallback/error rate; and
- p50, p95, and maximum end-to-end latency.

The evaluation should report regressions by query class. An aggregate score
must not hide a loss of exact-identifier precision behind gains on broad
semantic queries.

### 7.3 Test layers

- Small tests: deterministic ranking math, provenance normalization, and
  diagnostics naming.
- Medium tests: real search pipeline with deterministic stores and planner
  routing.
- Large tests: daemon restart, transport behavior, and end-to-end thought
  capture followed by search/read.
- Live dogfood: a small opt-in command that runs the corpus against the live
  daemon and stores only metrics, not query content or result payloads.

## 8. Workstream B: latency and transport diagnosis

### 8.1 Instrumentation

Add phase timing to the existing search diagnostics, with bounded fields for:

- request queue wait;
- embedding/provider time;
- keyword retrieval;
- vector retrieval;
- graph expansion;
- fusion/calibration/rerank;
- hydration;
- serialization; and
- daemon transport.

Each phase should report elapsed time and a small outcome code. Diagnostics
must remain safe to return when a phase fails or is skipped.

### 8.2 Investigation order

1. Compare daemon transport time with in-process search time.
2. Compare keyword-only, vector-only, and hybrid corpus queries.
3. Compare SQLite and shared Postgres where both are available.
4. Check embedding repair/backfill activity and provider admission state.
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

Rename or split `duplicate_candidate` so the output distinguishes:

- `lane_overlap_count`: a record appeared in multiple retrieval lanes;
- `final_duplicate_count`: duplicate IDs remained after fusion; and
- `missing_record_ids`: candidates could not be hydrated.

The existing hybrid provenance should remain available, but diagnostics should
not imply that final results contain duplicates when they only share candidates
between lanes. Add one small test per field for zero, positive, and degraded
cases.

## 11. Workstream E: upstream extraction

### 11.1 Candidate extraction: query-aware policy contracts

The mcp-memory upgrade exposed value in callbacks receiving a typed query
context rather than ambient filters or module state. Searchkernel should keep
this contract provider-neutral and document the planner's permission to skip a
lane when the requested limit is already satisfied.

### 11.2 Candidate extraction: cache epochs

mcp-memory's authoritative cache epochs and invalidation rules may be useful as
a generic searchkernel cache contract, but only if the abstraction describes
freshness/version tokens rather than SQLite or Postgres details. First define
the required lifecycle and stale-read semantics locally; then propose the
smallest upstream interface.

### 11.3 Candidate extraction: embedding integrity diagnostics

mcp-memory has concrete diagnostics for malformed embedding rows and safe
fallback behavior. A searchkernel contribution should be a backend-neutral
validation/result protocol, not a copy of mcp-memory's storage checks.

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

## 14. Open decisions

1. Should the evaluation corpus store stable memory references directly, or
   use a separate durable evaluation label that survives memory replacement?
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
