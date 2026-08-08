# Documentation Guide

This file is the shortest path to the repo's current documentation.

## Start here

If you need the current product/runtime story, read these first:

- `../README.md` — user-facing overview and operator quick start
- `specs/00-product-principles.md` — product invariants
- `specs/01-architecture-principles.md` — architecture invariants

## Status guide

Use `Status:` lines as the first navigation hint:

- `Active` — current baseline; safe default source of truth
- `Active, partially implemented` — current direction with shipped behavior plus explicit deferred work
- `Accepted direction` — adopted architectural direction, but not necessarily complete end-state implementation
- `Draft` — working design/backlog doc, not canonical runtime truth
- `Vision` — target/product-direction doc, not canonical runtime truth
- `Proposed direction` — candidate architectural direction, not canonical runtime truth

When a non-active spec conflicts with an active spec, the active spec wins.

## Core runtime and architecture

- `specs/07-background-task-architecture.md` — durable task model and runtime support loops
- `specs/08-system-resilience.md` — boot, shutdown, transport, and runtime-safety behavior
- `specs/14-work-item-execution-architecture.md` — proposed work-item execution direction
- `specs/15-search-repair-queue-architecture.md` — proposed repair-queue direction

## Storage and shared mode

- `specs/16-storage-backend-selection-and-shared-mode.md` — backend authority and shared-mode rules
- `specs/17-shared-mode-readthrough-cache.md` — local shared-mode cache plus current narrow writeback scope
- `specs/18-postgres-server-side-vector-search.md` — Postgres semantic-search path and capability-gated `pgvector` behavior
- `postgres-shared-mode-runbook.md` — operator setup, smoke checks, and failure guidance

## Memory model and product behavior

- `specs/03-memory-management-spec.md` — active runtime memory behavior
- `specs/04-temporal-fact-graph-spec.md` — relational link and fact-graph semantics
- `specs/09-autonomous-agent-requirements.md` — maintenance-agent product intent
- `specs/10-testing-strategy.md` — testing expectations and layers
- `specs/11-e2e-user-flows.md` — end-to-end product flows
- `specs/19-curation-quality-and-family-ownership.md` — proposed curation quality, coverage, routing, and convergence policy
- `specs/20-memory-mutation-history-and-restore.md` — proposed reversible history, protection, and restore model

## UI and operator surfaces

- `specs/05-management-dashboard-ui.md` — dashboard direction
- `specs/12-nerd-metrics-analytics.md` — analytics/telemetry direction
- `specs/13-provider-admission-control.md` — provider routing/admission rules

## Design backlog and future direction

These are intentionally useful, but they are **not** the primary source of current runtime behavior:

- `specs/02-autonomous-memory-vision.md`
- `specs/05-management-dashboard-ui.md`
- `specs/09-autonomous-agent-requirements.md`
- `specs/12-nerd-metrics-analytics.md`
- `specs/14-work-item-execution-architecture.md`
- `specs/15-search-repair-queue-architecture.md`
- `specs/19-curation-quality-and-family-ownership.md`
- `specs/20-memory-mutation-history-and-restore.md`

## Plans and history

The `plans/` directory is useful context, but it is **not** the primary source of current behavior when a plan conflicts with a spec, runbook, or the root `README`.

Recommended entries:

- `plans/07-architecture-review-and-refactor-roadmap.md` — larger refactor context
- `plans/08-24h-health-follow-up-task-list.md` — operator follow-up context
- `plans/10-curation-harness-and-typed-planner.md` — active direct-agent curator contract
- `plans/11-curation-implementation-task-list.md` — small-agent implementation backlog for the curation direction
- `plans/09-dual-backend-storage-and-writeback-cache.md` — historical bridge for the storage/cache transition
- `plans/09-single-writer-task-state-resilience.md` — task-state hardening context

## ADRs

- `adrs/2026-04-01-stats-driven-batch-selection-for-agentic-maintenance.md`

## Quick reading order by goal

### I want to understand the product

1. `../README.md`
2. `specs/00-product-principles.md`
3. `specs/01-architecture-principles.md`
4. `specs/03-memory-management-spec.md`

### I want to operate shared Postgres mode

1. `../README.md`
2. `postgres-shared-mode-runbook.md`
3. `specs/16-storage-backend-selection-and-shared-mode.md`
4. `specs/17-shared-mode-readthrough-cache.md`

### I want to understand runtime internals

1. `specs/07-background-task-architecture.md`
2. `specs/08-system-resilience.md`
3. `specs/14-work-item-execution-architecture.md`
4. `specs/15-search-repair-queue-architecture.md`

### I want to implement safe automatic curation

1. `specs/19-curation-quality-and-family-ownership.md`
2. `specs/20-memory-mutation-history-and-restore.md`
3. `plans/10-curation-harness-and-typed-planner.md`
4. `plans/11-curation-implementation-task-list.md`

### I want design history without mistaking it for current truth

1. `specs/16-storage-backend-selection-and-shared-mode.md`
2. `specs/17-shared-mode-readthrough-cache.md`
3. `plans/09-dual-backend-storage-and-writeback-cache.md`
