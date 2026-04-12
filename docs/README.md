# Documentation Guide

This file is the shortest path to the repo's current documentation.

## Start here

If you need the current product/runtime story, read these first:

- `../README.md` — user-facing overview and operator quick start
- `specs/00-product-principles.md` — product invariants
- `specs/01-architecture-principles.md` — architecture invariants

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

## UI and operator surfaces

- `specs/05-management-dashboard-ui.md` — dashboard direction
- `specs/12-nerd-metrics-analytics.md` — analytics/telemetry direction
- `specs/13-provider-admission-control.md` — provider routing/admission rules

## Plans and history

The `plans/` directory is useful context, but it is **not** the primary source of current behavior when a plan conflicts with a spec, runbook, or the root `README`.

Recommended entries:

- `plans/07-architecture-review-and-refactor-roadmap.md` — larger refactor context
- `plans/08-24h-health-follow-up-task-list.md` — operator follow-up context
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

### I want design history without mistaking it for current truth

1. `specs/16-storage-backend-selection-and-shared-mode.md`
2. `specs/17-shared-mode-readthrough-cache.md`
3. `plans/09-dual-backend-storage-and-writeback-cache.md`