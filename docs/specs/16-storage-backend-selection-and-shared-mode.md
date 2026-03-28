# Architecture Decision Record: Storage Backend Selection, Shared Mode, and Deferred Writeback

**Status:** Active direction, partially implemented

## 1. Decision

`mcp-memory` supports two storage backends with distinct product roles:

- **SQLite** remains the default backend for local, zero-config, single-machine use.
- **Postgres** is the explicit shared-mode backend for multi-machine or cloud-hosted use.

The system must not silently fall back from a broken Postgres configuration to SQLite.
When shared mode is selected, Postgres is authoritative.

Local cache and writeback behavior are explicitly deferred.
They are not part of the first stable shared-mode contract.

## 2. Why this decision exists

The current local-first SQLite model is still the right default for most single-machine users:

- zero configuration
- no external service dependency
- fast startup
- simple operator story

It is not the right answer for roaming users who want one shared authoritative memory store across machines.
Syncing a live SQLite store through Syncthing or similar file-sync tools is operationally unsafe and has already produced the failure modes the product warns about:

- lock contention around a live database
- sync-conflict artifacts
- WAL / SHM interference
- corruption recovery work
- daemon timeouts caused by storage contention rather than daemon death

The correct product answer is therefore backend selection, not “better live SQLite syncing.”

## 3. Current implementation baseline

As of March 2026, this decision is not purely aspirational.
The codebase already includes:

- storage config models for `sqlite`, `postgres`, and reserved cache settings
- a backend factory under `src/mcp_memory/storage/`
- backend-aware runtime composition through `ApplicationContext.storage_backend`
- Postgres connection management, bootstrap, and migrations
- Postgres-backed repositories for the current operational surface
- backend-aware health/status reporting in the management layer
- focused Postgres integration coverage for runtime, daemon health, hooks, search, and backup skipping

The remaining work is mostly about tightening the abstraction boundary, aligning docs/specs, and closing operator gaps.

## 4. Product modes

### 4.1 Local SQLite mode

**Default:** yes

**Target user:** local single-machine user

**Behavior:**
- SQLite is the runtime source of truth.
- File-based backup and repair workflows apply.
- Zero external database service is required.

### 4.2 Shared Postgres mode

**Default:** no

**Target user:** roaming user or shared/cloud deployment

**Behavior:**
- Postgres is the authoritative memory store.
- The daemon still runs locally as the MCP/worker runtime.
- Operator-facing health and status surfaces must report that Postgres is active.
- SQLite-specific backup/repair guidance does not apply as the primary storage story.

### 4.3 Shared mode with local cache / writeback

**Default:** no

**Behavior:**
- reserved for future work
- not part of the current supported shared-mode contract

Any current config fields related to cache/writeback are preparatory only and must not be documented as active shared-mode functionality.

## 5. Authority and fallback rules

The system must follow these invariants:

1. **SQLite remains fully supported and remains the default.**
2. **When `storage.backend = "postgres"`, Postgres is authoritative.**
3. **There is no hidden fallback from broken Postgres to SQLite.**
4. **Cache or writeback mode must never silently fork history.**
5. **Workspace identity remains metadata for ranking/analytics, not a storage tenancy boundary.**

Fail-fast behavior matters here.
Invalid Postgres configuration, unavailable drivers, or bootstrap failures should stop shared mode clearly rather than degrading into a second implicit storage system.

## 6. Runtime and filesystem semantics

Shared mode does **not** mean “no local filesystem state exists.”
It means “the local filesystem is not the authoritative memory store.”

In Postgres mode:

- the daemon still runs locally
- local runtime paths may still exist for support concerns such as state, sockets, logs, embeddings, or future cache sidecars
- those local paths are non-authoritative unless a future cache design explicitly says otherwise

This distinction must be documented clearly so operators do not mistake the presence of local files or directories for a second source of truth.

## 7. Architecture boundary

The intended storage architecture has four layers:

1. storage config + backend factory
2. connection/session provider
3. repository/query layer
4. search adapter layer

This seam already exists in broad shape, but the codebase still has cleanup work remaining.
Several helper/operator modules still branch on connection API shape and adapt SQL by backend instead of depending on stable protocol-level interfaces.

The intended direction remains:

- runtime composition should depend on backend-neutral interfaces where practical
- backend-specific SQL should live in backend-specific repository/query implementations
- helper and operator surfaces should not need to probe whether they received SQLite or Postgres

## 8. Search decision

### 8.1 Current baseline

Postgres lexical search is in scope for the first stable shared-mode backend.
The current implementation uses weighted `tsvector` construction at query time over memories plus aggregated tags.

This is an acceptable functional baseline because:

- lexical candidate retrieval remains backend-specific
- later ranking and fusion logic can remain shared
- it avoids coupling the backend switch to a second large search-maintenance redesign

### 8.2 Future optimization option

A dedicated Postgres search projection table remains a valid future optimization if query-time document construction becomes too expensive at realistic corpus sizes.

That should be treated as a performance/operability decision, not as a prerequisite for the initial shared-mode contract.

### 8.3 Semantic search

The system should not require `pgvector` to make Postgres shared mode viable.
Semantic storage can remain behind its own abstraction until there is a stronger product reason to standardize on a specific shared-mode embedding strategy.

## 9. Migration policy

The storage story needs an explicit migration path for existing local users.

Required direction:

- provide a **one-way SQLite → Postgres migration/import path**
- do not require manual SQL surgery to adopt shared mode
- do not introduce dual-write or active-active replication between SQLite and Postgres

The migration/import tool should preserve the current runtime contract as closely as possible:

- memories
- workspace mappings
- tags
- links
- journal/task/log state where appropriate for the target workflow
- enough metadata/lineage to preserve retrieval quality and management visibility

Whether all operational telemetry should migrate is a product decision; memory integrity matters more than perfect telemetry continuity.

## 10. Operational policy

### 10.1 SQLite mode

Operational guidance remains file-oriented:

- local filesystem store
- file snapshot backups
- corruption recovery / rebuild workflows
- explicit warnings against syncing a live SQLite database across machines

### 10.2 Postgres mode

Operational guidance becomes service-oriented:

- Postgres owns durability, backup, and restore semantics
- local SQLite snapshot tooling is not the primary backup story
- health/status output must disclose the active backend
- startup must fail fast on invalid configuration or missing driver dependencies

## 11. Deferred work

The following are intentionally deferred from the first stable shared-mode contract:

- local read-only cache
- writeback outbox and replay
- offline multi-writer conflict handling
- transparent SQLite/Postgres dual-write
- full semantic-search backend convergence
- `pgvector` as a required dependency

## 12. Immediate follow-up work

The highest-value next steps after this ADR are:

1. align `README.md` and other user-facing docs with the shipped storage split
2. replace remaining backend-detection shims in helper/operator modules with protocol-driven interfaces
3. decide whether the current Postgres lexical-search baseline is sufficient or needs a projection-table follow-up
4. add a one-way SQLite → Postgres migration/import tool
5. write a shared-mode operator runbook with setup, backup/restore, and failure guidance
6. define explicit performance and operability exit criteria for Postgres search, queue claiming, and startup health

## 13. Consequences

This decision keeps the product honest:

- local-first remains simple
- shared mode has a real first-class answer
- the repo does not pretend that syncing a live SQLite database is an acceptable substitute for backend design
- cache/writeback can be designed deliberately later instead of being smuggled into the first shared-mode release

The main cost is that the repo must now carry two backend implementations and keep docs/specs disciplined enough that the product story stays coherent across both.