# Plan: Dual-Backend Storage, Postgres FTS, and Optional Writeback Cache

> **Status note (12 April 2026):** this document is now partly historical. The shipped runtime already has SQLite as the default local backend, Postgres as the authoritative backend in shared mode, an active shared-mode local SQLite sidecar cache, and two active cache modes: `readonly` for readthrough/degraded cached search+read behavior, and a deliberately narrow `writeback` mode for `record_thought` via durable local outbox queueing, opportunistic foreground flush, and a daemon-owned periodic background flusher. Broader offline mutation and generalized writeback remain deferred.

## 1. Context

`mcp-memory` no longer assumes SQLite is the only runtime source of truth.
SQLite is still the right default for local, single-machine use:

- zero configuration
- no external service dependency
- fast startup
- local-first behavior
- simple operator story

It is not the right answer for users who switch between machines frequently and want one shared authoritative store.
Using Syncthing or other live file-sync tooling against an active SQLite database is operationally unsafe and has already produced the exact failure modes the current warnings describe:

- shared-storage lock contention
- sync-conflict artifacts
- WAL / SHM interference
- corruption recovery work
- daemon request timeouts caused by storage contention rather than daemon death

The product should therefore support two first-class modes:

1. **Local mode** — SQLite, zero-config, single-machine optimized
2. **Shared mode** — Postgres, explicitly configured, multi-machine / cloud-backed

A third mode is desirable later:

3. **Shared mode with local resilience cache** — Postgres primary plus local cache and optional writeback queue for flaky connectivity

## 2. Decision Summary

The recommended architecture direction is:

- keep SQLite as the default backend
- add Postgres as an optional primary backend behind explicit config
- extract a storage abstraction boundary so the daemon/runtime depends on backend-neutral interfaces instead of SQLite details
- implement Postgres full-text search using native Postgres FTS
- treat broader writeback beyond the shipped narrow `record_thought` path as follow-up work after the primary Postgres path is stable
- design the cache seam now so phase-two work does not require a second architecture rewrite

## 2.1 Current implementation snapshot

As of March 2026, the codebase is already partway through this plan.

The following are already present in code:

- storage config models for `sqlite`, `postgres`, and active cache settings
- a runtime storage factory under `src/mcp_memory/storage/`
- backend-aware runtime composition via `ApplicationContext.storage_backend`
- Postgres bootstrap + migrations
- Postgres repositories for core operational stores
- an active shared-mode local SQLite sidecar cache under `.../memories/cache/shared_read_cache.sqlite3` when enabled
- shipped `readonly` cache behavior for readthrough/degraded cached search+read
- shipped narrow `writeback` behavior for `record_thought` only
- backend-aware management health/status reporting
- focused Postgres integration coverage for runtime, daemon health, hooks, search, and backup skipping

That means the main remaining work is no longer “add a storage seam from scratch.”
The real remaining work is:

- align docs/specs with the shipped implementation
- replace residual duck-typed backend branching in operator/helper layers with cleaner protocol-level seams
- decide whether the current Postgres lexical-search implementation is the intended steady state or an interim step toward a projection table
- document Postgres-mode filesystem semantics and active-vs-deferred cache behavior explicitly

## 3. Product Goals

### 3.1 User-facing goals

- preserve a zero-config local install path
- support one shared cloud-hosted store for roaming users
- eliminate the need to sync live SQLite data directories between machines
- make backend choice explicit and inspectable
- allow future degraded/offline behavior without silent data divergence

### 3.2 Engineering goals

- remove direct runtime dependence on `sqlite3`, `PRAGMA`, `BEGIN IMMEDIATE`, and FTS5-specific assumptions from high-level runtime code
- keep search semantics and tool contracts stable across backends where practical
- improve task-queue concurrency behavior under Postgres
- preserve current SQLite behavior while refactoring
- widen tests deliberately instead of forcing full cross-backend parity on day one

## 4. Non-Goals for the First Postgres Slice

The first Postgres implementation should **not** attempt to ship:

- active-active replication between SQLite and Postgres
- full offline multi-writer conflict resolution
- transparent cross-device merge semantics
- perfect offline semantic search parity
- provider/daemon execution in fully disconnected writeback mode
- backend-agnostic SQL by forcing both databases into one least-common-denominator query shape

The goal is a clean primary backend option, not instant universal distributed systems glory.

## 5. Proposed Product Modes

### 5.1 Mode A — Local SQLite

**Target user:** single-machine user, zero-config install

**Characteristics:**
- local runtime source of truth
- zero external dependencies
- existing daemon model preserved
- current backup model preserved
- no change to basic install experience

**Default:** yes

### 5.2 Mode B — Shared Postgres

**Target user:** roaming user or team deployment with one shared authoritative database

**Characteristics:**
- Postgres is the primary source of truth
- daemon still runs locally as the MCP/worker runtime
- storage moves off the local synced directory and into the DB service
- local filesystem state may still exist for runtime support concerns, but it is not the authoritative memory store
- backend config is explicit
- SQLite-specific backup/repair logic no longer applies as the primary operational path

**Default:** no

### 5.3 Mode C — Shared Postgres + Local Cache

**Target user:** roaming user with intermittent connectivity

**Characteristics:**
- Postgres remains authoritative when reachable
- local cache serves recent reads
- current implementation can also queue `record_thought` locally during qualifying outages
- broader replay remains a deliberate future design area

**Default:** no

**Current implementation:** active, but intentionally narrow

## 6. Configuration Model

Add a storage section to config.

Suggested shape:

```toml
[storage]
backend = "sqlite" # or "postgres"

[storage.sqlite]
path = "" # optional; empty means existing XDG default

[storage.postgres]
dsn = ""
pool_min = 1
pool_max = 10
statement_timeout_ms = 30000
lock_timeout_ms = 5000
application_name = "mcp-memory"

[storage.cache]
enabled = false
mode = "readonly" # readonly | writeback
max_cached_search_docs = 50000
max_outbox_entries = 10000
```

`[storage.cache]` is no longer purely reserved.

Current implementation:

- `readonly` is active for readthrough/degraded cached search+read behavior
- `writeback` is active only for `record_thought`

Still deferred:

- broader offline mutation
- generalized replay/conflict handling

Additional CLI / health requirements:

- daemon health/status must report active backend
- startup must fail fast on invalid Postgres config
- cache mode must be visible in operator health/overview output when enabled

## 7. Architecture Direction

## 7.1 Current coupling to remove

The current runtime is tightly coupled to SQLite through:

- `src/mcp_memory/mcp/runtime.py`
- `src/mcp_memory/utils/db.py`
- `src/mcp_memory/utils/db_schema.py`
- `src/mcp_memory/core/tasks.py`
- `src/mcp_memory/core/journal.py`
- `src/mcp_memory/relational/repository.py`
- `src/mcp_memory/relational/search.py`
- `src/mcp_memory/runtime_log_store.py`
- `src/mcp_memory/provider_usage_store.py`
- `src/mcp_memory/work_item_store.py`
- `src/mcp_memory/embedding_repair_store.py`
- `src/mcp_memory/sqlite_backup.py`

SQLite-specific assumptions currently include:

- `sqlite3.Connection`
- `PRAGMA` configuration and integrity checks
- `BEGIN IMMEDIATE`
- FTS5 virtual tables
- WAL/SHM behavior
- file-based backup logic
- stringly schema bootstrapping tied to SQLite DDL semantics

## 7.2 Required new seams

Introduce a backend-neutral storage architecture with four layers:

1. **Storage config + backend factory**
2. **Connection/session provider**
3. **Repository layer**
4. **Search adapter layer**

### 7.2.1 Backend factory

Create a runtime backend factory that provides:

- connection/session manager
- migration/bootstrap runner
- repository set
- search adapter
- optional backup/maintenance hooks
- optional local cache hooks

## 7.2.2 Repository protocols

Formalize backend-neutral protocols for:

- memory repository
- journal repository
- task queue / task runs
- runtime logs
- provider usage / AI conversations
- work items
- embedding repair queue
- telemetry stores

The runtime should compose these protocols, not instantiate SQLite repositories directly.

Current status:

- the runtime already composes backend-specific resources through a factory
- several surfaces still carry `Any`-typed resources and branch on connection API shape rather than true protocol boundaries

The remaining architectural task is to replace connection-shape probing and backend-specific SQL adaptation in operator/helper code with explicit repository/query interfaces.

### 7.2.3 Search adapter

Create a search abstraction with backend-specific lexical retrieval implementations:

- SQLite lexical search via FTS5
- Postgres lexical search via `tsvector`

The later fusion/ranking pipeline should operate on a shared candidate representation.

Current status:

- SQLite lexical retrieval uses the existing FTS5 path
- Postgres lexical retrieval is already implemented with weighted `tsvector` construction in query-time CTEs over memories + aggregated tags

The remaining decision is whether that inline Postgres FTS shape is the intended long-term design or whether the system should later move to an explicit search projection table.

## 8. Phase Plan and Current Status

### 8.1 Phase 0 — ADR, interfaces, and invariants

### Status
Mostly complete. This plan exists and the dedicated storage ADR/spec now lives in `docs/specs/16-storage-backend-selection-and-shared-mode.md`.

### Goal
Define the storage seam and operational rules before changing behavior.

### Deliverables

- this planning document
- a storage ADR/spec update in `docs/specs/`
- backend-neutral interface definitions
- explicit invariants for cache/writeback behavior

### Key invariants

- SQLite remains fully supported and is still the default
- when `storage.backend = postgres`, Postgres is authoritative
- backend must be surfaced in daemon health/status
- no hidden fallback from broken Postgres to SQLite
- cache/writeback must never silently fork history

### Acceptance criteria

- architecture agreed in docs
- interface boundaries identified
- no code behavior change yet

### 8.2 Phase 1 — Extract the storage boundary

### Status
Mostly complete in code. The backend factory and storage package exist, but protocol formalization and helper-layer cleanup remain.

### Goal
Refactor runtime composition to stop hard-coding SQLite creation.

### Work

1. Introduce `StorageBackendKind` and storage config models
2. Replace direct `DatabaseManager(spec.memory_path / "indices" / "memory.db")` construction with a backend factory
3. Separate current `DatabaseManager` responsibilities into:
   - connection/session management
   - schema bootstrap / migration management
4. Add repository protocols and SQLite-backed implementations conforming to them
5. Update `ApplicationContext` and runtime composition to depend on protocol-level components rather than concrete SQLite classes where practical
6. Remove residual backend-detection shims from helper/operator modules that still branch on connection API shape

### Files likely touched

- `src/mcp_memory/config.py`
- `src/mcp_memory/context.py`
- `src/mcp_memory/mcp/runtime.py`
- new storage/backend modules under `src/mcp_memory/storage/` or similar

### Acceptance criteria

- SQLite mode behaves exactly as before
- tests continue passing in SQLite mode
- runtime composition no longer hard-codes SQLite DB path at the top level
- operator/helper layers no longer need placeholder rewriting or backend probing for routine reporting/telemetry flows

### 8.3 Phase 2 — Add Postgres schema and migration system

### Status
Largely complete for the current operational surface, though the docs/spec story has not caught up.

### Goal
Create a first-class Postgres storage bootstrap path.

### Work

1. Add Postgres migration runner
2. Create versioned Postgres DDL/migrations for core tables:
   - `memories`
   - `memory_workspaces`
   - `tags`
   - `memory_tags`
   - `links`
   - `system1_journal`
   - `tasks`
   - `task_runs`
   - `runtime_logs`
   - `provider_usage`
   - `task_execution_attempts`
   - `ai_conversations`
   - `provider_admission_state`
   - `memory_tool_events`
   - `provider_policy_events`
   - `work_items`
   - `embedding_repair_queue`
   - `schema_metadata`
3. Define index strategy explicitly for Postgres
4. Add health/bootstrap checks for Postgres connectivity and schema version

### Data-type recommendations

Use native Postgres features rather than SQLite-shaped emulation where it improves correctness:

- `uuid` for stable IDs where compatible
- `jsonb` for metadata and result payloads
- `timestamptz` for timestamps
- `text` where dynamic string semantics are important

If timestamp format compatibility is sensitive, normalize at the repository boundary rather than keeping the schema artificially SQLite-shaped.

### Acceptance criteria

- empty Postgres DB initializes cleanly
- daemon can boot against Postgres
- schema version reporting works for both backends

### 8.4 Phase 3 — Implement Postgres repository layer

### Status
Substantially complete for the current runtime surface. The main remaining work is boundary cleanup rather than first implementation.

### Goal
Support core runtime behavior on Postgres.

### Work

Implement Postgres-backed versions of:

- relational memory repository
- journal store
- task queue and task runs
- runtime log repository
- provider usage / AI conversation repository
- work item repository
- embedding repair queue

### Concurrency changes

Replace SQLite-specific queue semantics with Postgres-native transactional patterns.

#### Current SQLite patterns to remove

- `BEGIN IMMEDIATE`
- busy-timeout-driven retry logic
- WAL-based assumptions about writer arbitration

#### Postgres replacements

Use patterns like:

- `SELECT ... FOR UPDATE SKIP LOCKED`
- transactional claim/update/return flows
- statement/lock timeouts configured at the connection/session level

This is especially important for:

- `tasks`
- `work_items`
- `embedding_repair_queue`

### Operational lessons to carry forward

A prior Postgres-backed design note already surfaced two important implementation hazards:

1. pooled connection handles must return to the pool on close/exit
2. DB-backed logging must tolerate pre-migration startup ordering

So the repository/session layer must ensure:

- safe pool lifecycle
- central connection wrapper behavior
- startup log sink tolerance for missing tables before migrations complete

### Acceptance criteria

- memory CRUD works on Postgres
- task worker can claim/complete/fail tasks correctly on Postgres
- runtime logs and provider telemetry persist correctly
- no pool leaks under startup/logging load

### 8.5 Phase 4 — Implement Postgres FTS

### Status
Partially complete. Postgres lexical search exists, but the document still proposes a different recommended long-term shape than the code currently uses.

### Goal
Provide first-class lexical search in Postgres.

### Decision
Postgres FTS is in scope for the initial Postgres backend.

### Current implementation

The current code uses weighted `tsvector` construction at query time over `memories` plus aggregated tags.
That is enough for the first functional Postgres slice and keeps the lexical-candidate stage backend-specific while leaving later ranking/fusion logic shared.

### Future option

If query-time document construction becomes too expensive at realistic corpus sizes, move to a dedicated Postgres search projection table instead of reproducing SQLite FTS inline.

Suggested table:

- `memory_search_documents`
  - `memory_id`
  - `workspace_ids_text` or derived search hints if needed
  - `title`
  - `summary`
  - `content`
  - `tags_text`
  - `status`
  - `type`
  - `search_tsv`
  - `updated_at`

### Why a projection table

- avoids hot-path multi-join complexity for every query
- gives explicit control over weighted FTS document construction
- makes reindex/rebuild logic clearer
- keeps backend-specific search maintenance isolated

### Search ranking shape

Build weighted `tsvector` documents, for example:

- title = weight A
- summary = weight A/B
- tags = weight B
- content = weight C

Then retrieve lexical matches via:

- `websearch_to_tsquery` or `plainto_tsquery` depending on query semantics
- `ts_rank` / `ts_rank_cd`
- GIN index on `search_tsv`

### Integration rule

Only the lexical candidate retrieval layer should differ by backend.
The later ranking pipeline should stay backend-neutral as much as possible.

### Semantic search question

Do **not** couple Postgres backend support to immediate `pgvector` adoption.

Recommended order:

1. ship Postgres lexical search via FTS
2. keep existing semantic layer behind its own abstraction
3. decide later whether semantic embeddings should:
   - remain local/backend-specific
   - move into Postgres with `pgvector`
   - or support both

### Acceptance criteria

- keyword search quality is acceptable on Postgres
- memory search contracts remain stable
- lexical retrieval performance is acceptable with realistic corpus sizes
- the plan explicitly states whether inline `tsvector` construction remains acceptable or has been superseded by a projection-table design

### 8.6 Phase 5 — Runtime/bootstrap/ops integration

### Status
Partially complete. Backend reporting and backup skipping exist, but docs and operator guidance are still SQLite-heavy.

### Goal
Make backend selection operationally real and observable.

### Work

1. Report backend type in daemon health/status
2. Add backend-aware startup validation
3. Add backend-aware maintenance hooks
4. Update docs for:
   - local SQLite mode
   - shared Postgres mode
   - operator setup expectations
5. Adjust backup/maintenance language:
   - SQLite backup remains file-snapshot oriented
   - Postgres backup becomes DB-service-managed and no longer uses `sqlite_backup.py`

### Acceptance criteria

- backend appears in status output
- configuration errors are explicit
- docs explain why Syncthing is not the shared-storage solution
- docs explain which local filesystem paths still exist in Postgres mode and which ones are non-authoritative

### 8.7 Phase 6 — Test strategy

### Status
Partially complete. Focused Postgres integration tests exist, but the testing docs still describe a mostly SQLite-only posture.

### Goal
Add confidence without exploding the matrix immediately.

### SQLite lane

Keep SQLite as the default CI/test path.

### Postgres lane

Add focused integration tests for:

- memory CRUD
- tag/workspace/link operations
- task queue claim/complete/fail
- runtime logs
- provider usage / conversations
- daemon startup with Postgres config
- Postgres FTS retrieval

### Guidance

Do not force the entire existing suite to run on both backends initially.
Start with targeted Postgres coverage for the core operational paths, then widen as confidence grows.

Also keep the testing strategy docs aligned with reality so contributors know which Postgres paths already have active coverage.

## 9. Writeback cache design

### 9.1 Recommendation

Treat **broader** local caching/writeback as phase-two work.

That remains the right order even though a narrow slice is now shipped, because:

- primary backend correctness is hard enough on its own
- writeback caching adds replay, idempotency, versioning, and conflict semantics
- conflating both efforts will hide root causes and slow validation

Implementation note: config keys for cache mode are now active, but the docs must still state clearly that `writeback` does **not** mean general offline functionality.

### 9.2 Cache architecture target

When enabled under Postgres mode, the local cache should be a **local SQLite sidecar** used for:

- recent reads
- search document cache
- mutation outbox
- connectivity/degraded-mode bookkeeping

This should be a separate local cache database, not a second source of truth.

### Source-of-truth rule

- Postgres remains authoritative
- local cache is derivative plus queued intents

### 9.3 Cache modes

#### `disabled`
- no local cache
- online-only shared mode

#### `readonly`
- local read cache only
- reads may succeed from cache when offline
- writes require server connectivity

#### `writeback`
- local read cache
- writes go to local outbox when offline
- replay to Postgres when connectivity returns

### Shipping order

1. `disabled`
2. `readonly`
3. `writeback`

### 9.4 Read caching behavior

#### Safe first scope

Cache:

- recently read memory records
- search projection rows for lexical search
- maybe recent telemetry for UX

#### Offline behavior

When offline:

- reads may return cached data
- results must be marked degraded/stale where appropriate
- semantic search may be partial or disabled
- background mutation agents should not silently mutate authoritative state from cache alone

### 9.5 Writeback behavior

#### Core design

Use a local durable outbox table for pending mutations.
Each queued mutation must include:

- mutation type
- payload
- idempotency key
- local enqueue time
- target entity ID
- expected server version / precondition metadata where needed
- replay status
- last replay error

#### Replay requirements

Replay must be:

- idempotent
- ordered where necessary
- version-aware
- observable

#### Conflict policy

If replay encounters server-side version drift:

- do not silently overwrite
- either reject, rebase, or send to explicit conflict handling

#### Initial restriction

The first writeback version should only support a limited set of mutation families, for example:

- `record_thought`
- selected append/create operations

Do **not** initially allow the full maintenance mutation surface to run in disconnected writeback mode.

Current status: only `record_thought` is shipped today.

### 9.6 Search behavior with cache

#### Recommended model

- online: query Postgres primary, optionally warm local cache
- offline readonly: search local cached lexical projection only
- offline writeback: same as readonly plus pending local-mutation awareness where safe

#### Important constraint

Do not promise perfect offline search parity.
A degraded but explicit offline mode is acceptable.
Silent divergence is not.

### 9.7 Observability for cache/writeback

When cache or writeback mode is enabled, expose:

- connectivity state
- last successful sync time
- outbox depth
- oldest pending mutation age
- replay success/failure counts
- cached-read hit rate
- degraded-search rate

## 10. Implementation order

Recommended remaining sequence:

1. update `README.md` and relevant specs to reflect the shipped storage split and new ADR
2. formalize repository/query protocols instead of relying on `Any`-typed resources and backend probing
3. move operator/helper SQL out of generic helpers and into backend-specific repository or service adapters
4. ratify the current inline Postgres FTS design or deliberately replace it with a projection table
5. document Postgres-mode local filesystem semantics and active-vs-deferred cache config behavior
6. widen focused Postgres coverage only where the remaining abstraction cleanup touches real behavior
7. add optional read-only local cache
8. add optional writeback outbox

## 11. Acceptance milestones

### 11.1 Milestone A — backend seam landed

- SQLite default path unchanged
- no direct top-level runtime dependency on SQLite DB path construction
- repository interfaces exist

### 11.2 Milestone B — Postgres backend functional

- daemon boots with Postgres config
- CRUD + tasks + logs work
- no connection-pool leaks
- health/status surfaces backend details

### 11.3 Milestone C — Postgres FTS functional

- lexical search works acceptably on Postgres
- result contracts unchanged
- ranking pipeline remains coherent

### 11.4 Milestone D — read-only cache functional

- cached reads available during outages
- degraded mode explicit
- no local authoritative writes

### 11.5 Milestone E — writeback queue functional

- supported mutation families replay safely
- idempotency enforced
- conflict cases surfaced explicitly

## 12. Open design questions

These need explicit answers during implementation, not accidental drift:

1. Should semantic embeddings remain backend-specific or become a separate independent storage layer?
2. Should Postgres mode use native `uuid` / `timestamptz` everywhere, or preserve string/timestamp compatibility more aggressively?
3. How much search projection denormalization is acceptable before rebuild cost becomes too high?
4. Which mutation families are safe enough for first writeback support?
5. Should offline mode allow background deterministic maintenance at all, or restrict to user-initiated writes only?
6. Should cache replay happen inside the daemon only, or can CLI commands trigger manual sync repair?

## 13. Immediate next coding tranche

The next tranche should focus on matching the docs and boundaries to the implementation that already exists:

1. update this plan, the new storage ADR, and the top-level docs to distinguish shipped behavior from deferred cache/writeback work
2. formalize protocol-level interfaces for repositories, queues, logs, and query/reporting helpers
3. remove connection-shape branching from modules such as reporting, hook reminders, and retrieval telemetry
4. decide whether the current inline Postgres FTS implementation is final for the first stable shared-mode release or a temporary step toward a projection table
5. document what local filesystem state still exists in Postgres mode and what remains authoritative
6. add a one-way SQLite-to-Postgres migration/import path so existing local users can adopt shared mode without manual SQL surgery
7. define explicit performance/operability exit criteria for Postgres FTS, queue claiming, and startup health checks
8. write an operator runbook for shared mode covering setup, backup/restore responsibility, failure modes, and “do not sync live SQLite” guidance

That sequence tightens the architecture without pretending the project is still at day-zero backend extraction.

## 14. Final recommendation

This project is worth doing.
It cleanly solves the multi-machine use case without harming the local-first default.

The correct product framing is:

- **SQLite by default**
- **Postgres for shared/cloud mode**
- **shared-mode cache active today, with broader writeback later**

That keeps the zero-config path intact, gives roaming users a first-class answer, and avoids repeating the operational problems caused by syncing a live SQLite store across machines.

At this point the plan should treat Postgres support as an in-flight implementation with remaining cleanup work, not as a purely future architecture sketch.
