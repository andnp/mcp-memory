# Architecture Decision Record: Background Task Architecture

**Status:** Current durable-task baseline

## 1. Context
The daemon runs background work while still needing to tolerate process restarts and short-lived client sessions.

## 2. Current Decision
Use a SQLite-backed `tasks` table with a worker loop.

### Current Benefits
- durable enqueue before execution
- retry tracking
- failed-task visibility
- restart-safe background processing

## 3. Current Task Lifecycle

### 3.1 Enqueue
- `record_thought` writes to `system1_journal`
- threshold logic enqueues ingest work
- runtime bootstrap ensures recurring maintenance tasks exist
- when local embeddings are enabled, ingest still runs as a durable task but may cluster pending thoughts into smaller semantic mini-batches before consolidation

### 3.2 Execute
- the daemon starts a runtime worker
- the worker claims ready tasks from SQLite
- handlers process tasks, update status, and append immutable run logs to `task_runs`

### 3.3 Failure
- retries increment on failure
- tasks stop retrying after `max_retries`
- permanently failed tasks remain visible for inspection

### 3.4 Observability
- every execution attempt is logged in SQLite via `task_runs`
- overview/dashboard and CLI stats read directly from those task-run records
- task recency, average duration, and compressed-line totals are derived from the run log

## 4. Current Implemented Task Families
- ingest
- summarize
- graph linker
- conflict detector
- defragmenter
- taxonomist
- fact checker
- project manager
- sweeper

## 5. Current Non-Goals
The present runtime does **not** require:
- Huey
- reference-count-based shutdown semantics
- strategy-roulette scheduling
- always-on daemon behavior

## 6. Future Work
Potential future additions:
- richer scheduling policies
- broader maintenance agent roster
- more explicit per-task observability in the dashboard
- lifecycle changes if the daemon transport model evolves

The current runtime may also evolve from the SQLite-backed embedding table to a more specialized local vector engine later, but the initial semantic layer intentionally stays embedded and local-first.
