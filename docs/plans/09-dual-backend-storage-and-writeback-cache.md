# Plan History: Dual-Backend Storage and Writeback Cache

## Status

This document is now a **historical bridge**, not the canonical source of current behavior.

The storage split and shared-mode cache work described here are no longer purely planned:

- SQLite remains the default backend
- Postgres is the authoritative shared-mode backend
- shared-mode local cache behavior is shipped
- narrow `record_thought` writeback is shipped
- a daemon-owned background flusher is shipped

Use the following documents as the current source of truth:

- `docs/specs/16-storage-backend-selection-and-shared-mode.md`
- `docs/specs/17-shared-mode-readthrough-cache.md`
- `docs/specs/18-postgres-server-side-vector-search.md`
- `docs/postgres-shared-mode-runbook.md`
- `README.md`

## Why this plan existed

This plan captured the transition from a SQLite-only mental model toward:

1. explicit backend selection
2. Postgres shared mode as the correct answer for roaming/shared use
3. a local shared-mode cache sidecar
4. eventual writeback resilience for flaky connectivity

It was useful while the architecture was still being extracted and the storage boundary was only partially real.

## What is now shipped

### Backend model

- **SQLite** is still the zero-config default for local use.
- **Postgres** is the explicit shared-mode backend.
- shared mode does **not** silently fall back to SQLite.

### Shared-mode cache

- `storage.cache.mode = "readonly"` is active.
- `storage.cache.mode = "writeback"` is active, but only for a narrow resilience path.
- the local SQLite sidecar cache is derivative and non-authoritative.

### Narrow writeback support

Current writeback support is intentionally limited to `record_thought`:

- foreground writes can degrade into a durable local outbox during outage/timeout conditions
- later successful authoritative writes can opportunistically flush queued thoughts
- the daemon also runs a periodic background flusher

Broader offline mutation and generalized writeback remain deferred.

### Semantic search direction

The Postgres vector-search path is no longer just a proposal:

- capability detection is shipped
- diagnostics report the active semantic-search mode
- server-side `pgvector` search is used when the backend is ready
- Python fallback remains available when it is not

## What is still deferred

The following are still future work rather than shipped product guarantees:

- broader offline mutation/writeback beyond `record_thought`
- generalized conflict-aware replay semantics for larger mutation families
- full offline parity for shared-mode search
- final consolidation of all backend-neutral repository/query seams
- ANN/vector-tuning work beyond the current capability-gated baseline

## Remaining cleanup opportunities

The main remaining work after the current implementation wave is documentation and boundary cleanup, not greenfield storage design:

1. keep specs, runbooks, and README aligned as cache/writeback expands
2. continue replacing helper-layer backend probing with protocol-driven interfaces
3. decide how much broader writeback should become, if at all
4. improve operator visibility for cache hit rate, outbox depth, replay health, and degraded-mode behavior

## Why this file still exists

This file remains useful as a short historical handoff because it explains **why** the storage split and local shared-mode cache were introduced, while pointing readers at the canonical docs that now describe the active system.
