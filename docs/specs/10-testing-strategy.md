# Specification: Testing Strategy

## 1. Overview
The test suite uses the `small` / `medium` / `large` structure and prefers real local components over brittle mocks.

## 2. Test Categories

### 2.1 Small Tests
- location: `tests/small/`
- scope
  - isolated logic
  - schema/repository helpers
  - deterministic ranking math
  - FTS candidate retrieval
  - graph-aware reranking and expansion behavior
  - importer normalization
- rules
  - no network
  - no real provider execution
  - use real temporary SQLite when needed

### 2.2 Medium Tests
- location: `tests/medium/`
- scope
  - integration between repository, runtime, daemon helpers, task queue, and provider wrappers
  - agentic maintenance handler behavior and task routing
  - daemon stop/restart hardening and stale-owner recovery
- rules
  - no external network
  - provider behavior should use deterministic fakes
  - prefer real runtime wiring over mocking internals

### 2.3 Large Tests
- location: `tests/large/`
- scope
  - end-to-end daemon/proxy/runtime flows
  - locking and lifecycle behavior
  - full user journeys such as thought capture -> ingest -> search/read
- rules
  - local process startup is allowed
  - external provider execution should remain opt-in, not default

Live dogfooding remains valuable for this repo because the product is the workflow: targeted live CLI/daemon checks are encouraged when the change touches agentic maintenance, queue behavior, or daemon lifecycle semantics.

## 3. Fixture Strategy

### 3.1 Real SQLite First
Prefer real temporary SQLite databases instead of mocking repository/database behavior.

### 3.2 Fake Providers
Use deterministic fake providers for:
- ingest actions
- summary generation
- subprocess-backed provider wrappers

### 3.3 Focused Seed Data
Seed only the tables that are part of the active runtime contract.
Avoid keeping fixtures for deleted indexing-era schema.

## 4. Coverage Priorities
- workspace-aware search ordering
- staged ranking invariants (RRF, calibration, recency, canonical-support tuning, working-memory boost/dampening, graph authority, graph expansion, degradation)
- ranking ladder coverage for generic mixed-memory queries
- summary-first search/read behavior
- daemon startup and locking
- daemon provenance, stale-metadata recovery, duplicate-owner cleanup, and stale-process-group shutdown
- durable task retries and failed-task visibility
- provider fallback behavior
- agentic ingest/deduplicator/curator behavior on the internal MCP maintenance surface
- relational import and typed-link behavior
- ZMQ transport concurrency and stale-socket recovery once the transport migration begins

## 5. Non-Goals
The test strategy does not assume:
- chunking pipelines
- indexing-era compatibility layers

Vector retrieval remains optional in tests: suites should not require local embedding support unless the test is explicitly about semantic ranking.

Current tests no longer treat HTTP as the authoritative transport; the runtime is ZMQ-first. Continue adding focused large tests for ROUTER/DEALER concurrency, heartbeat loss, and stale-socket recovery rather than treating transport migration as invisible infrastructure churn.
