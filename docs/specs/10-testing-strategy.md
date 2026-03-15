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
  - importer normalization
- rules
  - no network
  - no real provider execution
  - use real temporary SQLite when needed

### 2.2 Medium Tests
- location: `tests/medium/`
- scope
  - integration between repository, runtime, daemon helpers, task queue, and provider wrappers
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
- staged ranking invariants (RRF, calibration, recency, working-memory boost, degradation)
- ranking ladder coverage for generic mixed-memory queries
- summary-first search/read behavior
- daemon startup and locking
- durable task retries and failed-task visibility
- provider fallback behavior
- relational import and typed-link behavior

## 5. Non-Goals
The test strategy does not assume:
- ZMQ transport
- chunking pipelines
- indexing-era compatibility layers

Vector retrieval remains optional in tests: suites should not require local embedding support unless the test is explicitly about semantic ranking.
