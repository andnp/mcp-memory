# Specification: End-to-End User Flows

This document defines the highest-value user journeys that should stay green as `mcp-memory` evolves.

## 1. Thought Capture to Searchable Memory

**Goal:** a user records quick thoughts, background ingestion promotes them into a structured relational memory, and the result becomes searchable/readable.

**Runtime path:**
- `record_thought`
- `System1Journal`
- task queue threshold rule
- background ingest worker
- relational repository
- relational search/read tools

**Key invariants:**
- the ingest task is created once at the threshold and reused afterward
- journal entries are marked processed
- at least one relational memory is created
- the created memory is searchable and readable through MCP tools

## 2. Shared Runtime Across Clients

**Goal:** multiple clients attached to the same workspace share one runtime and see the same state immediately.

**Runtime path:**
- `DaemonLifecycleController.acquire_runtime()`
- MCP tool handlers on the shared runtime
- relational repository/search service

**Key invariants:**
- two clients acquire the same runtime object
- writes from one client are visible to the other without restart
- releasing one client does not shut down the runtime while another is active

## 3. Import and Persistence Across Restart

**Goal:** user imports markdown and creates records, then the daemon shuts down cleanly; a later session reopens the workspace and sees the same data.

**Runtime path:**
- `import_markdown_memory_file`
- `create_memory_record`
- relational SQLite storage
- daemon shutdown/reacquire
- `list_memory_records`, `search_memory_records`, `read_memory_record`, `get_memory_stats`

**Key invariants:**
- imported and created records survive runtime shutdown
- stats and search results remain consistent after reacquire
- new runtime acquisition after shutdown gets a fresh runtime object backed by the same data

## 4. Coverage Mapping

- `tests/large/test_e2e_user_flows.py::test_e2e_record_thought_ingests_and_becomes_searchable`
- `tests/large/test_e2e_user_flows.py::test_e2e_shared_runtime_persists_records_across_shutdown`

These tests complement the smaller runtime, contract, and daemon lifecycle suites by proving the full user journeys work together.