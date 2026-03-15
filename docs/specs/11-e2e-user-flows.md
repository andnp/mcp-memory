# Specification: End-to-End User Flows

This document defines the highest-value user journeys that should stay green as `mcp-memory` evolves.

## 1. Thought Capture to Searchable Memory

**Goal:** a user records quick thoughts, background ingestion promotes them into a structured relational memory, and the result becomes searchable/readable.

**Runtime path:**
- `record_thought`
- `system1_journal`
- task queue threshold rule
- background ingest worker
- relational repository
- `search_memory_records`
- `read_memory_record`

**Key invariants:**
- the ingest task is created once at the threshold and reused afterward
- journal entries are marked processed
- at least one relational memory is created
- the created memory is searchable and readable through MCP tools

## 2. Explicit Link Authoring

**Goal:** a client can create or remove a typed relationship without mutating memory body content.

**Runtime path:**
- `create_memory_link`
- `delete_memory_link`
- relational `links` table
- `read_memory_record`

**Key invariants:**
- creating a link makes it visible in the read payload immediately
- deleting a link removes it from the read payload immediately
- invalid source or target references fail safely
- explicit `ext:` targets remain supported

## 3. Shared Runtime Across Clients

**Goal:** multiple clients attached to the same workspace share one runtime and see the same state immediately.

**Runtime path:**
- daemon startup/bootstrap
- MCP tool handlers on the shared runtime
- relational repository/search service

**Key invariants:**
- writes from one client are visible to the other without restart
- the shared store stays coherent across clients for the same workspace

## 4. Import and Persistence Across Restart

**Goal:** records seeded into the relational store survive runtime shutdown; a later session reopens the workspace and can still search and read them through the MCP surface.

**Runtime path:**
- relational SQLite storage
- daemon shutdown/reacquire
- `search_memory_records`
- `read_memory_record`

**Key invariants:**
- seeded records survive runtime shutdown
- search and read results remain consistent after reacquire
- new runtime acquisition after shutdown gets a fresh runtime object backed by the same data
