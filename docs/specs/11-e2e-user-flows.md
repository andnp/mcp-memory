# Specification: End-to-End User Flows

**Status:** Active

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

## 3. Shared Global Runtime Across Clients

**Goal:** multiple clients share one global runtime and see the same state immediately, while workspace context only affects retrieval relevance and analytics.

**Runtime path:**
- daemon startup/bootstrap
- MCP tool handlers on the shared runtime
- relational repository/search service

**Key invariants:**
- writes from one client are visible to the other without restart
- the shared store stays coherent across clients regardless of workspace context
- workspace context does not create a separate daemon, queue, or storage partition

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
