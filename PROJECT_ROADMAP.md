# mcp-memory Project Roadmap

This roadmap outlines the path to transforming the extracted memory subsystem into a lean, standalone, high-performance MCP server.

---

## Epic 1: Project Foundation & Refactoring
**Goal**: Establish a clean, self-contained codebase free of legacy dependencies.

> Status note: the old file-backed runtime manager/search/tool path has been removed. Current roadmap work should assume a relational-first runtime rather than preserving compatibility with the extracted markdown-era architecture.

### Task 1.1: Namespace & Import Cleanup
- **What**: Rename internal modules and fix all absolute/relative imports to use the `mcp_memory` namespace.
- **Why**: To ensure the project is a fully portable package that doesn't conflict with or rely on the `mcp-markdown-ragdocs` structure.

### Task 1.2: Model & Configuration Pruning
- **What**: Remove all document-search-specific Pydantic models (e.g., `DocumentChunk`, `ChunkMetadata`) and configuration sections (e.g., `[search.rerank]`, `[indexing.watcher]`).
- **Why**: To reduce cognitive load and prevent "feature creep" from the original project. A focused configuration is easier for users to manage.

---

## Epic 2: Core Architecture Simplification
**Goal**: Remove the overhead of the multiprocess "worker" architecture.

### Task 2.1: Implement Lean `MemoryContext`
- **What**: Replace the complex `ApplicationContext` with a lightweight, single-process state manager.
- **Why**: Memory operations (writing small MD files) are fast and low-latency. The overhead of a multiprocess worker, IPC, and task queues is unnecessary and adds complexity.

### Task 2.2: Decommission IPC & Worker Logic
- **What**: Identify and remove all code related to `LifecycleCoordinator`, `Queue`, `Snapshot`, and `WorkerProcess`.
- **Why**: Simplifies the codebase, reduces memory footprint, and removes several layers of potential failure in single-process environments.

---

## Epic 3: Index System Modernization
**Goal**: Transition from snapshot-based synchronization to direct persistence.

### Task 3.1: Streamline Indices for Direct Persistence
- **What**: Remove snapshotting, versioning, and "read-only" sync logic from `VectorIndex`, `KeywordIndex`, and `GraphStore`.
- **Why**: In a single-process model, indices can write directly to their persistence layer without needing to coordinate versioning with a secondary process.

### Task 3.2: Implement Embedding Provider Pattern
- **What**: Create an abstract base class for embedding generation with support for multiple backends.
- **Why**: To allow a "Low Resource" mode using remote APIs (OpenAI/Anthropic) vs. a "Privacy" mode using local `sentence-transformers`. This makes the server accessible on machines without high RAM/GPU resources.

---

## Epic 4: Graph & Relationship Refinement
**Goal**: Enhance cross-corpus linking and temporal logic.

### Task 4.1: Abstract External Reference "Ghost Nodes"
- **What**: Update the `GraphStore` to treat any wikilink targeting a non-memory file (e.g., `[[src/main.py]]`) as a generic external reference.
- **Why**: To maintain the "linked memory" functionality without requiring the server to index or even have access to the referenced source code files.

### Task 4.2: Temporal Relationship Queries
- **What**: Refine `get_memory_relationships` to better traverse version chains (`SUPERSEDES`) and handle contradictions.
- **Why**: To allow AI assistants to understand the lifecycle of their own knowledge—identifying when a plan is outdated or when a new observation conflicts with a stored fact.

---

## Epic 5: MCP Interface Overhaul
**Goal**: Leverage modern SDK patterns for better tool discovery and validation.

### Task 5.1: Pydantic-Driven Tool Schemas
- **What**: Define all MCP tool inputs and outputs using Pydantic models.
- **Why**: To ensure strict type safety and leverage automatic schema generation for the MCP `list_tools` capability.

### Task 5.2: SDK Decorator Migration
- **What**: Migrate tool registration from manual handler mapping to a decorator-based approach (`@server.call_tool`).
- **Why**: To make the entry point (`server.py`) more readable and idiomatic, following standard MCP SDK patterns.

---

## Epic 6: Testing & Validation
**Goal**: Ensure a high-quality, stable standalone release.

### Task 6.1: Extract and Adapt Unit Tests
- **What**: Move memory-specific tests from the original repo and adapt them to the new `MemoryContext`.
- **Why**: To maintain the high level of test coverage from the parent project.

### Task 6.2: Lightweight Integration Suite
- **What**: Create an end-to-end test suite that verifies tool execution (Create -> Search -> Update -> Delete) via a mock MCP client.
- **Why**: To ensure the core value proposition (AI persistent memory) is functionally perfect before the first release.

---

## Epic 7: Daemonized Architecture & IPC
**Goal**: Transform the server into a shared background daemon with multiple "thin client" connections.

### Task 7.1: Daemon Lifecycle & Lock Management
- **What**: Implement a daemon controller that uses PID files and platform-specific locking to ensure only one instance of the "Brain" runs at a time.
- **Why**: To prevent index corruption and race conditions that occur when multiple processes try to write to the same memory files or SQLite database simultaneously.

### Task 7.2: Shared IPC Layer (Domain Sockets/Named Pipes)
- **What**: Create an internal communication layer for the daemon to receive requests from multiple "Thin MCP Proxies."
- **Why**: To allow multiple IDEs or CLI tools to share the same memory state in real-time. Unix Domain Sockets provide a secure, low-latency path for this local-only traffic.

### Task 7.3: Reference Counting & Autoshutdown
- **What**: Implement a "Ref-Count" system in the daemon that tracks active client connections. If the count hits zero, the daemon performs a graceful shutdown after a configurable grace period.
- **Why**: To ensure the system is "ephemeral" and only consumes system resources when it is actually being used by an agent or tool.

---

## Epic 8: Agentic Background Worker
**Goal**: Move heavy "thinking" tasks out of the request-response loop.

### Task 8.1: Background Task Scheduler
- **What**: Implement a low-priority background scheduler (e.g., using `Huey` or a simple `asyncio` loop) for "agentic" maintenance.
- **Why**: Tasks like "memory consolidation" or "global cross-linking" are computationally expensive. Running them in the background ensures the MCP tool responses remain near-instant.

### Task 8.2: Automatic Memory "Defrag" & Consolidation
- **What**: Create a background agent that looks for highly similar memories or fragmented journals and suggests (or performs) merges.
- **Why**: As an agent stores more memories, the "noise" increases. Automatic consolidation keeps the most relevant information condensed and retrievable.

---

## Epic 9: Management Dashboard (Web UI)
**Goal**: Provide a human-readable interface for the memory bank.

### Task 9.1: Localhost Management API
- **What**: Add a REST/FastAPI layer to the daemon to expose memory stats, raw file access, and index status.
- **Why**: To provide a backend for the UI and allow for external management scripts.

### Task 9.2: Memory Bank Dashboard (Frontend)
- **What**: Build a small, visually appealing web dashboard (hosted on localhost) to visualize the memory graph, edit memories, and view agentic "defrag" logs.
- **Why**: To give the user visibility into what their AI assistant is "remembering" and provide a way to manually prune or correct facts.
