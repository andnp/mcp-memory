# Architecture Principles: Relational Memory Server

## 1. Relational-First Persistence
The `mcp-memory` server treats AI memory as **structured data** in a single, global SQLite database. This allows for atomic transactions and robust cross-memory linking, eliminating the fragility of filesystem-based Markdown.

## 2. "Soft" Project Context (Workspace IDs)
Projects are no longer hard silos. Instead of tying context to brittle, absolute file paths:
- **Workspace IDs**: Every client connection establishes a Workspace ID (e.g., hashing the local Git origin URL or the repo's first commit SHA).
- **Origin Tracking**: Every thought and memory stores the `workspace_id` of the client that created it.
- **Contextual Search**: When searching, the Orchestrator applies a significant **contextual boost** to memories associated with the active Workspace ID. This allows seamless cross-pollination while maintaining project relevance.

## 3. Human & AI Asymmetry (The Human CLI)
While the AI accesses memory via the background daemon and MCP proxy, humans need a zero-friction way to inject knowledge without breaking flow.
- **The Human CLI**: A tool like `mcp-memory stash "The new auth endpoint needs a bearer token"` instantly appends to the System 1 journal.
- This ensures the daemon can ingest high-signal human insights asynchronously without requiring the user to open a Web UI or explain things to an AI chat window.

## 4. Progressive Discovery (Search vs. Read)
To minimize context window bloat, the server enforces a two-stage discovery process:
- **Search Tool**: Returns IDs, titles, types, and a **two-sentence summary**.
- **Read Tool**: Returns the full content.
Agents are instructed to search first and only "read" memories that are highly relevant to the task.

## 5. Shared Daemon & Reference Counting
The system operates as a persistent background daemon ("The Brain"). It remains alive only as long as an active client is connected, performing a graceful shutdown when the last client disconnects.

## 6. Agentic Maintenance
The daemon runs low-priority, resource-gated background agents:
- **Summarizer**: Maintains the two-sentence summary for every memory.
- **Ingestor**: Promotes System 1 thoughts to System 2.
- **Consolidator**: Merges redundant knowledge and detects contradictions.
