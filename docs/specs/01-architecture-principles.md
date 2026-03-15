# Architecture Principles: Relational Memory Server

## 1. Relational-First Persistence
The runtime source of truth is SQLite.

- memories, links, tags, workspaces, journal entries, and tasks all persist relationally
- new runtime features should extend the relational model rather than reintroducing file-backed state management

## 2. Workspace Context Without Hard Silos
Workspace context should guide relevance without fragmenting the store.

- each runtime resolves an active `workspace_id`
- thoughts, tasks, and memories carry workspace association
- search applies workspace-aware boosting rather than hard partitioning

## 3. Progressive Discovery
The public API should prefer search-first, read-second workflows.

- search returns compact summaries and metadata
- read returns full content plus structural context
- future improvements should refine ranking quality, not inflate payload size by default

## 4. Thin Proxy, Shared Daemon
The MCP-facing process should stay thin.

- `mcp-memory run` is a proxy
- the workspace daemon owns runtime state
- daemon responsibilities include background work, management API, and persistence coordination

## 5. Explicit Provider Boundaries
Provider-specific behavior belongs in provider-specific wrappers.

- avoid generic CLI-flag abstractions
- keep invocation details inside each wrapper
- keep user-facing configuration minimal and explicit

## 6. Conservative Background Automation
Autonomous mutation must degrade safely.

- deterministic fallback paths should exist where practical
- failed tasks must remain inspectable
- new agents should be added only when their behavior is explicit and testable
