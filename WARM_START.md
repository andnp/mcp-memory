# Warm Start Guide

## Current Architecture

- relational-first runtime backed by SQLite
- one shared global memory store under XDG data home
- workspace identity derived from git root when available
- global daemon owns runtime state and background workers
- `mcp_memory.mcp.runtime` is the runtime composition root for storage, providers, and typed capabilities
- `mcp_memory.daemon_runtime` is the daemon composition/lifecycle root; `daemon_app` only adapts FastAPI routes and request scope
- `mcp_memory.daemon_background` owns embedding warmup, dashboard build, backup, and writeback coordination
- workspace context is metadata for ranking and analytics, not an execution boundary
- `mcp-memory run` is a thin MCP stdio proxy that auto-starts the daemon
- `mcp-memory admin dashboard open` ensures the daemon is running and opens the dashboard URL

## First Steps for a New Session

1. Confirm the current runtime shape
   - read `README.md`
   - read `docs/plans/07-cleanup-checklist.md`
   - scan `docs/specs/` for target-state gaps

2. Verify the codebase state
   - run the test suite
   - inspect `src/mcp_memory/daemon.py`
   - inspect `src/mcp_memory/mcp/runtime.py`
   - inspect `src/mcp_memory/core/agent_runtime.py`

3. Keep work aligned with the shipped architecture
   - prefer relational repository flows
   - prefer explicit typed links in the `links` table
   - avoid reintroducing indexing, chunking, wikilinks, or project-registry logic
   - keep provider-specific invocation inside provider wrappers

## Near-Term Priorities

- finish deleting residual legacy schema/docs/test baggage
- align the specs with the daemon-first relational runtime
- decide which spec features are true v1 scope
- implement the next chosen feature slice without reintroducing compatibility shims
