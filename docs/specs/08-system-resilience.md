# Architecture Decision Record: System Resilience & Runtime Safety

**Status:** Current implementation baseline

## 1. Boot Coordination

### Decision
Use strict filesystem locking around daemon startup.

### Current Behavior
- the thin proxy resolves workspace identity
- the proxy attempts to acquire a workspace-specific lock
- if a healthy daemon already exists, the proxy reuses it
- otherwise the proxy spawns the daemon and waits for readiness metadata

## 2. Proxy-Daemon Communication

### Current Decision
Use localhost HTTP between the MCP thin proxy and the daemon.

### Current Rationale
- simple framing
- easy local debugging
- easy dashboard/API reuse
- enough for current single-user localhost scope

### Explicit Non-Goal
ZeroMQ is **not** part of the current implementation.
It remains a future option, not a present dependency or runtime requirement.

## 3. Durable Task Safety

### Decision
Use a SQLite-backed `tasks` table for background work.

### Current Behavior
- tasks are persisted before background processing
- failed tasks are retried up to their configured retry limit
- permanently failed tasks remain visible for inspection
- worker state survives daemon restarts through the database

## 4. Deadlettering

### Current Behavior
- each task tracks `retries_count`
- tasks stop retrying after `max_retries`
- failed tasks remain queryable through the management surface

## 5. Configuration Reliability

### Decision
Use explicit config files, not shell-environment assumptions, for critical daemon behavior.

### Current Behavior
- config is loaded from `~/.config/mcp-memory/config.toml`
- a safe default config is created on first start if missing
- provider behavior is wrapper-owned, not driven by generic CLI-flag config

## 6. Telemetry Growth Control

### Current Behavior
The Sweeper task deletes old completed-task and old processed-journal telemetry so the daemon does not accumulate unbounded operational noise.

## 7. Open Decisions

The following are still product decisions, not current guarantees:

- whether to keep localhost HTTP as the long-term proxy/daemon transport
- whether to add reference counting and daemon autoshutdown
- whether to add richer recovery/repair flows beyond current startup locking and durable tasks
