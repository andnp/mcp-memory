# Architecture Decision Record: System Resilience & Runtime Safety

**Status:** Active

## 1. Boot Coordination

### Decision
Use strict filesystem locking around daemon startup.

### Current Behavior
- the thin proxy resolves runtime context, including optional workspace metadata for search ranking
- the proxy attempts to acquire a global daemon lock
- if a healthy global daemon already exists, the proxy reuses it
- otherwise the proxy spawns the daemon and waits for readiness metadata
- if daemon metadata exists but the owner is unhealthy or stale, recovery now terminates the real daemon process group before removing metadata and continuing boot

## 2. Proxy-Daemon Communication

### Current Decision
Use ZeroMQ over a stable local IPC socket between the MCP thin proxy and the daemon.

**Status:** Active. The daemon/proxy runtime boundary is now ZMQ-first. Any remaining HTTP-facing management routes are compatibility or test scaffolding, not the authoritative transport.

### Current Rationale
- stable global endpoint simplifies duplicate-owner detection and operator inspection
- ROUTER/DEALER fits many thin proxies talking to one daemon runtime
- IPC keeps the transport local, explicit, and independent of dynamic port selection
- transport metadata can expose stronger provenance than ad hoc localhost probing

### Current Timeout Behavior
- the MCP thin proxy applies a 60s client timeout budget for daemon-backed tool calls
- the daemon transport enforces bounded execution deadlines instead of allowing requests to hang forever
- slower request classes such as search, read, `record_thought`, and internal tool paths use an extended 60s transport budget; ordinary paths keep a shorter default deadline
- when a transport deadline expires, the daemon returns a structured timeout payload (`status = "error"`, `error = "daemon_request_timed_out"`, plus path/timeout metadata)
- repeated client-observed timeout failures trigger background daemon metadata refresh and can escalate to a forced daemon restart

### Remaining Limits
- stale-socket and mixed-binary recovery still need stronger migration coverage
- the daemon hello contract still needs richer runtime-state reporting beyond basic ready health
- some operator-facing naming still reflects older dashboard/HTTP language and should keep being cleaned up
- shutdown diagnostics could still be more explicit about whether escalation to `SIGKILL` was required and which child subprocess blocked graceful exit

### Authoritative Direction
Move to ZeroMQ over a stable Unix domain socket for the daemon/proxy boundary.

#### Intended Contract
- daemon owns one global `ROUTER` socket
- thin proxies connect with `DEALER` sockets
- endpoint is stable and global, for example `ipc://.../mcp-memory/daemon.sock`
- workspace context flows inside requests as metadata for search ranking and analytics only
- workspace context must never influence daemon identity or transport endpoint selection

#### Required Handshake Data
The daemon hello/status contract should expose enough provenance to detect stale or competing owners:
- `pid`
- `binary_path`
- `version`
- `started_at`
- `runtime_state` such as `booting`, `ready`, `draining`, `shutting_down`
- explicit global scope marker

#### Duplicate-Owner Recovery
The transport layer should make duplicate-daemon detection explicit:
- acquire the global boot lock first
- probe the stable socket endpoint
- if a healthy daemon responds, reuse it
- if the socket exists without a healthy owner, remove the stale socket and continue boot
- if metadata exists but the live owner disagrees on provenance, surface a loud recovery signal instead of silently continuing
- if metadata exists for an unhealthy owner whose PID is still running, terminate the real daemon process group before clearing metadata so the system does not enter a split-brain state where the old daemon is still alive but no longer registered

### Explicit Non-Goal
Reintroducing dynamic localhost HTTP as the authoritative proxy-daemon boundary is **not** a goal.
ZeroMQ over the stable IPC socket is the current runtime authority; any remaining HTTP-facing management pieces are compatibility or operator-facing scaffolding.

## 3. Durable Task Safety

### Decision
Use the authoritative backend's durable `tasks` store for background work.

### Current Behavior
- tasks are persisted before background processing
- failed tasks are retried up to their configured retry limit
- permanently failed tasks remain visible for inspection
- worker state survives daemon restarts through the database
- worker recovery now complements the hardened daemon stop path: stale daemon shutdown no longer relies on metadata removal alone and instead proves the underlying process is gone

SQLite remains the default local task backend.
In shared mode, Postgres is authoritative for the task system.

The `record_thought` writeback flusher is a separate daemon-owned support loop, not a durable maintenance task family.

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

- the final socket location and retention policy for stale-socket cleanup
- whether to add reference counting and daemon autoshutdown
- whether to add richer recovery/repair flows beyond current startup locking, stale-socket cleanup, and durable tasks

## 8. Current Stop/Restart Safety Notes

- daemon stop/restart now targets the daemon process group, not only the top-level daemon PID
- graceful shutdown still starts with `SIGTERM`
- if the daemon does not exit within the configured grace window, shutdown escalates to `SIGKILL`
- stale socket cleanup runs after stop/recovery so the next boot does not inherit a dead transport endpoint
- this shipped behavior exists specifically to avoid the previously observed failure mode where CLI/status could show “no daemon registered” while the old daemon process was still alive and interfering with memory reads/writes
