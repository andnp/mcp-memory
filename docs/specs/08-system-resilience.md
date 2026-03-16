# Architecture Decision Record: System Resilience & Runtime Safety

**Status:** Current implementation baseline

## 1. Boot Coordination

### Decision
Use strict filesystem locking around daemon startup.

### Current Behavior
- the thin proxy resolves runtime context, including optional workspace metadata for search ranking
- the proxy attempts to acquire a global daemon lock
- if a healthy global daemon already exists, the proxy reuses it
- otherwise the proxy spawns the daemon and waits for readiness metadata

## 2. Proxy-Daemon Communication

### Current Decision
Use localhost HTTP between the MCP thin proxy and the daemon.

**Status:** Transitional. HTTP remains the current transport, but it is no longer the intended long-term boundary.

### Current Rationale
- simple framing
- easy local debugging
- easy dashboard/API reuse
- enough for current single-user localhost scope with one global runtime authority

### Known Limits
- dynamic port discovery is weaker than a stable IPC endpoint for duplicate-owner detection
- HTTP health checks do not provide a strong daemon hello/provenance contract
- the current transport leaves stale-port and mixed-binary recovery less explicit than desired

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

### Explicit Non-Goal
ZeroMQ is **not** part of the current implementation yet.
It is the planned transport direction, but not a present runtime dependency.

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

- the exact migration sequence from HTTP to ZMQ
- the final socket location and retention policy for stale-socket cleanup
- whether to add reference counting and daemon autoshutdown
- whether to add richer recovery/repair flows beyond current startup locking and durable tasks
