# Architecture Decision Record: System Resilience & IPC

## 1. Context & Problem
While the logical data model and agent workflows are sound, the physical execution of a daemonized, multi-client system introduces complex failure modes. This document codifies the "defensive architecture" required to prevent race conditions, data corruption, and infinite crash loops.

## 2. The Boot Race Condition (Locking)
**Problem:** Multiple IDEs launching simultaneously could cause multiple Thin Proxies to attempt to spawn the central daemon at the same time, leading to corrupted indices.
**Decision:** We will use strict filesystem-level exclusive locking (`flock` on Unix, `msvcrt` on Windows) on a `.lock` file.
**Flow:**
1. Proxy attempts to acquire the lock.
2. If successful, it checks if the daemon is alive. If not, it spawns the daemon, waits for it to bind its socket, and then releases the lock.
3. If the lock is already held, the proxy blocks (waits) until the lock is released, then connects to the newly spawned daemon.

## 3. Proxy-Daemon Communication (ZeroMQ IPC)
**Problem:** Multiple proxies streaming JSON-RPC over `stdio` to a single daemon requires robust message framing and multiplexing to prevent garbled data.
**Decision:** We will use **ZeroMQ (ZMQ)** for the IPC layer between the Thin Proxies and the Daemon.
**Why ZMQ:**
- **ZeroMQ `ROUTER` / `DEALER` Pattern:** The daemon runs a `ROUTER` socket. Each proxy runs a `DEALER` socket. ZMQ automatically handles the framing, queuing, and multiplexing of messages. 
- The daemon knows exactly which proxy sent a request and can route the response back seamlessly.
- ZMQ handles the complexities of Unix Domain Sockets or TCP loopback automatically.

## 4. SQLite vs. FAISS Desync (Split-Brain)
**Problem:** If the daemon crashes after writing metadata to SQLite but before writing vectors to the FAISS `.bin` file, the indices are out of sync.
**Decision:** Boot-time Reconciliation.
**Flow:**
1. Upon booting, the daemon counts the number of `active` memories in SQLite.
2. It compares this to the `ntotal` count in the FAISS index.
3. If there is a mismatch, the daemon triggers a `RebuildVectorIndex` background task. It reads all memories, re-embeds them, and overwrites the FAISS index to ensure total consistency.

## 5. Explicit Configuration (No Env Vars)
**Problem:** GUI applications (like VS Code launched from a dock) do not inherit shell environment variables (like `OPENAI_API_KEY`), causing background tasks to fail silently.
**Decision:** The daemon will *not* rely on environment variables for critical agentic authentication.
**Flow:**
- All configuration (storage paths, AI provider settings, API keys) must be explicitly defined in a globally accessible file: `~/.config/mcp-memory/config.toml`.
- The daemon reads this file at boot. If critical keys are missing, the background agents gracefully pause and flag an error in the Web UI.

## 6. The "Poison Pill" (Deadlettering Tasks)
**Problem:** A malformed thought causes the `Ingestor` LLM to return bad JSON, crashing the worker. Because the task queue is durable, the daemon will retry the task on every boot, creating an infinite crash loop.
**Decision:** Implement Deadlettering in the `tasks` schema.
**Schema Update:** Add `retries_count (INTEGER DEFAULT 0)` to the `tasks` table.
**Flow:**
- If a task raises an unhandled exception, `retries_count` is incremented.
- If `retries_count >= 3`, the task status is changed from `pending` to `failed`.
- The worker ignores `failed` tasks. These are surfaced in the Web UI for the user to manually inspect, fix, or delete.

## 7. Infinite Table Growth (Telemetry Bloat)
**Problem:** The `tasks` table and `system1_journal` are constantly written to. If completed tasks and merged thoughts are never deleted, the SQLite database will eventually bloat to gigabytes, slowing down queries.
**Decision:** The "Sweeper" Background Agent.
**Flow:**
- A low-priority background task runs daily (or on daemon boot if it hasn't run in 24 hours).
- It executes: `DELETE FROM tasks WHERE status = 'completed' AND updated_at < date('now', '-7 days')`.
- It executes: `DELETE FROM system1_journal WHERE status = 'merged' AND timestamp < date('now', '-7 days')`.
- This ensures the daemon retains enough recent history for UI observability and debugging without suffering from infinite data growth.
