# Architecture Decision Record: Ephemeral Background Tasks

## 1. Context & Problem
The `mcp-memory` daemon is designed to be **ephemeral**; it spins up when a client (like an IDE or CLI) connects, and it gracefully shuts down when the last client disconnects to save system resources. 

However, the daemon is also responsible for running asynchronous, long-running agentic tasks (like the `MemoryIngestor` and the `Defragmenter`). If these tasks are managed via in-memory queues (e.g., `asyncio.Queue`), shutting down the daemon will result in lost tasks and unprocessed System 1 thoughts.

## 2. Decision: SQLite-Backed Task Queue
We will use a durable, database-backed queue mechanism. All background tasks must be persisted to disk before execution begins.

### 2.1 Implementation Options
1. **Huey (with SQLite Storage)**: `huey` is a lightweight Python task queue that supports SQLite as a backend. It handles retries, delays, and cron-like scheduling natively.
2. **Custom `tasks` Table**: A simple custom implementation using a `tasks` table (`id`, `task_name`, `payload`, `status`, `created_at`) processed by a background `asyncio` loop polling the DB. 
*(Recommendation: Start with a custom `tasks` table to minimize dependencies, or use Huey if complex retry logic is needed immediately).*

## 3. The Task Lifecycle in an Ephemeral Daemon

Because the daemon can shut down at any time, the task execution loop must be highly deliberate:

1. **Task Enqueueing**:
   - When the user calls `record_thought`, the thought is saved to `system1_journal`. If the threshold is reached, an `IngestSystem1` task is written to the `tasks` table.
   - When the server boots, it schedules a `RunStrategyRoulette` task with a delay (if one doesn't already exist).

2. **Daemon Boot**:
   - Upon starting (triggered by a client connection), the daemon spawns a dedicated "Worker Task" in the event loop.
   - This worker immediately queries the `tasks` table for anything marked `pending` or `failed` and begins processing them.

3. **Graceful Shutdown**:
   - When the reference counter hits zero (last client disconnects), the daemon enters a "Shutting Down" state.
   - It stops accepting *new* tasks from clients.
   - It allows the *currently executing* background task a short grace period (e.g., 5-10 seconds) to finish.
   - If the task finishes, its status is updated to `completed`.
   - If the task is still running after the grace period, the daemon forcefully exits. Because the task was never marked `completed` in the database, it will automatically be picked up and retried the next time the daemon boots.

## 4. Why This Works
This architecture perfectly bridges the gap between an "Ephemeral Server" and "Persistent Agentic Work." 
- You can rapidly stash thoughts via the CLI and instantly close your laptop. The thought is safe in the DB.
- The next time you open VS Code (days later), the daemon boots, sees the pending ingestion task, and the background agent silently processes your thoughts while you begin coding.
