# MCP Memory Server

A standalone Model Context Protocol (MCP) server for persistent AI memory management.

## 🚀 Overview

The MCP Memory Server provides a persistent memory bank for AI assistants, enabling them to store, retrieve, and organize knowledge across sessions. The current runtime is relational, ZMQ-backed, and supports two storage modes:

- **SQLite** as the default local, zero-config backend
- **Postgres** as the explicit shared-mode backend for multi-machine or cloud-hosted use
- **Local SQLite cache sidecars** as optional, non-authoritative support state in shared Postgres mode

## 🚧 Current status

The runtime is now **relational-first**.

- New memory CRUD, search, read, and markdown import flows run through the relational runtime.
- Background task handling is now part of the active runtime surface.
- The MCP stdio entrypoint is now a thin proxy that auto-starts a proper workspace daemon on demand.
- Shared Postgres mode now has an active local SQLite sidecar cache for readthrough and degraded-read behavior.
- `storage.cache.mode = "writeback"` is now shipped in a narrow form for `record_thought`: durable local outbox queueing on connectivity/timeout-ish authoritative failures, opportunistic foreground flush after later successful authoritative writes, and a daemon-owned periodic background flusher.
- Broader offline mutation and generalized writeback remain deferred.
- Daemon-backed MCP requests now use a 60s client timeout budget, and the transport returns structured timeout errors instead of hanging indefinitely.

### Provider token economics

Provider billing is token-based. Authoritative input, output, cache, reasoning,
and total token telemetry is the cost measure; provider-call counts remain a
throughput measure.

The maintenance optimization target is therefore **useful completed work per
provider call and per token**, not merely lower latency.

That product goal shapes the maintenance architecture:

- keep one provider execution alive across additional compatible work when safe
- let the agent discover, reprioritize, and continue through internal tools during the same run
- use deterministic prep only when it increases the amount of useful work a provider call can finish, not just to make the call faster
- prefer richer work packets and compatibility-group continuation when it reduces repeated context and token use

### Key Features

- **Graph-Aware Search Ranking**: Search now combines weighted BM25 keyword retrieval, optional semantic candidates, soft workspace-aware semantic ordering, RRF fusion, sigmoid calibration, graph-aware reranking, one-hop graph expansion, and degradation penalties.
- **Memory-Specific Recency Boost**: Automatically prioritizes recent memories (journals, plans) while preserving long-term facts.
- **Typed Relational Links**: Memory records can carry explicit typed relationships such as `SUPERSEDES`, `EXTENDS`, and `CONTRADICTS`.
- **Categorized Memories**: Built-in support for `journal`, `plan`, `fact`, `observation`, and `reflection` types.
- **Relational Memory Foundation**: UUID-backed relational memory records with workspace IDs, tags, and typed links.
- **Stable Agent References**: Agent-facing search and read payloads use persisted `mem-<number>` references, while canonical UUIDs remain available for debugging and internal lineage.
- **Local Semantic Search**: Fully local embeddings via `sentence-transformers` enrich search and ingest without any external AI provider.
- **Shared Global Storage**: All memories live in one shared XDG data directory, with workspace identity attached to thoughts, tasks, and memories.
- **Shared-Mode Local Cache**: In Postgres shared mode, an optional local SQLite sidecar can serve fresh exact search hits, validated read hits, and degraded cached search/read fallbacks without becoming a second source of truth.
- **Global Daemon**: One global daemon per user environment owns runtime state, background workers, and transport coordination.
- **Explicit Composition Roots**: `mcp/runtime.py` assembles storage, providers, and typed capability bundles; `daemon_runtime.py` owns daemon startup, transport, workers, and graceful shutdown while `daemon_app.py` remains the FastAPI adapter.
- **Stable IPC Transport**: The thin MCP proxy talks to the daemon over a stable ZeroMQ ROUTER/DEALER transport on a Unix socket.
- **Agentic Maintenance Agents**: The canonical `memory-curator` campaign and ingest can run through a trusted internal MCP maintenance surface when an agentic provider is configured. Historical cleanup task names are compatibility aliases, not independent campaigns.
- **Crash-Safe Ingest Finalization**: Agentic ingest preserves durable journal claim/delete/release semantics while still allowing direct MCP create/append mutations.
- **Hardened Daemon Recovery**: Daemon stop/restart now terminates stale process groups, escalates from `SIGTERM` to `SIGKILL` when needed, and cleans stale sockets instead of merely dropping metadata.
- **Operational Views**: Health, overview, and lineage inspection remain available through the daemon management surface and CLI.
- **Task-Level Provider Telemetry**: AI provider usage is attributed to the background task that made the call so you can see which agents are actually consuming model time.

## 🛠 Tech Stack

- **Python 3.13+**
- **MCP (Model Context Protocol)**: Standard interface for AI tool integration.
- **SQLite (FTS5)**: Robust keyword search and relational storage.
- **Postgres**: Shared-mode relational storage for multi-machine deployments.
- **Sentence Transformers (optional)**: Local embeddings for semantic retrieval and thought clustering.
- **ZeroMQ (`pyzmq`)**: Local daemon/proxy IPC transport over Unix domain sockets.
- **FastAPI**: In-process compatibility and test harness for management routes.

## 📂 Project Structure

```text
mcp-memory/
├── src/
│   └── mcp_memory/
│       ├── core/           # Journal, task runtime, provider wrappers, and shared utilities
│       ├── management/     # Dashboard service models and static assets
│       ├── relational/     # Relational repository, importer, and search services
│       ├── mcp/            # MCP tool handlers and runtime wiring
│       ├── utils/          # Shared utilities (Config, DB, IO)
│       ├── cli.py          # Command-line interface
│       ├── daemon.py       # Daemon autostart, stop, and recovery logic
│       ├── daemon_runtime.py # Global daemon composition and lifecycle
│       ├── daemon_background.py # Backup, writeback, warmup, and build loops
│       ├── daemon_app.py   # FastAPI management adapter
│       └── server.py       # MCP stdio thin proxy
├── tests/                  # Comprehensive test suite
├── pyproject.toml          # Dependency management (uv/hatch)
└── README.md               # You are here
```

## 📚 Documentation map

If you want the repo's canonical docs without spelunking every markdown file, start here:

- `docs/README.md` — documentation index and entry points

Most readers will usually want one of these first:

- `docs/specs/00-product-principles.md` — product invariants
- `docs/specs/01-architecture-principles.md` — architecture invariants
- `docs/postgres-shared-mode-runbook.md` — operator path for Postgres shared mode
- `docs/specs/16-storage-backend-selection-and-shared-mode.md` — storage authority and backend rules
- `docs/specs/17-shared-mode-readthrough-cache.md` — shared-mode cache and narrow writeback behavior
- `docs/specs/18-postgres-server-side-vector-search.md` — Postgres semantic-search architecture

## 📋 Current MCP Tools

The following tools are exposed via the MCP server:

### Minimal Public Surface
- `record_thought`: Record a raw system-1 thought in the system-1 journal. In shared Postgres `writeback` mode, this can degrade into a durable local outbox queue when the authoritative write times out or connectivity fails.
- `search_memory_records`: Search relational memory records with compact summary-first results and staged ranking over weighted keyword + optional semantic retrieval. Debug mode exposes score/workspace/ranking details.
- `read_memory_record`: Read one memory record by stable reference or legacy UUID with only the reference, title, and content by default. Relationships, superseded breadcrumbs, and metadata are explicit opt-ins.
- `read_memory_records`: Read up to 20 records by stable references in one call; unresolved references are included only when present.

Agent-facing search results use `memory_ref` values such as `mem-123`. These references are persisted with each memory and are stable across restarts and migrations. Read tools accept both stable references and legacy UUIDs; UUIDs remain canonical in storage, telemetry, lineage, and debug/admin payloads.

Everything else is intentionally kept out of the public MCP surface. Admin, migration, browsing, and operational views belong in the daemon management surface or CLI, not in the assistant-facing protocol.

For trusted maintenance agents, the repo also now includes a workspace-local internal MCP surface exposed through `uv run mcp-memory internal-run`. The bundled `.gemini/settings.json` enables that internal tool surface only inside this workspace.

### Admin / Maintenance Commands
- `uv run mcp-memory memory stash "Remember to normalize workspace metadata."`: record one raw thought into the System 1 journal.
- `uv run mcp-memory admin install`: install local tool integrations.
- `uv run mcp-memory admin dashboard open`: ensure the daemon is running and open the operator dashboard.
- `uv run mcp-memory admin agent run memory-curator`: trigger one background agent for the active workspace.
- `uv run mcp-memory admin agent run --all`: enqueue the active background campaigns for the active workspace.
- `uv run mcp-memory admin health --json`: print an AI-friendly health snapshot, including active backend and cache state.
- `uv run mcp-memory admin overview`: print background task and memory statistics from the active backend.
- `uv run mcp-memory admin search health`: show semantic search health for the current runtime context.
- `uv run mcp-memory memory import-markdown /path/to/memory.md`: import one markdown memory file into the relational store.
- `uv run mcp-memory memory import-markdown /path/to/one.md '/path/to/*.md'`: import explicit files and globbed markdown files in one command.

Current maintenance/runtime highlights:

- Existing-memory cleanup and curation run through `memory-curator`; its policy covers normalization, linking, deduplication, decomposition, retention, and specialist review routing.
- Historical cleanup names such as `deduplicator`, `graph-linker`, and `project-manager` redirect to `memory-curator` for manual triggers. `ingest-system1` remains a separate write-stream campaign.
- Agentic ingest prefers ingest-specific internal tools, preserving lineage and summarize-task enqueueing in the tool layer rather than in prompt-only behavior.
- Deduplicator observation absorption stays deterministic; provider-assisted rewriting is reserved for fact-to-fact merges.
- Provider usage reporting in `admin overview` and the dashboard is task-attributed, so you can see which background task is actually consuming model time.
- Search debug output exposes graph/ranking fields useful for live tuning and dogfooding.

For deeper architecture/history context, use `docs/README.md` rather than treating this README as the entire design archive.

## ⚙️ Configuration

Preferred config location:

- `~/.config/mcp-memory/config.toml`

The runtime stores relational state in a shared global directory, typically:

- `~/.local/share/mcp-memory/memories/indices/memory.db`

If `XDG_DATA_HOME` is set, that location is used instead of `~/.local/share`.

The active workspace ID is derived from the current git root when available, with a stable path-based fallback outside git repos.

### Shared Postgres mode

If you want one shared memory store across multiple machines, use Postgres instead of syncing a live SQLite database.

- local SQLite remains the default
- Postgres is the shared-mode backend
- Postgres is authoritative whenever shared mode is selected
- shared mode must not silently fall back to SQLite
- if `[storage.cache]` is enabled in shared mode, the runtime creates a non-authoritative local SQLite sidecar at `.../memories/cache/shared_read_cache.sqlite3`
- `storage.cache.mode = "readonly"` enables readthrough and degraded cached search/read behavior
- `storage.cache.mode = "writeback"` currently extends that cache with a narrow durable outbox for `record_thought` only

For local Postgres startup and operator guidance, see:

- `compose.postgres.yml`
- `docs/postgres-shared-mode-runbook.md`

### Example `config.toml`

```toml
[ai]
provider = "gemini-cli"
model = "gemini-3-flash-preview"
timeout_seconds = 900
max_retries = 1

[gemini_cli]
command = "gemini"

[daemon]
host = "127.0.0.1"
port = 4242
auto_start_timeout_seconds = 10.0
shutdown_grace_seconds = 5.0
healthcheck_interval_seconds = 0.05

[backups]
enabled = true
interval_seconds = 3600.0
max_snapshots = 24
create_startup_snapshot = true
warn_on_shared_storage = true

[embeddings]
provider = "sentence-transformers"
model = "sentence-transformers/all-MiniLM-L6-v2"
batch_size = 32
ollama_base_url = "http://localhost:11434"

[search_ranking]
rrf_k = 60.0
calibration_threshold = 0.035
calibration_steepness = 150.0
workspace_multiplier = 1.2
degradation_multiplier = 0.3
access_half_life_days = 7.0
access_bonus_scale = 0.1
authority_link_step = 0.02
authority_link_cap = 10

[memory]
enabled = true
checkpoint_interval_ops = 10
checkpoint_interval_secs = 300

[memory.recency_journal]
boost_window_days = 14
max_boost_amount = 0.2
boost_decay_rate = 0.95

[memory.recency_plan]
boost_window_days = 30
max_boost_amount = 0.15
boost_decay_rate = 0.97

[memory.recency_fact]
boost_window_days = 180
max_boost_amount = 0.05
boost_decay_rate = 0.99

[memory.recency_observation]
boost_window_days = 14
max_boost_amount = 0.2
boost_decay_rate = 0.95

[memory.recency_reflection]
boost_window_days = 180
max_boost_amount = 0.05
boost_decay_rate = 0.99
```

The current code also applies two shipped ranking behaviors on top of those exposed knobs:
- a small bonus for graph-supported stable memories (`fact`, `observation`, `reflection`)
- a damped access bonus for unsupported transient `plan` / `journal` memories

Those two behaviors are intentionally still code-owned while we keep dogfooding the tuned search path.

## 🏃 Getting Started

1. **Install dependencies**:
   ```bash
   uv sync
   ```

2. **Create the config directory**:
   ```bash
   mkdir -p ~/.config/mcp-memory
   ```

3. **Write your config**:
   Save the example above to:
   ```text
   ~/.config/mcp-memory/config.toml
   ```

4. **Authenticate your AI provider**:
  If you set `ai.provider = "gemini-cli"`, make sure the Gemini CLI is installed and authenticated before relying on AI-assisted background tasks.

### Local shared-mode quick start

Bring up a local Postgres instance:

```bash
docker compose -f compose.postgres.yml up -d
```

Set `~/.config/mcp-memory/config.toml` to use Postgres:

```toml
[storage]
backend = "postgres"

[storage.postgres]
dsn = "postgresql://mcp_memory:change-me@127.0.0.1:5432/mcp_memory"
```

Optional shared-mode cache settings:

```toml
[storage.cache]
enabled = true
mode = "readonly" # or "writeback"
```

Use `mode = "readonly"` for readthrough/degraded cached search+read behavior.
Use `mode = "writeback"` only if you want the current narrow `record_thought` outbox fallback; broader offline mutation is still deferred.

If you already have SQLite data, dry-run the migration first:

```bash
uv run mcp-memory admin migrate-sqlite-to-postgres --dry-run --postgres-dsn 'postgresql://mcp_memory:change-me@127.0.0.1:5432/mcp_memory'
```

Then perform the import:

```bash
uv run mcp-memory admin migrate-sqlite-to-postgres --postgres-dsn 'postgresql://mcp_memory:change-me@127.0.0.1:5432/mcp_memory'
```

`embeddings.provider` defaults to `sentence-transformers`. To use an Ollama
embedding model instead, set the provider, model, and Ollama endpoint:

```toml
[embeddings]
provider = "ollama"
model = "qwen3-embedding:0.6b"
ollama_base_url = "http://localhost:11434"
```

For the full operator path, backup notes, and smoke checklist, see `docs/postgres-shared-mode-runbook.md`.

5. **Run the MCP proxy**:
   ```bash
   uv run mcp-memory run
   ```

Local semantic embeddings are now part of the default runtime behavior. The runtime will prefer `sentence-transformers` locally and falls back to a deterministic local hashing embedder if the configured model cannot be loaded yet.

On daemon startup, the runtime also makes a best-effort background attempt to download and cache the configured embedding model locally.

Daemon-backed MCP requests use a 60s client timeout budget. If a daemon request exceeds its transport deadline, the client now receives a structured timeout error instead of waiting forever.

   This command auto-starts the global daemon if it is not already running.

6. **Open the operator dashboard**:
   ```bash
   uv run mcp-memory admin dashboard open
   ```

   This command ensures the daemon is running, opens the dashboard in your browser, and prints the dashboard URL.

7. **Run the daemon manually** (optional):
   ```bash
   uv run mcp-memory daemon start
   ```

8. **Stash one thought quickly** (optional admin flow):
   ```bash
   uv run mcp-memory memory stash "I prefer snake_case JSON keys."
   ```

9. **Import legacy markdown** (optional admin flow):
   ```bash
   uv run mcp-memory memory import-markdown /path/to/memory.md
   ```

10. **Trigger or inspect background agents** (optional admin flow):
   ```bash
   uv run mcp-memory admin agent run memory-curator
   uv run mcp-memory admin agent run --all
   uv run mcp-memory admin health --json
   uv run mcp-memory admin overview
   uv run mcp-memory admin search health
   ```

11. **Install local tool integrations** (optional admin flow):
   ```bash
   uv run mcp-memory admin install
   uv run mcp-memory admin install --tool copilot --component hooks
   uv run mcp-memory admin install --tool claude --scope user --component hooks
   uv run mcp-memory admin install --tool gemini --component mcp
   ```

   This installer can write:
   - workspace Copilot hook config in `.github/hooks/mcp-memory.json`
   - Claude-format hook config in `.claude/settings.local.json` or `~/.claude/settings.json`
   - Gemini MCP config in `.gemini/settings.json`

   The installed hook commands forward `SessionStart`, `PostToolUse`, and `Stop` payloads into the workspace daemon through the hidden `hook-runner` command.

12. **Configure with your AI Assistant**:
   Add the following to your MCP configuration (e.g., Claude Desktop):
   ```json
   {
     "mcpServers": {
       "memory": {
         "command": "uv",
         "args": ["--directory", "/path/to/mcp-memory", "run", "mcp-memory", "run"]
       }
     }
   }
   ```

13. **Smoke test the system**:
   - confirm `uv run mcp-memory daemon status` reports a healthy daemon with a stable `ipc://...` endpoint
   - if using SQLite, confirm `~/.local/share/mcp-memory/memories/indices/memory.db` exists
   - if using Postgres, confirm `uv run mcp-memory admin health --json` reports the Postgres backend
   - if shared-mode cache is enabled, confirm `uv run mcp-memory admin health --json` reports the expected cache mode/state/path
   - record a thought through your MCP client
   - verify that the thought becomes searchable and readable through the MCP client
   - run `uv run mcp-memory admin overview` and confirm the task/memory metrics look sane

## Local verification

For normal local iteration, use this fast ladder:

```bash
uv run ruff check .
uv run pyright
uv run pytest tests/small/
```

These checks are always safe to run locally and do not require Docker or a running Postgres instance.

Widen deliberately when the change touches broader integration surfaces:

- `uv run pytest tests/medium/` for in-process integration work, especially storage integration and management/API behavior
- `uv run pytest tests/large/` for end-to-end flows, daemon lifecycle, and other full-runtime paths

Some DB-backed `tests/medium/` / `tests/large/` targets require Docker-backed Postgres fixtures. If your local target is configured to use the pgvector-backed image, pre-pull it first so fixture setup does not fail opaquely:

```bash
docker pull pgvector/pgvector:pg17
docker run --rm -d --name mcp-memory-test-pgvector -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=password -p 5432:5432 pgvector/pgvector:pg17
```

The repo's shared-mode operator quick start in `compose.postgres.yml` is separate from these test fixtures and uses the normal `postgres:17` image.

## Backup

In SQLite mode, back up the local data directory periodically:

```bash
cp -R ~/.local/share/mcp-memory ~/.local/share/mcp-memory.backup
```

The daemon now also supports periodic SQLite snapshots into `~/.local/share/mcp-memory/backups/` through the `[backups]` config block. This is useful for local SQLite mode.

In Postgres shared mode, use normal Postgres backup/restore procedures instead. See `docs/postgres-shared-mode-runbook.md`.

If shared-mode cache is enabled, treat the local SQLite sidecar as disposable support state, not as a backup or second source of truth.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.
