   Development helpers:
   ```bash
   uv run mcp-memory daemon-status
   uv run mcp-memory daemon-stop
   uv run mcp-memory daemon-restart
   ```
# MCP Memory Server

A standalone Model Context Protocol (MCP) server for persistent AI memory management.

## 🚀 Overview

The MCP Memory Server provides a persistent memory bank for AI assistants, enabling them to store, retrieve, and organize knowledge across sessions. The current runtime is relational, ZMQ-backed, and centered on one shared global SQLite store.

## 🚧 Current status

The runtime is now **relational-first**.

- New memory CRUD, search, read, and markdown import flows run through the relational runtime.
- Background task handling is now part of the active runtime surface.
- The MCP stdio entrypoint is now a thin proxy that auto-starts a proper workspace daemon on demand.

### Key Features

- **Graph-Aware Search Ranking**: Search now combines weighted BM25 keyword retrieval, optional semantic candidates, soft workspace-aware semantic ordering, RRF fusion, sigmoid calibration, graph-aware reranking, one-hop graph expansion, and degradation penalties.
- **Memory-Specific Recency Boost**: Automatically prioritizes recent memories (journals, plans) while preserving long-term facts.
- **Typed Relational Links**: Memory records can carry explicit typed relationships such as `SUPERSEDES`, `EXTENDS`, and `CONTRADICTS`.
- **Categorized Memories**: Built-in support for `journal`, `plan`, `fact`, `observation`, and `reflection` types.
- **Relational Memory Foundation**: UUID-backed relational memory records with workspace IDs, tags, and typed links.
- **Local Semantic Search**: Fully local embeddings via `sentence-transformers` enrich search and ingest without any external AI provider.
- **Shared Global Storage**: All memories live in one shared XDG data directory, with workspace identity attached to thoughts, tasks, and memories.
- **Global Daemon**: One global daemon per user environment owns runtime state, background workers, and transport coordination.
- **Stable IPC Transport**: The thin MCP proxy talks to the daemon over a stable ZeroMQ ROUTER/DEALER transport on a Unix socket.
- **Agentic Maintenance Agents**: Curator, deduplicator, and ingest now run through a trusted internal MCP maintenance surface when an agentic provider is configured.
- **Crash-Safe Ingest Finalization**: Agentic ingest preserves durable journal claim/delete/release semantics while still allowing direct MCP create/append mutations.
- **Hardened Daemon Recovery**: Daemon stop/restart now terminates stale process groups, escalates from `SIGTERM` to `SIGKILL` when needed, and cleans stale sockets instead of merely dropping metadata.
- **Operational Views**: Health, overview, and lineage inspection remain available through the daemon management surface and CLI.
- **Task-Level Provider Telemetry**: AI provider usage is attributed to the background task that made the call so you can see which agents are actually consuming model time.

## 🛠 Tech Stack

- **Python 3.13+**
- **MCP (Model Context Protocol)**: Standard interface for AI tool integration.
- **SQLite (FTS5)**: Robust keyword search and relational storage.
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
│       ├── daemon.py       # Workspace daemon and autostart logic
│       └── server.py       # MCP stdio thin proxy
├── tests/                  # Comprehensive test suite
├── pyproject.toml          # Dependency management (uv/hatch)
└── README.md               # You are here
```

## 📋 Current MCP Tools

The following tools are exposed via the MCP server:

### Minimal Public Surface
- `record_thought`: Record a raw system-1 thought in the local journal.
- `search_memory_records`: Search relational memory records with summary-first results and staged ranking over weighted keyword + optional semantic retrieval.
- `read_memory_record`: Read a relational memory record with relationships and superseded breadcrumbs.

Everything else is intentionally kept out of the public MCP surface. Admin, migration, browsing, and operational views belong in the daemon management surface or CLI, not in the assistant-facing protocol.

For trusted maintenance agents, the repo also now includes a workspace-local internal MCP surface exposed through `uv run mcp-memory internal-run`. The bundled `.gemini/settings.json` enables that internal tool surface only inside this workspace.

### Admin / Maintenance Commands
- `uv run mcp-memory stash "Remember to normalize workspace metadata."`: record one raw thought into the System 1 journal.
- `uv run mcp-memory dashboard`: ensure the daemon is running and print the active transport endpoint.
- `uv run mcp-memory agents run sweeper`: trigger one background agent for the active workspace.
- `uv run mcp-memory agents run-all`: enqueue all background agents for the active workspace.
- `uv run mcp-memory stats`: print background task and memory statistics from SQLite.
- `uv run mcp-memory import-markdown /path/to/memory.md`: import one markdown memory file into the relational store.
- `uv run mcp-memory import-markdown /path/to/one.md '/path/to/*.md'`: import explicit files and globbed markdown files in one command.

Background maintenance now also includes a `deduplicator` agent that can merge highly similar fact memories into a canonical fact and absorb matching observation memories into that fact while archiving the source memories with lineage links.

When an agentic AI provider is configured and the internal maintenance MCP surface is available, the current maintenance stack can now:
- run `memory-curator` through internal MCP maintenance tools
- run `deduplicator` through the same agentic MCP mode
- run `ingest-system1` through agentic MCP mode with internal batch claiming and direct append/create mutations while preserving handler-owned journal finalization

Agentic ingest now prefers dedicated ingest-specific internal tools for append/create mutations, so `ingest_task_id`, entry lineage, workspace propagation, and summarize-task enqueueing live in the tool layer instead of prompt instructions.

Deduplicator observation absorption now stays deterministic and reserves provider-assisted rewriting for fact-to-fact merges, which reduces per-run provider fan-out without changing the recurring maintenance cadence.

Split maintenance now records richer lineage structure: split children share a `split_group_id`, include per-part ordering/count metadata plus sibling IDs, and the original memory records the full child set/count.

Agentic ingest also now normalizes legacy provider payloads that return `results` instead of `actions`, so append opportunities are no longer silently lost when the provider uses the older shape.

Provider usage reporting in `uv run mcp-memory stats` and the dashboard now includes the task name responsible for each provider-usage aggregate row, making it easier to tell whether work is coming from `ingest-system1`, `defragmenter`, `memory-curator`, or another agentic task.

Search debug output now exposes graph- and ranking-specific fields such as `authority_supporting_links`, `authority_contradicting_links`, `expanded_by_graph`, `graph_seed_id`, `graph_link_type`, and `graph_rrf_score`, making live search tuning and dogfooding much easier.

Markdown import keeps its tiny file-scanning logic local to the importer instead of preserving a separate file-backed helper layer in the active package structure.

## ⚙️ Configuration

Preferred config location:

- `~/.config/mcp-memory/config.toml`

The runtime stores relational state in a shared global directory, typically:

- `~/.local/share/mcp-memory/memories/indices/memory.db`

If `XDG_DATA_HOME` is set, that location is used instead of `~/.local/share`.

The active workspace ID is derived from the current git root when available, with a stable path-based fallback outside git repos.

### Example `config.toml`

```toml
[ai]
provider = "none"
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
model = "sentence-transformers/all-MiniLM-L6-v2"
batch_size = 32

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

5. **Run the MCP proxy**:
   ```bash
   uv run mcp-memory run
   ```

Local semantic embeddings are now part of the default runtime behavior. The runtime will prefer `sentence-transformers` locally and falls back to a deterministic local hashing embedder if the configured model cannot be loaded yet.

On daemon startup, the runtime also makes a best-effort background attempt to download and cache the configured embedding model locally.

   This command auto-starts the global daemon if it is not already running.

6. **Inspect the active daemon transport**:
   ```bash
   uv run mcp-memory dashboard
   ```

   This command ensures the daemon is running and prints the active `ipc://...` transport endpoint.

7. **Run the daemon manually** (optional):
   ```bash
   uv run mcp-memory daemon
   ```

8. **Stash one thought quickly** (optional admin flow):
   ```bash
   uv run mcp-memory stash "I prefer snake_case JSON keys."
   ```

9. **Import legacy markdown** (optional admin flow):
   ```bash
   uv run mcp-memory import-markdown /path/to/memory.md
   ```

10. **Trigger or inspect background agents** (optional admin flow):
   ```bash
   uv run mcp-memory agents run sweeper
   uv run mcp-memory agents run-all
   uv run mcp-memory stats
   ```

11. **Install local tool integrations** (optional admin flow):
   ```bash
   uv run mcp-memory install
   uv run mcp-memory install --tool copilot --component hooks
   uv run mcp-memory install --tool claude --scope user --component hooks
   uv run mcp-memory install --tool gemini --component mcp
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
   - confirm the daemon command prints a stable `ipc://...` endpoint
   - confirm `~/.local/share/mcp-memory/memories/indices/memory.db` exists
   - record a thought through your MCP client
   - verify that the thought becomes searchable and readable through the MCP client
   - run `uv run mcp-memory stats` and confirm the task/memory metrics look sane

## Backup

Because the runtime now uses one shared memory store, back up the directory periodically:

```bash
cp -R ~/.local/share/mcp-memory ~/.local/share/mcp-memory.backup
```

The daemon now also supports periodic SQLite snapshots into `~/.local/share/mcp-memory/backups/` through the `[backups]` config block. This is especially useful if the store lives inside Syncthing or any other shared-storage setup that can create conflict files around a live SQLite database.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.
