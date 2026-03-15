# MCP Memory Server

A standalone Model Context Protocol (MCP) server for persistent AI memory management.

## 🚀 Overview

The MCP Memory Server provides a persistent memory bank for AI assistants, enabling them to store, retrieve, and organize knowledge across sessions. The current runtime is relational, workspace-aware, and backed by one shared global SQLite store.

## 🚧 Current status

The runtime is now **relational-first**.

- New memory CRUD, search, read, and markdown import flows run through the relational runtime.
- Background task handling is now part of the active runtime surface.
- The MCP stdio entrypoint is now a thin proxy that auto-starts a proper workspace daemon on demand.

### Key Features

- **Memory-Specific Recency Boost**: Automatically prioritizes recent memories (journals, plans) while preserving long-term facts.
- **Typed Relational Links**: Memory records can carry explicit typed relationships such as `SUPERSEDES`, `EXTENDS`, and `CONTRADICTS`.
- **Categorized Memories**: Built-in support for `journal`, `plan`, `fact`, `observation`, and `reflection` types.
- **Relational Memory Foundation**: UUID-backed relational memory records with workspace IDs, tags, and typed links.
- **Local Semantic Search**: Fully local embeddings via `sentence-transformers` enrich search and ingest without any external AI provider.
- **Shared Global Storage**: All memories live in one shared XDG data directory, with workspace identity attached to thoughts, tasks, and memories.
- **Workspace Daemon**: One localhost daemon per workspace owns runtime state, background workers, and the management API.
- **Local Management API**: The daemon exposes a small read-only dashboard/API for runtime health, overview, and lineage inspection.

## 🛠 Tech Stack

- **Python 3.13+**
- **MCP (Model Context Protocol)**: Standard interface for AI tool integration.
- **SQLite (FTS5)**: Robust keyword search and relational storage.
- **Sentence Transformers (optional)**: Local embeddings for semantic retrieval and thought clustering.
- **FastAPI + Uvicorn**: Local daemon and management API.

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
- `search_memory_records`: Search relational memory records with summary-first results and local semantic ranking.
- `read_memory_record`: Read a relational memory record with relationships and superseded breadcrumbs.

Everything else is intentionally kept out of the public MCP surface. Admin, migration, browsing, and operational views belong in the dashboard/API or CLI, not in the assistant-facing protocol.

### Admin / Maintenance Commands
- `uv run mcp-memory dashboard`: ensure the daemon is running and print the dashboard URL.
- `uv run mcp-memory agents run sweeper`: trigger one background agent for the active workspace.
- `uv run mcp-memory agents run-all`: enqueue all background agents for the active workspace.
- `uv run mcp-memory stats`: print background task and memory statistics from SQLite.
- `uv run mcp-memory import-markdown /path/to/memory.md`: import one markdown memory file into the relational store.
- `uv run mcp-memory import-markdown /path/to/one.md '/path/to/*.md'`: import explicit files and globbed markdown files in one command.

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
timeout_seconds = 60
max_retries = 1

[gemini_cli]
command = "gemini"

[daemon]
host = "127.0.0.1"
auto_start_timeout_seconds = 10.0
shutdown_grace_seconds = 5.0
healthcheck_interval_seconds = 0.05

[embeddings]
model = "sentence-transformers/all-MiniLM-L6-v2"
batch_size = 32

[memory]
enabled = true
checkpoint_interval_ops = 10
checkpoint_interval_secs = 300

[memory.recency_journal]
boost_window_days = 14
max_boost_amount = 0.2
boost_decay_rate = 0.95

[memory.recency_plan]
boost_window_days = 7
max_boost_amount = 0.5
boost_decay_rate = 0.9

[memory.recency_fact]
boost_window_days = 60
max_boost_amount = 0.2
boost_decay_rate = 0.99

[memory.recency_observation]
boost_window_days = 14
max_boost_amount = 0.2
boost_decay_rate = 0.95

[memory.recency_reflection]
boost_window_days = 30
max_boost_amount = 0.15
boost_decay_rate = 0.98
```

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

   This command auto-starts the workspace daemon if it is not already running.

6. **Get the dashboard URL**:
   ```bash
   uv run mcp-memory dashboard
   ```

   This command ensures the daemon is running and prints the dashboard URL.

7. **Run the daemon manually** (optional):
   ```bash
   uv run mcp-memory daemon
   ```

8. **Import legacy markdown** (optional admin flow):
   ```bash
   uv run mcp-memory import-markdown /path/to/memory.md
   ```

9. **Trigger or inspect background agents** (optional admin flow):
   ```bash
   uv run mcp-memory agents run sweeper
   uv run mcp-memory agents run-all
   uv run mcp-memory stats
   ```

10. **Configure with your AI Assistant**:
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

11. **Smoke test the system**:
   - confirm the printed dashboard URL loads
   - confirm `~/.local/share/mcp-memory/memories/indices/memory.db` exists
   - record a thought through your MCP client
   - verify that the thought becomes searchable and readable through the MCP client
   - run `uv run mcp-memory stats` and confirm the task/memory metrics look sane

## Backup

Because the runtime now uses one shared memory store, back up the directory periodically:

```bash
cp -R ~/.local/share/mcp-memory ~/.local/share/mcp-memory.backup
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.
