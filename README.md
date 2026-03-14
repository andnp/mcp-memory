# MCP Memory Server

A standalone Model Context Protocol (MCP) server for persistent AI memory management. This project extracts the high-performance memory subsystem from `mcp-markdown-ragdocs` into a dedicated, lightweight service.

## 🚀 Overview

The MCP Memory Server provides a persistent "Memory Bank" for AI assistants, enabling them to store, retrieve, and organize knowledge across sessions. It uses a sophisticated hybrid search engine (Vector + Keyword + Graph) with specialized recency boosting and cross-corpus linking.

## 🚧 Current status

The runtime is now **relational-first**.

- New memory CRUD, search, read, and markdown import flows run through the relational runtime.
- The old file-backed manager/search/tool path has been removed from the active codebase.
- Background task handling is now part of the runtime surface rather than a future-only design note.
- Hybrid index modules still exist in `indices/`, but they are no longer the primary runtime path.

### Key Features

- **Hybrid Search**: Combines semantic vector search (FAISS), keyword search (SQLite FTS5), and relationship graph search.
- **Memory-Specific Recency Boost**: Automatically prioritizes recent memories (journals, plans) while preserving long-term facts.
- **Graph-Based Linking**: Supports `[[wikilinks]]` between memories and external documents with context-aware edges.
- **Categorized Memories**: Built-in support for `journal`, `plan`, `fact`, `observation`, and `reflection` types.
- **Relational Memory Foundation**: UUID-backed relational memory records with workspaces, tags, and typed links.
- **Project-Scoped Runtime State**: The runtime resolves a per-project memory directory and stores relational state in `indices/memory.db`.
- **Local Management API**: A read-only FastAPI dashboard/API exposes runtime health, failed tasks, recent memories, and lineage inspection over localhost.

## 🛠 Tech Stack

- **Python 3.13+**
- **MCP (Model Context Protocol)**: Standard interface for AI tool integration.
- **LlamaIndex**: Core indexing and retrieval infrastructure.
- **FAISS**: High-performance vector similarity search.
- **SQLite (FTS5)**: Robust keyword search and relational storage.
- **Sentence-Transformers**: Local embedding generation.
- **NetworkX**: Relationship graph management.

## 📂 Project Structure

```text
mcp-memory/
├── src/
│   └── mcp_memory/
│       ├── core/           # Journal, task runtime, and shared utilities
│       ├── relational/     # Relational repository, importer, and search services
│       ├── indices/        # Hybrid index experiments and storage backends
│       ├── mcp/            # MCP server implementation and tools
│       ├── models/         # Pydantic data models
│       ├── utils/          # Shared utilities (Config, IO, Similarity)
│       ├── cli.py          # Command-line interface
│       └── server.py       # Main entry point
├── tests/                  # Comprehensive test suite
├── pyproject.toml          # Dependency management (uv/hatch)
└── README.md               # You are here
```

## 📋 Current MCP Tools

The following tools are exposed via the MCP server:

### Journal Runtime
- `record_thought`: Record a raw system-1 thought in the local journal.
- `get_pending_thoughts`: List pending journal thoughts awaiting consolidation.

### Relational Memory Records
- `create_memory_record`: Create a UUID-backed relational memory record.
- `get_memory_record`: Fetch a relational memory record by ID.
- `list_memory_records`: List relational memory records with optional filters.
- `search_memory_records`: Search relational memory records with summary-first results.
- `read_memory_record`: Read a relational memory record with relationships and superseded breadcrumbs.
- `import_markdown_memory_file`: Import a legacy markdown memory file into the relational store.

### Runtime Statistics
- `get_memory_stats`: Return basic runtime statistics for the journal, relational records, and indexed documents.

The public MCP surface is relational-first; removed file-backed tools are intentionally not part of the supported API.

## ⚙️ Configuration

Configured via `config.toml` or environment variables:

```toml
[memory]
enabled = true
storage_strategy = "project"  # or "user"
score_threshold = 0.1

[memory.recency_journal]
boost_window_days = 14
max_boost_amount = 0.2
boost_decay_rate = 0.95
```

The current runtime resolves a per-project memory directory and stores relational runtime state in `indices/memory.db` beneath that directory.

## 🏃 Getting Started

1. **Install dependencies**:
   ```bash
   uv sync
   ```

2. **Run the server**:
   ```bash
   uv run mcp-memory run
   ```

3. **Run the management dashboard**:
  ```bash
  uv run mcp-memory dashboard --host 127.0.0.1 --port 8765
  ```

  Then open `http://127.0.0.1:8765/` for the read-only operations snapshot.

4. **Configure with your AI Assistant**:
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

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.
