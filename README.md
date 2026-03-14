# MCP Memory Server

A standalone Model Context Protocol (MCP) server for persistent AI memory management. This project extracts the high-performance memory subsystem from `mcp-markdown-ragdocs` into a dedicated, lightweight service.

## 🚀 Overview

The MCP Memory Server provides a persistent "Memory Bank" for AI assistants, enabling them to store, retrieve, and organize knowledge across sessions. It uses a sophisticated hybrid search engine (Vector + Keyword + Graph) with specialized recency boosting and cross-corpus linking.

### Key Features

- **Hybrid Search**: Combines semantic vector search (FAISS), keyword search (SQLite FTS5), and relationship graph search.
- **Memory-Specific Recency Boost**: Automatically prioritizes recent memories (journals, plans) while preserving long-term facts.
- **Graph-Based Linking**: Supports `[[wikilinks]]` between memories and external documents with context-aware edges.
- **Categorized Memories**: Built-in support for `journal`, `plan`, `fact`, `observation`, and `reflection` types.
- **Full CRUD Suite**: Tools for creating, reading, updating, appending, and merging memories.
- **Flexible Storage**: Store memories project-locally (`.memories/`) or in a global user directory.

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
│       ├── core/           # Core memory logic (Manager, Search, Storage)
│       ├── indices/        # Hybrid indices (Vector, Keyword, Graph)
│       ├── mcp/            # MCP server implementation and tools
│       ├── models/         # Pydantic data models
│       ├── utils/          # Shared utilities (Config, IO, Similarity)
│       ├── cli.py          # Command-line interface
│       └── server.py       # Main entry point
├── tests/                  # Comprehensive test suite
├── pyproject.toml          # Dependency management (uv/hatch)
└── README.md               # You are here
```

## 📋 Tool Definitions

The following tools are exposed via the MCP server:

### CRUD Operations
- `create_memory`: Create a new memory file with metadata.
- `read_memory`: Retrieve the full content of a memory.
- `update_memory`: Replace a memory's content entirely.
- `append_memory`: Add content to the end of an existing memory.
- `delete_memory`: Soft-delete a memory (moves to `.trash/`).

### Search & Retrieval
- `search_memories`: Hybrid search with recency boosting and type filtering.
- `search_linked_memories`: Find memories linking to specific external documents.
- `get_memory_relationships`: Query version history (SUPERSEDES), dependencies (DEPENDS_ON), or contradictions.

### Maintenance
- `get_memory_stats`: Get statistics on the memory bank (count, size, tags).
- `merge_memories`: Consolidate multiple related memories into a single summary.

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

## 🏃 Getting Started

1. **Install dependencies**:
   ```bash
   uv sync
   ```

2. **Run the server**:
   ```bash
   uv run mcp-memory run
   ```

3. **Configure with your AI Assistant**:
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
