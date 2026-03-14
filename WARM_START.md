# Warm-Start Guide: Extracting the Memory Subsystem

This guide outlines the steps to complete the transition of the memory subsystem from `mcp-markdown-ragdocs` to this standalone `mcp-memory` project.

## 🏁 Prerequisites

- [uv](https://github.com/astral-sh/uv) installed.
- Access to the `mcp-markdown-ragdocs` codebase.

## 📋 Extraction Steps

### 1. File Migration

Copy the following core components from `mcp-markdown-ragdocs/src/` to `mcp-memory/src/mcp_memory/`:

| Source in `ragdocs` | Destination in `mcp-memory` | Notes |
| :--- | :--- | :--- |
| `memory/*.py` | `core/` | Rename `init.py` to `__init__.py` and adjust imports. |
| `indices/vector.py` | `indices/vector.py` | Simplify: remove non-memory logic. |
| `indices/keyword.py` | `indices/keyword.py` | Focus on SQLite FTS5. |
| `indices/graph.py` | `indices/graph.py` | Keep ghost node logic. |
| `storage/db.py` | `utils/db.py` | Lightweight SQLite manager. |
| `utils/atomic_io.py` | `utils/io.py` | Basic atomic file operations. |
| `utils/similarity.py` | `utils/similarity.py` | Cosine similarity logic. |
| `search/base_orchestrator.py` | `core/orchestrator.py` | Base for memory search. |

### 2. Refactoring & Pruning

- **Path Normalization**: Replace all `from src.memory` imports with relative imports or `from mcp_memory.core`.
- **Config Management**: Simplify `Config` to only include `MemoryConfig`. Remove doc-search-specific settings.
- **Model Consolidation**: Merge `src/models.py` (Chunk, Document) into `mcp_memory/models.py`. Keep only what's needed for memories.
- **Chunking**: Simplify `HeaderChunker`. Since memories are usually small, you can simplify the tree-sitter logic or use a lightweight markdown parser.

### 3. MCP Server Setup

- Use the logic in `src/mcp/server.py` and `src/memory/tools.py` from the original project to build the new `src/mcp_memory/server.py`.
- Ensure all 9 tools are correctly registered with the new MCP Python SDK.

## 🧪 Testing Strategy

1. **Unit Tests**: Extract tests from `tests/unit/memory/` and adapt them to the new structure.
2. **Integration Tests**: Verify that `MemoryIndexManager` correctly initializes indices and persists data in both `project` and `user` storage modes.
3. **E2E Tests**: Use a lightweight MCP client to verify tool execution.

## 📦 Deployment

This server can be run as a standard MCP server via `stdio`. Ensure the environment has access to the `uv` binary if using the recommended `uv run` configuration.

---

**Next Step**: Start by copying the `core/` logic and fixing imports.
