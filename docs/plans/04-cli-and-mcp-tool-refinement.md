# Plan: CLI & MCP Tool Refinement

## 1. Context
This plan finalizes the user-facing contract. It enforces **Progressive Discovery** (to save context tokens) and introduces the **Zero-Friction Stash** for human developers.

## 2. Interface Specifications

### 2.1 The `stash` Command (COMPLETED)
- **Usage**: `mcp-memory stash "I prefer to use snake_case for all JSON keys."`
- **Output**: `"Thought stashed successfully (ID: 42)"`.
- **Side Effect**: Capture current `workspace_id` via `git rev-parse --show-toplevel`.

### 2.2 Tool Contract: `search_memory_records` (COMPLETED)
- **Goal**: Breadth over depth.
- **Output Schema**: ✅ Matches `RelationalSearchResult`.
- **Summarization**: ✅ Smart truncate fallback now applies when a formal summary is missing.

### 2.3 Tool Contract: `read_memory_record` (COMPLETED)
- **Goal**: Depth and Traceability.
- **Output Schema**: ✅ Matches `RelationalReadResult`.
- **Breadcrumbs**: ✅ `superseded` list included.

---

## 3. Implementation Tasks

### Task 1: CLI Enhancements
- [x] Add `stash` command to `mcp_memory/cli.py`.
- [x] Ensure `stash` can handle multi-line input via `stdin` if no text argument is provided.

### Task 2: Summarization Logic
- [x] In the `search_memory_records` handler (or `RelationalSearchResult`), implement a **Smart Truncate** (First 200 chars, truncated at the last complete sentence) if the formal `summary` is missing.

### Task 3: Testing
- [x] `tests/small/test_cli_stash.py`: Verify Git-based `workspace_id` resolution.
- [x] `tests/medium/test_relational_runtime_tools.py` and `tests/medium/test_repository_tools.py`: Verify that `search` returns summaries and `read` returns full content.
