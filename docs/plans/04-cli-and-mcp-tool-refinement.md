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
- **Output Schema**: ✅ Compact summary-first result projection for agent use.
- **Noise Control**: ✅ Omits `score` and `workspace_ids` by default; debug mode exposes ranking/audit fields.
- **Summarization**: ✅ Smart truncate fallback now applies when a formal summary is missing.

### 2.3 Tool Contract: `read_memory_record` (COMPLETED)
- **Goal**: Depth on demand without paying the traceability cost on every read.
- **Output Schema**: ✅ Minimal `record` by default.
- **Breadcrumbs**: ✅ `superseded` list is available through `include_superseded`.
- **Relationships**: ✅ Incoming/outgoing links are available through `include_relationships`.
- **Metadata**: ✅ Maintenance metadata and workspace IDs are available through `include_metadata`.

### 2.4 Background Cleanup Direction
- **Default stance**: Do not add another always-on cleaning loop just to hide noisy tool payloads; payload shape should be fixed at the MCP boundary first.
- **Maintenance stance**: Keep durable content cleanup in the existing task framework. Summarizer, curator, defragmenter, taxonomist, and sweeper jobs should own memory quality improvements such as oversized records, stale metadata, and tag normalization.
- **Future candidate**: Add a bounded metadata-hygiene maintenance task only if stored metadata itself becomes harmful to retrieval quality or dashboard usability. It should normalize lineage keys and prune obsolete bookkeeping, not remove audit information needed for maintenance traceability.

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

### Task 4: Token Hygiene
- [x] `search_memory_records`: Keep public results compact unless `debug=true`.
- [x] `read_memory_record`: Return relationships, superseded records, and metadata only when explicitly requested by external callers.
- [x] Shared read-cache paths: Compact legacy cached payloads before returning them to external callers.

### Task 5: Maintenance Feedback-Loop Diagnostics
- [x] Sweeper reports lineage/relationship hotspots before cleanup: active split originals, oversized lineage metadata, and high relationship-density records.
- [x] SQLite now gets the same dead maintenance metadata key pruning behavior as Postgres.
- [ ] Future guardrail: prevent curator split/merge work from reprocessing recent split or merge lineage without an explicit repair reason.
