# Vision: Autonomous Memory Consolidation

**Status:** Vision

## 1. Overview
Current memory systems place high cognitive load on the AI, requiring it to manually organize, format, and structure files. This vision introduces a **Self-Healing Knowledge Graph** for `mcp-memory`:
- **System 1 (Thought Cache):** A fast, append-only scratchpad for raw thoughts.
- **System 2 (Structured Memory):** The durable, relational Markdown memory graph.

An **Autonomous Background Agent** (operating within the `mcp-memory` daemon) periodically consolidates System 1 into System 2 and "defragments" System 2 through semantic merging and obsolete fact pruning.

## 2. System 1: The "Thought" Cache
Active agents (and humans via CLI) can quickly stash findings using:
- **`record_thought(text: str)`**: Appends a timestamped entry to `system1_journal`. This captures the `workspace_id` automatically.

## 3. The "Strategy Roulette" (Maintenance Loops)
To keep the graph lean and connected, background agents run maintenance cycles every 6-12 hours. They do not just process the "most recent" data; they use a **Strategy Roulette** to select memory batches:
1. **Semantic Clustering**: Merging redundant facts.
2. **Pure Noise**: Random sampling to find lateral links.
3. **Cold Storage**: Pruning memories that haven't been read in months.
4. **Never Surfaced**: Investigating why certain memories never appear in search.
5. **Anomalies**: Handling exceptionally large or fragmented memories.

## 4. Tiered AI Provider Layer
Maintenance is performed for free using the CLI tools already on the user's machine. The daemon selects the model "tier" based on task complexity:
- **Strong (Reasoning)**: `gemini-cli` or `copilot-cli`. Used for **Conflict Detection** and **Defragmentation**.
- **Medium**: Local models via `ollama` or `opencode`. Used for **Ingestion** and **Graph Linking**.
- **Weak/Fast**: Small local models. Used for **Summarization** and **Tag Normalization**.

## 5. Safety & Human Observability
Because the agent autonomously modifies the knowledge base, safety is built-in:
- **Soft Deletes**: Files are moved to `.trash/` (or marked as `archived` in DB).
- **The Conflict Inbox**: Direct contradictions are flagged for the user in the Web UI.
- **Audit Logs**: Every agent action (merges, links, status changes) is logged and reversible.

## 6. Goal
A memory bank that grows smarter while you sleep, staying dense, organized, and free of contradictions without requiring a single manual organization task from the user.
