# Vision: Autonomous Memory Consolidation

## 1. Overview
Current memory systems place high cognitive load on the AI, requiring it to manually organize, format, and structure Markdown files. This often leads to fragmentation and "memory decay" as information is stashed in silos. 

This vision introduces a **Dual-System Memory Architecture** for `mcp-memory`:
- **System 1 (Ephemeral Cache):** A fast, low-friction, append-only scratchpad for raw thoughts.
- **System 2 (Structured Memory):** The existing, durable Markdown memory graph.

An **Autonomous Background Agent** (operating within the `mcp-memory` daemon) periodically consolidates System 1 into System 2 and "defragments" System 2 through semantic merging and obsolete fact pruning.

## 2. System 1: The "Thought" Cache
Active AI agents can quickly stash context or findings using a lightweight tool:
- **`record_thought(text: str)`**: Appends a timestamped entry to a `system1_journal` table in SQLite.

This allows the agent to maintain focus on its primary task (e.g., coding) while ensuring no critical context is lost.

## 3. Agentic Background Tasks
The `mcp-memory` daemon runs periodic background tasks (using a simple scheduler like `Huey` or `asyncio.sleep` loops) to maintain the memory graph.

### 3.1 Fast Consolidation (Flushing)
**Trigger**: When the System 1 journal reaches a certain threshold.
**Workflow**: 
1. Retrieve raw entries from System 1.
2. Search System 2 for related files.
3. Use a configured AI provider (CLI or API) to map raw thoughts into structured `create`, `update`, or `append` operations for System 2.

### 3.2 Deep Defragmentation (The "Strategy Roulette")
**Trigger**: A periodic cron task (e.g., every 6-12 hours).
**Workflow**: To prevent the agents from only optimizing the "most popular" memories (getting stuck in a local minimum), the Defragmenter and Linker agents use a **Strategy Roulette**. Every time they wake up, they randomly select a different heuristic to pull a batch of ~10-20 memories to review:

1. **Semantic Clustering**: (The default) Find dense groups of highly similar memories to merge redundant facts.
2. **Pure Random Walk**: Inject noise by picking memories completely at random. This is surprisingly effective at surfacing unexpected lateral connections or finding deeply buried, stale plans.
3. **The "Cold Storage" Pass**: Select the least frequently read memories, or those with the longest time since `last_accessed_at`. The agent decides if they should be archived, merged, or left alone.
4. **The "Anomaly" Pass**: Select the largest memories (maybe they need to be split/summarized) or the smallest memories (maybe they are fragments that need to be merged).
5. **The "Orphan" Pass**: Select memories with the fewest (or zero) tags or links to see if they can be connected to the broader graph.

By rotating through these strategies, the agents ensure the *entire* long-tail of the memory graph is continually groomed, not just the heavily trafficked areas.

## 4. AI Provider Layer
To support autonomous consolidation without requiring API keys or incurring cloud costs, `mcp-memory` uses a pluggable **AI Provider** layer strictly targeting local or CLI-based tools. 

Supported backends include:
- **`copilot-cli`**: Executes `gh copilot suggest ...` using the user's existing GitHub Copilot authentication.
- **`gemini-cli`**: Executes `gemini ask --json ...` using the user's existing Gemini CLI setup.
- **`opencode`**: Leverages the local opencode CLI.
- **`ollama`**: Connects to a local Ollama instance for completely offline, zero-cost processing.

This approach guarantees that the background daemon can "think" for free, leveraging the tools the developer already has installed on their machine.

## 5. Safety & Trust
Because the agent can autonomously modify or delete files, safety is paramount:
- **Soft Deletes**: Deleted files are moved to `.trash/` with a timestamp.
- **Audit Logs**: Every autonomous action is logged in `consolidation_audit.md`.
- **Reference Counting**: Consolidation tasks only run when the daemon is active and "background maintenance" is enabled.
- **Rollback**: Users can manually revert any autonomous change via the Management Dashboard.

## 6. Goal
The ultimate goal of autonomous consolidation is to provide an AI with a "self-healing" long-term memory that remains dense, organized, and free of contradictions without requiring constant manual maintenance.
