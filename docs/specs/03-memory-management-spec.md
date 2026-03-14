# Specification: Memory Management System (Relational)

**Status:** ✅ Standalone Migration in Progress

> Current implementation note:
> - The relational schema, repository, importer, and relational search service back the active MCP runtime.
> - The old file-backed manager/search runtime path has been removed.
> - Public MCP tools already expose relational create/get/list/search/read/import flows, while some advanced ranking and maintenance behaviors remain implementation work.

## 1. Relational Schema

### 1.1 Core Tables
- **`memories`**: `id (UUID PRIMARY KEY)`, `title (TEXT)`, `content (TEXT)`, `summary (TEXT)`, `type (VARCHAR)`, `status (VARCHAR: active, stale, degraded, archived)`, `access_score (REAL)`, `last_accessed_at (DATETIME)`, `created_at (DATETIME)`, `updated_at (DATETIME)`.
    - **Types**: `journal`, `plan`, `fact`, `observation`, `reflection`.
- **`memory_workspaces`**: `memory_id (UUID)`, `workspace_id (VARCHAR)`. (Maps memories to 1 or more workspaces).
- **`links`**: `source_id (UUID)`, `target_id (UUID)`, `type (VARCHAR)`, `context (TEXT)`.
- **`system1_journal`**: `id (INTEGER PRIMARY KEY)`, `content (TEXT)`, `workspace_id (VARCHAR)`, `author (VARCHAR)`, `timestamp (DATETIME)`, `status (VARCHAR)`.

## 2. Progressive Discovery Tools

To minimize context window bloat, the active AI interacts via a two-stage discovery process:

1. **`search_memory_records`**: 
    - **Inputs**: `query`, `limit`. (The `workspace_id` is automatically injected by the MCP proxy).
    - **Outputs**: Returns metadata and a **two-sentence summary** of matches.
2. **`read_memory_record`**: 
    - **Inputs**: `memory_id`.
    - **Outputs**: Returns the **full content**, all structural relationships, AND a list of **superseded memories**. If the read memory supersedes older memories (e.g., a Reflection that summarizes 3 older Journals), the response explicitly lists the IDs and titles of those superseded memories. This allows the AI to trace the "why" and dig into historical context if needed.
    - **Side Effect**: Triggers the **Access Score Decay** logic (see below).
3. **`record_thought`**: Quickly stashes raw context into System 1.

## 3. Search Ranking & Scoring Pipeline
The `search_memories` tool uses a highly calibrated, 6-stage pipeline ported and refined from `ragdocs`:

### Stage 1: RRF Fusion (Base Retrieval)
Combines dense vector retrieval (FAISS) and sparse keyword retrieval (FTS5) using Reciprocal Rank Fusion. This establishes the initial candidate pool and generates a base rank score.

### Stage 2: Sigmoid Calibration
Expands compressed RRF scores (typically 0.01-0.05) into an interpretable absolute confidence range (0.0 to 1.0) using a sigmoid function:
`score = 1.0 / (1.0 + exp(-steepness * (rrf_score - threshold)))`

### Stage 3: Type-Aware Recency Boost (Exponential Additive)
Recent memories receive an exponentially decaying bonus *added* to their calibrated score. The decay curve depends entirely on the memory `type`:
- **`journal` / `observation`**: High initial boost, fast decay (e.g., 14-day window). Relevant for current tasks.
- **`plan`**: Medium boost, medium decay (e.g., 30-day window). Remains relevant while a feature is built.
- **`fact` / `reflection`**: Low initial boost, extremely slow decay. Evergreen knowledge that shouldn't age out.
*(Old memories beyond their boost window retain their full base score without penalty).*

### Stage 4: Workspace Relevance (Contextual Boost)
If the querying client's `workspace_id` exists in the memory's `memory_workspaces` mapping, apply a flat score multiplier (e.g., 1.2x). This ensures project-specific facts rise to the top without hard-siloing the data. A global or cross-project memory naturally accumulates multiple workspace associations over time as it is updated from different contexts.

### Stage 5: Time-Decayed Access Score (Working Memory Boost)
Apply a logarithmic bonus based on the memory's `access_score`. Because this score decays over time (like a half-life), it effectively measures "Working Memory." A memory read 3 times today gets a higher boost than a memory read 50 times last year.

### Stage 6: Graph Authority (In-Degree / PageRank)
Apply a slight multiplier based on the memory's "In-Degree" (how many other memories link *to* it). If an architectural decision has 15 other plans depending on it, its foundational authority naturally boosts it up the rankings.

### Stage 7: Degradation Penalty
If a memory's `status` is `stale` (e.g. an untouched, old plan) or `degraded` (e.g. references missing external files), apply a severe penalty multiplier (e.g., 0.3x) to its final score. This gracefully down-weights obsolete information without aggressively deleting it, allowing the agent to naturally reconstruct new context if needed.

## 4. The Decaying Access Score Mechanics
To prevent the "Rich Get Richer" algorithmic feedback loop, the system uses a **Time-Decayed Access Score**.

When `read_memory_record` is called:
1. The system calculates the time elapsed since `last_accessed_at`.
2. The current `access_score` is decayed using a half-life formula (e.g., losing 50% of its value every 7 days).
3. A value of `1.0` is added to the decayed score.
4. `last_accessed_at` is updated to `now()`.

## 5. Maintenance Agents (The Background Daemon)
The server runs several asynchronous background agents that keep the memory bank pristine without slowing down the active AI coder. 

### 5.1 The Strategy Roulette (Intelligent Sampling)
To prevent the maintenance agents from only optimizing highly-ranked or recently used memories (getting stuck in a local minimum), they use a **Strategy Roulette**. When an agent like the Graph Linker or Defragmenter wakes up, it randomly selects a heuristic to pull its batch of memories:
- **The "Cold Storage" Pass**: Selects the least frequently read memories or those with the longest time since `updated_at`.
- **The "Anomaly" Pass**: Selects the absolute largest memories (which may need chunking) or the smallest (which may need merging).
- **The "Never Surfaced" Pass**: Selects memories that consistently score low in search and have never been read.
- **The "Noise" Pass**: Injects pure randomness, selecting 10 random memories to find unexpected lateral connections or surface deeply buried context.
- **The "Semantic Cluster" Pass**: (Default) Selects dense groups of highly similar memories.

### 5.2 The Agent Roster
1. **`The Ingestor` (System 1 → System 2)**:
    - Wakes up when the System 1 journal has N items. It uses an LLM to group raw thoughts, queries System 2 for related context, and then creates new structured memories or appends to existing ones.
    - **Cross-Pollination**: If a thought from `workspace_B` is appended to an existing memory that only had `workspace_A`, the Ingestor adds `workspace_B` to the `memory_workspaces` junction table for that memory. This structurally records that the memory is now relevant to both workspaces.
2. **`The Summarizer`**:
    - Listens for `memories` table inserts/updates. It calls an LLM to generate or update the critical 2-sentence `summary` field used by the `search_memories` tool.
3. **`The Graph Linker`**:
    - Runs periodically using the *Strategy Roulette* to pull a batch of memories. It asks an LLM if any memories in the batch are related, and if so, creates `DEPENDS_ON` or `AMENDS` edges.
4. **`The Conflict Detector` (The Immune System)**:
    - Scans for highly similar `fact` or `plan` memories with different timestamps. It prompts an LLM to check for contradictions. If found, it creates a `CONTRADICTS` edge and flags it in the Web UI for human resolution.
5. **`The Defragmenter` (The Garbage Collector)**:
    - Uses the *Strategy Roulette* to pull batches. If it finds a dense cluster of old `journal` memories, it prompts an LLM to write a single `reflection` memory summarizing the effort, then creates `SUPERSEDES` links pointing to the old journals.
6. **`The Taxonomist`**:
    - Periodically reviews the global `tags` table and `memory_tags` mappings. It condenses and deletes redundant tags (e.g., merging `unit-test` and `testing` into `pytest`) to keep the ontology clean without requiring human intervention.
7. **`The Fact Checker`**:
    - Scans for memories containing external links (e.g., `ext:src/auth.py`). It verifies if the file still exists in the associated `workspace_id`. If the file is missing or significantly moved, it marks the dependent memories as `degraded` so they are naturally down-weighted in search.
8. **`The Project Manager`**:
    - Scans for `plan` memories that have not been modified or superseded in over 60 days. It flags them as `stale`, preventing the active AI from confidently retrieving a 6-month-old "To-Do" list as current architecture.
