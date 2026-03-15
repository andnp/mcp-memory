# Specification: Memory Management System (Relational)

**Status:** ✅ Design Complete (Ready for Implementation)

## 1. Relational Schema

### 1.1 Core Tables
- **`memories`**: `id (UUID PRIMARY KEY)`, `title (TEXT)`, `content (TEXT)`, `summary (TEXT)`, `type (VARCHAR)`, `status (VARCHAR: active, stale, degraded, archived)`, `access_score (REAL)`, `last_accessed_at (DATETIME)`, `last_surfaced_at (DATETIME)`, `created_at (DATETIME)`, `updated_at (DATETIME)`, `metadata (JSON)`.
    - **Types**: `journal`, `plan`, `fact`, `observation`, `reflection`.
- **`memory_workspaces`**: `memory_id (UUID)`, `workspace_id (VARCHAR)`. (Many-to-many junction for "Soft Projects").
- **`links`**: `source_id (UUID)`, `target_id (UUID)`, `type (VARCHAR)`, `context (TEXT)`.
    - **Edge Types**: `SUPERSEDES`, `DEPENDS_ON`, `AMENDS`, `CONTRADICTS`.
- **`system1_journal`**: `id (INTEGER PRIMARY KEY)`, `content (TEXT)`, `workspace_id (VARCHAR)`, `author (VARCHAR: ai, human)`, `timestamp (DATETIME)`, `status (VARCHAR: pending, merged)`.

## 2. Progressive Discovery Tools

1. **`search_memories`**: 
    - **Inputs**: `query`, `limit`. (`workspace_id` injected by proxy).
    - **Outputs**: Returns metadata and a **two-sentence summary** of matches.
    - **Telemetry**: Updates `last_surfaced_at` for all matches.
2. **`read_memory`**: 
    - **Inputs**: `memory_id`.
    - **Outputs**: Returns **full content**, relationships, and a list of **superseded IDs** (breadcrumb trail).
    - **Side Effect**: Decays `access_score` and adds `+1.0` (Working Memory boost).
3. **`record_thought`**: Quickly stashes raw context into System 1.

## 3. Search Ranking & Scoring Pipeline (The 7-Stage Gauntlet)

1. **RRF Fusion**: Combines Vector (FAISS) and Keyword (FTS5) rankings.
2. **Sigmoid Calibration**: Expands scores to an absolute `0.0 - 1.0` range.
3. **Type-Aware Recency Boost**: Exponentially decaying bonus based on `type` (Journals/Plans decay fast; Facts/Reflections stay evergreen).
4. **Workspace Boost**: Flat `1.2x` multiplier if the memory is linked to the current `workspace_id`.
5. **Access Score Boost**: Logarithmic bonus based on the `access_score` (favoring "Working Memory").
6. **Graph Authority**: Slight multiplier based on "In-Degree" (how many memories link *to* this one).
7. **Degradation Penalty**: Severe `0.3x` penalty if status is `stale` or `degraded`.

## 4. Maintenance Agents (The Background Daemon)

### 4.1 The Strategy Roulette (Intelligent Sampling)
When an agent wakes up, it selects a batch of memories using one of these heuristics:
- **Semantic Cluster**: Dense groups of similar memories (for merging).
- **Pure Noise**: 10 random memories (to find lateral connections).
- **Cold Storage**: Least recently read memories (to check for obsolescence).
- **Never Surfaced**: Memories that consistently score low in search.
- **Anomalies**: Absolute largest or smallest memories (for splitting/merging).

### 4.2 The Agent Roster
1. **The Ingestor**: Flushes System 1 → System 2. Cross-pollinates `memory_workspaces` if a thought is appended to a memory from a different project.
2. **The Summarizer**: Maintains the 2-sentence `summary` field for all memories.
3. **The Graph Linker**: Discovers new semantic relationships between silos using the Strategy Roulette.
4. **The Conflict Detector**: Identifies contradictions and flags them in the Web UI inbox.
5. **The Defragmenter**: Synthesizes clusters of old journals into single `reflection` memories and creates `SUPERSEDES` links.
6. **The Taxonomist**: Normalizes and deduplicates the tag ontology autonomously.
7. **The Fact Checker**: Verifies `ext:` file paths still exist; marks memories as `degraded` if missing.
8. **The Project Manager**: Flags `plan` memories older than 60 days as `stale`.
9. **The Sweeper**: Purges telemetry (`tasks`, `journal`) older than 7 days.
