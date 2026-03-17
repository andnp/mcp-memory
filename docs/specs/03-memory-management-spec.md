# Specification: Memory Management System (Relational)

**Status:** Active runtime baseline

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
    - **Inputs**: `query`, `limit`, with optional `memory_type` / `status`.
    - **Workspace Context**: caller workspace is runtime context supplied by the client/proxy layer, not an agent-controlled search argument.
    - **Outputs**: Returns metadata and summary-first matches.
    - **Telemetry**: Updates `last_surfaced_at` for all returned results in one batch write.
2. **`read_memory`**:
    - **Inputs**: `memory_id`.
    - **Outputs**: Returns full content, relationships, and a superseded breadcrumb trail.
    - **Side Effect**: Decays `access_score` and adds `+1.0` (Working Memory boost).
3. **`record_thought`**: Quickly stashes raw context into System 1.

## 3. Search Ranking & Scoring Pipeline (The 10-Stage Gauntlet)

The active runtime now uses a staged ranking pipeline:

1. **Keyword Candidate Retrieval**: uses SQLite FTS5 / weighted BM25 over `title`, `summary`, `content`, and `tags`.
2. **Semantic Candidate Retrieval**: optional local embedding/vector retrieval participates when the semantic layer is available.
3. **Soft Workspace-Aware Semantic Ordering**: semantic candidates are still global-first, but workspace-local candidates are softly preferred during semantic ordering instead of relying only on later-stage boosts.
4. **RRF Fusion**: combines keyword and semantic candidate rankings into one reciprocal-rank score space.
5. **Sigmoid Calibration**: converts the fused RRF score into a bounded `0.0 - 1.0` confidence score.
6. **Type-Aware Recency Boost**: adds an exponentially decaying freshness bonus based on the candidate's actual `type`.
7. **Canonical Graph-Support Bonus**: adds a small bounded bonus for stable memories (`fact`, `observation`, `reflection`) that have supporting incoming links.
8. **Workspace + Working-Memory Adjustment**: applies workspace boost and a time-decayed access bonus, while dampening unsupported transient `plan` / `journal` memories so they do not dominate as easily.
9. **Graph-Aware Authority + Expansion**: authority uses link-type-aware weighting (`DEPENDS_ON` / `AMENDS` stronger than `CONTRADICTS`, `SUPERSEDES` not treated as trust), and the engine may expand one hop from the strongest primary matches with discounted scores.
10. **Degradation Penalty**: multiplies stale / degraded records by `0.3x` to push them toward the bottom.

### 3.2 Workspace Semantics Guardrail
- search remains global-first even when workspace context is present
- workspace context is used for ranking bias only, not silent result filtering
- explicit workspace filtering belongs to dashboards and analytics, not normal memory retrieval

### 3.1 Current Search Notes
- keyword retrieval uses weighted BM25 over `title`, `summary`, `content`, and `tags`
- semantic retrieval is optional and only participates when the local embedder/vector store is configured
- semantic candidate ordering now softly prefers workspace-local memories without turning workspace into a hidden filter
- the explicit `memory_type` query argument no longer applies a separate compatibility boost; ranking now relies on the staged pipeline and type-aware recency only
- the ranking weights are tunable via `[search_ranking]` in `config.toml`
- search debug output can expose ranking details including keyword/semantic participation, graph authority counts, graph expansion provenance, and final score composition

## 4. Maintenance Agents (The Background Daemon)

### 4.1 The Strategy Roulette (Intelligent Sampling)
When an agent wakes up, it selects a batch of memories using one of these heuristics:
- **Semantic Cluster**: Dense groups of similar memories (for merging).
- **Pure Noise**: 10 random memories (to find lateral connections).
- **Cold Storage**: Least recently read memories (to check for obsolescence).
- **Never Surfaced**: Memories that consistently score low in search.
- **Anomalies**: Absolute largest or smallest memories (for splitting/merging).

### 4.2 The Agent Roster
1. **The Ingestor**: Flushes System 1 → System 2. In the current agentic path it claims journal batches through an internal MCP tool, performs direct MCP create/append mutations, and still relies on handler-owned claim finalization so delete/release semantics stay crash-safe. It preserves and accumulates memory workspace associations as relevance metadata when thoughts from additional workspaces are incorporated.
2. **The Summarizer**: Maintains the 2-sentence `summary` field for all memories.
3. **The Graph Linker**: Discovers new semantic relationships between silos using the Strategy Roulette.
4. **The Conflict Detector**: Identifies contradictions and flags them in the Web UI inbox.
5. **The Defragmenter**: Synthesizes clusters of old journals into single `reflection` memories and creates `SUPERSEDES` links.
6. **The Taxonomist**: Normalizes and deduplicates the tag ontology autonomously.
7. **The Fact Checker**: Verifies `ext:` file paths still exist; marks memories as `degraded` if missing.
8. **The Project Manager**: Flags `plan` memories older than 60 days as `stale`.
9. **The Sweeper**: Purges telemetry (`tasks`, `journal`) older than 7 days.

Trusted maintenance agents currently include fully agentic curator, deduplicator, and ingest workflows backed by the internal MCP maintenance surface.
