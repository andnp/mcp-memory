# Architecture Decision Record: Database Schema & Future-Proofing

## 1. Context
The `mcp-memory` server relies on a single, global SQLite database for all state. As the system evolves, we will invent new background agents, new scoring heuristics (like the Strategy Roulette), and new ways to track memory health. 

If we strictly type every new idea as a dedicated SQLite column, we will constantly be running database migrations and breaking backward compatibility. We need a schema that is **structurally rigid for core relationships** but **highly flexible for telemetry and agentic state**.

## 2. Core Schema Design

The database is divided into three logical domains: Core Content, Graph Relationships, and Telemetry/Extensibility.

### Domain A: Core Content
*These fields are fundamental to the existence of a memory and are strongly typed for indexing and fast retrieval.*

**Table: `memories`**
- `id`: `UUID PRIMARY KEY`
- `title`: `TEXT`
- `content`: `TEXT` (The full markdown)
- `summary`: `TEXT` (The 2-sentence summary for search)
- `type`: `VARCHAR` (e.g., `fact`, `plan`, `journal`)
- `status`: `VARCHAR` (e.g., `active`, `stale`, `degraded`, `archived`)
- `created_at`: `DATETIME`
- `updated_at`: `DATETIME` (Tracks edits/writes)

### Domain B: Graph & Indexing
*These tables map the relationships between memories and their external contexts.*

**Table: `memory_workspaces`** (Many-to-Many)
- `memory_id`: `UUID`
- `workspace_id`: `VARCHAR`

**Table: `links`** (The Edges)
- `source_id`: `UUID`
- `target_id`: `UUID`
- `type`: `VARCHAR` (`SUPERSEDES`, `DEPENDS_ON`, `CONTRADICTS`, etc.)
- `context`: `TEXT` (The anchor text explaining *why* the link exists)

**Table: `tags` & `memory_tags`**
- Standard many-to-many tag ontology.

### Domain C: Telemetry & Extensibility (The Future-Proofing Layer)
*This is where the magic happens. We must track how the AI uses the memory bank to fuel the background agents.*

**Updates to `memories` table:**
- `last_accessed_at`: `DATETIME` (When was `read_memory` last called on this ID?)
- `last_surfaced_at`: `DATETIME` (When was this last returned as a result in `search_memories`?)
- `access_score`: `REAL` (The time-decayed working memory score).

**The Extensibility Column:**
- `metadata`: `JSON` (or `TEXT` storing JSON).
  * **Why?** This is our ultimate future-proofing tool. If we decide tomorrow that we want to track the "average_search_score", or if the `Taxonomist` agent wants to leave a hidden note like `"needs_review": true`, it goes into the `metadata` JSON blob. We do not need a SQL schema migration for agentic scratchpad data. SQLite 3.38+ has excellent built-in JSON functions (`json_extract`), meaning we can still query against these flexible fields if needed.

## 3. The Telemetry Loop
To support the Strategy Roulette, the system must maintain these metrics accurately:

1. **On `search_memories`**:
   - The Orchestrator returns a list of results.
   - For every memory returned, it updates `last_surfaced_at = NOW()`.
   - (Optional) It can append the search score to a running average inside the `metadata` JSON blob.

2. **On `read_memory`**:
   - Updates `last_accessed_at = NOW()`.
   - Recalculates and updates the `access_score` (decaying the old value and adding +1).

3. **On `update_memory` / `append_memory`**:
   - Updates `updated_at = NOW()`.

## 4. Why This Works
When the `Defragmenter` wakes up and spins the "Strategy Roulette", it has exactly the data it needs:
- **"Cold Storage" Pass**: `SELECT * FROM memories ORDER BY last_accessed_at ASC`
- **"Never Surfaced" Pass**: `SELECT * FROM memories WHERE last_surfaced_at IS NULL OR last_surfaced_at < date('-30 days')`
- **"Anomaly" Pass**: `SELECT * FROM memories ORDER BY length(content) DESC` (Largest)

And if we invent a new pass tomorrow (e.g., "The Agent Confidence Pass"), the coding agents can simply read/write to the `metadata` JSON field without needing to write complex Alembic/SQLite migration scripts.
