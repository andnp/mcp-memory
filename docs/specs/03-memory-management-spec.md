# Specification: Memory Management System (Relational)

**Status:** Active

## 1. Relational Schema

### 1.1 Core Tables
- **`memories`**: `id (UUID PRIMARY KEY)`, `memory_ref (unique positive integer)`, `title (TEXT)`, `content (TEXT)`, `summary (TEXT)`, `type (VARCHAR)`, `status (VARCHAR: active, stale, degraded, archived)`, `access_score (REAL)`, `last_accessed_at (DATETIME)`, `last_surfaced_at (DATETIME)`, `created_at (DATETIME)`, `updated_at (DATETIME)`, `metadata (JSON)`.
    - **Types**: `journal`, `plan`, `fact`, `observation`, `reflection`.
    - **Current metadata conventions** may include ingest lineage (`source_entry_ids`, `appended_entry_ids`, `ingest_task_id`), merge lineage (`merged_source_ids`), and split lineage (`split_group_id`, `split_from_memory_id`, `split_part_index`, `split_part_count`, `split_child_memory_ids`, `split_sibling_memory_ids`).
- **`memory_workspaces`**: `memory_id (UUID)`, `workspace_id (VARCHAR)`. (Many-to-many junction for "Soft Projects").
- **`links`**: `source_id (UUID)`, `target_id (UUID)`, `type (VARCHAR)`, `context (TEXT)`.
    - **Edge Types**: `SUPERSEDES`, `DEPENDS_ON`, `AMENDS`, `CONTRADICTS`.
- **`system1_journal`**: `id (INTEGER PRIMARY KEY)`, `content (TEXT)`, `workspace_id (VARCHAR)`, `author (VARCHAR: ai, human)`, `timestamp (DATETIME)`, `status (VARCHAR: pending, merged)`.

## 2. Progressive Discovery Tools

1. **`search_memories`**:
    - **Inputs**: `query`, `limit`, with optional `memory_type` / `status`.
    - **Workspace Context**: caller workspace is runtime context supplied by the client/proxy layer, not an agent-controlled search argument.
    - **Outputs**: Returns compact summary-first matches by default: `memory_ref`, title, and summary. `memory_ref` is rendered as `mem-<number>` for agents; canonical UUIDs remain available in debug/audit fields.
    - **Telemetry**: Updates `last_surfaced_at` for all returned results in one batch write.
2. **`read_memory`**:
    - **Inputs**: `memory_ref` or a legacy UUID.
    - **Outputs**: Returns only the stable reference, title, and full record content by default. Relationship edges, superseded breadcrumbs, and metadata/workspace routing fields are explicit opt-ins for callers that need traceability.
    - **Side Effect**: Decays `access_score` and adds `+1.0` (Working Memory boost).
3. **`read_memory_records`**:
    - **Inputs**: Up to 20 `memory_ref` values or legacy UUIDs.
    - **Outputs**: Returns all found records and a separate missing-reference list without aborting the batch.
4. **`record_thought`**: Quickly stashes raw context into System 1.

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
- when keyword candidates exist, semantic-only candidates are penalized and low keyword-coverage matches are damped so exact lexical intent can anchor the top of the ranking more reliably
- when no keyword candidates exist, low-confidence semantic-only result sets may abstain entirely instead of surfacing weak guesses as confident matches
- graph expansion remains a recall aid, but graph-only candidates are downranked behind direct keyword/semantic evidence so relationship traversal does not dominate routine query precision
- the ranking weights are tunable via `[search_ranking]` in `config.toml`
- search debug output can expose ranking details including keyword/semantic participation, graph authority counts, graph expansion provenance, and final score composition

## 4. Maintenance Agents (The Background Daemon)

### 4.1 The Strategy Roulette (Intelligent Sampling)
Target behavior for entropy-capable maintenance agents:

When a maintenance agent wakes up, it should select a batch of memories using one of these heuristics:
- **Semantic Cluster**: Dense groups of similar memories (for merging).
- **Bounded Noise**: Random sample from under-touched, non-recently-mutated candidates (to find lateral connections without wasting runs).
- **Cold Storage**: Least recently read memories (to check for obsolescence).
- **Never Surfaced**: Memories that consistently score low in search.
- **Anomalies**: Absolute largest or smallest memories (for splitting/merging).
- **Graph Bridge**: Records that appear semantically connected to other clusters but lack explicit links.
- **Orphan / Low Support**: Records with weak graph support and little evidence of long-term value.
- **Cooldown Escape**: Records outside the recent maintenance hot set so repeated runs do not keep revisiting the same memories.
- **Conflict Frontier**: Fact/plan candidates that look potentially incompatible but are not yet linked by `CONTRADICTS`.

Entropy is a **selection-layer concern** only. The durable task queue, ingest claim/delete/release semantics, and mutation safety rails remain deterministic.

Selection design requirements:
- strategy choice should be reproducible for one task run via a task-scoped seed
- each batch acquisition should return explicit strategy metadata for observability
- sampling should be bounded and auditable rather than free-form randomness throughout the run
- agents may combine deterministic prioritization with stochastic tie-breaking or seed selection

Suggested initial assignment:
- `memory-curator`: anomaly, cold-storage, never-surfaced, orphan/low-support, bounded-noise, semantic, graph-bridge, conflict-frontier, and cooldown-escape
- `ingest-system1`: deterministic FIFO claim ordering with semantic grouping only inside the claimed batch

The curator is the single production campaign for existing-memory content and graph cleanup. The historical specialist names remain policy and routing concepts, but are not independently scheduled or manually executed.

### 4.2 The Agent Roster

Existing-memory curation has one production entrypoint: `memory-curator`. The specialist roles below describe policy ownership and review routing inside that campaign; they are not independent cleanup campaigns.

1. **The Ingestor**: Flushes System 1 → System 2. In the current agentic path it claims journal batches through an internal MCP tool, performs direct MCP create/append mutations through ingest-specific internal tools, and still relies on handler-owned claim finalization so delete/release semantics stay crash-safe. It preserves and accumulates memory workspace associations as relevance metadata when thoughts from additional workspaces are incorporated.
2. **The Summarizer policy**: Maintains the 2-sentence `summary` field for all memories when curation selects summary work.
3. **The Graph Linker policy**: Discovers new semantic relationships between silos and routes approved relationship work through the curator campaign.
4. **The Conflict Detector policy**: Identifies contradictions and routes them for review without silently resolving durable claims.
5. **The Defragmenter policy**: Synthesizes clusters of old journals into single `reflection` memories and creates `SUPERSEDES` links through curator execution.
6. **The Taxonomist policy**: Normalizes and deduplicates the tag ontology within the approved curation policy.
7. **The Fact Checker policy**: Verifies `ext:` file paths still exist and routes degradation evidence through the curation policy.
8. **The Project Manager policy**: Flags `plan` memories older than 60 days as `stale` through the appropriate review route.
9. **The Sweeper**: Purges telemetry (`tasks`, `journal`) older than 7 days.
    - Also reports lineage/relationship hotspots before cleanup: active split originals, oversized lineage metadata, and high relationship-density memories.
    - Metadata garbage collection for dead internal maintenance keys must stay backend-parity across SQLite and Postgres.

Potential future addition: **The Metadata Hygienist**. This should be a bounded maintenance task, not an always-on request-path cleaner. It may normalize lineage metadata, prune obsolete bookkeeping keys, and compact oversized audit payloads when doing so improves retrieval/dashboard quality. It must preserve auditability for ingest, merge, split, and workspace-routing decisions.

Trusted maintenance agents currently include fully agentic curator, deduplicator, and ingest workflows backed by the internal MCP maintenance surface.

### 4.2.1 Agentic Mutation Guardrails
- agentic maintenance prompts should optimize for small-to-medium focused memories rather than giant multi-topic records
- durable long-term knowledge is preferred; transient execution chatter, debugging narration, and one-off status updates should stay separate or be ignored unless they carry reusable value
- mutations are optional, not mandatory; no-op outcomes are valid when a change would not improve retrieval quality or coherence
- agents must preserve project / product / repository boundaries and should not combine records merely because they share generic engineering words like `architecture`, `infra`, `testing`, `roadmap`, or `migration`
- when the scope is mixed or uncertain, agents should prefer read/search, split, or link operations over forcing a merge or append

### 4.2.2 Provider Token Economics
- provider billing is token-based; authoritative input, output, cache, reasoning, and total token telemetry is the cost measure
- provider-call counts remain useful throughput measures, so the daemon should optimize for **useful work completed per provider call and per token**
- a provider maintenance run should behave like a long-lived adaptive working session that can inspect, mutate, reprioritize, and continue through internal tools while it is still productive
- deterministic prep is valuable only when it increases same-call yield or reduces repeated context and token use; latency reduction alone is not sufficient justification
- compatibility-group continuation and richer work packets are preferred when they allow one provider run to finish more related work safely

### 4.3 Current Maintenance Notes
- current runtime is still mostly deterministic and recency-biased for candidate acquisition; full strategy roulette is planned but not yet broadly implemented
- `memory-curator` currently adds limited anomaly pressure for oversized memories, including a periodic larger-memory pass keyed from task identity
- deduplicator observation absorption now stays deterministic; provider-assisted rewriting is reserved for fact-to-fact merges
- split maintenance preserves richer lineage structure through shared split-group metadata, part ordering, sibling ids, and original child-set metadata
- sweeper diagnostics now surface lineage churn warning signs before metadata GC so operators can see whether oversized metadata or dense auto-linking may be driving token pressure
