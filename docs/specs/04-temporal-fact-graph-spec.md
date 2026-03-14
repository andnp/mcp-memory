# Specification: Temporal Fact Graph (Relational)

## 1. Overview
The **Temporal Fact Graph** transforms the memory bank from a flat storage system into a lineage-aware knowledge base. By moving from filename-based wikilinks to **UUID-based relational edges**, we ensure that relationships remain durable even as content is renamed, refactored, or merged.

## 2. Problem Statement
Flat memory systems struggle with conflicting information (e.g., "Decision: Use SQLite" vs. "Decision: Move to PostgreSQL"). A relational graph allows the system to structuralize "current truth" versus "historical context" using explicit, typed relationships.

## 3. Relational Relationship Model
Relationships are stored in a dedicated `links` table in SQLite, separate from the memory content.

### 3.1 Edge Types (The Schema)
The system supports the following typed relationships between memory IDs:

| Edge Type | Description |
|-----------|-------------|
| `SUPERSEDES` | The source memory is the new "truth"; the target is now historical context. |
| `DEPRECATES` | The source memory marks the target as obsolete without a direct replacement. |
| `AMENDS` | The source memory adds detail or updates to the target without fully replacing it. |
| `DEPENDS_ON` | The source memory requires the target for context or logical consistency. |
| `CONTRADICTS` | An undirected edge indicating a potential factual conflict detected by the agent. |

## 4. Lineage-Aware Retrieval

### 4.1 Default Search (Hiding the Past)
The `search_memories` tool automatically filters or heavily deprioritizes any memory that has an incoming `SUPERSEDES` edge. This guarantees that when an active AI asks for the "current architecture," it only sees the Head of the version chain, keeping its context window clean.

### 4.2 Progressive Historical Discovery (The Breadcrumb Trail)
While old memories are hidden from default search, they are **never deleted**. The system surfaces historical context progressively via the `read_memory` tool:
- When an AI calls `read_memory` on a current fact (e.g., the "March CSS Reflection"), the tool returns the full content **plus a list of all memories that this fact `SUPERSEDES`** (e.g., `journal_a.md`, `journal_b.md`).
- If the AI needs to understand *why* a decision was made, it can explicitly call `read_memory` on those superseded IDs, walking backward through the lineage tree as far as needed.

### 4.3 SQL-Driven Traversal
Because links are stored in a relational table, the system can perform recursive CTE (Common Table Expression) queries to:
- Find the "Head" of a version chain.
- Retrieve the full audit trail for a specific decision.
- Identify "Orphaned" memories that have no relationships.

## 5. Agentic Graph Management
Maintenance agents use specialized tools to manage the graph:

- **`GraphLinker`**: A background agent that periodically performs a "Relationship Sweep," identifying missing links between memories using vector similarity and LLM verification.
- **`ConflictDetector`**: A specialized agent that looks for memories with high semantic similarity but conflicting claims, creating `CONTRADICTS` edges for human or autonomous resolution.
- **`Defragmenter`**: A specialized agent that writes summary reflections of old journals and creates `SUPERSEDES` links to tuck the old journals away into the historical record.

## 6. Goal
To provide a self-healing knowledge graph where the AI can understand the evolution of its own reasoning, ensuring that the "current truth" is always what is presented by default, while the complete historical context is just one `read_memory` call away.
