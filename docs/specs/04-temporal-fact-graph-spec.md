# Specification: Temporal Fact Graph (Relational)

**Status:** Active relational-link baseline with future expansion points.

## 1. Overview
The memory graph is represented by explicit typed rows in the `links` table.

This allows:
- lineage tracking through `SUPERSEDES`
- dependency tracking through explicit typed links
- progressive historical discovery through `read_memory_record`
- durable relationships independent of memory titles or markdown syntax

## 2. Current Relationship Model

### 2.1 Storage
Relationships live in SQLite.

- `source_id`
- `target_id`
- `type`
- `context`

### 2.2 Active Retrieval Behavior
- `search_memory_records` hides superseded memories by default
- `read_memory_record` returns incoming/outgoing relationships
- `read_memory_record` also returns superseded breadcrumbs

### 2.3 Current Link Authoring Boundary
The runtime stores typed links relationally, but link authoring is not part of the minimal public MCP surface.

Current behavior:
- links may target another memory ID or an explicit `ext:` target
- link metadata is stored relationally, not inferred from markdown syntax
- lineage is primarily surfaced through reads, not through standalone graph-management tools

## 3. Current Typed Link Use Cases
Common active link types include:
- `SUPERSEDES`
- `DEPENDS_ON`
- `REFERENCES`
- `CONTRADICTS`
- `EXTENDS`

The system does not currently enforce a closed enum in the database layer, but clients should prefer stable uppercase relationship names.

## 4. Current Non-Goals
The current runtime does **not** assume:
- wikilink parsing
- graph-store sidecar persistence
- automatic graph-link inference as part of every write

## 5. Future Work
Possible future additions:
- Graph Linker agent
- Conflict Detector agent
- lineage-specific UI affordances
- stronger link-type normalization rules
- recursive lineage traversal helpers
