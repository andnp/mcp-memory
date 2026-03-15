# Specification: Memory Management System (Relational)

**Status:** Active runtime baseline

## 1. Scope

This document describes the memory system that currently backs the shipped `mcp-memory` runtime.

- relational-first storage
- shared global SQLite database
- workspace-aware retrieval and background maintenance
- explicit typed links stored in the `links` table
- no indexing-era vector, chunking, or wikilink runtime behavior

## 2. Active Schema

### 2.1 Core Tables
- `memories`
  - `id`, `title`, `content`, `summary`, `type`, `status`, `created_at`, `updated_at`
  - `access_score`, `last_accessed_at`, `last_surfaced_at`, `metadata`
- `memory_workspaces`
  - maps a memory to one or more `workspace_id` values
- `links`
  - typed relationships between memories or explicit external targets
  - fields: `source_id`, `target_id`, `type`, `context`
- `tags`
- `memory_tags`
- `system1_journal`
  - raw thought capture before ingest
- `tasks`
  - durable background task queue
- `schema_metadata`

### 2.2 Status Values
- memory status
  - `active`
  - `stale`
  - `degraded`
  - `archived`
- memory types
  - `journal`
  - `plan`
  - `fact`
  - `observation`
  - `reflection`

## 3. Active MCP Tools

- `record_thought`
- `search_memory_records`
- `read_memory_record`

The public MCP surface is intentionally minimal:

- one lightweight capture tool
- one discovery tool
- one detail tool

Administrative inspection, migration/import, and storage-oriented CRUD are not part of the assistant-facing contract.

## 4. Search Behavior

### 4.1 Current Ranking Model
The active relational search path is intentionally simple.

- token-based matching across title, summary, content, and tags
- additive type-aware recency boost
- time-decayed access-score boost
- workspace match multiplier
- stale/degraded penalty
- superseded memories hidden by default

### 4.2 Current Non-Goals
The following are **not** part of the current shipped search path:

- vector retrieval
- FAISS
- chunk retrieval
- RRF fusion
- sigmoid calibration
- graph-authority ranking

These may be added later, but they are not current runtime requirements.

## 5. Access Score Mechanics

When `read_memory_record` is called:

1. the current access score is decayed based on elapsed time
2. the decayed score is incremented
3. `last_accessed_at` is updated
4. the refreshed record is returned with relationships and superseded breadcrumbs

## 6. Background Maintenance

### 6.1 Currently Implemented
- Ingestor
  - promotes grouped System 1 thoughts into relational observation memories
- Summarizer
  - updates the search summary field for created memories
- Fact Checker
  - marks memories as `degraded` when explicit `ext:` targets no longer resolve
- Project Manager
  - marks old plans as `stale`
- Sweeper
  - purges old task/journal telemetry

### 6.2 Not Yet Implemented
- Graph Linker
- Conflict Detector
- Defragmenter
- Taxonomist
- strategy roulette sampling
- cross-workspace append/merge behavior during ingest

## 7. Markdown Import

The markdown importer supports:

- YAML frontmatter parsing
- title derivation from filename when needed
- type/status/tag normalization
- import provenance metadata
- assignment of imported memories to explicit workspace IDs

The importer does **not** currently perform automatic relationship extraction from markdown bodies.

Markdown import is a maintenance/bootstrap capability, not a public MCP tool.

## 8. Future Work

Potential future additions:

- richer calibrated search pipeline
- explicit relational-link authoring workflows
- richer ingest merge behavior
- advanced maintenance agents
- broader provider support
