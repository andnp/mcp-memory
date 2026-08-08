# Architecture Decision Record: Database Schema Design

**Status:** Active

## 1. Core Principle
SQLite is the default local backend for operational state. When shared mode
selects Postgres, Postgres is authoritative; any local SQLite state is
derivative support state rather than a second source of truth.

The schema should be:
- rigid for core memory/task/link relationships
- flexible for metadata and future agent state
- small enough to avoid carrying deleted indexing-era concepts

## 2. Active Logical Domains

### 2.1 Core Memory Content
- `memories`
  - identity, content, summary, type, status, timestamps
  - access/surfacing telemetry
  - JSON metadata for extensibility

### 2.2 Workspace and Tag Mapping
- `memory_workspaces`
- `tags`
- `memory_tags`

### 2.3 Relationships
- `links`
  - explicit typed edges between memories or explicit external targets

### 2.4 Operational Queues and Journal
- `system1_journal`
- `tasks`
- `schema_metadata`

## 3. Metadata Philosophy
The `metadata` JSON field on `memories` is the main extensibility layer.

Use it for:
- import provenance
- source-entry tracking
- agent annotations that do not justify first-class columns yet

Do not use it to hide core relational data that deserves its own table.

## 4. Telemetry Loop
Current runtime telemetry updates include:
- `last_surfaced_at` on search results
- `last_accessed_at` and `access_score` on read
- `updated_at` on explicit memory updates

## 5. Current Non-Goals
The schema no longer includes active support for:
- vector index storage
- chunk tables
- graph-store sidecar tables
- indexing-era search shadow tables

## 6. Future Work
Possible future schema additions should be driven by real runtime behavior, such as:
- richer agent annotations
- stronger link semantics
- explicit conflict-review state
- richer task observability if the dashboard requires it
