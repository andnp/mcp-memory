from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime
from uuid import uuid4

from mcp_memory.core.ports.memory import (
    FTS_QUERY_TOKEN_PATTERN,
    MemoryLink,
    MemoryReadContext,
    MemoryRecord,
    RankedMemoryCandidate,
    VALID_MEMORY_STATUSES,
    VALID_MEMORY_TYPES,
    _DEFAULT_CANDIDATE_LIMIT,
    _DEFAULT_LOW_SUPPORT_MAX,
    _DEFAULT_OVERSIZED_CANDIDATE_MIN_CHARS,
    _DEFAULT_QUALITY_OVERSIZED_MIN_CHARS,
    _DEFAULT_THIN_CANDIDATE_MAX_CHARS,
    _QUALITY_SIGNAL_ALIASES,
    build_memory_summary,
    build_read_cache_validation_token,
    parse_memory_ref,
)
from mcp_memory.utils.db import DatabaseManager

_SUMMARY_UNSET = object()
_NONCRITICAL_WRITE_TIMEOUT_SECONDS = 0.1


logger = logging.getLogger(__name__)


RelationalMemoryRecord = MemoryRecord
RelationalMemoryReadContext = MemoryReadContext


class RelationalMemoryRepository:
    def __init__(self, db_manager: DatabaseManager) -> None:
        self._db = db_manager

    def get_search_epochs(self) -> dict[str, int]:
        rows = self._db.get_connection().execute(
            "SELECT key, value FROM schema_metadata WHERE key IN (?, ?, ?)",
            (
                "search_epoch_keyword",
                "search_epoch_vector",
                "search_epoch_graph",
            ),
        ).fetchall()
        values = {str(row[0]): int(row[1]) for row in rows}
        return {
            "keyword": values["search_epoch_keyword"],
            "vector": values["search_epoch_vector"],
            "graph": values["search_epoch_graph"],
        }

    def get_read_cache_validation_tokens(self, memory_ids: list[str]) -> dict[str, str]:
        tokens: dict[str, str] = {}
        for memory_id in self._normalize_values(memory_ids):
            record = self.get_memory(memory_id)
            if record is None:
                continue
            outgoing = self.get_links(memory_id, direction="outgoing")
            incoming = self.get_links(memory_id, direction="incoming")
            superseded_targets = [
                target
                for link in outgoing
                if link.link_type == "SUPERSEDES"
                for target in [self.get_memory(link.target_id)]
                if target is not None
            ]
            tokens[memory_id] = build_read_cache_validation_token(
                record=record,
                outgoing_links=outgoing,
                incoming_links=incoming,
                superseded_records=superseded_targets,
            )
        return tokens

    def get_searchable_memories(
        self,
        memory_ids: list[str],
        *,
        status: str | None = None,
        include_superseded: bool = False,
    ) -> list[RelationalMemoryRecord]:
        normalized_ids = self._normalize_values(memory_ids)
        if not normalized_ids:
            return []
        resolved_ids = [
            resolved_id
            for memory_id in normalized_ids
            for resolved_id in [self.resolve_memory_id(memory_id)]
            if resolved_id is not None
        ]
        if not resolved_ids:
            return []

        conn = self._db.get_connection()
        placeholders = ",".join("?" for _ in resolved_ids)
        clauses = [f"memories.id IN ({placeholders})"]
        params: list[object] = [*resolved_ids]
        if status is not None:
            clauses.append("memories.status = ?")
            params.append(status)
        if not include_superseded:
            clauses.append(
                "NOT EXISTS (SELECT 1 FROM links supersedes WHERE supersedes.target_id = memories.id AND supersedes.type = 'SUPERSEDES')"
            )

        rows = conn.execute(
            """
            SELECT
                memories.*,
                COALESCE(workspace_agg.workspace_ids, '') AS workspace_ids_csv,
                COALESCE(tag_agg.tags, '') AS tags_csv
            FROM memories
            LEFT JOIN (
                SELECT memory_id, GROUP_CONCAT(DISTINCT workspace_id) AS workspace_ids
                FROM memory_workspaces
                GROUP BY memory_id
            ) workspace_agg ON workspace_agg.memory_id = memories.id
            LEFT JOIN (
                SELECT memory_tags.memory_id, GROUP_CONCAT(DISTINCT tags.name) AS tags
                FROM memory_tags
                JOIN tags ON tags.id = memory_tags.tag_id
                GROUP BY memory_tags.memory_id
            ) tag_agg ON tag_agg.memory_id = memories.id
            WHERE
            """
            + " AND ".join(clauses),
            params,
        ).fetchall()

        records_by_id: dict[str, RelationalMemoryRecord] = {}
        for row in rows:
            metadata = json.loads(row["metadata"] or "{}")
            records_by_id[str(row["id"])] = RelationalMemoryRecord(
                id=row["id"],
                memory_ref=row["memory_ref"],
                title=row["title"],
                content=row["content"],
                summary=row["summary"],
                type=row["type"],
                status=row["status"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                archived_at=row["archived_at"],
                read_count=int(row["read_count"] or 0),
                access_score=row["access_score"],
                last_accessed_at=row["last_accessed_at"],
                last_surfaced_at=row["last_surfaced_at"],
                metadata=metadata,
                workspace_ids=_split_csv_values(row["workspace_ids_csv"]),
                tags=_split_csv_values(row["tags_csv"]),
            )
        return [records_by_id[memory_id] for memory_id in resolved_ids if memory_id in records_by_id]

    def create_memory(
        self,
        title: str,
        content: str,
        workspace_ids: list[str],
        tags: list[str] | None = None,
        summary: str | None = None,
        memory_type: str = "journal",
        status: str = "active",
        metadata: dict[str, object] | None = None,
        memory_id: str | None = None,
        created_at: str | None = None,
        updated_at: str | None = None,
    ):
        now = self._utc_now()
        created_timestamp = created_at or now
        updated_timestamp = updated_at or created_timestamp
        normalized_title = title.strip()
        normalized_content = content.strip()
        normalized_workspace_ids = self._normalize_values(workspace_ids)
        normalized_tags = self._normalize_values(tags or [])
        normalized_type = self._validate_memory_type(memory_type)
        normalized_status = self._validate_memory_status(status)
        archived_timestamp = self._utc_now() if normalized_status == "archived" else None
        self._validate_required_text("title", normalized_title)
        self._validate_required_text("content", normalized_content)
        if not normalized_workspace_ids:
            raise ValueError("workspace_ids must contain at least one non-empty value")

        summary_text = summary or self._build_summary(
            title=normalized_title,
            content=normalized_content,
            memory_type=normalized_type,
        )
        payload = json.dumps(metadata or {}, sort_keys=True)
        record_id = memory_id or str(uuid4())

        conn = self._db.get_connection()
        with conn:
            next_ref_row = conn.execute(
                "SELECT COALESCE(MAX(memory_ref), 0) + 1 FROM memories"
            ).fetchone()
            memory_ref = int(next_ref_row[0]) if next_ref_row is not None else 1
            conn.execute(
                """
                INSERT INTO memories (
                    id, memory_ref, title, content, summary, type, status, created_at, updated_at,
                    archived_at, read_count, access_score, last_accessed_at, last_surfaced_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record_id,
                    memory_ref,
                    normalized_title,
                    normalized_content,
                    summary_text,
                    normalized_type,
                    normalized_status,
                    created_timestamp,
                    updated_timestamp,
                    archived_timestamp,
                    0,
                    0.0,
                    None,
                    None,
                    payload,
                ),
            )
            self._replace_workspace_mappings(conn, record_id, normalized_workspace_ids)
            self._replace_tag_mappings(conn, record_id, normalized_tags)
            self._replace_fts_row(
                conn,
                record_id,
                title=normalized_title,
                summary=summary_text,
                content=normalized_content,
                tags=normalized_tags,
            )

        return self.get_memory(record_id)

    def add_link(
        self,
        source_id: str,
        target_id: str,
        link_type: str,
        context: str = "",
    ):
        normalized_link_type = self._normalize_link_type(link_type)
        conn = self._db.get_connection()
        with conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO links (source_id, target_id, type, context)
                VALUES (?, ?, ?, ?)
                """,
                (source_id, target_id, normalized_link_type, context.strip()),
            )
        return MemoryLink(
            source_id=source_id,
            target_id=target_id,
            link_type=normalized_link_type,
            context=context.strip(),
        )

    def remove_link(
        self,
        source_id: str,
        target_id: str,
        link_type: str,
    ) -> bool:
        normalized_link_type = self._normalize_link_type(link_type)
        conn = self._db.get_connection()
        with conn:
            cursor = conn.execute(
                "DELETE FROM links WHERE source_id = ? AND target_id = ? AND type = ?",
                (source_id, target_id, normalized_link_type),
            )
        return cursor.rowcount > 0

    def delete_memory(self, memory_id: str):
        resolved_memory_id = self.resolve_memory_id(memory_id)
        if resolved_memory_id is None:
            return None
        existing = self.get_memory(resolved_memory_id)
        if existing is None:
            return None

        conn = self._db.get_connection()
        with conn:
            conn.execute(
                "DELETE FROM links WHERE source_id = ? OR target_id = ?",
                (resolved_memory_id, resolved_memory_id),
            )
            conn.execute("DELETE FROM memories_fts WHERE memory_id = ?", (resolved_memory_id,))
            conn.execute("DELETE FROM memories WHERE id = ?", (resolved_memory_id,))
        return existing

    def search_keyword_memory_ids(
        self,
        query: str,
        *,
        workspace_id: str | None = None,
        memory_type: str | None = None,
        status: str | None = None,
        tags: Sequence[str] | None = None,
        include_superseded: bool = False,
        limit: int = 50,
    ) -> list[str]:
        # Keep the established public query semantics unchanged; maintenance
        # search uses the private variant below for its type filter.
        return self._search_keyword_memory_ids(
            query,
            workspace_id=workspace_id,
            memory_type=None,
            status=status,
            tags=tags,
            include_superseded=include_superseded,
            limit=limit,
        )

    def _search_keyword_memory_ids(
        self,
        query: str,
        *,
        workspace_id: str | None = None,
        memory_type: str | None = None,
        status: str | None = None,
        tags: Sequence[str] | None = None,
        include_superseded: bool = False,
        limit: int = 50,
    ) -> list[str]:
        tokens = [match.group(0).lower() for match in FTS_QUERY_TOKEN_PATTERN.finditer(query)]
        normalized_query = " OR ".join(f'"{token}"' for token in tokens)
        if not normalized_query:
            return []

        conn = self._db.get_connection()
        clauses = ["memories_fts MATCH ?"]
        params: list[object] = [normalized_query]
        joins = ["JOIN memories ON memories.id = memories_fts.memory_id"]
        if workspace_id is not None:
            joins.append("JOIN memory_workspaces ON memory_workspaces.memory_id = memories.id")
            clauses.append("memory_workspaces.workspace_id = ?")
            params.append(workspace_id)
        if memory_type is not None:
            clauses.append("memories.type = ?")
            params.append(memory_type)
        if status is not None:
            clauses.append("memories.status = ?")
            params.append(status)
        for tag in tags or ():
            clauses.append(
                "EXISTS (SELECT 1 FROM memory_tags JOIN tags ON tags.id = memory_tags.tag_id "
                "WHERE memory_tags.memory_id = memories.id AND tags.name = ?)"
            )
            params.append(tag)
        if not include_superseded:
            clauses.append(
                "NOT EXISTS (SELECT 1 FROM links supersedes WHERE supersedes.target_id = memories.id AND supersedes.type = 'SUPERSEDES')"
            )

        params.append(limit)
        query_sql = (
            "SELECT DISTINCT memories.id, bm25(memories_fts, 3.0, 2.0, 1.0, 1.5) AS rank FROM memories_fts "
            + " ".join(joins)
            + " WHERE "
            + " AND ".join(clauses)
            + " ORDER BY rank ASC, memories.updated_at DESC LIMIT ?"
        )
        rows = conn.execute(query_sql, params).fetchall()
        return [str(row["id"]) for row in rows]

    def get_ranking_candidates(
        self,
        memory_ids: list[str],
        *,
        status: str | None = None,
        include_superseded: bool = False,
    ) -> list[RankedMemoryCandidate]:
        normalized_ids = self._normalize_values(memory_ids)
        if not normalized_ids:
            return []
        resolved_ids = [
            resolved_id
            for memory_id in normalized_ids
            for resolved_id in [self.resolve_memory_id(memory_id)]
            if resolved_id is not None
        ]
        if not resolved_ids:
            return []

        conn = self._db.get_connection()
        placeholders = ",".join("?" for _ in resolved_ids)
        clauses = [f"memories.id IN ({placeholders})"]
        params: list[object] = [*resolved_ids]
        if status is not None:
            clauses.append("memories.status = ?")
            params.append(status)
        if not include_superseded:
            clauses.append("COALESCE(link_counts.has_incoming_supersedes, 0) = 0")

        rows = conn.execute(
            """
            SELECT
                memories.*,
                COALESCE(workspace_agg.workspace_ids, '') AS workspace_ids_csv,
                COALESCE(tag_agg.tags, '') AS tags_csv,
                COALESCE(link_counts.incoming_links_count, 0) AS incoming_links_count,
                COALESCE(link_counts.has_incoming_supersedes, 0) AS has_incoming_supersedes,
                COALESCE(link_counts.incoming_depends_on_count, 0) AS incoming_depends_on_count,
                COALESCE(link_counts.incoming_amends_count, 0) AS incoming_amends_count,
                COALESCE(link_counts.incoming_contradicts_count, 0) AS incoming_contradicts_count,
                COALESCE(link_counts.incoming_supersedes_count, 0) AS incoming_supersedes_count
            FROM memories
            LEFT JOIN (
                SELECT memory_id, GROUP_CONCAT(DISTINCT workspace_id) AS workspace_ids
                FROM memory_workspaces
                GROUP BY memory_id
            ) workspace_agg ON workspace_agg.memory_id = memories.id
            LEFT JOIN (
                SELECT memory_tags.memory_id, GROUP_CONCAT(DISTINCT tags.name) AS tags
                FROM memory_tags
                JOIN tags ON tags.id = memory_tags.tag_id
                GROUP BY memory_tags.memory_id
            ) tag_agg ON tag_agg.memory_id = memories.id
            LEFT JOIN (
                SELECT
                    target_id AS memory_id,
                    COUNT(*) AS incoming_links_count,
                    MAX(CASE WHEN type = 'SUPERSEDES' THEN 1 ELSE 0 END) AS has_incoming_supersedes,
                    SUM(CASE WHEN type = 'DEPENDS_ON' THEN 1 ELSE 0 END) AS incoming_depends_on_count,
                    SUM(CASE WHEN type = 'AMENDS' THEN 1 ELSE 0 END) AS incoming_amends_count,
                    SUM(CASE WHEN type = 'CONTRADICTS' THEN 1 ELSE 0 END) AS incoming_contradicts_count,
                    SUM(CASE WHEN type = 'SUPERSEDES' THEN 1 ELSE 0 END) AS incoming_supersedes_count
                FROM links
                GROUP BY target_id
            ) link_counts ON link_counts.memory_id = memories.id
            WHERE
            """
            + " AND ".join(clauses),
            params,
        ).fetchall()

        ranked_by_id: dict[str, RankedMemoryCandidate] = {}
        for row in rows:
            metadata = json.loads(row["metadata"] or "{}")
            ranked_by_id[str(row["id"])] = RankedMemoryCandidate(
                record=RelationalMemoryRecord(
                    id=row["id"],
                    memory_ref=row["memory_ref"],
                    title=row["title"],
                    content=row["content"],
                    summary=row["summary"],
                    type=row["type"],
                    status=row["status"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    read_count=int(row["read_count"] or 0),
                    access_score=row["access_score"],
                    last_accessed_at=row["last_accessed_at"],
                    last_surfaced_at=row["last_surfaced_at"],
                    metadata=metadata,
                    workspace_ids=_split_csv_values(row["workspace_ids_csv"]),
                    tags=_split_csv_values(row["tags_csv"]),
                ),
                incoming_links_count=int(row["incoming_links_count"] or 0),
                has_incoming_supersedes=bool(row["has_incoming_supersedes"]),
                incoming_link_type_counts={
                    "DEPENDS_ON": int(row["incoming_depends_on_count"] or 0),
                    "AMENDS": int(row["incoming_amends_count"] or 0),
                    "CONTRADICTS": int(row["incoming_contradicts_count"] or 0),
                    "SUPERSEDES": int(row["incoming_supersedes_count"] or 0),
                },
            )
        return [ranked_by_id[memory_id] for memory_id in resolved_ids if memory_id in ranked_by_id]

    def query_maintenance_candidates(
        self,
        strategy: str,
        *,
        limit: int = _DEFAULT_CANDIDATE_LIMIT,
        workspace_id: str | None = None,
        status: str | None = None,
        include_superseded: bool = False,
        oversized_min_chars: int = _DEFAULT_OVERSIZED_CANDIDATE_MIN_CHARS,
        thin_max_chars: int = _DEFAULT_THIN_CANDIDATE_MAX_CHARS,
        low_support_max: int = _DEFAULT_LOW_SUPPORT_MAX,
        quality_signal: str | None = None,
        retrieval_min_searches: int = 2,
        retrieval_max_conversion_rate: float = 0.25,
        seed: int | str = 0,
    ) -> list[RelationalMemoryRecord]:
        """Return a bounded, globally scoped maintenance candidate frontier.

        The strategy is evaluated and ordered in SQLite, so this method does not
        load a recent window or hydrate the corpus before applying the limit.
        Results are ordered with an explicit record-id tie-breaker.  A workspace
        filter is opt-in; ``None`` means the shared global corpus.

        Supported strategies are ``cold-storage``, ``never-surfaced``,
        ``oversized/thin``, ``orphan/low-support``, ``quality-signal``, and
        ``seeded-random``.  The random strategy uses a SHA-256 key derived from
        ``seed`` and the record ID, rather than SQLite's process-dependent
        ``random()``.
        """
        normalized_strategy = strategy.strip().lower()
        strategy_aliases = {
            "cold": "cold-storage",
            "never_surfaced": "never-surfaced",
            "oversized-thin": "oversized/thin",
            "orphan-low-support": "orphan/low-support",
            "quality": "quality-signal",
            "retrieval_quality": "retrieval-quality",
            "seeded_random": "seeded-random",
        }
        normalized_strategy = strategy_aliases.get(normalized_strategy, normalized_strategy)
        supported_strategies = {
            "cold-storage",
            "never-surfaced",
            "oversized/thin",
            "orphan/low-support",
            "quality-signal",
            "retrieval-quality",
            "seeded-random",
        }
        if normalized_strategy not in supported_strategies:
            raise ValueError(f"unsupported maintenance candidate strategy: {strategy!r}")
        if limit <= 0:
            return []
        if oversized_min_chars < 0 or thin_max_chars < 0 or low_support_max < 0:
            raise ValueError("candidate thresholds must be non-negative")
        if retrieval_min_searches <= 0 or not 0 <= retrieval_max_conversion_rate <= 1:
            raise ValueError("retrieval-quality thresholds are invalid")

        clauses = ["1 = 1"]
        params: list[object] = []
        if workspace_id is not None:
            clauses.append(
                "EXISTS (SELECT 1 FROM memory_workspaces scoped_workspace "
                "WHERE scoped_workspace.memory_id = memories.id "
                "AND scoped_workspace.workspace_id = ?)"
            )
            params.append(workspace_id)
        if status is not None:
            clauses.append("memories.status = ?")
            params.append(status)
        if not include_superseded:
            clauses.append(
                "NOT EXISTS (SELECT 1 FROM links supersedes "
                "WHERE supersedes.target_id = memories.id AND supersedes.type = 'SUPERSEDES')"
            )

        order_by = "memories.updated_at ASC, memories.created_at ASC, memories.id ASC"
        if normalized_strategy == "cold-storage":
            order_by = (
                "memories.last_accessed_at IS NULL DESC, memories.last_accessed_at ASC, "
                "memories.updated_at ASC, memories.id ASC"
            )
        elif normalized_strategy == "never-surfaced":
            clauses.append("memories.last_surfaced_at IS NULL")
            order_by = "memories.updated_at ASC, memories.created_at ASC, memories.id ASC"
        elif normalized_strategy == "oversized/thin":
            clauses.append(
                "(LENGTH(COALESCE(memories.content, '')) > ? OR "
                "(LENGTH(COALESCE(memories.content, '')) <= ? "
                "AND memories.metadata LIKE '%\"split_from_memory_id\"%'))"
            )
            params.extend([oversized_min_chars, thin_max_chars])
            order_by = (
                "LENGTH(COALESCE(memories.content, '')) DESC, memories.read_count ASC, "
                "memories.updated_at ASC, memories.id ASC"
            )
        elif normalized_strategy == "orphan/low-support":
            clauses.append("COALESCE(incoming_counts.incoming_links_count, 0) <= ?")
            params.append(low_support_max)
            order_by = (
                "COALESCE(incoming_counts.incoming_links_count, 0) ASC, memories.read_count ASC, "
                "memories.access_score ASC, memories.last_surfaced_at IS NULL DESC, "
                "memories.last_surfaced_at ASC, memories.id ASC"
            )
        elif normalized_strategy == "quality-signal":
            quality_clause, quality_params = _quality_signal_clause(quality_signal)
            clauses.append(quality_clause)
            params.extend(quality_params)
            order_by = "memories.updated_at ASC, memories.created_at ASC, memories.id ASC"
        elif normalized_strategy == "retrieval-quality":
            clauses.append(
                "COALESCE(retrieval_stats.search_count, 0) >= ? AND "
                "COALESCE(retrieval_stats.read_count, 0) * 1.0 / "
                "retrieval_stats.search_count <= ?"
            )
            params.extend([retrieval_min_searches, retrieval_max_conversion_rate])
            order_by = (
                "COALESCE(retrieval_stats.search_count, 0) DESC, "
                "COALESCE(retrieval_stats.read_count, 0) ASC, memories.updated_at ASC, memories.id ASC"
            )
        else:
            conn = self._db.get_connection()
            conn.create_function("stable_seeded_random_key", 2, _stable_seeded_random_key, deterministic=True)
            order_by = "stable_seeded_random_key(?, memories.id) ASC, memories.id ASC"
            params.append(str(seed))

        conn = self._db.get_connection()
        rows = conn.execute(
            """
            SELECT
                memories.*,
                COALESCE(workspace_agg.workspace_ids, '') AS workspace_ids_csv,
                COALESCE(tag_agg.tags, '') AS tags_csv
            FROM memories
            LEFT JOIN (
                SELECT memory_id, GROUP_CONCAT(DISTINCT workspace_id) AS workspace_ids
                FROM memory_workspaces
                GROUP BY memory_id
            ) workspace_agg ON workspace_agg.memory_id = memories.id
            LEFT JOIN (
                SELECT memory_tags.memory_id, GROUP_CONCAT(DISTINCT tags.name) AS tags
                FROM memory_tags
                JOIN tags ON tags.id = memory_tags.tag_id
                GROUP BY memory_tags.memory_id
            ) tag_agg ON tag_agg.memory_id = memories.id
            LEFT JOIN (
                SELECT target_id AS memory_id, COUNT(*) AS incoming_links_count
                FROM links
                GROUP BY target_id
            ) incoming_counts ON incoming_counts.memory_id = memories.id
            LEFT JOIN (
                SELECT memory_id,
                       SUM(CASE WHEN event_kind = 'search' THEN 1 ELSE 0 END) AS search_count,
                       SUM(CASE WHEN event_kind = 'read' THEN 1 ELSE 0 END) AS read_count
                FROM memory_tool_events
                WHERE memory_id IS NOT NULL
                GROUP BY memory_id
            ) retrieval_stats ON retrieval_stats.memory_id = memories.id
            WHERE
            """
            + " AND ".join(clauses)
            + " ORDER BY "
            + order_by
            + " LIMIT ?",
            [*params, limit],
        ).fetchall()
        return [self._candidate_record_from_row(row) for row in rows]

    def query_cold_candidates(
        self,
        *,
        limit: int = _DEFAULT_CANDIDATE_LIMIT,
        workspace_id: str | None = None,
        status: str | None = None,
        include_superseded: bool = False,
    ) -> list[RelationalMemoryRecord]:
        return self.query_maintenance_candidates(
            "cold-storage",
            limit=limit,
            workspace_id=workspace_id,
            status=status,
            include_superseded=include_superseded,
        )

    def query_never_surfaced_candidates(
        self,
        *,
        limit: int = _DEFAULT_CANDIDATE_LIMIT,
        workspace_id: str | None = None,
        status: str | None = None,
        include_superseded: bool = False,
    ) -> list[RelationalMemoryRecord]:
        return self.query_maintenance_candidates(
            "never-surfaced",
            limit=limit,
            workspace_id=workspace_id,
            status=status,
            include_superseded=include_superseded,
        )

    def query_oversized_thin_candidates(
        self,
        *,
        limit: int = _DEFAULT_CANDIDATE_LIMIT,
        workspace_id: str | None = None,
        status: str | None = None,
        include_superseded: bool = False,
        oversized_min_chars: int = _DEFAULT_OVERSIZED_CANDIDATE_MIN_CHARS,
        thin_max_chars: int = _DEFAULT_THIN_CANDIDATE_MAX_CHARS,
    ) -> list[RelationalMemoryRecord]:
        return self.query_maintenance_candidates(
            "oversized/thin",
            limit=limit,
            workspace_id=workspace_id,
            status=status,
            include_superseded=include_superseded,
            oversized_min_chars=oversized_min_chars,
            thin_max_chars=thin_max_chars,
        )

    def query_orphan_low_support_candidates(
        self,
        *,
        limit: int = _DEFAULT_CANDIDATE_LIMIT,
        workspace_id: str | None = None,
        status: str | None = None,
        include_superseded: bool = False,
        low_support_max: int = _DEFAULT_LOW_SUPPORT_MAX,
    ) -> list[RelationalMemoryRecord]:
        return self.query_maintenance_candidates(
            "orphan/low-support",
            limit=limit,
            workspace_id=workspace_id,
            status=status,
            include_superseded=include_superseded,
            low_support_max=low_support_max,
        )

    def query_quality_signal_candidates(
        self,
        *,
        limit: int = _DEFAULT_CANDIDATE_LIMIT,
        workspace_id: str | None = None,
        status: str | None = None,
        include_superseded: bool = False,
        quality_signal: str | None = None,
    ) -> list[RelationalMemoryRecord]:
        return self.query_maintenance_candidates(
            "quality-signal",
            limit=limit,
            workspace_id=workspace_id,
            status=status,
            include_superseded=include_superseded,
            quality_signal=quality_signal,
        )

    def query_retrieval_quality_candidates(
        self, *, limit: int = _DEFAULT_CANDIDATE_LIMIT, workspace_id: str | None = None,
        status: str | None = None, include_superseded: bool = False,
        min_searches: int = 2, max_conversion_rate: float = 0.25,
    ) -> list[RelationalMemoryRecord]:
        return self.query_maintenance_candidates(
            "retrieval-quality", limit=limit, workspace_id=workspace_id, status=status,
            include_superseded=include_superseded, retrieval_min_searches=min_searches,
            retrieval_max_conversion_rate=max_conversion_rate,
        )

    def query_seeded_random_candidates(
        self,
        seed: int | str,
        *,
        limit: int = _DEFAULT_CANDIDATE_LIMIT,
        workspace_id: str | None = None,
        status: str | None = None,
        include_superseded: bool = False,
    ) -> list[RelationalMemoryRecord]:
        return self.query_maintenance_candidates(
            "seeded-random",
            limit=limit,
            workspace_id=workspace_id,
            status=status,
            include_superseded=include_superseded,
            seed=seed,
        )

    def get_links(
        self,
        memory_id: str,
        direction: str = "outgoing",
        link_type: str | None = None,
    ):
        resolved_memory_id = self.resolve_memory_id(memory_id)
        if resolved_memory_id is None:
            return []
        conn = self._db.get_connection()
        if direction == "incoming":
            clause = "target_id = ?"
            order_by = "ORDER BY source_id ASC, target_id ASC"
        else:
            clause = "source_id = ?"
            order_by = "ORDER BY source_id ASC, target_id ASC"

        query = f"SELECT source_id, target_id, type, context FROM links WHERE {clause}"
        params: list[str] = [resolved_memory_id]
        if link_type is not None:
            query += " AND type = ?"
            params.append(self._normalize_link_type(link_type))
        query += f" {order_by}"

        rows = conn.execute(query, params).fetchall()
        return [
            MemoryLink(
                source_id=row["source_id"],
                target_id=row["target_id"],
                link_type=row["type"],
                context=row["context"],
            )
            for row in rows
        ]

    def has_incoming_link(self, memory_id: str, link_type: str):
        normalized_link_type = self._normalize_link_type(link_type)
        resolved_memory_id = self.resolve_memory_id(memory_id)
        if resolved_memory_id is None:
            return False
        conn = self._db.get_connection()
        row = conn.execute(
            "SELECT 1 FROM links WHERE target_id = ? AND type = ? LIMIT 1",
            (resolved_memory_id, normalized_link_type),
        ).fetchone()
        return row is not None

    def count_incoming_links(self, memory_id: str) -> int:
        resolved_memory_id = self.resolve_memory_id(memory_id)
        if resolved_memory_id is None:
            return 0
        row = self._db.get_connection().execute(
            "SELECT COUNT(*) FROM links WHERE target_id = ?",
            (resolved_memory_id,),
        ).fetchone()
        return 0 if row is None else int(row[0])

    def touch_last_surfaced(self, memory_ids: list[str], surfaced_at: str, *, best_effort: bool = False):
        normalized_ids = self._normalize_values(memory_ids)
        if not normalized_ids:
            return 0
        resolved_ids = [
            resolved_id
            for memory_id in normalized_ids
            for resolved_id in [self.resolve_memory_id(memory_id)]
            if resolved_id is not None
        ]
        if not resolved_ids:
            return 0

        conn = self._db.get_connection() if not best_effort else self._db.open_connection(timeout_seconds=_NONCRITICAL_WRITE_TIMEOUT_SECONDS)
        placeholders = ",".join("?" for _ in resolved_ids)
        try:
            with conn:
                cursor = conn.execute(
                    f"UPDATE memories SET last_surfaced_at = ? WHERE id IN ({placeholders})",
                    [surfaced_at, *resolved_ids],
                )
            return cursor.rowcount
        except sqlite3.OperationalError as exc:
            if not best_effort or "locked" not in str(exc).lower():
                raise
            logger.debug("Skipping last_surfaced_at update due to SQLite lock contention")
            return 0
        finally:
            if best_effort:
                conn.close()

    def record_access(
        self,
        memory_id: str,
        access_score: float,
        accessed_at: str,
        increment_read_count: bool = False,
    ):
        resolved_memory_id = self.resolve_memory_id(memory_id)
        if resolved_memory_id is None:
            return None
        conn = self._db.get_connection()
        read_count_clause = "read_count = read_count + 1, " if increment_read_count else ""
        with conn:
            cursor = conn.execute(
                f"""
                UPDATE memories
                SET {read_count_clause}access_score = ?, last_accessed_at = ?
                WHERE id = ?
                """,
                (access_score, accessed_at, resolved_memory_id),
            )
        if cursor.rowcount == 0:
            return None
        return self.get_memory(resolved_memory_id)

    def list_most_read_memories(
        self,
        workspace_id: str | None = None,
        limit: int = 10,
        status: str | None = None,
    ):
        conn = self._db.get_connection()
        params: list[object] = []
        query = "SELECT DISTINCT memories.* FROM memories"
        where_clauses = ["memories.read_count > 0"]
        if workspace_id is not None:
            query += " JOIN memory_workspaces ON memory_workspaces.memory_id = memories.id"
            params.append(workspace_id)
            where_clauses.insert(0, "memory_workspaces.workspace_id = ?")
        if status is not None:
            params.append(status)
            where_clauses.append("memories.status = ?")
        query += " WHERE " + " AND ".join(where_clauses)
        query += " ORDER BY memories.read_count DESC, memories.last_accessed_at DESC, memories.updated_at DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(query, params).fetchall()
        return [self._hydrate_record(conn, row) for row in rows]

    def append_workspace_ids(self, memory_id: str, workspace_ids: list[str]):
        resolved_memory_id = self.resolve_memory_id(memory_id)
        if resolved_memory_id is None:
            return None
        normalized_workspace_ids = self._normalize_values(workspace_ids)
        if not normalized_workspace_ids:
            return self.get_memory(resolved_memory_id)

        current = self.get_memory(resolved_memory_id)
        if current is None:
            return None

        merged_workspace_ids = self._normalize_values(
            [*current.workspace_ids, *normalized_workspace_ids]
        )
        conn = self._db.get_connection()
        with conn:
            self._replace_workspace_mappings(conn, resolved_memory_id, merged_workspace_ids)
            conn.execute(
                "UPDATE memories SET updated_at = ? WHERE id = ?",
                (self._utc_now(), resolved_memory_id),
            )
        return self.get_memory(resolved_memory_id)

    def get_memory(self, memory_id: str):
        conn = self._db.get_connection()
        resolved_memory_id = self.resolve_memory_id(memory_id)
        if resolved_memory_id is None:
            return None
        row = conn.execute(
            "SELECT * FROM memories WHERE id = ?",
            (resolved_memory_id,),
        ).fetchone()
        if row is None:
            return None
        return self._hydrate_record(conn, row)

    def peek_memory(self, memory_id: str) -> RelationalMemoryReadContext | None:
        """Read authoritative memory context without changing retrieval telemetry."""
        record = self.get_memory(memory_id)
        if record is None:
            return None

        outgoing = self.get_links(memory_id, direction="outgoing")
        incoming = self.get_links(memory_id, direction="incoming")
        superseded = [
            target
            for link in outgoing
            if link.link_type == "SUPERSEDES"
            for target in [self.get_memory(link.target_id)]
            if target is not None
        ]
        return RelationalMemoryReadContext(
            record=record,
            relationships={"outgoing": outgoing, "incoming": incoming},
            superseded=superseded,
        )

    def search_memories_for_maintenance(
        self,
        query: str,
        *,
        workspace_id: str | None = None,
        memory_type: str | None = None,
        status: str | None = None,
        include_superseded: bool = False,
        limit: int = 50,
    ) -> list[RelationalMemoryReadContext]:
        """Search authoritative SQLite records without surfacing or accessing them."""
        memory_ids = self._search_keyword_memory_ids(
            query,
            workspace_id=workspace_id,
            memory_type=memory_type,
            status=status,
            include_superseded=include_superseded,
            limit=limit,
        )
        contexts: list[RelationalMemoryReadContext] = []
        for memory_id in memory_ids:
            context = self.peek_memory(memory_id)
            if context is not None:
                contexts.append(context)
        return contexts

    def update_memory(
        self,
        memory_id: str,
        title: str | None = None,
        content: str | None = None,
        summary: str | None | object = _SUMMARY_UNSET,
        memory_type: str | None = None,
        status: str | None = None,
        metadata: dict[str, object] | None = None,
        workspace_ids: list[str] | None = None,
        tags: list[str] | None = None,
        access_score: float | None = None,
        last_accessed_at: str | None = None,
        last_surfaced_at: str | None = None,
    ):
        conn = self._db.get_connection()
        resolved_memory_id = self.resolve_memory_id(memory_id)
        if resolved_memory_id is None:
            return None

        normalized_type = (
            self._validate_memory_type(memory_type) if memory_type is not None else None
        )
        normalized_status = (
            self._validate_memory_status(status) if status is not None else None
        )
        normalized_title = title.strip() if title is not None else None
        normalized_content = content.strip() if content is not None else None
        if normalized_title is not None:
            self._validate_required_text("title", normalized_title)
        if normalized_content is not None:
            self._validate_required_text("content", normalized_content)
        if workspace_ids is not None and not self._normalize_values(workspace_ids):
            raise ValueError("workspace_ids must contain at least one non-empty value")

        existing = self.get_memory(resolved_memory_id)
        if existing is None:
            return None

        resolved_summary = summary
        if summary is _SUMMARY_UNSET and (
            normalized_title is not None or normalized_content is not None or normalized_type is not None
        ):
            resolved_summary = self._build_summary(
                title=normalized_title or existing.title,
                content=normalized_content or existing.content,
                memory_type=normalized_type or existing.type,
            )

        columns = []
        values = []
        updates = {
            "title": normalized_title,
            "content": normalized_content,
            "summary": None if resolved_summary is _SUMMARY_UNSET else resolved_summary,
            "type": normalized_type,
            "status": normalized_status,
            "access_score": access_score,
            "last_accessed_at": last_accessed_at,
            "last_surfaced_at": last_surfaced_at,
        }
        for column_name, value in updates.items():
            if value is None:
                continue
            columns.append(f"{column_name} = ?")
            values.append(value)

        if metadata is not None:
            columns.append("metadata = ?")
            values.append(json.dumps(metadata, sort_keys=True))

        if normalized_status is not None and normalized_status != existing.status:
            columns.append("archived_at = ?")
            values.append(
                self._utc_now() if normalized_status == "archived" else None
            )

        columns.append("updated_at = ?")
        values.append(self._utc_now())
        values.append(resolved_memory_id)

        with conn:
            conn.execute(
                f"UPDATE memories SET {', '.join(columns)} WHERE id = ?",
                values,
            )
            if workspace_ids is not None:
                self._replace_workspace_mappings(
                    conn,
                    resolved_memory_id,
                    self._normalize_values(workspace_ids),
                )
            if tags is not None:
                self._replace_tag_mappings(
                    conn,
                    resolved_memory_id,
                    self._normalize_values(tags),
                )
            current_row = conn.execute(
                "SELECT title, summary, content FROM memories WHERE id = ?",
                (resolved_memory_id,),
            ).fetchone()
            current_tags = (
                self._normalize_values(tags)
                if tags is not None
                else [row[0] for row in conn.execute(
                    """
                    SELECT tags.name
                    FROM tags
                    JOIN memory_tags ON memory_tags.tag_id = tags.id
                    WHERE memory_tags.memory_id = ?
                    ORDER BY tags.name ASC
                    """,
                    (resolved_memory_id,),
                ).fetchall()]
            )
            assert current_row is not None
            self._replace_fts_row(
                conn,
                resolved_memory_id,
                title=str(current_row["title"]),
                summary=str(current_row["summary"] or ""),
                content=str(current_row["content"]),
                tags=current_tags,
            )

        return self.get_memory(resolved_memory_id)

    def list_memories(
        self,
        workspace_id: str | None = None,
        memory_type: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ):
        conn = self._db.get_connection()
        clauses = []
        params: list[object] = []

        query = "SELECT DISTINCT memories.* FROM memories"
        if workspace_id is not None:
            query += " JOIN memory_workspaces ON memory_workspaces.memory_id = memories.id"
            clauses.append("memory_workspaces.workspace_id = ?")
            params.append(workspace_id)
        if memory_type is not None:
            clauses.append("memories.type = ?")
            params.append(memory_type)
        if status is not None:
            clauses.append("memories.status = ?")
            params.append(status)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY memories.updated_at DESC, memories.created_at DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(query, params).fetchall()
        return [self._hydrate_record(conn, row) for row in rows]

    def count_memories_updated_since(
        self,
        cutoff: str,
        *,
        workspace_id: str | None = None,
    ) -> int:
        conn = self._db.get_connection()
        clauses = ["memories.updated_at >= ?"]
        params: list[object] = [cutoff]

        query = "SELECT COUNT(DISTINCT memories.id) FROM memories"
        if workspace_id is not None:
            query += " JOIN memory_workspaces ON memory_workspaces.memory_id = memories.id"
            clauses.append("memory_workspaces.workspace_id = ?")
            params.append(workspace_id)
        query += " WHERE " + " AND ".join(clauses)

        row = conn.execute(query, params).fetchone()
        return int(row[0]) if row is not None else 0

    def list_memory_ids(
        self,
        workspace_id: str | None = None,
        memory_type: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[str]:
        conn = self._db.get_connection()
        clauses = []
        params: list[object] = []

        query = "SELECT DISTINCT memories.id FROM memories"
        if workspace_id is not None:
            query += " JOIN memory_workspaces ON memory_workspaces.memory_id = memories.id"
            clauses.append("memory_workspaces.workspace_id = ?")
            params.append(workspace_id)
        if memory_type is not None:
            clauses.append("memories.type = ?")
            params.append(memory_type)
        if status is not None:
            clauses.append("memories.status = ?")
            params.append(status)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY memories.updated_at DESC, memories.created_at DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(query, params).fetchall()
        return [str(row[0]) for row in rows]

    def _hydrate_record(self, conn, row):
        workspace_rows = conn.execute(
            "SELECT workspace_id FROM memory_workspaces WHERE memory_id = ? ORDER BY workspace_id ASC",
            (row["id"],),
        ).fetchall()
        tag_rows = conn.execute(
            """
            SELECT tags.name
            FROM tags
            JOIN memory_tags ON memory_tags.tag_id = tags.id
            WHERE memory_tags.memory_id = ?
            ORDER BY tags.name ASC
            """,
            (row["id"],),
        ).fetchall()
        metadata = json.loads(row["metadata"] or "{}")
        return RelationalMemoryRecord(
            id=row["id"],
            memory_ref=row["memory_ref"],
            title=row["title"],
            content=row["content"],
            summary=row["summary"],
            type=row["type"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            read_count=int(row["read_count"] or 0),
            access_score=row["access_score"],
            last_accessed_at=row["last_accessed_at"],
            last_surfaced_at=row["last_surfaced_at"],
            metadata=metadata,
            workspace_ids=[workspace_row[0] for workspace_row in workspace_rows],
            tags=[tag_row[0] for tag_row in tag_rows],
        )

    def _candidate_record_from_row(self, row) -> RelationalMemoryRecord:
        return RelationalMemoryRecord(
            id=row["id"],
            memory_ref=row["memory_ref"],
            title=row["title"],
            content=row["content"],
            summary=row["summary"],
            type=row["type"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            read_count=int(row["read_count"] or 0),
            access_score=row["access_score"],
            last_accessed_at=row["last_accessed_at"],
            last_surfaced_at=row["last_surfaced_at"],
            metadata=json.loads(row["metadata"] or "{}"),
            workspace_ids=_split_csv_values(row["workspace_ids_csv"]),
            tags=_split_csv_values(row["tags_csv"]),
        )

    def resolve_memory_id(self, memory_id: str) -> str | None:
        conn = self._db.get_connection()
        row = conn.execute("SELECT id FROM memories WHERE id = ?", (memory_id,)).fetchone()
        if row is not None:
            return str(row[0])

        memory_ref = parse_memory_ref(memory_id)
        if memory_ref is None:
            return None
        row = conn.execute(
            "SELECT id FROM memories WHERE memory_ref = ?",
            (memory_ref,),
        ).fetchone()
        return None if row is None else str(row[0])

    def _replace_workspace_mappings(self, conn, memory_id: str, workspace_ids: list[str]):
        conn.execute("DELETE FROM memory_workspaces WHERE memory_id = ?", (memory_id,))
        for workspace_id in workspace_ids:
            conn.execute(
                "INSERT INTO memory_workspaces (memory_id, workspace_id) VALUES (?, ?)",
                (memory_id, workspace_id),
            )

    def _replace_tag_mappings(self, conn, memory_id: str, tags: list[str]):
        conn.execute("DELETE FROM memory_tags WHERE memory_id = ?", (memory_id,))
        for tag in tags:
            conn.execute("INSERT OR IGNORE INTO tags (name) VALUES (?)", (tag,))
            tag_row = conn.execute("SELECT id FROM tags WHERE name = ?", (tag,)).fetchone()
            if tag_row is None:
                continue
            conn.execute(
                "INSERT INTO memory_tags (memory_id, tag_id) VALUES (?, ?)",
                (memory_id, tag_row[0]),
            )

    def _replace_fts_row(
        self,
        conn,
        memory_id: str,
        *,
        title: str,
        summary: str,
        content: str,
        tags: list[str],
    ):
        conn.execute("DELETE FROM memories_fts WHERE memory_id = ?", (memory_id,))
        conn.execute(
            "INSERT INTO memories_fts(memory_id, title, summary, content, tags) VALUES (?, ?, ?, ?, ?)",
            (memory_id, title, summary, content, " ".join(tags)),
        )

    def _normalize_values(self, values: list[str]):
        normalized_values = []
        seen = set()
        for value in values:
            normalized_value = value.strip()
            if not normalized_value or normalized_value in seen:
                continue
            seen.add(normalized_value)
            normalized_values.append(normalized_value)
        return normalized_values

    def _validate_memory_type(self, memory_type: str):
        normalized_type = memory_type.strip()
        if normalized_type not in VALID_MEMORY_TYPES:
            raise ValueError(
                f"invalid memory_type: {memory_type!r}. Expected one of {sorted(VALID_MEMORY_TYPES)}"
            )
        return normalized_type

    def _validate_memory_status(self, status: str):
        normalized_status = status.strip()
        if normalized_status not in VALID_MEMORY_STATUSES:
            raise ValueError(
                f"invalid status: {status!r}. Expected one of {sorted(VALID_MEMORY_STATUSES)}"
            )
        return normalized_status

    def _validate_required_text(self, field_name: str, value: str):
        if not value:
            raise ValueError(f"{field_name} must be non-empty")

    def _normalize_link_type(self, link_type: str) -> str:
        normalized_link_type = re.sub(r"[\s-]+", "_", link_type.strip()).upper()
        if not normalized_link_type:
            raise ValueError("link_type must be non-empty")
        return normalized_link_type

    def _utc_now(self):
        return datetime.now(UTC).isoformat()

    def _build_summary(self, *, title: str, content: str, memory_type: str | None = None):
        return build_memory_summary(title=title, content=content, memory_type=memory_type)


SQLiteRelationalMemoryRepository = RelationalMemoryRepository


def _split_csv_values(value: str | None) -> list[str]:
    if not value:
        return []
    return [item for item in value.split(",") if item]


def _quality_signal_clause(quality_signal: str | None) -> tuple[str, list[object]]:
    signal = None if quality_signal is None else quality_signal.strip().lower()
    if signal is not None:
        signal = _QUALITY_SIGNAL_ALIASES.get(signal, signal)
    clauses: dict[str, tuple[str, list[object]]] = {
        "trace_like_memory_count": (
            "(LOWER(TRIM(memories.title)) LIKE 'task_complete%' OR "
            "LOWER(TRIM(memories.title)) LIKE 'task complete%' OR "
            "LOWER(TRIM(memories.title)) LIKE 'task_complete_record%')",
            [],
        ),
        "generic_summary_count": (
            "(LOWER(TRIM(COALESCE(memories.summary, ''))) LIKE 'covers %' OR "
            "LOWER(TRIM(COALESCE(memories.summary, ''))) LIKE 'added %')",
            [],
        ),
        "untagged_observation_count": (
            "memories.type = 'observation' AND NOT EXISTS ("
            "SELECT 1 FROM memory_tags untagged WHERE untagged.memory_id = memories.id)",
            [],
        ),
        "oversized_memory_count": (
            "LENGTH(COALESCE(memories.content, '')) >= ?",
            [_DEFAULT_QUALITY_OVERSIZED_MIN_CHARS],
        ),
        "raw_ingress_count": (
            "(memories.metadata LIKE '%\"created_via_ingest\": true%' "
            "AND memories.type IN ('journal', 'observation') "
            "AND (NOT EXISTS (SELECT 1 FROM memory_tags raw_untagged "
            "WHERE raw_untagged.memory_id = memories.id) "
            "OR LENGTH(COALESCE(memories.content, '')) > 1600 "
            "OR LOWER(TRIM(COALESCE(memories.summary, ''))) LIKE 'covers %' "
            "OR LOWER(TRIM(COALESCE(memories.summary, ''))) LIKE 'added %'))",
            [],
        ),
    }
    if signal is None:
        return "(" + " OR ".join(clause for clause, _ in clauses.values()) + ")", [
            parameter for _, values in clauses.values() for parameter in values
        ]
    if signal not in clauses:
        supported = sorted([*clauses, *_QUALITY_SIGNAL_ALIASES])
        raise ValueError(f"unsupported quality signal: {quality_signal!r}; expected one of {supported}")
    return clauses[signal]


def _stable_seeded_random_key(seed: object, memory_id: object) -> str:
    payload = f"{seed}\x00{memory_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
