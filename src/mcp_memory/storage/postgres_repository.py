from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import cast
from uuid import uuid4

from mcp_memory.core.summaries import build_deterministic_summary
from mcp_memory.relational.repository import (
    FTS_QUERY_TOKEN_PATTERN,
    MemoryLink,
    RankedMemoryCandidate,
    RelationalMemoryRecord,
    build_read_cache_validation_token,
    VALID_MEMORY_STATUSES,
    VALID_MEMORY_TYPES,
)
from mcp_memory.storage.buffered_writer import BufferedWriter
from mcp_memory.storage.session import CursorLike, DbConnectionLike, SessionManager


_SUMMARY_UNSET = object()


class PostgresRelationalMemoryRepository:
    def __init__(self, session_manager: SessionManager[DbConnectionLike]) -> None:
        self._sessions = session_manager
        self._last_surfaced_writer = BufferedWriter[tuple[str, str]](
            self._flush_last_surfaced_batch,
            name="postgres-last-surfaced",
            low_watermark=1,
            high_watermark=128,
            flush_interval_seconds=0.01,
        )

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

        clauses = ["memories.id = ANY(%s::text[])"]
        params: list[object] = [normalized_ids]
        if status is not None:
            clauses.append("memories.status = %s")
            params.append(status)
        if not include_superseded:
            clauses.append(
                "NOT EXISTS (SELECT 1 FROM links supersedes WHERE supersedes.target_id = memories.id AND supersedes.type = 'SUPERSEDES')"
            )

        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                        memories.id,
                        memories.title,
                        memories.content,
                        memories.summary,
                        memories.type,
                        memories.status,
                        memories.created_at,
                        memories.updated_at,
                        memories.read_count,
                        memories.access_score,
                        memories.last_accessed_at,
                        memories.last_surfaced_at,
                        memories.metadata,
                        COALESCE(workspace_agg.workspace_ids, ARRAY[]::text[]) AS workspace_ids,
                        COALESCE(tag_agg.tags, ARRAY[]::text[]) AS tags
                    FROM memories
                    LEFT JOIN (
                        SELECT memory_id, array_agg(DISTINCT workspace_id ORDER BY workspace_id) AS workspace_ids
                        FROM memory_workspaces
                        GROUP BY memory_id
                    ) workspace_agg ON workspace_agg.memory_id = memories.id
                    LEFT JOIN (
                        SELECT memory_tags.memory_id, array_agg(DISTINCT tags.name ORDER BY tags.name) AS tags
                        FROM memory_tags
                        JOIN tags ON tags.id = memory_tags.tag_id
                        GROUP BY memory_tags.memory_id
                    ) tag_agg ON tag_agg.memory_id = memories.id
                    WHERE
                    """
                    + " AND ".join(clauses),
                    tuple(params),
                )
                rows = cursor.fetchall()

        records_by_id: dict[str, RelationalMemoryRecord] = {}
        for row in rows:
            memory_id = str(row[0])
            records_by_id[memory_id] = RelationalMemoryRecord(
                id=memory_id,
                title=str(row[1]),
                content=str(row[2]),
                summary=None if row[3] is None else str(row[3]),
                type=str(row[4]),
                status=str(row[5]),
                created_at=str(row[6]),
                updated_at=str(row[7]),
                read_count=self._coerce_int(row[8]),
                access_score=self._coerce_float(row[9]),
                last_accessed_at=None if row[10] is None else str(row[10]),
                last_surfaced_at=None if row[11] is None else str(row[11]),
                metadata=self._load_metadata(row[12]),
                workspace_ids=self._load_text_values(row[13]),
                tags=self._load_text_values(row[14]),
            )
        return [records_by_id[memory_id] for memory_id in normalized_ids if memory_id in records_by_id]

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
        self._validate_required_text("title", normalized_title)
        self._validate_required_text("content", normalized_content)
        if not normalized_workspace_ids:
            raise ValueError("workspace_ids must contain at least one non-empty value")

        summary_text = summary or self._build_summary(
            title=normalized_title,
            content=normalized_content,
            memory_type=normalized_type,
        )
        record_id = memory_id or str(uuid4())
        metadata_payload = json.dumps(metadata or {}, sort_keys=True)

        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO memories (
                        id, title, content, summary, type, status, created_at, updated_at,
                        read_count, access_score, last_accessed_at, last_surfaced_at, metadata
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                    """,
                    (
                        record_id,
                        normalized_title,
                        normalized_content,
                        summary_text,
                        normalized_type,
                        normalized_status,
                        created_timestamp,
                        updated_timestamp,
                        0,
                        0.0,
                        None,
                        None,
                        metadata_payload,
                    ),
                )
                self._replace_workspace_mappings(cursor, record_id, normalized_workspace_ids)
                self._replace_tag_mappings(cursor, record_id, normalized_tags)
                self._upsert_search_document(
                    cursor,
                    memory_id=record_id,
                    title=normalized_title,
                    summary=summary_text,
                    content=normalized_content,
                    tags=normalized_tags,
                )
            connection.commit()

        return self.get_memory(record_id)

    def get_memory(self, memory_id: str):
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                row = self._select_memory_row(cursor, memory_id)
                if row is None:
                    return None
                return self._hydrate_record(cursor, row)

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
        existing = self.get_memory(memory_id)
        if existing is None:
            return None

        normalized_type = self._validate_memory_type(memory_type) if memory_type is not None else None
        normalized_status = self._validate_memory_status(status) if status is not None else None
        normalized_title = title.strip() if title is not None else None
        normalized_content = content.strip() if content is not None else None
        if normalized_title is not None:
            self._validate_required_text("title", normalized_title)
        if normalized_content is not None:
            self._validate_required_text("content", normalized_content)
        if workspace_ids is not None and not self._normalize_values(workspace_ids):
            raise ValueError("workspace_ids must contain at least one non-empty value")

        resolved_summary = summary
        if summary is _SUMMARY_UNSET and (
            normalized_title is not None or normalized_content is not None or normalized_type is not None
        ):
            resolved_summary = self._build_summary(
                title=normalized_title or existing.title,
                content=normalized_content or existing.content,
                memory_type=normalized_type or existing.type,
            )

        columns: list[str] = []
        values: list[object] = []
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
            columns.append(f"{column_name} = %s")
            values.append(value)

        if metadata is not None:
            columns.append("metadata = %s::jsonb")
            values.append(json.dumps(metadata, sort_keys=True))

        columns.append("updated_at = %s")
        values.append(self._utc_now())
        values.append(memory_id)

        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"UPDATE memories SET {', '.join(columns)} WHERE id = %s",
                    tuple(values),
                )
                if workspace_ids is not None:
                    self._replace_workspace_mappings(
                        cursor,
                        memory_id,
                        self._normalize_values(workspace_ids),
                    )
                if tags is not None:
                    self._replace_tag_mappings(
                        cursor,
                        memory_id,
                        self._normalize_values(tags),
                    )
                effective_tags = existing.tags if tags is None else self._normalize_values(tags)
                self._upsert_search_document(
                    cursor,
                    memory_id=memory_id,
                    title=normalized_title or existing.title,
                    summary=cast(str | None, existing.summary if resolved_summary is _SUMMARY_UNSET else resolved_summary),
                    content=normalized_content or existing.content,
                    tags=effective_tags,
                )
            connection.commit()

        return self.get_memory(memory_id)

    def list_memories(
        self,
        workspace_id: str | None = None,
        memory_type: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ):
        joins: list[str] = []
        clauses: list[str] = []
        params: list[object] = []
        if workspace_id is not None:
            joins.append("JOIN memory_workspaces ON memory_workspaces.memory_id = memories.id")
            clauses.append("memory_workspaces.workspace_id = %s")
            params.append(workspace_id)
        if memory_type is not None:
            clauses.append("memories.type = %s")
            params.append(memory_type)
        if status is not None:
            clauses.append("memories.status = %s")
            params.append(status)

        query = (
            "SELECT DISTINCT id, title, content, summary, type, status, created_at, updated_at, "
            "read_count, access_score, last_accessed_at, last_surfaced_at, metadata "
            "FROM memories "
        )
        if joins:
            query += " ".join(joins) + " "
        if clauses:
            query += "WHERE " + " AND ".join(clauses) + " "
        query += "ORDER BY memories.updated_at DESC, memories.created_at DESC LIMIT %s"
        params.append(limit)

        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, tuple(params))
                rows = cursor.fetchall()
                return [self._hydrate_record(cursor, row) for row in rows]

    def search_keyword_memory_ids(
        self,
        query: str,
        *,
        workspace_id: str | None = None,
        memory_type: str | None = None,
        status: str | None = None,
        include_superseded: bool = False,
        limit: int = 50,
    ) -> list[str]:
        tokens = [match.group(0).lower() for match in FTS_QUERY_TOKEN_PATTERN.finditer(query)]
        if not tokens:
            return []

        clauses = [
            """
            EXISTS (
                SELECT 1
                FROM unnest(%s::text[]) AS token
                WHERE memory_search_documents.search_document @@ plainto_tsquery('simple', token)
            )
            """
        ]
        params: list[object] = [tokens, tokens]
        joins: list[str] = []
        if workspace_id is not None:
            joins.append("JOIN memory_workspaces ON memory_workspaces.memory_id = memories.id")
            clauses.append("memory_workspaces.workspace_id = %s")
            params.append(workspace_id)
        if memory_type is not None:
            clauses.append("memories.type = %s")
            params.append(memory_type)
        if status is not None:
            clauses.append("memories.status = %s")
            params.append(status)
        if not include_superseded:
            clauses.append(
                "NOT EXISTS (SELECT 1 FROM links supersedes WHERE supersedes.target_id = memories.id AND supersedes.type = 'SUPERSEDES')"
            )

        params.append(limit)
        query_sql = (
            """
            SELECT DISTINCT memories.id,
                memories.updated_at,
                (
                    SELECT COALESCE(SUM(ts_rank_cd(memory_search_documents.search_document, plainto_tsquery('simple', token))), 0.0)
                    FROM unnest(%s::text[]) AS token
                    WHERE memory_search_documents.search_document @@ plainto_tsquery('simple', token)
                ) AS rank
            FROM memory_search_documents
            JOIN memories ON memories.id = memory_search_documents.memory_id
            """
            + (" ".join(joins) + " " if joins else "")
            + "WHERE "
            + " AND ".join(clauses)
            + " ORDER BY rank DESC, memories.updated_at DESC LIMIT %s"
        )

        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query_sql, tuple(params))
                rows = cursor.fetchall()
                return [str(row[0]) for row in rows]

    def get_ranking_candidates(
        self,
        memory_ids: list[str],
        *,
        status: str | None = None,
        include_superseded: bool = False,
        timing_ms: dict[str, float] | None = None,
    ) -> list[RankedMemoryCandidate]:
        normalized_ids = self._normalize_values(memory_ids)
        if not normalized_ids:
            return []

        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                clauses: list[str] = []
                params: list[object] = [normalized_ids]
                if status is not None:
                    clauses.append("memories.status = %s")
                    params.append(status)
                if not include_superseded:
                    clauses.append("COALESCE(link_counts.has_incoming_supersedes, 0) = 0")

                query_started = time.perf_counter()
                cursor.execute(
                    """
                    WITH input_ids AS (
                        SELECT memory_id, ordinality
                        FROM unnest(%s::text[]) WITH ORDINALITY AS requested(memory_id, ordinality)
                    ),
                    workspace_agg AS (
                        SELECT
                            memory_workspaces.memory_id,
                            array_agg(DISTINCT memory_workspaces.workspace_id ORDER BY memory_workspaces.workspace_id) AS workspace_ids
                        FROM memory_workspaces
                        JOIN input_ids ON input_ids.memory_id = memory_workspaces.memory_id
                        GROUP BY memory_workspaces.memory_id
                    ),
                    tag_agg AS (
                        SELECT
                            memory_tags.memory_id,
                            array_agg(DISTINCT tags.name ORDER BY tags.name) AS tags
                        FROM memory_tags
                        JOIN tags ON tags.id = memory_tags.tag_id
                        JOIN input_ids ON input_ids.memory_id = memory_tags.memory_id
                        GROUP BY memory_tags.memory_id
                    ),
                    link_counts AS (
                        SELECT
                            links.target_id AS memory_id,
                            COUNT(*) AS incoming_links_count,
                            MAX(CASE WHEN links.type = 'SUPERSEDES' THEN 1 ELSE 0 END) AS has_incoming_supersedes,
                            SUM(CASE WHEN links.type = 'DEPENDS_ON' THEN 1 ELSE 0 END) AS incoming_depends_on_count,
                            SUM(CASE WHEN links.type = 'AMENDS' THEN 1 ELSE 0 END) AS incoming_amends_count,
                            SUM(CASE WHEN links.type = 'CONTRADICTS' THEN 1 ELSE 0 END) AS incoming_contradicts_count,
                            SUM(CASE WHEN links.type = 'SUPERSEDES' THEN 1 ELSE 0 END) AS incoming_supersedes_count
                        FROM links
                        JOIN input_ids ON input_ids.memory_id = links.target_id
                        GROUP BY links.target_id
                    )
                    SELECT
                        memories.id,
                        memories.title,
                        memories.content,
                        memories.summary,
                        memories.type,
                        memories.status,
                        memories.created_at,
                        memories.updated_at,
                        memories.read_count,
                        memories.access_score,
                        memories.last_accessed_at,
                        memories.last_surfaced_at,
                        memories.metadata,
                        COALESCE(workspace_agg.workspace_ids, ARRAY[]::text[]) AS workspace_ids,
                        COALESCE(tag_agg.tags, ARRAY[]::text[]) AS tags,
                        COALESCE(link_counts.incoming_links_count, 0) AS incoming_links_count,
                        COALESCE(link_counts.has_incoming_supersedes, 0) AS has_incoming_supersedes,
                        COALESCE(link_counts.incoming_depends_on_count, 0) AS incoming_depends_on_count,
                        COALESCE(link_counts.incoming_amends_count, 0) AS incoming_amends_count,
                        COALESCE(link_counts.incoming_contradicts_count, 0) AS incoming_contradicts_count,
                        COALESCE(link_counts.incoming_supersedes_count, 0) AS incoming_supersedes_count
                    FROM input_ids
                    JOIN memories ON memories.id = input_ids.memory_id
                    LEFT JOIN workspace_agg ON workspace_agg.memory_id = memories.id
                    LEFT JOIN tag_agg ON tag_agg.memory_id = memories.id
                    LEFT JOIN link_counts ON link_counts.memory_id = memories.id
                    """
                    + ("WHERE " + " AND ".join(clauses) + " " if clauses else "")
                    + "ORDER BY input_ids.ordinality ASC",
                    tuple(params),
                )
                self._record_timing_ms(timing_ms, "candidate_hydration_query_execution", query_started)
                fetch_started = time.perf_counter()
                rows = cursor.fetchall()
                self._record_timing_ms(timing_ms, "candidate_hydration_row_fetch", fetch_started)
                if not rows:
                    return []

        candidate_build_started = time.perf_counter()
        candidates: list[RankedMemoryCandidate] = []
        for row in rows:
            has_incoming_supersedes = self._coerce_int(row[16]) > 0
            candidates.append(
                RankedMemoryCandidate(
                    record=RelationalMemoryRecord(
                        id=str(row[0]),
                        title=str(row[1]),
                        content=str(row[2]),
                        summary=None if row[3] is None else str(row[3]),
                        type=str(row[4]),
                        status=str(row[5]),
                        created_at=str(row[6]),
                        updated_at=str(row[7]),
                        read_count=self._coerce_int(row[8]),
                        access_score=self._coerce_float(row[9]),
                        last_accessed_at=None if row[10] is None else str(row[10]),
                        last_surfaced_at=None if row[11] is None else str(row[11]),
                        metadata=self._load_metadata(row[12]),
                        workspace_ids=self._load_text_values(row[13]),
                        tags=self._load_text_values(row[14]),
                    ),
                    incoming_links_count=self._coerce_int(row[15]),
                    has_incoming_supersedes=has_incoming_supersedes,
                    incoming_link_type_counts={
                        "DEPENDS_ON": self._coerce_int(row[17]),
                        "AMENDS": self._coerce_int(row[18]),
                        "CONTRADICTS": self._coerce_int(row[19]),
                        "SUPERSEDES": self._coerce_int(row[20]),
                    },
                )
            )
        self._record_timing_ms(timing_ms, "candidate_hydration_candidate_build", candidate_build_started)
        return candidates

    def touch_last_surfaced(self, memory_ids: list[str], surfaced_at: str, *, best_effort: bool = False):
        normalized_ids = self._normalize_values(memory_ids)
        if not normalized_ids:
            return 0

        if best_effort:
            self._last_surfaced_writer.write_many(
                [(memory_id, surfaced_at) for memory_id in normalized_ids]
            )
            return len(normalized_ids)

        self._update_last_surfaced_ids(normalized_ids, surfaced_at)
        return len(normalized_ids)

    def close(self) -> None:
        self._last_surfaced_writer.close()

    def flush(self) -> None:
        self._last_surfaced_writer.flush()

    def _flush_last_surfaced_batch(self, batch: list[tuple[str, str]]) -> None:
        latest_by_memory_id: dict[str, str] = {}
        for memory_id, pending_surfaced_at in batch:
            latest_by_memory_id[memory_id] = pending_surfaced_at

        memory_ids_by_timestamp: dict[str, list[str]] = {}
        for memory_id, pending_surfaced_at in latest_by_memory_id.items():
            memory_ids_by_timestamp.setdefault(pending_surfaced_at, []).append(memory_id)

        for pending_surfaced_at, pending_memory_ids in memory_ids_by_timestamp.items():
            self._update_last_surfaced_ids(pending_memory_ids, pending_surfaced_at)

    def _update_last_surfaced_ids(self, memory_ids: list[str], surfaced_at: str) -> None:

        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "UPDATE memories SET last_surfaced_at = %s WHERE id = ANY(%s::text[])",
                    (surfaced_at, memory_ids),
                )
            connection.commit()

    def append_workspace_ids(self, memory_id: str, workspace_ids: list[str]):
        normalized_workspace_ids = self._normalize_values(workspace_ids)
        if not normalized_workspace_ids:
            return self.get_memory(memory_id)

        current = self.get_memory(memory_id)
        if current is None:
            return None

        merged_workspace_ids = self._normalize_values([*current.workspace_ids, *normalized_workspace_ids])
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                self._replace_workspace_mappings(cursor, memory_id, merged_workspace_ids)
                cursor.execute(
                    "UPDATE memories SET updated_at = %s WHERE id = %s",
                    (self._utc_now(), memory_id),
                )
            connection.commit()
        return self.get_memory(memory_id)

    def add_link(
        self,
        source_id: str,
        target_id: str,
        link_type: str,
        context: str = "",
    ):
        normalized_link_type = self._normalize_link_type(link_type)
        normalized_context = context.strip()
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO links (source_id, target_id, type, context)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (source_id, target_id, type)
                    DO UPDATE SET context = EXCLUDED.context
                    """,
                    (source_id, target_id, normalized_link_type, normalized_context),
                )
            connection.commit()
        return MemoryLink(
            source_id=source_id,
            target_id=target_id,
            link_type=normalized_link_type,
            context=normalized_context,
        )

    def get_links(
        self,
        memory_id: str,
        direction: str = "outgoing",
        link_type: str | None = None,
    ):
        if direction == "incoming":
            clause = "target_id = %s"
            order_by = "ORDER BY source_id ASC, target_id ASC"
        else:
            clause = "source_id = %s"
            order_by = "ORDER BY source_id ASC, target_id ASC"

        query = f"SELECT source_id, target_id, type, context FROM links WHERE {clause}"
        params: list[object] = [memory_id]
        if link_type is not None:
            query += " AND type = %s"
            params.append(self._normalize_link_type(link_type))
        query += f" {order_by}"

        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, tuple(params))
                rows = cursor.fetchall()
                return [
                    MemoryLink(
                        source_id=str(row[0]),
                        target_id=str(row[1]),
                        link_type=str(row[2]),
                        context=str(row[3]),
                    )
                    for row in rows
                ]

    def remove_link(
        self,
        source_id: str,
        target_id: str,
        link_type: str,
    ) -> bool:
        normalized_link_type = self._normalize_link_type(link_type)
        existing = self.get_links(source_id, direction="outgoing", link_type=normalized_link_type)
        if not any(link.target_id == target_id for link in existing):
            return False

        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "DELETE FROM links WHERE source_id = %s AND target_id = %s AND type = %s",
                    (source_id, target_id, normalized_link_type),
                )
            connection.commit()
        return True

    def has_incoming_link(self, memory_id: str, link_type: str):
        normalized_link_type = self._normalize_link_type(link_type)
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT 1 FROM links WHERE target_id = %s AND type = %s LIMIT 1",
                    (memory_id, normalized_link_type),
                )
                return cursor.fetchone() is not None

    def count_incoming_links(self, memory_id: str) -> int:
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT COUNT(*) FROM links WHERE target_id = %s",
                    (memory_id,),
                )
                row = cursor.fetchone()
                return self._coerce_int(row[0]) if row is not None else 0

    def delete_memory(self, memory_id: str):
        existing = self.get_memory(memory_id)
        if existing is None:
            return None

        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "DELETE FROM links WHERE source_id = %s OR target_id = %s",
                    (memory_id, memory_id),
                )
                cursor.execute("DELETE FROM memory_search_documents WHERE memory_id = %s", (memory_id,))
                cursor.execute("DELETE FROM memories WHERE id = %s", (memory_id,))
            connection.commit()
        return existing

    def record_access(
        self,
        memory_id: str,
        access_score: float,
        accessed_at: str,
        increment_read_count: bool = False,
    ):
        read_count_clause = "read_count = read_count + 1, " if increment_read_count else ""
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"UPDATE memories SET {read_count_clause}access_score = %s, last_accessed_at = %s WHERE id = %s",
                    (access_score, accessed_at, memory_id),
                )
            connection.commit()
        return self.get_memory(memory_id)

    def list_most_read_memories(
        self,
        workspace_id: str | None = None,
        limit: int = 10,
        status: str | None = None,
    ):
        joins: list[str] = []
        where_clauses = ["memories.read_count > 0"]
        params: list[object] = []
        if workspace_id is not None:
            joins.append("JOIN memory_workspaces ON memory_workspaces.memory_id = memories.id")
            where_clauses.insert(0, "memory_workspaces.workspace_id = %s")
            params.append(workspace_id)
        if status is not None:
            where_clauses.append("memories.status = %s")
            params.append(status)

        query = (
            "SELECT DISTINCT id, title, content, summary, type, status, created_at, updated_at, "
            "read_count, access_score, last_accessed_at, last_surfaced_at, metadata FROM memories "
        )
        if joins:
            query += " ".join(joins) + " "
        query += "WHERE " + " AND ".join(where_clauses) + " "
        query += "ORDER BY memories.read_count DESC, memories.last_accessed_at DESC, memories.updated_at DESC LIMIT %s"
        params.append(limit)

        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, tuple(params))
                rows = cursor.fetchall()
                return [self._hydrate_record(cursor, row) for row in rows]

    def _select_memory_row(self, cursor: CursorLike, memory_id: str) -> tuple[object, ...] | None:
        cursor.execute(
            """
            SELECT id, title, content, summary, type, status, created_at, updated_at,
                   read_count, access_score, last_accessed_at, last_surfaced_at, metadata
            FROM memories
            WHERE id = %s
            """,
            (memory_id,),
        )
        return cursor.fetchone()

    def _workspace_ids_by_memory_id(
        self,
        cursor: CursorLike,
        memory_ids: list[str],
    ) -> dict[str, list[str]]:
        cursor.execute(
            """
            SELECT memory_id, workspace_id
            FROM memory_workspaces
            WHERE memory_id = ANY(%s::text[])
            ORDER BY memory_id ASC, workspace_id ASC
            """,
            (memory_ids,),
        )
        workspace_ids_by_memory_id: dict[str, list[str]] = {}
        for row in cursor.fetchall():
            memory_id = str(row[0])
            workspace_ids_by_memory_id.setdefault(memory_id, []).append(str(row[1]))
        return workspace_ids_by_memory_id

    def _tags_by_memory_id(
        self,
        cursor: CursorLike,
        memory_ids: list[str],
    ) -> dict[str, list[str]]:
        cursor.execute(
            """
            SELECT memory_tags.memory_id, tags.name
            FROM memory_tags
            JOIN tags ON tags.id = memory_tags.tag_id
            WHERE memory_tags.memory_id = ANY(%s::text[])
            ORDER BY memory_tags.memory_id ASC, tags.name ASC
            """,
            (memory_ids,),
        )
        tags_by_memory_id: dict[str, list[str]] = {}
        for row in cursor.fetchall():
            memory_id = str(row[0])
            tags_by_memory_id.setdefault(memory_id, []).append(str(row[1]))
        return tags_by_memory_id

    def _incoming_link_counts_by_memory_id(
        self,
        cursor: CursorLike,
        memory_ids: list[str],
    ) -> dict[str, tuple[int, dict[str, int]]]:
        cursor.execute(
            """
            SELECT
                target_id AS memory_id,
                COUNT(*) AS incoming_links_count,
                SUM(CASE WHEN type = 'DEPENDS_ON' THEN 1 ELSE 0 END) AS incoming_depends_on_count,
                SUM(CASE WHEN type = 'AMENDS' THEN 1 ELSE 0 END) AS incoming_amends_count,
                SUM(CASE WHEN type = 'CONTRADICTS' THEN 1 ELSE 0 END) AS incoming_contradicts_count,
                SUM(CASE WHEN type = 'SUPERSEDES' THEN 1 ELSE 0 END) AS incoming_supersedes_count
            FROM links
            WHERE target_id = ANY(%s::text[])
            GROUP BY target_id
            """,
            (memory_ids,),
        )
        link_counts_by_memory_id: dict[str, tuple[int, dict[str, int]]] = {}
        for row in cursor.fetchall():
            link_counts_by_memory_id[str(row[0])] = (
                self._coerce_int(row[1]),
                {
                    "DEPENDS_ON": self._coerce_int(row[2]),
                    "AMENDS": self._coerce_int(row[3]),
                    "CONTRADICTS": self._coerce_int(row[4]),
                    "SUPERSEDES": self._coerce_int(row[5]),
                },
            )
        return link_counts_by_memory_id

    def _hydrate_record(self, cursor: CursorLike, row: tuple[object, ...]) -> RelationalMemoryRecord:
        memory_id = str(row[0])
        cursor.execute(
            "SELECT workspace_id FROM memory_workspaces WHERE memory_id = %s ORDER BY workspace_id ASC",
            (memory_id,),
        )
        workspace_rows = cursor.fetchall()
        cursor.execute(
            """
            SELECT tags.name
            FROM tags
            JOIN memory_tags ON memory_tags.tag_id = tags.id
            WHERE memory_tags.memory_id = %s
            ORDER BY tags.name ASC
            """,
            (memory_id,),
        )
        tag_rows = cursor.fetchall()
        return RelationalMemoryRecord(
            id=memory_id,
            title=str(row[1]),
            content=str(row[2]),
            summary=None if row[3] is None else str(row[3]),
            type=str(row[4]),
            status=str(row[5]),
            created_at=str(row[6]),
            updated_at=str(row[7]),
            read_count=self._coerce_int(row[8]),
            access_score=self._coerce_float(row[9]),
            last_accessed_at=None if row[10] is None else str(row[10]),
            last_surfaced_at=None if row[11] is None else str(row[11]),
            metadata=self._load_metadata(row[12]),
            workspace_ids=[str(workspace_row[0]) for workspace_row in workspace_rows],
            tags=[str(tag_row[0]) for tag_row in tag_rows],
        )

    def _replace_workspace_mappings(self, cursor: CursorLike, memory_id: str, workspace_ids: list[str]) -> None:
        cursor.execute("DELETE FROM memory_workspaces WHERE memory_id = %s", (memory_id,))
        for workspace_id in workspace_ids:
            cursor.execute(
                "INSERT INTO memory_workspaces (memory_id, workspace_id) VALUES (%s, %s)",
                (memory_id, workspace_id),
            )

    def _replace_tag_mappings(self, cursor: CursorLike, memory_id: str, tags: list[str]) -> None:
        cursor.execute("DELETE FROM memory_tags WHERE memory_id = %s", (memory_id,))
        for tag in tags:
            cursor.execute(
                "INSERT INTO tags (name) VALUES (%s) ON CONFLICT (name) DO NOTHING",
                (tag,),
            )
            cursor.execute("SELECT id FROM tags WHERE name = %s", (tag,))
            tag_row = cursor.fetchone()
            if tag_row is None:
                continue
            cursor.execute(
                "INSERT INTO memory_tags (memory_id, tag_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                (memory_id, tag_row[0]),
            )

    def _upsert_search_document(
        self,
        cursor: CursorLike,
        *,
        memory_id: str,
        title: str,
        summary: str | None,
        content: str,
        tags: list[str],
    ) -> None:
        tags_text = " ".join(tags)
        cursor.execute(
            """
            INSERT INTO memory_search_documents (
                memory_id,
                title,
                summary,
                content,
                tags_text,
                search_document
            )
            VALUES (
                %s,
                %s,
                %s,
                %s,
                %s,
                (
                    setweight(to_tsvector('simple', COALESCE(%s, '')), 'A')
                    || setweight(to_tsvector('simple', COALESCE(%s, '')), 'A')
                    || setweight(to_tsvector('simple', COALESCE(%s, '')), 'B')
                    || setweight(to_tsvector('simple', COALESCE(%s, '')), 'C')
                )
            )
            ON CONFLICT (memory_id)
            DO UPDATE SET
                title = EXCLUDED.title,
                summary = EXCLUDED.summary,
                content = EXCLUDED.content,
                tags_text = EXCLUDED.tags_text,
                search_document = EXCLUDED.search_document
            """,
            (
                memory_id,
                title,
                summary or "",
                content,
                tags_text,
                title,
                summary or "",
                tags_text,
                content,
            ),
        )

    def _normalize_values(self, values: list[str]) -> list[str]:
        normalized_values: list[str] = []
        seen: set[str] = set()
        for value in values:
            normalized_value = value.strip()
            if not normalized_value or normalized_value in seen:
                continue
            seen.add(normalized_value)
            normalized_values.append(normalized_value)
        return normalized_values

    def _validate_memory_type(self, memory_type: str) -> str:
        normalized_type = memory_type.strip()
        if normalized_type not in VALID_MEMORY_TYPES:
            raise ValueError(
                f"invalid memory_type: {memory_type!r}. Expected one of {sorted(VALID_MEMORY_TYPES)}"
            )
        return normalized_type

    def _validate_memory_status(self, status: str) -> str:
        normalized_status = status.strip()
        if normalized_status not in VALID_MEMORY_STATUSES:
            raise ValueError(
                f"invalid status: {status!r}. Expected one of {sorted(VALID_MEMORY_STATUSES)}"
            )
        return normalized_status

    def _validate_required_text(self, field_name: str, value: str) -> None:
        if not value:
            raise ValueError(f"{field_name} must be non-empty")

    def _normalize_link_type(self, link_type: str) -> str:
        normalized_link_type = re.sub(r"[\s-]+", "_", link_type.strip()).upper()
        if not normalized_link_type:
            raise ValueError("link_type must be non-empty")
        return normalized_link_type

    def _utc_now(self) -> str:
        return datetime.now(UTC).isoformat()

    def _build_summary(self, *, title: str, content: str, memory_type: str | None = None) -> str:
        return build_deterministic_summary(
            title=title,
            content=content,
            memory_type=memory_type,
        )

    def _load_metadata(self, raw_value: object) -> dict[str, object]:
        if raw_value is None:
            return {}
        if isinstance(raw_value, Mapping):
            return {str(key): value for key, value in raw_value.items()}
        if isinstance(raw_value, str):
            loaded = json.loads(raw_value or "{}")
            if isinstance(loaded, dict):
                return {str(key): value for key, value in loaded.items()}
        raise TypeError("memory metadata must be a JSON object")

    def _load_text_values(self, raw_value: object) -> list[str]:
        if raw_value is None:
            return []
        if isinstance(raw_value, str):
            return [raw_value] if raw_value else []
        if isinstance(raw_value, list | tuple):
            return [str(value) for value in raw_value]
        raise TypeError("expected text array-compatible value")

    def _coerce_int(self, value: object) -> int:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            return int(value)
        raise TypeError("expected integer-compatible value")

    def _coerce_float(self, value: object) -> float:
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, int | float):
            return float(value)
        if isinstance(value, str):
            return float(value)
        raise TypeError("expected float-compatible value")

    def _record_timing_ms(self, timing_ms: dict[str, float] | None, key: str, started_at: float) -> None:
        if timing_ms is None:
            return
        elapsed_ms = round((time.perf_counter() - started_at) * 1000.0, 3)
        timing_ms[key] = round(timing_ms.get(key, 0.0) + elapsed_ms, 3)
