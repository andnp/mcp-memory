from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import cast
from uuid import uuid4

from mcp_memory.core.ports.memory import (
    _DEFAULT_CANDIDATE_LIMIT,
    _DEFAULT_LOW_SUPPORT_MAX,
    _DEFAULT_OVERSIZED_CANDIDATE_MIN_CHARS,
    _DEFAULT_QUALITY_OVERSIZED_MIN_CHARS,
    _DEFAULT_THIN_CANDIDATE_MAX_CHARS,
    _QUALITY_SIGNAL_ALIASES,
    FTS_QUERY_TOKEN_PATTERN,
    VALID_MEMORY_STATUSES,
    VALID_MEMORY_TYPES,
    MemoryCreateRequest,
    MemoryLink,
    MemoryReadContext,
    MemoryRecord,
    RankedMemoryCandidate,
    build_read_cache_validation_token,
    parse_memory_ref,
)
from mcp_memory.core.summaries import build_deterministic_summary
from mcp_memory.storage.buffered_writer import BufferedWriter
from mcp_memory.storage.session import CursorLike, DbConnectionLike, SessionManager

_SUMMARY_UNSET = object()
RelationalMemoryReadContext = MemoryReadContext
RelationalMemoryRecord = MemoryRecord


@dataclass(frozen=True, slots=True)
class _PreparedMemoryCreate:
    memory_id: str
    title: str
    content: str
    summary: str
    memory_type: str
    status: str
    created_at: str
    updated_at: str
    archived_at: str | None
    metadata: str
    workspace_ids: list[str]
    tags: list[str]


class PostgresRelationalMemoryRepository:
    def get_search_epochs(self) -> dict[str, int]:
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT key, value FROM schema_metadata WHERE key = ANY(%s)",
                    ([
                        "search_epoch_keyword",
                        "search_epoch_vector",
                        "search_epoch_graph",
                    ],),
                )
                rows = cursor.fetchall()
        values = {str(row[0]): self._coerce_int(row[1]) for row in rows}
        return {
            "keyword": values["search_epoch_keyword"],
            "vector": values["search_epoch_vector"],
            "graph": values["search_epoch_graph"],
        }

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
        normalized_ids = self._normalize_values(memory_ids)
        if not normalized_ids:
            return {}

        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                memory_rows_by_id = self._memory_rows_by_id(cursor, normalized_ids)
                if not memory_rows_by_id:
                    return {}

                existing_memory_ids = [
                    memory_id for memory_id in normalized_ids if memory_id in memory_rows_by_id
                ]
                outgoing_links_by_memory_id = self._links_by_memory_id(
                    cursor,
                    existing_memory_ids,
                    direction="outgoing",
                )
                incoming_links_by_memory_id = self._links_by_memory_id(
                    cursor,
                    existing_memory_ids,
                    direction="incoming",
                )
                superseded_target_ids = self._normalize_values(
                    [
                        link.target_id
                        for memory_id in existing_memory_ids
                        for link in outgoing_links_by_memory_id.get(memory_id, [])
                        if link.link_type == "SUPERSEDES"
                    ]
                )
                superseded_rows_by_id = self._memory_rows_by_id(cursor, superseded_target_ids)

        tokens: dict[str, str] = {}
        for memory_id in normalized_ids:
            record_row = memory_rows_by_id.get(memory_id)
            if record_row is None:
                continue
            outgoing = outgoing_links_by_memory_id.get(memory_id, [])
            incoming = incoming_links_by_memory_id.get(memory_id, [])
            superseded_targets = [
                self._record_from_memory_row(superseded_rows_by_id[link.target_id])
                for link in outgoing
                if link.link_type == "SUPERSEDES" and link.target_id in superseded_rows_by_id
            ]
            tokens[memory_id] = build_read_cache_validation_token(
                record=self._record_from_memory_row(record_row),
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

        clauses = ["memories.id = ANY(%s::text[])"]
        params: list[object] = [resolved_ids]
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
                        memories.memory_ref,
                        memories.archived_at,
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
                memory_ref=self._coerce_int(row[13]),
                archived_at=None if row[14] is None else str(row[14]),
                workspace_ids=self._load_text_values(row[15]),
                tags=self._load_text_values(row[16]),
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
        created = self.create_memories(
            [
                MemoryCreateRequest(
                    title=title,
                    content=content,
                    workspace_ids=workspace_ids,
                    tags=tags or (),
                    summary=summary,
                    memory_type=memory_type,
                    status=status,
                    metadata=metadata,
                    memory_id=memory_id,
                    created_at=created_at,
                    updated_at=updated_at,
                )
            ]
        )
        return self.get_memory(created[0].id) if created else None

    def create_memories(self, requests: Sequence[MemoryCreateRequest]) -> list[MemoryRecord]:
        prepared = [self._prepare_memory_create(request) for request in requests]
        if not prepared:
            return []

        memory_ids = [item.memory_id for item in prepared]
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                for item in prepared:
                    cursor.execute(
                        """
                        INSERT INTO memories (
                            id, title, content, summary, type, status, created_at, updated_at,
                            read_count, access_score, last_accessed_at, last_surfaced_at, metadata, archived_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)
                        """,
                        (
                            item.memory_id,
                            item.title,
                            item.content,
                            item.summary,
                            item.memory_type,
                            item.status,
                            item.created_at,
                            item.updated_at,
                            0,
                            0.0,
                            None,
                            None,
                            item.metadata,
                            item.archived_at,
                        ),
                    )
                    self._replace_workspace_mappings(cursor, item.memory_id, item.workspace_ids)
                    self._replace_tag_mappings(cursor, item.memory_id, item.tags)
                    self._upsert_search_document(
                        cursor,
                        memory_id=item.memory_id,
                        title=item.title,
                        summary=item.summary,
                        content=item.content,
                        tags=item.tags,
                    )
            connection.commit()

        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                rows_by_id = self._memory_rows_by_id(cursor, memory_ids)
                workspaces_by_id = self._workspace_ids_by_memory_id(cursor, memory_ids)
                tags_by_id = self._tags_by_memory_id(cursor, memory_ids)
                return [
                    self._record_from_memory_row(
                        rows_by_id[memory_id],
                        workspace_ids=workspaces_by_id.get(memory_id, []),
                        tags=tags_by_id.get(memory_id, []),
                    )
                    for memory_id in memory_ids
                ]

    def _prepare_memory_create(self, request: MemoryCreateRequest) -> _PreparedMemoryCreate:
        now = self._utc_now()
        created_timestamp = request.created_at or now
        updated_timestamp = request.updated_at or created_timestamp
        normalized_title = request.title.strip()
        normalized_content = request.content.strip()
        normalized_workspace_ids = self._normalize_values(list(request.workspace_ids))
        normalized_tags = self._normalize_values(list(request.tags))
        normalized_type = self._validate_memory_type(request.memory_type)
        normalized_status = self._validate_memory_status(request.status)
        self._validate_required_text("title", normalized_title)
        self._validate_required_text("content", normalized_content)
        if not normalized_workspace_ids:
            raise ValueError("workspace_ids must contain at least one non-empty value")

        summary_text = request.summary or self._build_summary(
            title=normalized_title,
            content=normalized_content,
            memory_type=normalized_type,
        )
        return _PreparedMemoryCreate(
            memory_id=request.memory_id or str(uuid4()),
            title=normalized_title,
            content=normalized_content,
            summary=summary_text,
            memory_type=normalized_type,
            status=normalized_status,
            created_at=created_timestamp,
            updated_at=updated_timestamp,
            archived_at=self._utc_now() if normalized_status == "archived" else None,
            metadata=json.dumps(request.metadata or {}, sort_keys=True),
            workspace_ids=normalized_workspace_ids,
            tags=normalized_tags,
        )

    def get_memory(self, memory_id: str):
        resolved_memory_id = self.resolve_memory_id(memory_id)
        if resolved_memory_id is None:
            return None
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                row = self._select_memory_row(cursor, resolved_memory_id)
                if row is None:
                    return None
                return self._hydrate_record(cursor, row)

    def resolve_memory_id(self, memory_id: str) -> str | None:
        memory_ref = parse_memory_ref(memory_id)
        if memory_ref is None:
            return memory_id

        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT id FROM memories WHERE memory_ref = %s",
                    (memory_ref,),
                )
                row = cursor.fetchone()
                return None if row is None else str(row[0])

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
        tags: Sequence[str] | None = None,
        include_superseded: bool = False,
        limit: int = 50,
    ) -> list[RelationalMemoryReadContext]:
        """Search authoritative Postgres records without surfacing or accessing them."""
        memory_ids = self.search_keyword_memory_ids(
            query,
            workspace_id=workspace_id,
            memory_type=memory_type,
            status=status,
            tags=tags,
            include_superseded=include_superseded,
            limit=limit,
        )
        return self._hydrate_maintenance_contexts(memory_ids)

    def _hydrate_maintenance_contexts(
        self,
        memory_ids: list[str],
    ) -> list[RelationalMemoryReadContext]:
        if not memory_ids:
            return []

        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                memory_rows_by_id = self._memory_rows_by_id(cursor, memory_ids)
                existing_memory_ids = [
                    memory_id for memory_id in memory_ids if memory_id in memory_rows_by_id
                ]
                if not existing_memory_ids:
                    return []

                workspace_ids_by_memory_id = self._workspace_ids_by_memory_id(
                    cursor,
                    existing_memory_ids,
                )
                tags_by_memory_id = self._tags_by_memory_id(cursor, existing_memory_ids)
                outgoing_links_by_memory_id = self._links_by_memory_id(
                    cursor,
                    existing_memory_ids,
                    direction="outgoing",
                )
                incoming_links_by_memory_id = self._links_by_memory_id(
                    cursor,
                    existing_memory_ids,
                    direction="incoming",
                )
                superseded_target_ids = self._normalize_values(
                    [
                        link.target_id
                        for memory_id in existing_memory_ids
                        for link in outgoing_links_by_memory_id.get(memory_id, [])
                        if link.link_type == "SUPERSEDES"
                    ]
                )
                superseded_rows_by_id = self._memory_rows_by_id(cursor, superseded_target_ids)
                superseded_workspace_ids_by_memory_id = self._workspace_ids_by_memory_id(
                    cursor,
                    superseded_target_ids,
                ) if superseded_target_ids else {}
                superseded_tags_by_memory_id = (
                    self._tags_by_memory_id(cursor, superseded_target_ids)
                    if superseded_target_ids
                    else {}
                )

        contexts: list[RelationalMemoryReadContext] = []
        for memory_id in existing_memory_ids:
            record = self._record_from_memory_row(
                memory_rows_by_id[memory_id],
                workspace_ids=workspace_ids_by_memory_id.get(memory_id, []),
                tags=tags_by_memory_id.get(memory_id, []),
            )
            outgoing = outgoing_links_by_memory_id.get(memory_id, [])
            superseded = [
                self._record_from_memory_row(
                    superseded_rows_by_id[link.target_id],
                    workspace_ids=superseded_workspace_ids_by_memory_id.get(link.target_id, []),
                    tags=superseded_tags_by_memory_id.get(link.target_id, []),
                )
                for link in outgoing
                if link.link_type == "SUPERSEDES" and link.target_id in superseded_rows_by_id
            ]
            contexts.append(
                RelationalMemoryReadContext(
                    record=record,
                    relationships={
                        "outgoing": outgoing,
                        "incoming": incoming_links_by_memory_id.get(memory_id, []),
                    },
                    superseded=superseded,
                )
            )
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
        resolved_memory_id = self.resolve_memory_id(memory_id)
        if resolved_memory_id is None:
            return None
        existing = self.get_memory(resolved_memory_id)
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

        if normalized_status is not None and normalized_status != existing.status:
            columns.append("archived_at = %s")
            values.append(self._utc_now() if normalized_status == "archived" else None)

        columns.append("updated_at = %s")
        values.append(self._utc_now())
        values.append(resolved_memory_id)

        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"UPDATE memories SET {', '.join(columns)} WHERE id = %s",
                    tuple(values),
                )
                if workspace_ids is not None:
                    self._replace_workspace_mappings(
                        cursor,
                        resolved_memory_id,
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
                    memory_id=resolved_memory_id,
                    title=normalized_title or existing.title,
                    summary=cast(str | None, existing.summary if resolved_summary is _SUMMARY_UNSET else resolved_summary),
                    content=normalized_content or existing.content,
                    tags=effective_tags,
                )
            connection.commit()

        return self.get_memory(resolved_memory_id)

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
            "SELECT id, title, content, summary, type, status, created_at, updated_at, "
            "read_count, access_score, last_accessed_at, last_surfaced_at, metadata, memory_ref, archived_at "
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

                memory_ids = [str(row[0]) for row in rows]
                workspace_ids_by_memory_id = self._workspace_ids_by_memory_id(cursor, memory_ids)
                tags_by_memory_id = self._tags_by_memory_id(cursor, memory_ids)

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
                memory_ref=self._coerce_int(row[13]),
                archived_at=None if row[14] is None else str(row[14]),
                workspace_ids=workspace_ids_by_memory_id.get(memory_id, []),
                tags=tags_by_memory_id.get(memory_id, []),
            )
        return [records_by_id[memory_id] for memory_id in memory_ids if memory_id in records_by_id]

    def list_skill_review_observations(
        self,
        *,
        limit: int,
        offset: int,
    ) -> list[RelationalMemoryRecord]:
        """List the authoritative global open skill-observation ledger slice."""
        if limit <= 0 or offset < 0:
            raise ValueError("ledger limit and offset are invalid")
        query = """
            SELECT DISTINCT memories.id, memories.title, memories.content, memories.summary,
                memories.type, memories.status, memories.created_at, memories.updated_at,
                memories.read_count, memories.access_score, memories.last_accessed_at,
                memories.last_surfaced_at, memories.metadata, memories.memory_ref, memories.archived_at
            FROM memories
            WHERE memories.status = 'active'
              AND memories.type = 'observation'
              AND EXISTS (
                  SELECT 1
                  FROM memory_tags
                  JOIN tags ON tags.id = memory_tags.tag_id
                  WHERE memory_tags.memory_id = memories.id
                    AND tags.name = 'skill-observation'
              )
            ORDER BY memories.updated_at ASC, memories.created_at ASC, memories.id ASC
            LIMIT %s OFFSET %s
        """
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, (limit, offset))
                rows = cursor.fetchall()
                memory_ids = [str(row[0]) for row in rows]
                workspace_ids_by_memory_id = self._workspace_ids_by_memory_id(cursor, memory_ids)
                tags_by_memory_id = self._tags_by_memory_id(cursor, memory_ids)
        return [
            RelationalMemoryRecord(
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
                memory_ref=self._coerce_int(row[13]),
                archived_at=None if row[14] is None else str(row[14]),
                workspace_ids=workspace_ids_by_memory_id.get(str(row[0]), []),
                tags=tags_by_memory_id.get(str(row[0]), []),
            )
            for row in rows
        ]

    def list_memory_ids(
        self,
        workspace_id: str | None = None,
        memory_type: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[str]:
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

        query = "SELECT memories.id FROM memories "
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
                return [str(row[0]) for row in rows]

    def count_memories_updated_since(
        self,
        cutoff: str,
        *,
        workspace_id: str | None = None,
    ) -> int:
        joins: list[str] = []
        clauses: list[str] = []
        params: list[object] = []
        if workspace_id is not None:
            joins.append("JOIN memory_workspaces ON memory_workspaces.memory_id = memories.id")
            clauses.append("memory_workspaces.workspace_id = %s")
            params.append(workspace_id)
        clauses.append("memories.updated_at >= %s")
        params.append(cutoff)

        query = "SELECT COUNT(*) FROM memories "
        if joins:
            query += " ".join(joins) + " "
        if clauses:
            query += "WHERE " + " AND ".join(clauses) + " "

        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, tuple(params))
                row = cursor.fetchone()
                return self._coerce_int(row[0]) if row is not None else 0

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
        for tag in tags or ():
            clauses.append(
                "EXISTS (SELECT 1 FROM memory_tags JOIN tags ON tags.id = memory_tags.tag_id "
                "WHERE memory_tags.memory_id = memories.id AND tags.name = %s)"
            )
            params.append(tag)
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
        resolved_ids = [
            resolved_id
            for memory_id in normalized_ids
            for resolved_id in [self.resolve_memory_id(memory_id)]
            if resolved_id is not None
        ]
        if not resolved_ids:
            return []

        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                clauses: list[str] = []
                params: list[object] = [resolved_ids]
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
                        memories.memory_ref,
                        memories.archived_at,
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
            has_incoming_supersedes = self._coerce_int(row[18]) > 0
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
                        memory_ref=self._coerce_int(row[13]),
                        archived_at=None if row[14] is None else str(row[14]),
                        workspace_ids=self._load_text_values(row[15]),
                        tags=self._load_text_values(row[16]),
                    ),
                    incoming_links_count=self._coerce_int(row[17]),
                    has_incoming_supersedes=has_incoming_supersedes,
                    incoming_link_type_counts={
                        "DEPENDS_ON": self._coerce_int(row[19]),
                        "AMENDS": self._coerce_int(row[20]),
                        "CONTRADICTS": self._coerce_int(row[21]),
                        "SUPERSEDES": self._coerce_int(row[22]),
                    },
                )
            )
        self._record_timing_ms(timing_ms, "candidate_hydration_candidate_build", candidate_build_started)
        return candidates

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
        """Return a bounded, globally scoped maintenance candidate frontier."""
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

        clauses = ["TRUE"]
        params: list[object] = []
        if workspace_id is not None:
            clauses.append(
                "EXISTS (SELECT 1 FROM memory_workspaces scoped_workspace "
                "WHERE scoped_workspace.memory_id = memories.id "
                "AND scoped_workspace.workspace_id = %s)"
            )
            params.append(workspace_id)
        if status is not None:
            clauses.append("memories.status = %s")
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
        elif normalized_strategy == "oversized/thin":
            clauses.append(
                "(LENGTH(COALESCE(memories.content, '')) > %s OR "
                "(LENGTH(COALESCE(memories.content, '')) <= %s "
                "AND memories.metadata ? 'split_from_memory_id'))"
            )
            params.extend([oversized_min_chars, thin_max_chars])
            order_by = (
                "LENGTH(COALESCE(memories.content, '')) DESC, memories.read_count ASC, "
                "memories.updated_at ASC, memories.id ASC"
            )
        elif normalized_strategy == "orphan/low-support":
            clauses.append("COALESCE(incoming_counts.incoming_links_count, 0) <= %s")
            params.append(low_support_max)
            order_by = (
                "COALESCE(incoming_counts.incoming_links_count, 0) ASC, memories.read_count ASC, "
                "memories.access_score ASC, memories.last_surfaced_at IS NULL DESC, "
                "memories.last_surfaced_at ASC, memories.id ASC"
            )
        elif normalized_strategy == "quality-signal":
            quality_clause, quality_params = self._quality_signal_clause(quality_signal)
            clauses.append(quality_clause)
            params.extend(quality_params)
        elif normalized_strategy == "retrieval-quality":
            clauses.append(
                "COALESCE(retrieval_stats.search_count, 0) >= %s AND "
                "COALESCE(retrieval_stats.read_count, 0)::double precision / "
                "NULLIF(retrieval_stats.search_count, 0) <= %s"
            )
            params.extend([retrieval_min_searches, retrieval_max_conversion_rate])
            order_by = (
                "COALESCE(retrieval_stats.search_count, 0) DESC, "
                "COALESCE(retrieval_stats.read_count, 0) ASC, memories.updated_at ASC, memories.id ASC"
            )
        else:
            order_by = "md5(CONCAT(%s::text, ':', memories.id)) ASC, memories.id ASC"
            params.append(str(seed))

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
                        memories.memory_ref,
                        memories.archived_at,
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
                    LEFT JOIN (
                        SELECT target_id AS memory_id, COUNT(*) AS incoming_links_count
                        FROM links
                        GROUP BY target_id
                    ) incoming_counts ON incoming_counts.memory_id = memories.id
                    LEFT JOIN (
                        SELECT memory_id,
                               COUNT(*) FILTER (WHERE event_kind = 'search') AS search_count,
                               COUNT(*) FILTER (WHERE event_kind = 'read') AS read_count
                        FROM memory_tool_events
                        WHERE memory_id IS NOT NULL
                        GROUP BY memory_id
                    ) retrieval_stats ON retrieval_stats.memory_id = memories.id
                    WHERE
                    """
                    + " AND ".join(clauses)
                    + " ORDER BY "
                    + order_by
                    + " LIMIT %s",
                    tuple([*params, limit]),
                )
                rows = cursor.fetchall()

        return [
            RelationalMemoryRecord(
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
                memory_ref=self._coerce_int(row[13]),
                archived_at=None if row[14] is None else str(row[14]),
                workspace_ids=self._load_text_values(row[15]),
                tags=self._load_text_values(row[16]),
            )
            for row in rows
        ]

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

        if best_effort:
            self._last_surfaced_writer.write_many(
                [(memory_id, surfaced_at) for memory_id in resolved_ids]
            )
            return len(resolved_ids)

        self._update_last_surfaced_ids(resolved_ids, surfaced_at)
        return len(resolved_ids)

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
        resolved_memory_id = self.resolve_memory_id(memory_id)
        if resolved_memory_id is None:
            return None
        normalized_workspace_ids = self._normalize_values(workspace_ids)
        if not normalized_workspace_ids:
            return self.get_memory(resolved_memory_id)

        current = self.get_memory(resolved_memory_id)
        if current is None:
            return None

        merged_workspace_ids = self._normalize_values([*current.workspace_ids, *normalized_workspace_ids])
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                self._replace_workspace_mappings(cursor, resolved_memory_id, merged_workspace_ids)
                cursor.execute(
                    "UPDATE memories SET updated_at = %s WHERE id = %s",
                    (self._utc_now(), resolved_memory_id),
                )
            connection.commit()
        return self.get_memory(resolved_memory_id)

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
        resolved_memory_id = self.resolve_memory_id(memory_id)
        if resolved_memory_id is None:
            return []
        if direction == "incoming":
            clause = "target_id = %s"
            order_by = "ORDER BY source_id ASC, target_id ASC"
        else:
            clause = "source_id = %s"
            order_by = "ORDER BY source_id ASC, target_id ASC"

        query = f"SELECT source_id, target_id, type, context FROM links WHERE {clause}"
        params: list[object] = [resolved_memory_id]
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
        resolved_memory_id = self.resolve_memory_id(memory_id)
        if resolved_memory_id is None:
            return False
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT 1 FROM links WHERE target_id = %s AND type = %s LIMIT 1",
                    (resolved_memory_id, normalized_link_type),
                )
                return cursor.fetchone() is not None

    def count_incoming_links(self, memory_id: str) -> int:
        resolved_memory_id = self.resolve_memory_id(memory_id)
        if resolved_memory_id is None:
            return 0
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT COUNT(*) FROM links WHERE target_id = %s",
                    (resolved_memory_id,),
                )
                row = cursor.fetchone()
                return self._coerce_int(row[0]) if row is not None else 0

    def delete_memory(self, memory_id: str):
        resolved_memory_id = self.resolve_memory_id(memory_id)
        if resolved_memory_id is None:
            return None
        existing = self.get_memory(resolved_memory_id)
        if existing is None:
            return None

        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "DELETE FROM links WHERE source_id = %s OR target_id = %s",
                    (resolved_memory_id, resolved_memory_id),
                )
                cursor.execute("DELETE FROM memory_search_documents WHERE memory_id = %s", (resolved_memory_id,))
                cursor.execute("DELETE FROM memories WHERE id = %s", (resolved_memory_id,))
            connection.commit()
        return existing

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
        read_count_clause = "read_count = read_count + 1, " if increment_read_count else ""
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"UPDATE memories SET {read_count_clause}access_score = %s, last_accessed_at = %s WHERE id = %s",
                    (access_score, accessed_at, resolved_memory_id),
                )
            connection.commit()
        return self.get_memory(resolved_memory_id)

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
            "read_count, access_score, last_accessed_at, last_surfaced_at, metadata, memory_ref, archived_at FROM memories "
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
                   read_count, access_score, last_accessed_at, last_surfaced_at, metadata, memory_ref,
                   archived_at
            FROM memories
            WHERE id = %s
            """,
            (memory_id,),
        )
        return cursor.fetchone()

    def _memory_rows_by_id(
        self,
        cursor: CursorLike,
        memory_ids: list[str],
    ) -> dict[str, tuple[object, ...]]:
        if not memory_ids:
            return {}
        cursor.execute(
            """
            SELECT id, title, content, summary, type, status, created_at, updated_at,
                   read_count, access_score, last_accessed_at, last_surfaced_at, metadata, memory_ref,
                   archived_at
            FROM memories
            WHERE id = ANY(%s::text[])
            """,
            (memory_ids,),
        )
        return {str(row[0]): row for row in cursor.fetchall()}

    def _links_by_memory_id(
        self,
        cursor: CursorLike,
        memory_ids: list[str],
        *,
        direction: str,
    ) -> dict[str, list[MemoryLink]]:
        if not memory_ids:
            return {}
        if direction == "incoming":
            clause = "target_id = ANY(%s::text[])"
            grouping_index = 1
        else:
            clause = "source_id = ANY(%s::text[])"
            grouping_index = 0
        cursor.execute(
            f"SELECT source_id, target_id, type, context FROM links WHERE {clause} ORDER BY source_id ASC, target_id ASC",
            (memory_ids,),
        )
        links_by_memory_id: dict[str, list[MemoryLink]] = {}
        for row in cursor.fetchall():
            link = MemoryLink(
                source_id=str(row[0]),
                target_id=str(row[1]),
                link_type=str(row[2]),
                context=str(row[3]),
            )
            memory_id = link.target_id if grouping_index == 1 else link.source_id
            links_by_memory_id.setdefault(memory_id, []).append(link)
        return links_by_memory_id

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
            memory_ref=self._coerce_int(row[13]),
            archived_at=None if row[14] is None else str(row[14]),
            workspace_ids=[str(workspace_row[0]) for workspace_row in workspace_rows],
            tags=[str(tag_row[0]) for tag_row in tag_rows],
        )

    def _record_from_memory_row(
        self,
        row: tuple[object, ...],
        *,
        workspace_ids: list[str] | None = None,
        tags: list[str] | None = None,
    ) -> RelationalMemoryRecord:
        return RelationalMemoryRecord(
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
            memory_ref=self._coerce_int(row[13]),
            archived_at=None if row[14] is None else str(row[14]),
            workspace_ids=[] if workspace_ids is None else workspace_ids,
            tags=[] if tags is None else tags,
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

    def _quality_signal_clause(self, quality_signal: str | None) -> tuple[str, list[object]]:
        signal = None if quality_signal is None else quality_signal.strip().lower()
        if signal is not None:
            signal = _QUALITY_SIGNAL_ALIASES.get(signal, signal)
        clauses: dict[str, tuple[str, list[object]]] = {
            "trace_like_memory_count": (
                (
                    "(LOWER(TRIM(memories.title)) LIKE %s OR "
                    "LOWER(TRIM(memories.title)) LIKE %s OR "
                    "LOWER(TRIM(memories.title)) LIKE %s)"
                ),
                ["task_complete%", "task complete%", "task_complete_record%"],
            ),
            "generic_summary_count": (
                (
                    "(LOWER(TRIM(COALESCE(memories.summary, ''))) LIKE %s OR "
                    "LOWER(TRIM(COALESCE(memories.summary, ''))) LIKE %s)"
                ),
                ["covers %", "added %"],
            ),
            "untagged_observation_count": (
                "memories.type = 'observation' AND NOT EXISTS ("
                "SELECT 1 FROM memory_tags untagged WHERE untagged.memory_id = memories.id)",
                [],
            ),
            "oversized_memory_count": (
                "LENGTH(COALESCE(memories.content, '')) >= %s",
                [_DEFAULT_QUALITY_OVERSIZED_MIN_CHARS],
            ),
            "raw_ingress_count": (
                "(memories.metadata->>'created_via_ingest' = 'true' "
                "AND memories.type IN ('journal', 'observation') "
                "AND (NOT EXISTS (SELECT 1 FROM memory_tags raw_untagged "
                "WHERE raw_untagged.memory_id = memories.id) "
                "OR LENGTH(COALESCE(memories.content, '')) > 1600 "
                "OR LOWER(TRIM(COALESCE(memories.summary, ''))) LIKE %s "
                "OR LOWER(TRIM(COALESCE(memories.summary, ''))) LIKE %s))",
                ["covers %", "added %"],
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
