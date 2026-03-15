from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import uuid4

from mcp_memory.utils.db import DatabaseManager

VALID_MEMORY_TYPES = frozenset({"journal", "plan", "fact", "observation", "reflection"})
VALID_MEMORY_STATUSES = frozenset({"active", "stale", "degraded", "archived"})


@dataclass
class RelationalMemoryRecord:
    id: str
    title: str
    content: str
    summary: str | None
    type: str
    status: str
    created_at: str
    updated_at: str
    read_count: int
    access_score: float
    last_accessed_at: str | None
    last_surfaced_at: str | None
    metadata: dict[str, object] = field(default_factory=dict)
    workspace_ids: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


@dataclass
class MemoryLink:
    source_id: str
    target_id: str
    link_type: str
    context: str


class RelationalMemoryRepository:
    def __init__(self, db_manager: DatabaseManager) -> None:
        self._db = db_manager

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

        summary_text = summary or self._build_summary(normalized_content)
        payload = json.dumps(metadata or {}, sort_keys=True)
        record_id = memory_id or str(uuid4())

        conn = self._db.get_connection()
        with conn:
            conn.execute(
                """
                INSERT INTO memories (
                    id, title, content, summary, type, status, created_at, updated_at,
                    read_count, access_score, last_accessed_at, last_surfaced_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        conn = self._db.get_connection()
        with conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO links (source_id, target_id, type, context)
                VALUES (?, ?, ?, ?)
                """,
                (source_id, target_id, link_type, context),
            )
        return MemoryLink(
            source_id=source_id,
            target_id=target_id,
            link_type=link_type,
            context=context,
        )

    def remove_link(
        self,
        source_id: str,
        target_id: str,
        link_type: str,
    ) -> bool:
        conn = self._db.get_connection()
        with conn:
            cursor = conn.execute(
                "DELETE FROM links WHERE source_id = ? AND target_id = ? AND type = ?",
                (source_id, target_id, link_type),
            )
        return cursor.rowcount > 0

    def delete_memory(self, memory_id: str):
        existing = self.get_memory(memory_id)
        if existing is None:
            return None

        conn = self._db.get_connection()
        with conn:
            conn.execute(
                "DELETE FROM links WHERE source_id = ? OR target_id = ?",
                (memory_id, memory_id),
            )
            conn.execute("DELETE FROM memories_fts WHERE memory_id = ?", (memory_id,))
            conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        return existing

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
        normalized_query = " ".join(part.strip() for part in query.split() if part.strip())
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
        if not include_superseded:
            clauses.append(
                "NOT EXISTS (SELECT 1 FROM links supersedes WHERE supersedes.target_id = memories.id AND supersedes.type = 'SUPERSEDES')"
            )

        params.append(limit)
        query_sql = (
            "SELECT DISTINCT memories.id, bm25(memories_fts) AS rank FROM memories_fts "
            + " ".join(joins)
            + " WHERE "
            + " AND ".join(clauses)
            + " ORDER BY rank ASC, memories.updated_at DESC LIMIT ?"
        )
        rows = conn.execute(query_sql, params).fetchall()
        return [str(row["id"]) for row in rows]

    def get_links(
        self,
        memory_id: str,
        direction: str = "outgoing",
        link_type: str | None = None,
    ):
        conn = self._db.get_connection()
        if direction == "incoming":
            clause = "target_id = ?"
            order_by = "ORDER BY source_id ASC, target_id ASC"
        else:
            clause = "source_id = ?"
            order_by = "ORDER BY source_id ASC, target_id ASC"

        query = f"SELECT source_id, target_id, type, context FROM links WHERE {clause}"
        params: list[str] = [memory_id]
        if link_type is not None:
            query += " AND type = ?"
            params.append(link_type)
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
        conn = self._db.get_connection()
        row = conn.execute(
            "SELECT 1 FROM links WHERE target_id = ? AND type = ? LIMIT 1",
            (memory_id, link_type),
        ).fetchone()
        return row is not None

    def count_incoming_links(self, memory_id: str) -> int:
        row = self._db.get_connection().execute(
            "SELECT COUNT(*) FROM links WHERE target_id = ?",
            (memory_id,),
        ).fetchone()
        return 0 if row is None else int(row[0])

    def touch_last_surfaced(self, memory_ids: list[str], surfaced_at: str):
        normalized_ids = self._normalize_values(memory_ids)
        if not normalized_ids:
            return 0

        conn = self._db.get_connection()
        placeholders = ",".join("?" for _ in normalized_ids)
        with conn:
            cursor = conn.execute(
                f"UPDATE memories SET last_surfaced_at = ? WHERE id IN ({placeholders})",
                [surfaced_at, *normalized_ids],
            )
        return cursor.rowcount

    def record_access(
        self,
        memory_id: str,
        access_score: float,
        accessed_at: str,
        increment_read_count: bool = False,
    ):
        conn = self._db.get_connection()
        read_count_clause = "read_count = read_count + 1, " if increment_read_count else ""
        with conn:
            cursor = conn.execute(
                f"""
                UPDATE memories
                SET {read_count_clause}access_score = ?, last_accessed_at = ?
                WHERE id = ?
                """,
                (access_score, accessed_at, memory_id),
            )
        if cursor.rowcount == 0:
            return None
        return self.get_memory(memory_id)

    def list_most_read_memories(self, workspace_id: str | None = None, limit: int = 10):
        conn = self._db.get_connection()
        params: list[object] = []
        query = "SELECT DISTINCT memories.* FROM memories"
        if workspace_id is not None:
            query += " JOIN memory_workspaces ON memory_workspaces.memory_id = memories.id"
            params.append(workspace_id)
            query += " WHERE memory_workspaces.workspace_id = ? AND memories.read_count > 0"
        else:
            query += " WHERE memories.read_count > 0"
        query += " ORDER BY memories.read_count DESC, memories.last_accessed_at DESC, memories.updated_at DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(query, params).fetchall()
        return [self._hydrate_record(conn, row) for row in rows]

    def append_workspace_ids(self, memory_id: str, workspace_ids: list[str]):
        normalized_workspace_ids = self._normalize_values(workspace_ids)
        if not normalized_workspace_ids:
            return self.get_memory(memory_id)

        current = self.get_memory(memory_id)
        if current is None:
            return None

        merged_workspace_ids = self._normalize_values(
            [*current.workspace_ids, *normalized_workspace_ids]
        )
        conn = self._db.get_connection()
        with conn:
            self._replace_workspace_mappings(conn, memory_id, merged_workspace_ids)
            conn.execute(
                "UPDATE memories SET updated_at = ? WHERE id = ?",
                (self._utc_now(), memory_id),
            )
        return self.get_memory(memory_id)

    def get_memory(self, memory_id: str):
        conn = self._db.get_connection()
        row = conn.execute(
            "SELECT * FROM memories WHERE id = ?",
            (memory_id,),
        ).fetchone()
        if row is None:
            return None
        return self._hydrate_record(conn, row)

    def update_memory(
        self,
        memory_id: str,
        title: str | None = None,
        content: str | None = None,
        summary: str | None = None,
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
        if self.get_memory(memory_id) is None:
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

        columns = []
        values = []
        updates = {
            "title": normalized_title,
            "content": normalized_content,
            "summary": summary,
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

        columns.append("updated_at = ?")
        values.append(self._utc_now())
        values.append(memory_id)

        with conn:
            conn.execute(
                f"UPDATE memories SET {', '.join(columns)} WHERE id = ?",
                values,
            )
            if workspace_ids is not None:
                self._replace_workspace_mappings(
                    conn,
                    memory_id,
                    self._normalize_values(workspace_ids),
                )
            if tags is not None:
                self._replace_tag_mappings(
                    conn,
                    memory_id,
                    self._normalize_values(tags),
                )
            current_row = conn.execute(
                "SELECT title, summary, content FROM memories WHERE id = ?",
                (memory_id,),
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
                    (memory_id,),
                ).fetchall()]
            )
            assert current_row is not None
            self._replace_fts_row(
                conn,
                memory_id,
                title=str(current_row["title"]),
                summary=str(current_row["summary"] or ""),
                content=str(current_row["content"]),
                tags=current_tags,
            )

        return self.get_memory(memory_id)

    def list_memories(
        self,
        workspace_id: str | None = None,
        memory_type: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ):
        conn = self._db.get_connection()
        clauses = []
        params = []

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

    def _utc_now(self):
        return datetime.now(UTC).isoformat()

    def _build_summary(self, content: str):
        stripped = content.strip()
        if not stripped:
            return ""

        sentences = [segment.strip() for segment in stripped.split(".") if segment.strip()]
        if len(sentences) >= 2:
            return ". ".join(sentences[:2]) + "."
        if len(stripped) <= 220:
            return stripped
        return stripped[:217].rstrip() + "..."