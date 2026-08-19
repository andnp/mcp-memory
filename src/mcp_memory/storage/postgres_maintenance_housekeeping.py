from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import TypeVar

from mcp_memory.core.ports.maintenance import (
    ArchivedMemoryGcResult,
    DanglingLinkReconciliationResult,
    ExternalLinkRecord,
    LineageMemoryRecord,
    MaintenanceHousekeepingPort,
    MaintenanceHousekeepingTransaction,
)
from mcp_memory.storage.session import DbConnectionLike, SessionManager

_ResultT = TypeVar("_ResultT")
_ARCHIVED_MEMORY_GC_EXAMPLE_LIMIT = 10
_DANGLING_LINK_EXAMPLE_LIMIT = 10


class PostgresMaintenanceHousekeeping(MaintenanceHousekeepingPort):
    def __init__(self, sessions: SessionManager[DbConnectionLike]) -> None:
        self._sessions = sessions

    def execute(
        self,
        operation: Callable[[MaintenanceHousekeepingTransaction], _ResultT],
    ) -> _ResultT:
        with self._sessions.open_connection() as connection:
            try:
                result = operation(_PostgresMaintenanceHousekeepingTransaction(connection))
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return result


class _PostgresMaintenanceHousekeepingTransaction:
    def __init__(self, connection: DbConnectionLike) -> None:
        self._connection = connection

    def mark_stale_plans(self, cutoff: str, workspace_id: str | None) -> int:
        if workspace_id is None:
            query = """
                UPDATE memories
                SET status = 'stale'
                WHERE type = 'plan' AND status = 'active' AND updated_at < %s
            """
            params: tuple[object, ...] = (cutoff,)
        else:
            query = """
                UPDATE memories
                SET status = 'stale'
                WHERE id IN (
                    SELECT memories.id
                    FROM memories
                    JOIN memory_workspaces ON memory_workspaces.memory_id = memories.id
                    WHERE memories.type = 'plan'
                      AND memories.status = 'active'
                      AND memories.updated_at < %s
                      AND memory_workspaces.workspace_id = %s
                )
            """
            params = (cutoff, workspace_id)
        with self._connection.cursor() as cursor:
            cursor.execute(query, params)
            return _rowcount(cursor)

    def list_external_links(self) -> list[ExternalLinkRecord]:
        with self._connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT memories.id, memories.status, links.target_id, memory_workspaces.workspace_id
                FROM memories
                JOIN links ON links.source_id = memories.id
                LEFT JOIN memory_workspaces ON memory_workspaces.memory_id = memories.id
                WHERE links.target_id LIKE 'ext:%%'
                """
            )
            rows = cursor.fetchall()
        return [
            ExternalLinkRecord(
                memory_id=str(row[0]),
                status=None if row[1] is None else str(row[1]),
                target_id=str(row[2])[4:],
                workspace_id=None if row[3] is None else str(row[3]),
            )
            for row in rows
        ]

    def update_memory_statuses(
        self,
        memory_ids: Sequence[str],
        *,
        status: str,
        current_status: str | None = None,
    ) -> None:
        if not memory_ids:
            return
        placeholders = ",".join("%s" for _ in memory_ids)
        params: list[object] = [status, *sorted(memory_ids)]
        query = f"UPDATE memories SET status = %s WHERE id IN ({placeholders})"
        if current_status is not None:
            query += " AND status = %s"
            params.append(current_status)
        with self._connection.cursor() as cursor:
            cursor.execute(query, tuple(params))

    def list_lineage_memories(self) -> list[LineageMemoryRecord]:
        with self._connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, status, metadata
                FROM memories
                WHERE status != 'archived'
                """
            )
            metadata_rows = cursor.fetchall()
            cursor.execute(
                """
                SELECT memories.id,
                       COALESCE(outgoing.link_count, 0) AS outgoing_count,
                       COALESCE(incoming.link_count, 0) AS incoming_count
                FROM memories
                LEFT JOIN (
                    SELECT source_id AS memory_id, COUNT(*) AS link_count
                    FROM links
                    GROUP BY source_id
                ) outgoing ON outgoing.memory_id = memories.id
                LEFT JOIN (
                    SELECT target_id AS memory_id, COUNT(*) AS link_count
                    FROM links
                    GROUP BY target_id
                ) incoming ON incoming.memory_id = memories.id
                WHERE memories.status != 'archived'
                """
            )
            relationship_rows = cursor.fetchall()
        relationship_counts = {
            str(row[0]): _coerce_int(row[1]) + _coerce_int(row[2])
            for row in relationship_rows
        }
        return [
            LineageMemoryRecord(
                memory_id=str(row[0]),
                status=str(row[1]),
                metadata=row[2],
                relationship_count=relationship_counts.get(str(row[0]), 0),
            )
            for row in metadata_rows
        ]

    def reconcile_dangling_links(self, *, batch_size: int) -> DanglingLinkReconciliationResult:
        scanned = 0
        deleted = 0
        examples: list[dict[str, str]] = []
        while True:
            with self._connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT links.source_id, links.target_id, links.type
                    FROM links
                    WHERE NOT EXISTS (SELECT 1 FROM memories WHERE memories.id = links.source_id)
                       OR (
                           links.target_id NOT LIKE 'ext:%%'
                           AND NOT EXISTS (SELECT 1 FROM memories WHERE memories.id = links.target_id)
                       )
                    ORDER BY links.source_id ASC, links.target_id ASC, links.type ASC
                    LIMIT %s
                    """,
                    (batch_size,),
                )
                rows = cursor.fetchall()
            if not rows:
                break
            scanned += len(rows)
            predicates: list[str] = []
            link_params: list[object] = []
            for source_id, target_id, link_type in rows:
                predicates.append("(source_id = %s AND target_id = %s AND type = %s)")
                link_params.extend((source_id, target_id, link_type))
                if len(examples) < _DANGLING_LINK_EXAMPLE_LIMIT:
                    examples.append(
                        {
                            "source_id": str(source_id),
                            "target_id": str(target_id),
                            "type": str(link_type),
                        }
                    )
            with self._connection.cursor() as cursor:
                cursor.execute(
                    f"DELETE FROM links WHERE {' OR '.join(predicates)}",
                    tuple(link_params),
                )
                deleted += _rowcount(cursor)
        return {"scanned": scanned, "deleted": deleted, "deleted_link_examples": examples}

    def gc_archived_memories(
        self,
        *,
        cutoff: datetime,
        batch_size: int,
        mode: str,
    ) -> ArchivedMemoryGcResult:
        cutoff_text = cutoff.astimezone(UTC).isoformat()
        now_text = datetime.now(UTC).isoformat()
        with self._connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT memories.id,
                       EXISTS (
                           SELECT 1
                           FROM memory_protections
                           WHERE memory_protections.memory_id = memories.id
                             AND (memory_protections.expires_at IS NULL OR memory_protections.expires_at > %s)
                       ) AS is_protected,
                       EXISTS (
                           SELECT 1
                           FROM links
                           WHERE (links.source_id = memories.id OR links.target_id = memories.id)
                             AND links.type <> 'SUPERSEDES'
                       ) AS has_non_supersedes_link
                FROM memories
                WHERE memories.status = 'archived'
                  AND memories.archived_at IS NOT NULL
                  AND memories.archived_at < %s
                ORDER BY memories.archived_at ASC, memories.id ASC
                LIMIT %s
                """,
                (now_text, cutoff_text, batch_size),
            )
            rows = cursor.fetchall()
        eligible_ids: list[str] = []
        skipped_protected = 0
        skipped_linked = 0
        for memory_id, is_protected, has_non_supersedes_link in rows:
            if bool(is_protected):
                skipped_protected += 1
            elif bool(has_non_supersedes_link):
                skipped_linked += 1
            else:
                eligible_ids.append(str(memory_id))

        deleted_ids: list[str] = []
        if mode == "delete":
            for memory_id in eligible_ids:
                with self._connection.cursor() as cursor:
                    cursor.execute(
                        """
                        DELETE FROM memories
                        WHERE id = %s
                          AND status = 'archived'
                          AND archived_at IS NOT NULL
                          AND archived_at < %s
                          AND NOT EXISTS (
                              SELECT 1
                              FROM memory_protections
                              WHERE memory_protections.memory_id = memories.id
                                AND (memory_protections.expires_at IS NULL OR memory_protections.expires_at > %s)
                          )
                          AND NOT EXISTS (
                              SELECT 1
                              FROM links
                              WHERE (links.source_id = memories.id OR links.target_id = memories.id)
                                AND links.type <> 'SUPERSEDES'
                          )
                        """,
                        (memory_id, cutoff_text, now_text),
                    )
                    deleted = _rowcount(cursor)
                if deleted != 1:
                    continue
                with self._connection.cursor() as cursor:
                    cursor.execute(
                        "DELETE FROM links WHERE source_id = %s OR target_id = %s",
                        (memory_id, memory_id),
                    )
                    cursor.execute(
                        "DELETE FROM memory_search_documents WHERE memory_id = %s",
                        (memory_id,),
                    )
                    cursor.execute(
                        "DELETE FROM embeddings WHERE source_kind = 'memory' AND source_id = %s",
                        (memory_id,),
                    )
                    cursor.execute(
                        "DELETE FROM memory_protections WHERE memory_id = %s",
                        (memory_id,),
                    )
                deleted_ids.append(memory_id)
        return {
            "mode": mode,
            "cutoff": cutoff_text,
            "scanned": len(rows),
            "eligible": len(eligible_ids),
            "skipped_protected": skipped_protected,
            "skipped_linked": skipped_linked,
            "deleted": len(deleted_ids),
            "eligible_memory_examples": eligible_ids[:_ARCHIVED_MEMORY_GC_EXAMPLE_LIMIT],
            "deleted_memory_examples": deleted_ids[:_ARCHIVED_MEMORY_GC_EXAMPLE_LIMIT],
        }

    def delete_completed_tasks(self, cutoff_timestamp: float) -> int:
        with self._connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM tasks WHERE status = 'completed' AND updated_at < %s",
                (cutoff_timestamp,),
            )
            return _rowcount(cursor)

    def purge_recoverable_journal_entries(self, cutoff_timestamp: float) -> list[int]:
        with self._connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id
                FROM system1_journal
                WHERE status = 'recoverable'
                  AND recoverable_until IS NOT NULL
                  AND recoverable_until <= %s
                ORDER BY timestamp ASC
                """,
                (cutoff_timestamp,),
            )
            entry_ids = [_coerce_int(row[0]) for row in cursor.fetchall()]
            if not entry_ids:
                return []
            placeholders = ",".join("%s" for _ in entry_ids)
            cursor.execute(
                f"DELETE FROM system1_journal WHERE id IN ({placeholders}) AND status = 'recoverable'",
                tuple(entry_ids),
            )
        return entry_ids

    def purge_processed_journal_entries(self, cutoff_timestamp: float) -> int:
        with self._connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM system1_journal WHERE status IN ('processed', 'archived') AND timestamp < %s",
                (cutoff_timestamp,),
            )
            return _rowcount(cursor)

    def gc_dead_metadata_keys(self, keys: Sequence[str]) -> int:
        removal_expr = " - ".join(["metadata"] + [f"'{key}'" for key in keys])
        key_array = ", ".join(f"'{key}'" for key in keys)
        query = f"""
            UPDATE memories
            SET metadata = {removal_expr}
            WHERE metadata IS NOT NULL
              AND metadata != '{{}}'::jsonb
              AND metadata ?| ARRAY[{key_array}]
        """
        with self._connection.cursor() as cursor:
            cursor.execute(query)
            return _rowcount(cursor)


def _coerce_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return 0


def _rowcount(cursor: object) -> int:
    value = getattr(cursor, "rowcount", 0)
    return value if isinstance(value, int) else 0


__all__ = ["PostgresMaintenanceHousekeeping"]
