from __future__ import annotations

import json
import sqlite3
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
from mcp_memory.utils.db import DatabaseManager

_ResultT = TypeVar("_ResultT")
_ARCHIVED_MEMORY_GC_EXAMPLE_LIMIT = 10
_DANGLING_LINK_EXAMPLE_LIMIT = 10


class SQLiteMaintenanceHousekeeping(MaintenanceHousekeepingPort):
    def __init__(self, db_manager: DatabaseManager) -> None:
        self._db_manager = db_manager

    def execute(
        self,
        operation: Callable[[MaintenanceHousekeepingTransaction], _ResultT],
    ) -> _ResultT:
        connection = self._db_manager.get_connection()
        try:
            result = operation(_SQLiteMaintenanceHousekeepingTransaction(connection))
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        return result


class _SQLiteMaintenanceHousekeepingTransaction:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def mark_stale_plans(self, cutoff: str, workspace_id: str | None) -> int:
        if workspace_id is None:
            cursor = self._connection.execute(
                """
                UPDATE memories
                SET status = 'stale'
                WHERE type = 'plan' AND status = 'active' AND updated_at < ?
                """,
                (cutoff,),
            )
        else:
            cursor = self._connection.execute(
                """
                UPDATE memories
                SET status = 'stale'
                WHERE id IN (
                    SELECT memories.id
                    FROM memories
                    JOIN memory_workspaces ON memory_workspaces.memory_id = memories.id
                    WHERE memories.type = 'plan'
                      AND memories.status = 'active'
                      AND memories.updated_at < ?
                      AND memory_workspaces.workspace_id = ?
                )
                """,
                (cutoff, workspace_id),
            )
        return int(cursor.rowcount or 0)

    def list_external_links(self) -> list[ExternalLinkRecord]:
        rows = self._connection.execute(
            """
            SELECT memories.id, memories.status, links.target_id, memory_workspaces.workspace_id
            FROM memories
            JOIN links ON links.source_id = memories.id
            LEFT JOIN memory_workspaces ON memory_workspaces.memory_id = memories.id
            WHERE links.target_id LIKE 'ext:%'
            """
        ).fetchall()
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
        placeholders = ",".join("?" for _ in memory_ids)
        params: list[object] = [status, *sorted(memory_ids)]
        query = f"UPDATE memories SET status = ? WHERE id IN ({placeholders})"
        if current_status is not None:
            query += " AND status = ?"
            params.append(current_status)
        self._connection.execute(query, params)

    def list_lineage_memories(self) -> list[LineageMemoryRecord]:
        metadata_rows = self._connection.execute(
            """
            SELECT id, status, metadata
            FROM memories
            WHERE status != 'archived'
            """
        ).fetchall()
        relationship_rows = self._connection.execute(
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
        ).fetchall()
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
            rows = self._connection.execute(
                """
                SELECT links.source_id, links.target_id, links.type
                FROM links
                WHERE NOT EXISTS (SELECT 1 FROM memories WHERE memories.id = links.source_id)
                   OR (
                       links.target_id NOT LIKE 'ext:%'
                       AND NOT EXISTS (SELECT 1 FROM memories WHERE memories.id = links.target_id)
                   )
                ORDER BY links.source_id ASC, links.target_id ASC, links.type ASC
                LIMIT ?
                """,
                (batch_size,),
            ).fetchall()
            if not rows:
                break
            scanned += len(rows)
            predicates: list[str] = []
            link_params: list[object] = []
            for source_id, target_id, link_type in rows:
                predicates.append("(source_id = ? AND target_id = ? AND type = ?)")
                link_params.extend((source_id, target_id, link_type))
                if len(examples) < _DANGLING_LINK_EXAMPLE_LIMIT:
                    examples.append(
                        {
                            "source_id": str(source_id),
                            "target_id": str(target_id),
                            "type": str(link_type),
                        }
                    )
            cursor = self._connection.execute(
                f"DELETE FROM links WHERE {' OR '.join(predicates)}",
                link_params,
            )
            deleted += int(cursor.rowcount or 0)
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
        rows = self._connection.execute(
            """
            SELECT memories.id,
                   EXISTS (
                       SELECT 1
                       FROM memory_protections
                       WHERE memory_protections.memory_id = memories.id
                         AND (memory_protections.expires_at IS NULL OR memory_protections.expires_at > ?)
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
              AND memories.archived_at < ?
            ORDER BY memories.archived_at ASC, memories.id ASC
            LIMIT ?
            """,
            (now_text, cutoff_text, batch_size),
        ).fetchall()
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
                cursor = self._connection.execute(
                    """
                    DELETE FROM memories
                    WHERE id = ?
                      AND status = 'archived'
                      AND archived_at IS NOT NULL
                      AND archived_at < ?
                      AND NOT EXISTS (
                          SELECT 1
                          FROM memory_protections
                          WHERE memory_protections.memory_id = memories.id
                            AND (memory_protections.expires_at IS NULL OR memory_protections.expires_at > ?)
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
                if cursor.rowcount != 1:
                    continue
                self._connection.execute(
                    "DELETE FROM links WHERE source_id = ? OR target_id = ?",
                    (memory_id, memory_id),
                )
                self._connection.execute(
                    "DELETE FROM memories_fts WHERE memory_id = ?",
                    (memory_id,),
                )
                self._connection.execute(
                    "DELETE FROM embeddings WHERE source_kind = 'memory' AND source_id = ?",
                    (memory_id,),
                )
                self._connection.execute(
                    "DELETE FROM memory_protections WHERE memory_id = ?",
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
        cursor = self._connection.execute(
            "DELETE FROM tasks WHERE status = 'completed' AND updated_at < ?",
            (cutoff_timestamp,),
        )
        return int(cursor.rowcount or 0)

    def purge_recoverable_journal_entries(self, cutoff_timestamp: float) -> list[int]:
        rows = self._connection.execute(
            """
            SELECT id
            FROM system1_journal
            WHERE status = 'recoverable'
              AND recoverable_until IS NOT NULL
              AND recoverable_until <= ?
            ORDER BY timestamp ASC
            """,
            (cutoff_timestamp,),
        ).fetchall()
        entry_ids = [int(row[0]) for row in rows]
        if not entry_ids:
            return []
        placeholders = ",".join("?" for _ in entry_ids)
        self._connection.execute(
            f"DELETE FROM system1_journal WHERE id IN ({placeholders}) AND status = 'recoverable'",
            entry_ids,
        )
        return entry_ids

    def purge_processed_journal_entries(self, cutoff_timestamp: float) -> int:
        cursor = self._connection.execute(
            "DELETE FROM system1_journal WHERE status IN ('processed', 'archived') AND timestamp < ?",
            (cutoff_timestamp,),
        )
        return int(cursor.rowcount or 0)

    def gc_dead_metadata_keys(self, keys: Sequence[str]) -> int:
        rows = self._connection.execute(
            """
            SELECT id, metadata
            FROM memories
            WHERE metadata IS NOT NULL AND metadata != '{}'
            """
        ).fetchall()
        updated = 0
        dead_keys = set(keys)
        for memory_id, raw_metadata in rows:
            metadata = _decode_metadata(raw_metadata)
            if not metadata:
                continue
            pruned = {key: value for key, value in metadata.items() if key not in dead_keys}
            if pruned == metadata:
                continue
            self._connection.execute(
                "UPDATE memories SET metadata = ? WHERE id = ?",
                (json.dumps(pruned, sort_keys=True), memory_id),
            )
            updated += 1
        return updated


def _decode_metadata(raw_metadata: object) -> dict[str, object]:
    if isinstance(raw_metadata, dict):
        return dict(raw_metadata)
    if raw_metadata is None:
        return {}
    try:
        decoded = json.loads(str(raw_metadata) or "{}")
    except (TypeError, ValueError):
        return {}
    if not isinstance(decoded, dict):
        return {}
    return dict(decoded)


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


__all__ = ["SQLiteMaintenanceHousekeeping"]
