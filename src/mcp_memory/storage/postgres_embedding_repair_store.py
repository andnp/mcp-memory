from __future__ import annotations

import time
from uuid import uuid4

from mcp_memory.embedding_repair_store import (
    DEFAULT_EMBEDDING_REPAIR_LEASE_TTL_SECONDS,
    EMBEDDING_REPAIR_STATUS_COMPLETED,
    EMBEDDING_REPAIR_STATUS_PENDING,
    EMBEDDING_REPAIR_STATUS_RUNNING,
    EmbeddingRepairBacklogSnapshot,
    EmbeddingRepairQueueItem,
)
from mcp_memory.storage.session import DbConnectionLike, SessionManager


def _coerce_float(value: object) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"Expected float-compatible value, got {type(value)!r}")


def _coerce_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        return int(value)
    raise TypeError(f"Expected int-compatible value, got {type(value)!r}")


class PostgresEmbeddingRepairQueue:
    def __init__(self, session_manager: SessionManager[DbConnectionLike] | None) -> None:
        self._sessions = session_manager

    def enqueue_unique(
        self,
        *,
        memory_id: str,
        model_name: str,
        memory_updated_at: str,
        workspace_id: str | None = None,
        available_at: float | None = None,
    ) -> tuple[EmbeddingRepairQueueItem, bool]:
        if self._sessions is None:
            raise RuntimeError("embedding_repair_queue_unavailable")
        item_id = str(uuid4())
        now = time.time()
        ready_at = now if available_at is None else float(available_at)
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO embedding_repair_queue (
                        id, memory_id, workspace_id, model_name, memory_updated_at,
                        status, attempt_count, available_at, created_at, updated_at,
                        claimed_at, completed_at, lease_owner, lease_expires_at, last_error
                    ) VALUES (%s, %s, %s, %s, %s, %s, 0, %s, %s, %s, NULL, NULL, NULL, NULL, NULL)
                    ON CONFLICT (memory_id, model_name, memory_updated_at) DO NOTHING
                    """,
                    (
                        item_id,
                        memory_id,
                        workspace_id,
                        model_name,
                        memory_updated_at,
                        EMBEDDING_REPAIR_STATUS_PENDING,
                        ready_at,
                        now,
                        now,
                    ),
                )
                inserted = int(getattr(cursor, "rowcount", 0) or 0) == 1
            connection.commit()
        if not inserted:
            existing = self.find_by_version(memory_id=memory_id, model_name=model_name, memory_updated_at=memory_updated_at)
            if existing is None:
                raise RuntimeError("embedding_repair_unique_conflict_missing_row")
            return existing, False
        return self.get_item(item_id), True

    def find_by_version(self, *, memory_id: str, model_name: str, memory_updated_at: str) -> EmbeddingRepairQueueItem | None:
        if self._sessions is None:
            return None
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id, memory_id, workspace_id, model_name, memory_updated_at, status,
                           attempt_count, available_at, created_at, updated_at, claimed_at,
                           completed_at, lease_owner, lease_expires_at, last_error
                    FROM embedding_repair_queue
                    WHERE memory_id = %s AND model_name = %s AND memory_updated_at = %s
                    LIMIT 1
                    """,
                    (memory_id, model_name, memory_updated_at),
                )
                row = cursor.fetchone()
        return None if row is None else self._row_to_item(row)

    def claim_batch(
        self,
        *,
        lease_owner: str,
        limit: int,
        workspace_id: str | None = None,
        now: float | None = None,
        lease_ttl_seconds: float = DEFAULT_EMBEDDING_REPAIR_LEASE_TTL_SECONDS,
    ) -> list[EmbeddingRepairQueueItem]:
        if self._sessions is None:
            return []
        if limit < 1:
            raise ValueError("limit must be at least 1")
        current_time = time.time() if now is None else now
        lease_expires_at = current_time + lease_ttl_seconds
        clauses = [
            "((status = 'pending' AND available_at <= %s) OR (status = 'running' AND lease_expires_at IS NOT NULL AND lease_expires_at <= %s))",
        ]
        params: list[object] = [current_time, current_time]
        if workspace_id is None:
            clauses.append("workspace_id IS NULL")
        elif workspace_id != "*":
            clauses.append("workspace_id = %s")
            params.append(workspace_id)
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id
                    FROM embedding_repair_queue
                    WHERE """
                    + " AND ".join(clauses)
                    + " ORDER BY created_at ASC LIMIT %s FOR UPDATE SKIP LOCKED",
                    tuple([*params, limit]),
                )
                rows = cursor.fetchall()
                item_ids = [str(row[0]) for row in rows]
                if not item_ids:
                    connection.commit()
                    return []
                placeholders = ",".join(["%s"] * len(item_ids))
                cursor.execute(
                    f"""
                    UPDATE embedding_repair_queue
                    SET status = %s,
                        attempt_count = attempt_count + 1,
                        updated_at = %s,
                        claimed_at = %s,
                        lease_owner = %s,
                        lease_expires_at = %s,
                        last_error = NULL
                    WHERE id IN ({placeholders})
                    """,
                    (
                        EMBEDDING_REPAIR_STATUS_RUNNING,
                        current_time,
                        current_time,
                        lease_owner,
                        lease_expires_at,
                        *item_ids,
                    ),
                )
                cursor.execute(
                    f"""
                    SELECT id, memory_id, workspace_id, model_name, memory_updated_at, status,
                           attempt_count, available_at, created_at, updated_at, claimed_at,
                           completed_at, lease_owner, lease_expires_at, last_error
                    FROM embedding_repair_queue
                    WHERE id IN ({placeholders})
                    ORDER BY created_at ASC
                    """,
                    tuple(item_ids),
                )
                claimed_rows = cursor.fetchall()
            connection.commit()
        return [self._row_to_item(row) for row in claimed_rows]

    def complete_item(self, item_id: str, *, completed_at: float | None = None) -> EmbeddingRepairQueueItem:
        if self._sessions is None:
            raise RuntimeError("embedding_repair_queue_unavailable")
        now = time.time() if completed_at is None else completed_at
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE embedding_repair_queue
                    SET status = %s,
                        updated_at = %s,
                        completed_at = %s,
                        lease_owner = NULL,
                        lease_expires_at = NULL,
                        last_error = NULL
                    WHERE id = %s AND status = %s
                    """,
                    (EMBEDDING_REPAIR_STATUS_COMPLETED, now, now, item_id, EMBEDDING_REPAIR_STATUS_RUNNING),
                )
                if int(getattr(cursor, "rowcount", 0) or 0) != 1:
                    connection.rollback()
                    raise ValueError(f"Embedding repair item {item_id} is not running")
            connection.commit()
        return self.get_item(item_id)

    def release_item(self, item_id: str, *, released_at: float | None = None, error: str | None = None) -> EmbeddingRepairQueueItem:
        if self._sessions is None:
            raise RuntimeError("embedding_repair_queue_unavailable")
        now = time.time() if released_at is None else released_at
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE embedding_repair_queue
                    SET status = %s,
                        updated_at = %s,
                        lease_owner = NULL,
                        lease_expires_at = NULL,
                        last_error = %s
                    WHERE id = %s AND status = %s
                    """,
                    (EMBEDDING_REPAIR_STATUS_PENDING, now, error, item_id, EMBEDDING_REPAIR_STATUS_RUNNING),
                )
                if int(getattr(cursor, "rowcount", 0) or 0) != 1:
                    connection.rollback()
                    raise ValueError(f"Embedding repair item {item_id} is not running")
            connection.commit()
        return self.get_item(item_id)

    def get_item(self, item_id: str) -> EmbeddingRepairQueueItem:
        if self._sessions is None:
            raise RuntimeError("embedding_repair_queue_unavailable")
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id, memory_id, workspace_id, model_name, memory_updated_at, status,
                           attempt_count, available_at, created_at, updated_at, claimed_at,
                           completed_at, lease_owner, lease_expires_at, last_error
                    FROM embedding_repair_queue
                    WHERE id = %s
                    """,
                    (item_id,),
                )
                row = cursor.fetchone()
        if row is None:
            raise ValueError(f"Embedding repair item {item_id} was not found")
        return self._row_to_item(row)

    def list_items(
        self,
        *,
        status: str | None = None,
        workspace_id: str | None = None,
        lease_owner: str | None = None,
        limit: int = 50,
    ) -> list[EmbeddingRepairQueueItem]:
        if self._sessions is None:
            return []
        query = (
            "SELECT id, memory_id, workspace_id, model_name, memory_updated_at, status, attempt_count, "
            "available_at, created_at, updated_at, claimed_at, completed_at, lease_owner, lease_expires_at, last_error "
            "FROM embedding_repair_queue"
        )
        clauses: list[str] = []
        params: list[object] = []
        if status is not None:
            clauses.append("status = %s")
            params.append(status)
        if workspace_id is not None:
            clauses.append("workspace_id = %s")
            params.append(workspace_id)
        if lease_owner is not None:
            clauses.append("lease_owner = %s")
            params.append(lease_owner)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY updated_at DESC, created_at DESC LIMIT %s"
        params.append(limit)
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, tuple(params))
                return [self._row_to_item(row) for row in cursor.fetchall()]

    def backlog_snapshot(self, *, now: float | None = None) -> EmbeddingRepairBacklogSnapshot:
        if self._sessions is None:
            return EmbeddingRepairBacklogSnapshot(queued_count=0, running_count=0, oldest_queued_age_seconds=None)
        current_time = time.time() if now is None else now
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                        COALESCE(SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END), 0),
                        COALESCE(SUM(CASE WHEN status = 'running' THEN 1 ELSE 0 END), 0),
                        MIN(CASE WHEN status = 'pending' THEN created_at END)
                    FROM embedding_repair_queue
                    """,
                    (),
                )
                row = cursor.fetchone()
        queued_count = 0 if row is None or row[0] is None else _coerce_int(row[0])
        running_count = 0 if row is None or row[1] is None else _coerce_int(row[1])
        oldest_created_at = None if row is None else row[2]
        oldest_age = None if oldest_created_at is None else max(current_time - _coerce_float(oldest_created_at), 0.0)
        return EmbeddingRepairBacklogSnapshot(
            queued_count=queued_count,
            running_count=running_count,
            oldest_queued_age_seconds=oldest_age,
        )

    def prune_completed(
        self,
        *,
        older_than_seconds: float = 0.0,
        limit: int = 500,
        now: float | None = None,
    ) -> int:
        if self._sessions is None or limit < 1:
            return 0
        current_time = time.time() if now is None else now
        completed_before = current_time - max(float(older_than_seconds), 0.0)
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id FROM embedding_repair_queue
                    WHERE status = %s AND completed_at IS NOT NULL AND completed_at <= %s
                    ORDER BY completed_at ASC
                    LIMIT %s
                    """,
                    (EMBEDDING_REPAIR_STATUS_COMPLETED, completed_before, limit),
                )
                rows = cursor.fetchall()
                item_ids = [str(row[0]) for row in rows]
                if not item_ids:
                    connection.commit()
                    return 0
                placeholders = ",".join(["%s"] * len(item_ids))
                cursor.execute(
                    f"DELETE FROM embedding_repair_queue WHERE id IN ({placeholders})",
                    tuple(item_ids),
                )
                deleted = int(getattr(cursor, "rowcount", 0) or 0)
            connection.commit()
        return deleted

    def _row_to_item(self, row: tuple[object, ...]) -> EmbeddingRepairQueueItem:
        return EmbeddingRepairQueueItem(
            id=str(row[0]),
            memory_id=str(row[1]),
            workspace_id=None if row[2] is None else str(row[2]),
            model_name=str(row[3]),
            memory_updated_at=str(row[4]),
            status=str(row[5]),
            attempt_count=_coerce_int(row[6]),
            available_at=_coerce_float(row[7]),
            created_at=_coerce_float(row[8]),
            updated_at=_coerce_float(row[9]),
            claimed_at=None if row[10] is None else _coerce_float(row[10]),
            completed_at=None if row[11] is None else _coerce_float(row[11]),
            lease_owner=None if row[12] is None else str(row[12]),
            lease_expires_at=None if row[13] is None else _coerce_float(row[13]),
            last_error=None if row[14] is None else str(row[14]),
        )
