from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from uuid import uuid4

from mcp_memory.utils.db import DatabaseManager

EMBEDDING_REPAIR_STATUS_PENDING = "pending"
EMBEDDING_REPAIR_STATUS_RUNNING = "running"
EMBEDDING_REPAIR_STATUS_COMPLETED = "completed"
DEFAULT_EMBEDDING_REPAIR_LEASE_TTL_SECONDS = 1800.0


@dataclass(frozen=True)
class EmbeddingRepairQueueItem:
    id: str
    memory_id: str
    workspace_id: str | None
    model_name: str
    memory_updated_at: str
    status: str
    attempt_count: int
    available_at: float
    created_at: float
    updated_at: float
    claimed_at: float | None
    completed_at: float | None
    lease_owner: str | None
    lease_expires_at: float | None
    last_error: str | None


@dataclass(frozen=True)
class EmbeddingRepairBacklogSnapshot:
    queued_count: int
    running_count: int
    oldest_queued_age_seconds: float | None


class SQLiteEmbeddingRepairQueue:
    def __init__(self, db_manager: DatabaseManager) -> None:
        self._db = db_manager

    def enqueue_unique(
        self,
        *,
        memory_id: str,
        model_name: str,
        memory_updated_at: str,
        workspace_id: str | None = None,
        available_at: float | None = None,
    ) -> tuple[EmbeddingRepairQueueItem, bool]:
        item_id = str(uuid4())
        now = time.time()
        ready_at = now if available_at is None else float(available_at)
        conn = self._db.get_connection()
        try:
            conn.execute(
                """
                INSERT INTO embedding_repair_queue (
                    id,
                    memory_id,
                    workspace_id,
                    model_name,
                    memory_updated_at,
                    status,
                    attempt_count,
                    available_at,
                    created_at,
                    updated_at,
                    claimed_at,
                    completed_at,
                    lease_owner,
                    lease_expires_at,
                    last_error
                ) VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, ?, NULL, NULL, NULL, NULL, NULL)
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
            conn.commit()
        except sqlite3.IntegrityError:
            existing = self.find_by_version(memory_id=memory_id, model_name=model_name, memory_updated_at=memory_updated_at)
            if existing is None:
                raise
            return existing, False
        return self.get_item(item_id), True

    def find_by_version(self, *, memory_id: str, model_name: str, memory_updated_at: str) -> EmbeddingRepairQueueItem | None:
        row = self._db.get_connection().execute(
            """
            SELECT * FROM embedding_repair_queue
            WHERE memory_id = ? AND model_name = ? AND memory_updated_at = ?
            LIMIT 1
            """,
            (memory_id, model_name, memory_updated_at),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_item(row)

    def claim_batch(
        self,
        *,
        lease_owner: str,
        limit: int,
        workspace_id: str | None = None,
        now: float | None = None,
        lease_ttl_seconds: float = DEFAULT_EMBEDDING_REPAIR_LEASE_TTL_SECONDS,
    ) -> list[EmbeddingRepairQueueItem]:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        current_time = time.time() if now is None else now
        lease_expires_at = current_time + lease_ttl_seconds
        conn = self._db.get_connection()
        if conn.in_transaction:
            conn.commit()
        conn.execute("BEGIN IMMEDIATE")
        try:
            clauses = [
                "((status = 'pending' AND available_at <= ?) OR (status = 'running' AND lease_expires_at IS NOT NULL AND lease_expires_at <= ?))",
            ]
            params: list[object] = [current_time, current_time]
            if workspace_id is None:
                clauses.append("workspace_id IS NULL")
            elif workspace_id != "*":
                clauses.append("workspace_id = ?")
                params.append(workspace_id)
            rows = conn.execute(
                "SELECT id FROM embedding_repair_queue WHERE "
                + " AND ".join(clauses)
                + " ORDER BY created_at ASC LIMIT ?",
                [*params, limit],
            ).fetchall()
            if not rows:
                conn.commit()
                return []
            item_ids = [str(row["id"]) for row in rows]
            placeholders = ",".join("?" for _ in item_ids)
            conn.execute(
                f"""
                UPDATE embedding_repair_queue
                SET status = ?,
                    attempt_count = attempt_count + 1,
                    updated_at = ?,
                    claimed_at = ?,
                    lease_owner = ?,
                    lease_expires_at = ?,
                    last_error = NULL
                WHERE id IN ({placeholders})
                """,
                [
                    EMBEDDING_REPAIR_STATUS_RUNNING,
                    current_time,
                    current_time,
                    lease_owner,
                    lease_expires_at,
                    *item_ids,
                ],
            )
            claimed_rows = conn.execute(
                f"SELECT * FROM embedding_repair_queue WHERE id IN ({placeholders}) ORDER BY created_at ASC",
                item_ids,
            ).fetchall()
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        return [self._row_to_item(row) for row in claimed_rows]

    def complete_item(self, item_id: str, *, completed_at: float | None = None) -> EmbeddingRepairQueueItem:
        now = time.time() if completed_at is None else completed_at
        conn = self._db.get_connection()
        cursor = conn.execute(
            """
            UPDATE embedding_repair_queue
            SET status = ?,
                updated_at = ?,
                completed_at = ?,
                lease_owner = NULL,
                lease_expires_at = NULL,
                last_error = NULL
            WHERE id = ? AND status = ?
            """,
            (EMBEDDING_REPAIR_STATUS_COMPLETED, now, now, item_id, EMBEDDING_REPAIR_STATUS_RUNNING),
        )
        if cursor.rowcount != 1:
            conn.rollback()
            raise ValueError(f"Embedding repair item {item_id} is not running")
        conn.commit()
        return self.get_item(item_id)

    def release_item(self, item_id: str, *, released_at: float | None = None, error: str | None = None) -> EmbeddingRepairQueueItem:
        now = time.time() if released_at is None else released_at
        conn = self._db.get_connection()
        cursor = conn.execute(
            """
            UPDATE embedding_repair_queue
            SET status = ?,
                updated_at = ?,
                lease_owner = NULL,
                lease_expires_at = NULL,
                last_error = ?
            WHERE id = ? AND status = ?
            """,
            (EMBEDDING_REPAIR_STATUS_PENDING, now, error, item_id, EMBEDDING_REPAIR_STATUS_RUNNING),
        )
        if cursor.rowcount != 1:
            conn.rollback()
            raise ValueError(f"Embedding repair item {item_id} is not running")
        conn.commit()
        return self.get_item(item_id)

    def get_item(self, item_id: str) -> EmbeddingRepairQueueItem:
        row = self._db.get_connection().execute(
            "SELECT * FROM embedding_repair_queue WHERE id = ?",
            (item_id,),
        ).fetchone()
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
        query = "SELECT * FROM embedding_repair_queue"
        clauses: list[str] = []
        params: list[object] = []
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if workspace_id is not None:
            clauses.append("workspace_id = ?")
            params.append(workspace_id)
        if lease_owner is not None:
            clauses.append("lease_owner = ?")
            params.append(lease_owner)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY updated_at DESC, created_at DESC LIMIT ?"
        params.append(limit)
        rows = self._db.get_connection().execute(query, params).fetchall()
        return [self._row_to_item(row) for row in rows]

    def backlog_snapshot(self, *, now: float | None = None) -> EmbeddingRepairBacklogSnapshot:
        current_time = time.time() if now is None else now
        row = self._db.get_connection().execute(
            """
            SELECT
                COALESCE(SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END), 0) AS queued_count,
                COALESCE(SUM(CASE WHEN status = 'running' THEN 1 ELSE 0 END), 0) AS running_count,
                MIN(CASE WHEN status = 'pending' THEN created_at END) AS oldest_queued_created_at
            FROM embedding_repair_queue
            """
        ).fetchone()
        queued_count = 0 if row is None or row["queued_count"] is None else int(row["queued_count"])
        running_count = 0 if row is None or row["running_count"] is None else int(row["running_count"])
        oldest_created_at = None if row is None else row["oldest_queued_created_at"]
        oldest_age = None if oldest_created_at is None else max(current_time - float(oldest_created_at), 0.0)
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
        if limit < 1:
            return 0
        current_time = time.time() if now is None else now
        completed_before = current_time - max(float(older_than_seconds), 0.0)
        conn = self._db.get_connection()
        rows = conn.execute(
            """
            SELECT id FROM embedding_repair_queue
            WHERE status = ? AND completed_at IS NOT NULL AND completed_at <= ?
            ORDER BY completed_at ASC
            LIMIT ?
            """,
            (EMBEDDING_REPAIR_STATUS_COMPLETED, completed_before, limit),
        ).fetchall()
        if not rows:
            return 0
        item_ids = [str(row["id"]) for row in rows]
        placeholders = ",".join("?" for _ in item_ids)
        cursor = conn.execute(
            f"DELETE FROM embedding_repair_queue WHERE id IN ({placeholders})",
            item_ids,
        )
        conn.commit()
        return int(cursor.rowcount)

    def _row_to_item(self, row) -> EmbeddingRepairQueueItem:
        return EmbeddingRepairQueueItem(
            id=str(row["id"]),
            memory_id=str(row["memory_id"]),
            workspace_id=row["workspace_id"],
            model_name=str(row["model_name"]),
            memory_updated_at=str(row["memory_updated_at"]),
            status=str(row["status"]),
            attempt_count=int(row["attempt_count"]),
            available_at=float(row["available_at"]),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
            claimed_at=None if row["claimed_at"] is None else float(row["claimed_at"]),
            completed_at=None if row["completed_at"] is None else float(row["completed_at"]),
            lease_owner=row["lease_owner"],
            lease_expires_at=None if row["lease_expires_at"] is None else float(row["lease_expires_at"]),
            last_error=row["last_error"],
        )