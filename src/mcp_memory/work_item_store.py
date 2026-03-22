from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from mcp_memory.utils.db import DatabaseManager


WORK_ITEM_STATUS_PENDING = "pending"
WORK_ITEM_STATUS_RUNNING = "running"
WORK_ITEM_STATUS_COMPLETED = "completed"
WORK_ITEM_STATUS_DEFERRED = "deferred"

EXECUTION_LANE_DETERMINISTIC = "deterministic"
EXECUTION_LANE_AGENTIC = "agentic"
WORK_FAMILY_MEMORY_TAGGING = "memory_tagging"
DEFAULT_WORK_ITEM_LEASE_TTL_SECONDS = 1800.0


@dataclass(frozen=True)
class WorkItemRecord:
    id: str
    family_key: str
    execution_lane: str
    workspace_id: str | None
    payload: dict[str, Any]
    status: str
    priority: int
    attempt_count: int
    available_at: float
    created_at: float
    updated_at: float
    claimed_at: float | None
    completed_at: float | None
    lease_owner: str | None
    lease_expires_at: float | None
    idempotency_key: str | None
    last_error: str | None


class SQLiteWorkItemRepository:
    def __init__(self, db_manager: DatabaseManager) -> None:
        self._db = db_manager

    def enqueue_unique(
        self,
        *,
        family_key: str,
        execution_lane: str,
        payload: dict[str, Any] | None = None,
        workspace_id: str | None = None,
        priority: int = 100,
        available_at: float | None = None,
        idempotency_key: str | None = None,
    ) -> tuple[WorkItemRecord, bool]:
        item_id = str(uuid4())
        now = time.time()
        ready_at = now if available_at is None else available_at
        payload_json = json.dumps(payload or {}, sort_keys=True)
        conn = self._db.get_connection()
        try:
            conn.execute(
                """
                INSERT INTO work_items (
                    id,
                    family_key,
                    execution_lane,
                    workspace_id,
                    payload_json,
                    status,
                    priority,
                    attempt_count,
                    available_at,
                    created_at,
                    updated_at,
                    claimed_at,
                    completed_at,
                    lease_owner,
                    lease_expires_at,
                    idempotency_key,
                    last_error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, NULL, NULL, NULL, NULL, ?, NULL)
                """,
                (
                    item_id,
                    family_key,
                    execution_lane,
                    workspace_id,
                    payload_json,
                    WORK_ITEM_STATUS_PENDING,
                    priority,
                    ready_at,
                    now,
                    now,
                    idempotency_key,
                ),
            )
            conn.commit()
        except sqlite3.IntegrityError:
            existing = self.find_by_idempotency_key(idempotency_key)
            if existing is None:
                raise
            return existing, False
        return self.get_item(item_id), True

    def claim_batch(
        self,
        *,
        family_key: str,
        execution_lane: str,
        lease_owner: str,
        limit: int,
        workspace_id: str | None = None,
        now: float | None = None,
        lease_ttl_seconds: float = DEFAULT_WORK_ITEM_LEASE_TTL_SECONDS,
    ) -> list[WorkItemRecord]:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        current_time = time.time() if now is None else now
        lease_expires_at = current_time + lease_ttl_seconds
        conn = self._db.get_connection()
        conn.execute("BEGIN IMMEDIATE")
        try:
            clauses = [
                "family_key = ?",
                "execution_lane = ?",
                "((status IN ('pending', 'deferred') AND available_at <= ?) OR (status = 'running' AND lease_expires_at IS NOT NULL AND lease_expires_at <= ?))",
            ]
            params: list[object] = [family_key, execution_lane, current_time, current_time]
            if workspace_id is None:
                clauses.append("workspace_id IS NULL")
            elif workspace_id != "*":
                clauses.append("workspace_id = ?")
                params.append(workspace_id)
            rows = conn.execute(
                "SELECT id FROM work_items WHERE "
                + " AND ".join(clauses)
                + " ORDER BY priority ASC, created_at ASC LIMIT ?",
                [*params, limit],
            ).fetchall()
            if not rows:
                conn.commit()
                return []
            item_ids = [str(row["id"]) for row in rows]
            placeholders = ",".join("?" for _ in item_ids)
            conn.execute(
                f"""
                UPDATE work_items
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
                    WORK_ITEM_STATUS_RUNNING,
                    current_time,
                    current_time,
                    lease_owner,
                    lease_expires_at,
                    *item_ids,
                ],
            )
            claimed_rows = conn.execute(
                f"SELECT * FROM work_items WHERE id IN ({placeholders}) ORDER BY priority ASC, created_at ASC",
                item_ids,
            ).fetchall()
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        return [self._row_to_record(row) for row in claimed_rows]

    def heartbeat_item(
        self,
        item_id: str,
        *,
        lease_owner: str,
        lease_ttl_seconds: float = DEFAULT_WORK_ITEM_LEASE_TTL_SECONDS,
        heartbeated_at: float | None = None,
    ) -> WorkItemRecord:
        now = time.time() if heartbeated_at is None else heartbeated_at
        lease_expires_at = now + lease_ttl_seconds
        conn = self._db.get_connection()
        cursor = conn.execute(
            """
            UPDATE work_items
            SET updated_at = ?,
                lease_expires_at = ?
            WHERE id = ? AND status = ? AND lease_owner = ?
            """,
            (now, lease_expires_at, item_id, WORK_ITEM_STATUS_RUNNING, lease_owner),
        )
        if cursor.rowcount != 1:
            conn.rollback()
            raise ValueError(f"Work item {item_id} is not running for lease owner {lease_owner}")
        conn.commit()
        return self.get_item(item_id)

    def complete_item(
        self,
        item_id: str,
        *,
        completed_at: float | None = None,
    ) -> WorkItemRecord:
        now = time.time() if completed_at is None else completed_at
        conn = self._db.get_connection()
        cursor = conn.execute(
            """
            UPDATE work_items
            SET status = ?,
                updated_at = ?,
                completed_at = ?,
                lease_owner = NULL,
                lease_expires_at = NULL,
                last_error = NULL
            WHERE id = ? AND status = ?
            """,
            (WORK_ITEM_STATUS_COMPLETED, now, now, item_id, WORK_ITEM_STATUS_RUNNING),
        )
        if cursor.rowcount != 1:
            conn.rollback()
            raise ValueError(f"Work item {item_id} is not running")
        conn.commit()
        return self.get_item(item_id)

    def defer_item(
        self,
        item_id: str,
        *,
        error: str,
        retry_delay_seconds: float,
        deferred_at: float | None = None,
    ) -> WorkItemRecord:
        now = time.time() if deferred_at is None else deferred_at
        conn = self._db.get_connection()
        cursor = conn.execute(
            """
            UPDATE work_items
            SET status = ?,
                updated_at = ?,
                available_at = ?,
                lease_owner = NULL,
                lease_expires_at = NULL,
                last_error = ?
            WHERE id = ? AND status = ?
            """,
            (
                WORK_ITEM_STATUS_DEFERRED,
                now,
                now + max(retry_delay_seconds, 0.0),
                error,
                item_id,
                WORK_ITEM_STATUS_RUNNING,
            ),
        )
        if cursor.rowcount != 1:
            conn.rollback()
            raise ValueError(f"Work item {item_id} is not running")
        conn.commit()
        return self.get_item(item_id)

    def release_item(
        self,
        item_id: str,
        *,
        released_at: float | None = None,
    ) -> WorkItemRecord:
        now = time.time() if released_at is None else released_at
        conn = self._db.get_connection()
        cursor = conn.execute(
            """
            UPDATE work_items
            SET status = ?,
                updated_at = ?,
                lease_owner = NULL,
                lease_expires_at = NULL
            WHERE id = ? AND status = ?
            """,
            (WORK_ITEM_STATUS_PENDING, now, item_id, WORK_ITEM_STATUS_RUNNING),
        )
        if cursor.rowcount != 1:
            conn.rollback()
            raise ValueError(f"Work item {item_id} is not running")
        conn.commit()
        return self.get_item(item_id)

    def get_item(self, item_id: str) -> WorkItemRecord:
        row = self._db.get_connection().execute(
            "SELECT * FROM work_items WHERE id = ?",
            (item_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Work item {item_id} was not found")
        return self._row_to_record(row)

    def list_items(
        self,
        *,
        family_key: str | None = None,
        execution_lane: str | None = None,
        status: str | None = None,
        workspace_id: str | None = None,
        limit: int = 50,
    ) -> list[WorkItemRecord]:
        query = "SELECT * FROM work_items"
        clauses: list[str] = []
        params: list[object] = []
        if family_key is not None:
            clauses.append("family_key = ?")
            params.append(family_key)
        if execution_lane is not None:
            clauses.append("execution_lane = ?")
            params.append(execution_lane)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if workspace_id is not None:
            clauses.append("workspace_id = ?")
            params.append(workspace_id)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY updated_at DESC, created_at DESC LIMIT ?"
        params.append(limit)
        rows = self._db.get_connection().execute(query, params).fetchall()
        return [self._row_to_record(row) for row in rows]

    def find_by_idempotency_key(self, idempotency_key: str | None) -> WorkItemRecord | None:
        if idempotency_key is None:
            return None
        row = self._db.get_connection().execute(
            "SELECT * FROM work_items WHERE idempotency_key = ? LIMIT 1",
            (idempotency_key,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def _row_to_record(self, row) -> WorkItemRecord:
        payload_json = row["payload_json"] or "{}"
        return WorkItemRecord(
            id=str(row["id"]),
            family_key=str(row["family_key"]),
            execution_lane=str(row["execution_lane"]),
            workspace_id=row["workspace_id"],
            payload=dict(json.loads(payload_json)),
            status=str(row["status"]),
            priority=int(row["priority"]),
            attempt_count=int(row["attempt_count"]),
            available_at=float(row["available_at"]),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
            claimed_at=None if row["claimed_at"] is None else float(row["claimed_at"]),
            completed_at=None if row["completed_at"] is None else float(row["completed_at"]),
            lease_owner=row["lease_owner"],
            lease_expires_at=None if row["lease_expires_at"] is None else float(row["lease_expires_at"]),
            idempotency_key=row["idempotency_key"],
            last_error=row["last_error"],
        )
