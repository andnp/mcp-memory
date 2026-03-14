from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from mcp_memory.utils.db import DatabaseManager


@dataclass
class TaskRecord:
    id: str
    task_name: str
    data: dict[str, Any]
    workspace_id: str | None
    status: str
    priority: int
    retries_count: int
    max_retries: int
    created_at: float
    updated_at: float
    available_at: float
    claimed_at: float | None
    started_at: float | None
    completed_at: float | None
    last_error: str | None


class SQLiteTaskQueue:
    def __init__(self, db_manager: DatabaseManager) -> None:
        self._db = db_manager

    def enqueue(
        self,
        task_name: str,
        data: dict[str, Any] | None = None,
        workspace_id: str | None = None,
        priority: int = 100,
        max_retries: int = 3,
        available_at: float | None = None,
        task_id: str | None = None,
    ) -> TaskRecord:
        if not task_name or not task_name.strip():
            raise ValueError("task_name is required")

        now = time.time()
        task_payload = data or {}
        task_identifier = task_id or str(uuid4())
        ready_at = now if available_at is None else available_at
        conn = self._db.get_connection()
        conn.execute(
            """
            INSERT INTO tasks (
                id,
                task_name,
                workspace_id,
                data,
                status,
                priority,
                retries_count,
                max_retries,
                created_at,
                updated_at,
                available_at,
                claimed_at,
                started_at,
                completed_at,
                last_error
            ) VALUES (?, ?, ?, ?, 'pending', ?, 0, ?, ?, ?, ?, NULL, NULL, NULL, NULL)
            """,
            (
                task_identifier,
                task_name.strip(),
                workspace_id,
                json.dumps(task_payload, sort_keys=True),
                priority,
                max_retries,
                now,
                now,
                ready_at,
            ),
        )
        conn.commit()
        return self.get_task(task_identifier)

    def enqueue_unique(
        self,
        task_name: str,
        data: dict[str, Any] | None = None,
        workspace_id: str | None = None,
        priority: int = 100,
        max_retries: int = 3,
        available_at: float | None = None,
    ) -> tuple[TaskRecord, bool]:
        existing = self.find_open_task(task_name, workspace_id)
        if existing is not None:
            return existing, False

        try:
            task = self.enqueue(
                task_name=task_name,
                data=data,
                workspace_id=workspace_id,
                priority=priority,
                max_retries=max_retries,
                available_at=available_at,
            )
        except sqlite3.IntegrityError:
            existing = self.find_open_task(task_name, workspace_id)
            if existing is None:
                raise
            return existing, False

        return task, True

    def claim_next(self, now: float | None = None) -> TaskRecord | None:
        claimed_at = time.time() if now is None else now
        conn = self._db.get_connection()
        conn.execute("BEGIN IMMEDIATE")
        try:
            row = conn.execute(
                """
                SELECT *
                FROM tasks
                WHERE status = 'pending' AND available_at <= ?
                ORDER BY priority ASC, created_at ASC
                LIMIT 1
                """,
                (claimed_at,),
            ).fetchone()
            if row is None:
                conn.commit()
                return None

            updated = conn.execute(
                """
                UPDATE tasks
                SET status = 'running',
                    updated_at = ?,
                    claimed_at = ?,
                    started_at = COALESCE(started_at, ?)
                WHERE id = ? AND status = 'pending'
                """,
                (claimed_at, claimed_at, claimed_at, row["id"]),
            )
            if updated.rowcount != 1:
                conn.rollback()
                return None

            claimed_row = conn.execute(
                "SELECT * FROM tasks WHERE id = ?",
                (row["id"],),
            ).fetchone()
            conn.commit()
            if claimed_row is None:
                return None
            return self._row_to_record(claimed_row)
        except Exception:
            conn.rollback()
            raise

    def complete(self, task_id: str, completed_at: float | None = None) -> TaskRecord:
        now = time.time() if completed_at is None else completed_at
        conn = self._db.get_connection()
        cursor = conn.execute(
            """
            UPDATE tasks
            SET status = 'completed',
                updated_at = ?,
                completed_at = ?,
                last_error = NULL
            WHERE id = ? AND status = 'running'
            """,
            (now, now, task_id),
        )
        if cursor.rowcount != 1:
            conn.rollback()
            raise ValueError(f"Task {task_id} is not running")
        conn.commit()
        return self.get_task(task_id)

    def fail(
        self,
        task_id: str,
        error: str,
        retry_delay_seconds: float = 0.0,
        failed_at: float | None = None,
    ) -> TaskRecord:
        now = time.time() if failed_at is None else failed_at
        conn = self._db.get_connection()
        row = conn.execute(
            "SELECT retries_count, max_retries FROM tasks WHERE id = ? AND status = 'running'",
            (task_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Task {task_id} is not running")

        next_retries = int(row["retries_count"]) + 1
        terminal = next_retries >= int(row["max_retries"])
        status = "failed" if terminal else "pending"
        available_at = now if terminal else now + retry_delay_seconds
        completed_at = now if terminal else None
        cursor = conn.execute(
            """
            UPDATE tasks
            SET status = ?,
                retries_count = ?,
                updated_at = ?,
                available_at = ?,
                claimed_at = NULL,
                completed_at = ?,
                last_error = ?
            WHERE id = ? AND status = 'running'
            """,
            (status, next_retries, now, available_at, completed_at, error, task_id),
        )
        if cursor.rowcount != 1:
            conn.rollback()
            raise ValueError(f"Task {task_id} could not be updated")
        conn.commit()
        return self.get_task(task_id)

    def fail_permanently(
        self,
        task_id: str,
        error: str,
        failed_at: float | None = None,
    ) -> TaskRecord:
        now = time.time() if failed_at is None else failed_at
        conn = self._db.get_connection()
        row = conn.execute(
            "SELECT max_retries FROM tasks WHERE id = ? AND status = 'running'",
            (task_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Task {task_id} is not running")

        cursor = conn.execute(
            """
            UPDATE tasks
            SET status = 'failed',
                retries_count = max_retries,
                updated_at = ?,
                available_at = ?,
                claimed_at = NULL,
                completed_at = ?,
                last_error = ?
            WHERE id = ? AND status = 'running'
            """,
            (now, now, now, error, task_id),
        )
        if cursor.rowcount != 1:
            conn.rollback()
            raise ValueError(f"Task {task_id} could not be updated")
        conn.commit()
        return self.get_task(task_id)

    def get_task(self, task_id: str) -> TaskRecord:
        row = self._db.get_connection().execute(
            "SELECT * FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Task {task_id} was not found")
        return self._row_to_record(row)

    def find_open_task(
        self,
        task_name: str,
        workspace_id: str | None = None,
    ) -> TaskRecord | None:
        row = self._db.get_connection().execute(
            """
            SELECT *
            FROM tasks
            WHERE task_name = ?
              AND ((workspace_id IS NULL AND ? IS NULL) OR workspace_id = ?)
              AND status IN ('pending', 'running')
            ORDER BY created_at ASC
            LIMIT 1
            """,
            (task_name, workspace_id, workspace_id),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def count_by_status(self) -> dict[str, int]:
        rows = self._db.get_connection().execute(
            "SELECT status, COUNT(*) FROM tasks GROUP BY status"
        ).fetchall()
        return {str(row[0]): int(row[1]) for row in rows}

    def list_tasks(
        self,
        status: str | None = None,
        workspace_id: str | None = None,
        limit: int = 20,
    ) -> list[TaskRecord]:
        clauses: list[str] = []
        params: list[object] = []
        query = "SELECT * FROM tasks"

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

    def _row_to_record(self, row) -> TaskRecord:
        data = row["data"] or "{}"
        return TaskRecord(
            id=str(row["id"]),
            task_name=str(row["task_name"]),
            data=dict(json.loads(data)),
            workspace_id=row["workspace_id"],
            status=str(row["status"]),
            priority=int(row["priority"]),
            retries_count=int(row["retries_count"]),
            max_retries=int(row["max_retries"]),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
            available_at=float(row["available_at"]),
            claimed_at=(
                None if row["claimed_at"] is None else float(row["claimed_at"])
            ),
            started_at=(
                None if row["started_at"] is None else float(row["started_at"])
            ),
            completed_at=(
                None if row["completed_at"] is None else float(row["completed_at"])
            ),
            last_error=row["last_error"],
        )