from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

from mcp_memory.utils.db import DatabaseManager


@dataclass(frozen=True)
class TaskExecutionAttemptRecord:
    id: int
    task_id: str
    execution_epoch: int
    workspace_id: str | None
    task_name: str | None
    request_id: str | None
    subprocess_pid: int | None
    provider_key: str | None
    provider_name: str | None
    model_name: str | None
    status: str
    started_at: float
    last_heartbeat_at: float | None
    completed_at: float | None
    error_text: str | None
    termination_reason: str | None


class TaskExecutionAttemptRepository:
    def __init__(self, db_manager: DatabaseManager | None, *, workspace_id: str | None):
        self._db_manager = db_manager
        self._workspace_id = workspace_id

    def start_attempt(
        self,
        *,
        task_id: str,
        execution_epoch: int,
        task_name: str | None,
        request_id: str | None,
        subprocess_pid: int | None,
        provider_key: str | None,
        provider_name: str | None,
        model_name: str | None,
        started_at: float | None = None,
        status: str = "running",
    ) -> TaskExecutionAttemptRecord:
        if self._db_manager is None:
            raise ValueError("task_execution_attempts_not_initialized")
        now = time.time() if started_at is None else started_at
        reopen_terminal_attempt = """
            task_execution_attempts.completed_at IS NOT NULL
            AND CASE
                WHEN excluded.request_id IS NOT NULL
                    AND task_execution_attempts.request_id IS NOT NULL
                THEN excluded.request_id IS NOT task_execution_attempts.request_id
                WHEN excluded.subprocess_pid IS NOT NULL
                    AND task_execution_attempts.subprocess_pid IS NOT NULL
                THEN excluded.subprocess_pid IS NOT task_execution_attempts.subprocess_pid
                ELSE 1
            END
        """
        conn = self._db_manager.get_connection()
        conn.execute(
            f"""
            INSERT INTO task_execution_attempts (
                task_id,
                execution_epoch,
                workspace_id,
                task_name,
                request_id,
                subprocess_pid,
                provider_key,
                provider_name,
                model_name,
                status,
                started_at,
                last_heartbeat_at,
                completed_at,
                error_text,
                termination_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL)
            ON CONFLICT(task_id, execution_epoch) DO UPDATE SET
                workspace_id = COALESCE(excluded.workspace_id, task_execution_attempts.workspace_id),
                task_name = COALESCE(excluded.task_name, task_execution_attempts.task_name),
                request_id = CASE
                    WHEN task_execution_attempts.completed_at IS NULL THEN COALESCE(
                        excluded.request_id,
                        task_execution_attempts.request_id
                    )
                    WHEN {reopen_terminal_attempt} THEN excluded.request_id
                    ELSE task_execution_attempts.request_id
                END,
                subprocess_pid = CASE
                    WHEN task_execution_attempts.completed_at IS NULL THEN COALESCE(
                        excluded.subprocess_pid,
                        task_execution_attempts.subprocess_pid
                    )
                    WHEN {reopen_terminal_attempt} THEN excluded.subprocess_pid
                    ELSE task_execution_attempts.subprocess_pid
                END,
                provider_key = COALESCE(excluded.provider_key, task_execution_attempts.provider_key),
                provider_name = COALESCE(excluded.provider_name, task_execution_attempts.provider_name),
                model_name = COALESCE(excluded.model_name, task_execution_attempts.model_name),
                status = CASE
                    WHEN task_execution_attempts.completed_at IS NULL THEN excluded.status
                    WHEN {reopen_terminal_attempt} THEN excluded.status
                    ELSE task_execution_attempts.status
                END,
                started_at = CASE
                    WHEN task_execution_attempts.completed_at IS NULL THEN MIN(
                        task_execution_attempts.started_at,
                        excluded.started_at
                    )
                    WHEN {reopen_terminal_attempt} THEN excluded.started_at
                    ELSE task_execution_attempts.started_at
                END,
                last_heartbeat_at = CASE
                    WHEN task_execution_attempts.completed_at IS NULL THEN MAX(
                        COALESCE(task_execution_attempts.last_heartbeat_at, excluded.last_heartbeat_at),
                        excluded.last_heartbeat_at
                    )
                    WHEN {reopen_terminal_attempt} THEN excluded.last_heartbeat_at
                    ELSE task_execution_attempts.last_heartbeat_at
                END,
                completed_at = CASE
                    WHEN task_execution_attempts.completed_at IS NULL THEN NULL
                    WHEN {reopen_terminal_attempt} THEN NULL
                    ELSE task_execution_attempts.completed_at
                END,
                error_text = CASE
                    WHEN task_execution_attempts.completed_at IS NULL THEN NULL
                    WHEN {reopen_terminal_attempt} THEN NULL
                    ELSE task_execution_attempts.error_text
                END,
                termination_reason = CASE
                    WHEN task_execution_attempts.completed_at IS NULL THEN NULL
                    WHEN {reopen_terminal_attempt} THEN NULL
                    ELSE task_execution_attempts.termination_reason
                END
            """,
            (
                task_id,
                execution_epoch,
                self._workspace_id,
                task_name,
                request_id,
                subprocess_pid,
                provider_key,
                provider_name,
                model_name,
                status,
                now,
                now,
            ),
        )
        conn.commit()
        return self.get_attempt(task_id=task_id, execution_epoch=execution_epoch)

    def heartbeat_attempt(
        self,
        *,
        task_id: str,
        execution_epoch: int,
        heartbeat_at: float | None = None,
        request_id: str | None = None,
        subprocess_pid: int | None = None,
    ) -> TaskExecutionAttemptRecord:
        if self._db_manager is None:
            raise ValueError("task_execution_attempts_not_initialized")
        now = time.time() if heartbeat_at is None else heartbeat_at
        conn = self._db_manager.get_connection()
        cursor = conn.execute(
            """
            UPDATE task_execution_attempts
            SET last_heartbeat_at = CASE
                    WHEN completed_at IS NULL THEN MAX(COALESCE(last_heartbeat_at, ?), ?)
                    ELSE last_heartbeat_at
                END,
                request_id = CASE
                    WHEN completed_at IS NULL THEN COALESCE(?, request_id)
                    ELSE request_id
                END,
                subprocess_pid = CASE
                    WHEN completed_at IS NULL THEN COALESCE(?, subprocess_pid)
                    ELSE subprocess_pid
                END
            WHERE task_id = ? AND execution_epoch = ?
            """,
            (now, now, request_id, subprocess_pid, task_id, execution_epoch),
        )
        if cursor.rowcount != 1:
            conn.rollback()
            raise ValueError(f"Task execution attempt {task_id}@{execution_epoch} was not found")
        conn.commit()
        return self.get_attempt(task_id=task_id, execution_epoch=execution_epoch)

    def finish_attempt(
        self,
        *,
        task_id: str,
        execution_epoch: int,
        status: str,
        completed_at: float | None = None,
        request_id: str | None = None,
        subprocess_pid: int | None = None,
        error_text: str | None = None,
        termination_reason: str | None = None,
    ) -> TaskExecutionAttemptRecord:
        if self._db_manager is None:
            raise ValueError("task_execution_attempts_not_initialized")
        now = time.time() if completed_at is None else completed_at
        conn = self._db_manager.get_connection()
        cursor = conn.execute(
            """
            UPDATE task_execution_attempts
            SET status = CASE
                    WHEN completed_at IS NULL THEN ?
                    ELSE status
                END,
                completed_at = COALESCE(completed_at, ?),
                last_heartbeat_at = CASE
                    WHEN completed_at IS NULL THEN MAX(COALESCE(last_heartbeat_at, ?), ?)
                    ELSE last_heartbeat_at
                END,
                request_id = COALESCE(?, request_id),
                subprocess_pid = COALESCE(?, subprocess_pid),
                error_text = CASE
                    WHEN completed_at IS NULL THEN COALESCE(?, error_text)
                    ELSE error_text
                END,
                termination_reason = CASE
                    WHEN completed_at IS NULL THEN COALESCE(?, termination_reason)
                    ELSE termination_reason
                END
            WHERE task_id = ? AND execution_epoch = ?
            """,
            (
                status,
                now,
                now,
                now,
                request_id,
                subprocess_pid,
                error_text,
                termination_reason,
                task_id,
                execution_epoch,
            ),
        )
        if cursor.rowcount != 1:
            conn.rollback()
            raise ValueError(f"Task execution attempt {task_id}@{execution_epoch} was not found")
        conn.commit()
        return self.get_attempt(task_id=task_id, execution_epoch=execution_epoch)

    def get_attempt(self, *, task_id: str, execution_epoch: int) -> TaskExecutionAttemptRecord:
        if self._db_manager is None:
            raise ValueError("task_execution_attempts_not_initialized")
        row = self._db_manager.get_connection().execute(
            "SELECT * FROM task_execution_attempts WHERE task_id = ? AND execution_epoch = ?",
            (task_id, execution_epoch),
        ).fetchone()
        if row is None:
            raise ValueError(f"Task execution attempt {task_id}@{execution_epoch} was not found")
        return self._row_to_record(row)

    def list_attempts(
        self,
        *,
        task_id: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[TaskExecutionAttemptRecord]:
        if self._db_manager is None:
            return []
        query = "SELECT * FROM task_execution_attempts"
        clauses: list[str] = []
        params: list[object] = []
        if self._workspace_id is not None:
            clauses.append("workspace_id = ?")
            params.append(self._workspace_id)
        if task_id is not None:
            clauses.append("task_id = ?")
            params.append(task_id)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY started_at DESC, id DESC LIMIT ?"
        params.append(limit)
        rows = self._db_manager.get_connection().execute(query, params).fetchall()
        return [self._row_to_record(row) for row in rows]

    def _row_to_record(self, row: Any) -> TaskExecutionAttemptRecord:
        return TaskExecutionAttemptRecord(
            id=int(row["id"]),
            task_id=str(row["task_id"]),
            execution_epoch=int(row["execution_epoch"]),
            workspace_id=row["workspace_id"],
            task_name=row["task_name"],
            request_id=row["request_id"],
            subprocess_pid=None if row["subprocess_pid"] is None else int(row["subprocess_pid"]),
            provider_key=row["provider_key"],
            provider_name=row["provider_name"],
            model_name=row["model_name"],
            status=str(row["status"]),
            started_at=float(row["started_at"]),
            last_heartbeat_at=None if row["last_heartbeat_at"] is None else float(row["last_heartbeat_at"]),
            completed_at=None if row["completed_at"] is None else float(row["completed_at"]),
            error_text=row["error_text"],
            termination_reason=row["termination_reason"],
        )
