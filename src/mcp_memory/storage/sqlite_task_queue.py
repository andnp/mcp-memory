from __future__ import annotations

import json
import os
import sqlite3
import time
from collections.abc import Callable
from typing import Any
from uuid import uuid4

from mcp_memory.core.maintenance_schedule import BACKGROUND_CLEANUP_TASK_NAMES
from mcp_memory.core.ports.tasks import TaskRecord, TaskRunRecord, TaskRunSummary
from mcp_memory.core.task_results import TaskRunResultSource, coerce_task_run_result
from mcp_memory.utils.db import DatabaseManager


_ANY_WORKSPACE = object()
_RECOVERY_LOCK_RETRY_ATTEMPTS = 3
_RECOVERY_LOCK_RETRY_DELAY_SECONDS = 0.05


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
                execution_epoch,
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
            ) VALUES (?, ?, ?, ?, 'pending', 0, ?, 0, ?, ?, ?, ?, NULL, NULL, NULL, NULL)
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

    def claim_next(
        self,
        now: float | None = None,
        workspace_id: str | None | object = _ANY_WORKSPACE,
    ) -> TaskRecord | None:
        claimed_at = time.time() if now is None else now
        conn = self._db.get_connection()
        conn.execute("BEGIN IMMEDIATE")
        try:
            clauses = ["status = 'pending'", "available_at <= ?"]
            params: list[object] = [claimed_at]
            if workspace_id is None:
                clauses.append("workspace_id IS NULL")
            elif workspace_id is not _ANY_WORKSPACE:
                clauses.append("workspace_id = ?")
                params.append(workspace_id)

            cleanup_placeholders = ", ".join("?" for _ in BACKGROUND_CLEANUP_TASK_NAMES)
            cleanup_names = tuple(sorted(BACKGROUND_CLEANUP_TASK_NAMES))
            clauses.append(
                "(task_name NOT IN (" + cleanup_placeholders + ") "
                "OR NOT EXISTS ("
                "SELECT 1 FROM tasks AS running_cleanup "
                "WHERE running_cleanup.status = 'running' "
                "AND running_cleanup.task_name IN (" + cleanup_placeholders + ")"
                "))"
            )
            params.extend(cleanup_names)
            params.extend(cleanup_names)

            row = conn.execute(
                "SELECT * FROM tasks WHERE " + " AND ".join(clauses) + " ORDER BY priority ASC, created_at ASC LIMIT 1",
                params,
            ).fetchone()
            if row is None:
                conn.commit()
                return None

            updated = conn.execute(
                """
                UPDATE tasks
                SET status = 'running',
                    execution_epoch = execution_epoch + 1,
                    updated_at = ?,
                    claimed_at = ?,
                    started_at = COALESCE(started_at, ?),
                    last_error = NULL
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

    def complete(
        self,
        task_id: str,
        completed_at: float | None = None,
        run_result: TaskRunResultSource = None,
        execution_epoch: int | None = None,
    ) -> TaskRecord:
        now = time.time() if completed_at is None else completed_at
        normalized_result = coerce_task_run_result(run_result)
        conn = self._db.get_connection()
        row = self._get_running_task_row(conn, task_id, execution_epoch=execution_epoch)
        if row is None:
            raise self._running_task_mismatch_error(task_id, execution_epoch=execution_epoch)

        cursor = conn.execute(*self._running_task_update_statement(
            """
            UPDATE tasks
            SET status = 'completed',
                updated_at = ?,
                completed_at = ?,
                last_error = NULL,
                subprocess_pid = NULL,
                active_request_id = NULL
            """,
            task_id,
            now,
            now,
            execution_epoch=execution_epoch,
        ))
        if cursor.rowcount != 1:
            conn.rollback()
            raise self._running_task_mismatch_error(task_id, execution_epoch=execution_epoch)
        self._insert_task_run(
            conn,
            task_id=task_id,
            task_name=str(row["task_name"]),
            workspace_id=row["workspace_id"],
            status="completed",
            started_at=_coalesce_float(row["claimed_at"], row["started_at"], now),
            completed_at=now,
            result=normalized_result,
            error_text=None,
        )
        conn.commit()
        return self.get_task(task_id)

    def fail(
        self,
        task_id: str,
        error: str,
        retry_delay_seconds: float = 0.0,
        failed_at: float | None = None,
        execution_epoch: int | None = None,
    ) -> TaskRecord:
        now = time.time() if failed_at is None else failed_at
        conn = self._db.get_connection()
        row = self._get_running_task_row(conn, task_id, execution_epoch=execution_epoch)
        if row is None:
            raise self._running_task_mismatch_error(task_id, execution_epoch=execution_epoch)

        next_retries = int(row["retries_count"]) + 1
        terminal = next_retries >= int(row["max_retries"])
        status = "failed" if terminal else "pending"
        available_at = now if terminal else now + retry_delay_seconds
        completed_at = now if terminal else None
        cursor = conn.execute(*self._running_task_update_statement(
            """
            UPDATE tasks
            SET status = ?,
                retries_count = ?,
                updated_at = ?,
                available_at = ?,
                claimed_at = NULL,
                completed_at = ?,
                last_error = ?,
                subprocess_pid = NULL,
                active_request_id = NULL,
                cancellation_requested_at = NULL,
                cancelled_at = NULL,
                cancellation_reason = NULL,
                cancelled_by = NULL
            """,
            task_id,
            status,
            next_retries,
            now,
            available_at,
            completed_at,
            error,
            execution_epoch=execution_epoch,
        ))
        if cursor.rowcount != 1:
            conn.rollback()
            raise self._running_task_mismatch_error(task_id, execution_epoch=execution_epoch)
        self._insert_task_run(
            conn,
            task_id=task_id,
            task_name=str(row["task_name"]),
            workspace_id=row["workspace_id"],
            status="failed" if terminal else "retry",
            started_at=_coalesce_float(row["claimed_at"], row["started_at"], now),
            completed_at=now,
            result={},
            error_text=error,
        )
        conn.commit()
        return self.get_task(task_id)

    def fail_permanently(
        self,
        task_id: str,
        error: str,
        failed_at: float | None = None,
        execution_epoch: int | None = None,
    ) -> TaskRecord:
        now = time.time() if failed_at is None else failed_at
        conn = self._db.get_connection()
        row = self._get_running_task_row(conn, task_id, execution_epoch=execution_epoch)
        if row is None:
            raise self._running_task_mismatch_error(task_id, execution_epoch=execution_epoch)

        cursor = conn.execute(*self._running_task_update_statement(
            """
            UPDATE tasks
            SET status = 'failed',
                retries_count = max_retries,
                updated_at = ?,
                available_at = ?,
                claimed_at = NULL,
                completed_at = ?,
                last_error = ?,
                subprocess_pid = NULL,
                active_request_id = NULL
            """,
            task_id,
            now,
            now,
            now,
            error,
            execution_epoch=execution_epoch,
        ))
        if cursor.rowcount != 1:
            conn.rollback()
            raise self._running_task_mismatch_error(task_id, execution_epoch=execution_epoch)
        self._insert_task_run(
            conn,
            task_id=task_id,
            task_name=str(row["task_name"]),
            workspace_id=row["workspace_id"],
            status="failed",
            started_at=_coalesce_float(row["claimed_at"], row["started_at"], now),
            completed_at=now,
            result={},
            error_text=error,
        )
        conn.commit()
        return self.get_task(task_id)

    def retry_running_task(
        self,
        task_id: str,
        error: str,
        available_at: float | None = None,
        execution_epoch: int | None = None,
    ) -> TaskRecord:
        now = time.time() if available_at is None else available_at
        conn = self._db.get_connection()
        row = self._get_running_task_row(conn, task_id, execution_epoch=execution_epoch)
        if row is None:
            raise self._running_task_mismatch_error(task_id, execution_epoch=execution_epoch)

        cursor = conn.execute(*self._running_task_update_statement(
            """
            UPDATE tasks
            SET status = 'pending',
                updated_at = ?,
                available_at = ?,
                claimed_at = NULL,
                completed_at = NULL,
                last_error = ?,
                subprocess_pid = NULL,
                active_request_id = NULL,
                cancellation_requested_at = NULL,
                cancelled_at = NULL,
                cancellation_reason = NULL,
                cancelled_by = NULL
            """,
            task_id,
            now,
            now,
            error,
            execution_epoch=execution_epoch,
        ))
        if cursor.rowcount != 1:
            conn.rollback()
            raise self._running_task_mismatch_error(task_id, execution_epoch=execution_epoch)
        self._insert_task_run(
            conn,
            task_id=task_id,
            task_name=str(row["task_name"]),
            workspace_id=row["workspace_id"],
            status="retry",
            started_at=_coalesce_float(row["claimed_at"], row["started_at"], now),
            completed_at=now,
            result={"retry_reason": error},
            error_text=error,
        )
        conn.commit()
        return self.get_task(task_id)

    def set_running_process(
        self,
        task_id: str,
        *,
        subprocess_pid: int | None,
        request_id: str | None,
        updated_at: float | None = None,
        execution_epoch: int | None = None,
    ) -> TaskRecord:
        now = time.time() if updated_at is None else updated_at
        conn = self._db.get_connection()
        cursor = conn.execute(*self._running_task_update_statement(
            """
            UPDATE tasks
            SET subprocess_pid = ?,
                active_request_id = ?,
                updated_at = ?
            """,
            task_id,
            subprocess_pid,
            request_id,
            now,
            execution_epoch=execution_epoch,
        ))
        if cursor.rowcount != 1:
            conn.rollback()
            raise self._running_task_mismatch_error(task_id, execution_epoch=execution_epoch)
        conn.commit()
        return self.get_task(task_id)

    def clear_running_process(
        self,
        task_id: str,
        *,
        updated_at: float | None = None,
        execution_epoch: int | None = None,
    ) -> TaskRecord:
        return self.set_running_process(
            task_id,
            subprocess_pid=None,
            request_id=None,
            updated_at=updated_at,
            execution_epoch=execution_epoch,
        )

    def touch_running_task(
        self,
        task_id: str,
        *,
        updated_at: float | None = None,
        execution_epoch: int | None = None,
    ) -> TaskRecord:
        now = time.time() if updated_at is None else updated_at
        conn = self._db.get_connection()
        cursor = conn.execute(*self._running_task_update_statement(
            """
            UPDATE tasks
            SET updated_at = ?
            """,
            task_id,
            now,
            execution_epoch=execution_epoch,
        ))
        if cursor.rowcount != 1:
            conn.rollback()
            raise self._running_task_mismatch_error(task_id, execution_epoch=execution_epoch)
        conn.commit()
        return self.get_task(task_id)

    def request_cancel(
        self,
        task_id: str,
        *,
        cancelled_by: str,
        reason: str,
        requested_at: float | None = None,
    ) -> TaskRecord:
        now = time.time() if requested_at is None else requested_at
        conn = self._db.get_connection()
        row = conn.execute(
            "SELECT * FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Task {task_id} was not found")

        status = str(row["status"])
        if status == "pending":
            cursor = conn.execute(
                """
                UPDATE tasks
                SET status = 'cancelled',
                    updated_at = ?,
                    completed_at = ?,
                    cancelled_at = ?,
                    cancellation_reason = ?,
                    cancelled_by = ?,
                    last_error = ?,
                    subprocess_pid = NULL,
                    active_request_id = NULL
                WHERE id = ? AND status = 'pending'
                """,
                (now, now, now, reason, cancelled_by, reason, task_id),
            )
            if cursor.rowcount != 1:
                conn.rollback()
                raise ValueError(f"Task {task_id} could not be cancelled")
            self._insert_task_run(
                conn,
                task_id=task_id,
                task_name=str(row["task_name"]),
                workspace_id=row["workspace_id"],
                status="cancelled",
                started_at=now,
                completed_at=now,
                result={},
                error_text=reason,
            )
            conn.commit()
            return self.get_task(task_id)

        if status != "running":
            raise ValueError(f"Task {task_id} is not cancellable")

        cursor = conn.execute(
            """
            UPDATE tasks
            SET updated_at = ?,
                cancellation_requested_at = COALESCE(cancellation_requested_at, ?),
                cancellation_reason = ?,
                cancelled_by = ?
            WHERE id = ? AND status = 'running'
            """,
            (now, now, reason, cancelled_by, task_id),
        )
        if cursor.rowcount != 1:
            conn.rollback()
            raise ValueError(f"Task {task_id} could not be marked for cancellation")
        conn.commit()
        return self.get_task(task_id)

    def is_cancellation_requested(self, task_id: str) -> bool:
        row = self._db.get_connection().execute(
            "SELECT cancellation_requested_at FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        return row is not None and row["cancellation_requested_at"] is not None

    def finalize_cancellation(
        self,
        task_id: str,
        *,
        cancelled_at: float | None = None,
        execution_epoch: int | None = None,
    ) -> TaskRecord:
        now = time.time() if cancelled_at is None else cancelled_at
        conn = self._db.get_connection()
        row = self._get_running_task_row(conn, task_id, execution_epoch=execution_epoch)
        if row is None:
            raise self._running_task_mismatch_error(task_id, execution_epoch=execution_epoch)
        reason = str(row["cancellation_reason"] or "cancelled")
        cancelled_by = row["cancelled_by"]
        cursor = conn.execute(*self._running_task_update_statement(
            """
            UPDATE tasks
            SET status = 'cancelled',
                updated_at = ?,
                completed_at = ?,
                cancelled_at = ?,
                last_error = ?,
                subprocess_pid = NULL,
                active_request_id = NULL
            """,
            task_id,
            now,
            now,
            now,
            reason,
            execution_epoch=execution_epoch,
        ))
        if cursor.rowcount != 1:
            conn.rollback()
            raise self._running_task_mismatch_error(task_id, execution_epoch=execution_epoch)
        self._insert_task_run(
            conn,
            task_id=task_id,
            task_name=str(row["task_name"]),
            workspace_id=row["workspace_id"],
            status="cancelled",
            started_at=_coalesce_float(row["claimed_at"], row["started_at"], now),
            completed_at=now,
            result={"cancelled_by": cancelled_by, "reason": reason},
            error_text=reason,
        )
        conn.commit()
        return self.get_task(task_id)

    def recover_abandoned_running_tasks(
        self,
        *,
        workspace_id: str | None = None,
        stale_after_seconds: float = 300.0,
        now: float | None = None,
    ) -> list[TaskRecord]:
        current_time = time.time() if now is None else now
        recovered: list[TaskRecord] = []
        for task in self.list_tasks(status="running", workspace_id=workspace_id, limit=200):
            recent_activity_at = max(
                task.updated_at,
                task.started_at or task.updated_at,
                task.claimed_at or task.updated_at,
            )
            is_stale = max(current_time - recent_activity_at, 0.0) >= stale_after_seconds
            if task.subprocess_pid is not None:
                if _is_process_alive(task.subprocess_pid):
                    continue
                if not is_stale:
                    continue
                if task.cancellation_requested_at is not None:
                    recovered.append(
                        self._retry_recovery_transition(
                            lambda: self.finalize_cancellation(task.id, cancelled_at=current_time)
                        )
                    )
                else:
                    recovered.append(
                        self._retry_recovery_transition(
                            lambda: self.fail_permanently(
                                task.id,
                                f"Provider subprocess {task.subprocess_pid} exited unexpectedly",
                                failed_at=current_time,
                            )
                        )
                    )
                continue
            if task.cancellation_requested_at is not None:
                recovered.append(
                    self._retry_recovery_transition(
                        lambda: self.finalize_cancellation(task.id, cancelled_at=current_time)
                    )
                )
                continue
            if is_stale:
                recovered.append(
                    self._retry_recovery_transition(
                        lambda: self.fail(
                            task.id,
                            "Task was abandoned without an active provider subprocess",
                            retry_delay_seconds=5.0,
                            failed_at=current_time,
                        )
                    )
                )
        return recovered

    def _retry_recovery_transition(self, callback: Callable[[], TaskRecord]) -> TaskRecord:
        for attempt in range(_RECOVERY_LOCK_RETRY_ATTEMPTS):
            try:
                return callback()
            except sqlite3.OperationalError as exc:
                if "database is locked" not in str(exc).lower() or attempt + 1 >= _RECOVERY_LOCK_RETRY_ATTEMPTS:
                    raise
                time.sleep(_RECOVERY_LOCK_RETRY_DELAY_SECONDS * (attempt + 1))
        raise RuntimeError("recovery transition retries exhausted")

    def get_task(self, task_id: str) -> TaskRecord:
        row = self._db.get_connection().execute(
            "SELECT * FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Task {task_id} was not found")
        return self._row_to_record(row)

    def _get_running_task_row(self, conn, task_id: str, *, execution_epoch: int | None):
        query = "SELECT * FROM tasks WHERE id = ? AND status = 'running'"
        params: list[object] = [task_id]
        if execution_epoch is not None:
            query += " AND execution_epoch = ?"
            params.append(execution_epoch)
        return conn.execute(query, params).fetchone()

    def _running_task_update_statement(
        self,
        base_query: str,
        task_id: str,
        *params: object,
        execution_epoch: int | None,
    ) -> tuple[str, tuple[object, ...]]:
        query = base_query + "\n            WHERE id = ? AND status = 'running'"
        query_params: list[object] = [*params, task_id]
        if execution_epoch is not None:
            query += " AND execution_epoch = ?"
            query_params.append(execution_epoch)
        return query, tuple(query_params)

    def _running_task_mismatch_error(self, task_id: str, *, execution_epoch: int | None) -> ValueError:
        if execution_epoch is None:
            return ValueError(f"Task {task_id} is not running")
        return ValueError(f"Task {task_id} is not running for execution epoch {execution_epoch}")

    def update_pending_task(
        self,
        task_id: str,
        *,
        data: dict[str, Any] | None = None,
        available_at: float | None = None,
        priority: int | None = None,
    ) -> TaskRecord:
        conn = self._db.get_connection()
        row = conn.execute(
            "SELECT data, available_at, priority FROM tasks WHERE id = ? AND status = 'pending'",
            (task_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Task {task_id} is not pending")

        next_data = dict(json.loads(row["data"] or "{}")) if data is None else data
        next_available_at = float(row["available_at"]) if available_at is None else available_at
        next_priority = int(row["priority"]) if priority is None else priority
        now = time.time()
        cursor = conn.execute(
            "UPDATE tasks SET data = ?, available_at = ?, priority = ?, updated_at = ? WHERE id = ? AND status = 'pending'",
            (json.dumps(next_data, sort_keys=True), next_available_at, next_priority, now, task_id),
        )
        if cursor.rowcount != 1:
            conn.rollback()
            raise ValueError(f"Task {task_id} is not pending")
        conn.commit()
        return self.get_task(task_id)

    def extend_running_task_data_int_list(
        self,
        task_id: str,
        *,
        field_name: str,
        values: list[int],
    ) -> TaskRecord:
        if not field_name.strip():
            raise ValueError("field_name is required")

        normalized_values = [
            value
            for value in values
            if isinstance(value, int) and not isinstance(value, bool) and value > 0
        ]
        if not normalized_values:
            return self.get_task(task_id)


        def _merge(task_data: dict[str, Any]) -> bool:
            existing_values = task_data.get(field_name)
            if not isinstance(existing_values, list):
                existing_values = []
            merged_values = sorted(
                {
                    *[
                        int(value)
                        for value in existing_values
                        if isinstance(value, int) and not isinstance(value, bool) and value > 0
                    ],
                    *normalized_values,
                }
            )
            if merged_values == existing_values:
                return False
            task_data[field_name] = merged_values
            return True

        return self._mutate_running_task_data(task_id, mutate=_merge)

    def extend_running_task_data_object_list(
        self,
        task_id: str,
        *,
        field_name: str,
        values: list[dict[str, Any]],
    ) -> TaskRecord:
        if not field_name.strip():
            raise ValueError("field_name is required")

        normalized_values = [dict(value) for value in values if isinstance(value, dict)]
        if not normalized_values:
            return self.get_task(task_id)


        def _append(task_data: dict[str, Any]) -> bool:
            existing_values = task_data.get(field_name)
            if not isinstance(existing_values, list):
                existing_values = []
            task_data[field_name] = [*existing_values, *normalized_values]
            return True

        return self._mutate_running_task_data(task_id, mutate=_append)

    def clear_running_task_data_keys(
        self,
        task_id: str,
        *,
        field_names: list[str],
    ) -> TaskRecord:
        normalized_field_names = [field_name.strip() for field_name in field_names if isinstance(field_name, str) and field_name.strip()]
        if not normalized_field_names:
            return self.get_task(task_id)


        def _clear(task_data: dict[str, Any]) -> bool:
            updated = False
            for field_name in normalized_field_names:
                if field_name in task_data:
                    task_data.pop(field_name, None)
                    updated = True
            return updated

        return self._mutate_running_task_data(task_id, mutate=_clear)

    def _mutate_running_task_data(
        self,
        task_id: str,
        *,
        mutate: Callable[[dict[str, Any]], bool],
    ) -> TaskRecord:
        conn = self._db.get_connection()
        conn.execute("BEGIN IMMEDIATE")
        try:
            row = conn.execute(
                "SELECT data FROM tasks WHERE id = ? AND status IN ('pending', 'running')",
                (task_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"Task {task_id} is not open")

            task_data = _decode_json_object(row["data"])
            updated = mutate(task_data)

            if not updated:
                conn.rollback()
                return self.get_task(task_id)

            now = time.time()
            cursor = conn.execute(
                "UPDATE tasks SET data = ?, updated_at = ? WHERE id = ? AND status IN ('pending', 'running')",
                (json.dumps(task_data, sort_keys=True), now, task_id),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"Task {task_id} is not open")
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        return self.get_task(task_id)

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

    def find_open_task_any_workspace(
        self,
        task_name: str,
    ) -> TaskRecord | None:
        rows = self.list_open_tasks_any_workspace(task_name)
        if not rows:
            return None
        return rows[0]

    def list_open_tasks_any_workspace(
        self,
        task_name: str,
    ) -> list[TaskRecord]:
        rows = self._db.get_connection().execute(
            """
            SELECT *
            FROM tasks
            WHERE task_name = ?
              AND status IN ('pending', 'running')
            ORDER BY created_at ASC
            """,
            (task_name,),
        ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def find_open_task_with_data(
        self,
        task_name: str,
        *,
        workspace_id: str | None = None,
        data_fields: dict[str, Any],
    ) -> TaskRecord | None:
        if not data_fields:
            return self.find_open_task(task_name, workspace_id)

        rows = self._db.get_connection().execute(
            """
            SELECT *
            FROM tasks
            WHERE task_name = ?
              AND ((workspace_id IS NULL AND ? IS NULL) OR workspace_id = ?)
              AND status IN ('pending', 'running')
            ORDER BY created_at ASC
            """,
            (task_name, workspace_id, workspace_id),
        ).fetchall()
        for row in rows:
            record = self._row_to_record(row)
            if all(record.data.get(key) == value for key, value in data_fields.items()):
                return record
        return None

    def find_open_task_with_data_any_workspace(
        self,
        task_name: str,
        *,
        data_fields: dict[str, Any],
    ) -> TaskRecord | None:
        if not data_fields:
            return self.find_open_task_any_workspace(task_name)

        rows = self._db.get_connection().execute(
            """
            SELECT *
            FROM tasks
            WHERE task_name = ?
              AND status IN ('pending', 'running')
            ORDER BY created_at ASC
            """,
            (task_name,),
        ).fetchall()
        for row in rows:
            record = self._row_to_record(row)
            if all(record.data.get(key) == value for key, value in data_fields.items()):
                return record
        return None

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

    def list_task_runs(
        self,
        task_id: str | None = None,
        task_name: str | None = None,
        workspace_id: str | None = None,
        limit: int = 50,
    ) -> list[TaskRunRecord]:
        clauses: list[str] = []
        params: list[object] = []
        query = "SELECT * FROM task_runs"

        if task_id is not None:
            clauses.append("task_id = ?")
            params.append(task_id)

        if task_name is not None:
            clauses.append("task_name = ?")
            params.append(task_name)

        if workspace_id is not None:
            clauses.append("workspace_id = ?")
            params.append(workspace_id)

        if clauses:
            query += " WHERE " + " AND ".join(clauses)

        query += " ORDER BY completed_at DESC, started_at DESC LIMIT ?"
        params.append(limit)

        rows = self._db.get_connection().execute(query, params).fetchall()
        return [self._row_to_task_run(row) for row in rows]

    def get_latest_successful_task_completion(
        self,
        task_name: str,
    ) -> float | None:
        row = self._db.get_connection().execute(
            "SELECT MAX(completed_at) AS completed_at FROM task_runs WHERE task_name = ? AND status = 'completed'",
            (task_name,),
        ).fetchone()
        if row is None or row["completed_at"] is None:
            return None
        return float(row["completed_at"])

    def summarize_task_runs(
        self,
        task_names: list[str],
        workspace_id: str | None = None,
    ) -> list[TaskRunSummary]:
        if not task_names:
            return []

        summaries = {name: TaskRunSummary(task_name=name) for name in task_names}
        placeholders = ",".join("?" for _ in task_names)
        params: list[object] = [*task_names]
        query = f"SELECT * FROM task_runs WHERE task_name IN ({placeholders})"
        if workspace_id is not None:
            query += " AND workspace_id = ?"
            params.append(workspace_id)
        query += " ORDER BY completed_at DESC, started_at DESC"

        rows = self._db.get_connection().execute(query, params).fetchall()
        duration_totals: dict[str, float] = {name: 0.0 for name in task_names}
        for row in rows:
            task_name = str(row["task_name"])
            summary = summaries[task_name]
            summary.total_runs += 1
            duration_totals[task_name] += float(row["duration_seconds"] or 0.0)
            if row["status"] == "completed":
                summary.completed_runs += 1
            elif row["status"] == "failed":
                summary.failed_runs += 1
            elif row["status"] == "cancelled":
                summary.cancelled_runs += 1
            else:
                summary.retry_runs += 1

            result = coerce_task_run_result(row["result_json"])
            summary.total_lines_compressed += result.lines_compressed or 0

            if summary.last_completed_at is None:
                summary.last_status = str(row["status"])
                summary.last_started_at = float(row["started_at"])
                summary.last_completed_at = float(row["completed_at"])
                summary.last_error = row["error_text"]
                summary.last_result = result

        for summary in summaries.values():
            if summary.total_runs > 0:
                summary.avg_duration_seconds = duration_totals[summary.task_name] / summary.total_runs
        return [summaries[name] for name in task_names]

    def _row_to_record(self, row) -> TaskRecord:
        data = row["data"] or "{}"
        return TaskRecord(
            id=str(row["id"]),
            task_name=str(row["task_name"]),
            data=dict(json.loads(data)),
            workspace_id=row["workspace_id"],
            status=str(row["status"]),
            execution_epoch=int(row["execution_epoch"]),
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
            subprocess_pid=(None if row["subprocess_pid"] is None else int(row["subprocess_pid"])),
            active_request_id=row["active_request_id"],
            cancellation_requested_at=(
                None if row["cancellation_requested_at"] is None else float(row["cancellation_requested_at"])
            ),
            cancelled_at=(None if row["cancelled_at"] is None else float(row["cancelled_at"])),
            cancellation_reason=row["cancellation_reason"],
            cancelled_by=row["cancelled_by"],
        )

    def _row_to_task_run(self, row) -> TaskRunRecord:
        return TaskRunRecord(
            id=str(row["id"]),
            task_id=str(row["task_id"]),
            task_name=str(row["task_name"]),
            workspace_id=row["workspace_id"],
            status=str(row["status"]),
            started_at=float(row["started_at"]),
            completed_at=float(row["completed_at"]),
            duration_seconds=float(row["duration_seconds"] or 0.0),
            result=coerce_task_run_result(row["result_json"]),
            error_text=row["error_text"],
        )

    def _insert_task_run(
        self,
        conn,
        *,
        task_id: str,
        task_name: str,
        workspace_id: str | None,
        status: str,
        started_at: float,
        completed_at: float,
        result: TaskRunResultSource,
        error_text: str | None,
    ) -> None:
        normalized_result = coerce_task_run_result(result)
        duration_seconds = max(completed_at - started_at, 0.0)
        conn.execute(
            """
            INSERT INTO task_runs (
                id,
                task_id,
                task_name,
                workspace_id,
                status,
                started_at,
                completed_at,
                duration_seconds,
                result_json,
                error_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid4()),
                task_id,
                task_name,
                workspace_id,
                status,
                started_at,
                completed_at,
                duration_seconds,
                json.dumps(normalized_result.to_dict(), sort_keys=True),
                error_text,
            ),
        )


def _decode_json_object(value: object) -> dict[str, Any]:
    from mcp_memory.core import tasks as compatibility_tasks

    return compatibility_tasks._decode_json_object(value)


def _coalesce_float(*values: object) -> float:
    for value in values:
        if isinstance(value, bool):
            return float(int(value))
        if isinstance(value, (int, float, str)):
            return float(value)
    return time.time()


def _is_process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
