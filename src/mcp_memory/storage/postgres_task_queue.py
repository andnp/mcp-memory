from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from mcp_memory.storage.session import DbConnectionLike, SessionManager


_ANY_WORKSPACE = object()
_RECOVERY_LOCK_RETRY_ATTEMPTS = 3
_RECOVERY_LOCK_RETRY_DELAY_SECONDS = 0.05

_TASK_COLUMNS = (
    "id, task_name, workspace_id, data, status, execution_epoch, priority, retries_count, max_retries, "
    "created_at, updated_at, available_at, claimed_at, started_at, completed_at, last_error, subprocess_pid, "
    "active_request_id, cancellation_requested_at, cancelled_at, cancellation_reason, cancelled_by"
)
_TASK_RUN_COLUMNS = (
    "id, task_id, task_name, workspace_id, status, started_at, completed_at, duration_seconds, result_json, error_text"
)


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
    execution_epoch: int = 0
    subprocess_pid: int | None = None
    active_request_id: str | None = None
    cancellation_requested_at: float | None = None
    cancelled_at: float | None = None
    cancellation_reason: str | None = None
    cancelled_by: str | None = None


@dataclass
class TaskRunRecord:
    id: str
    task_id: str
    task_name: str
    workspace_id: str | None
    status: str
    started_at: float
    completed_at: float
    duration_seconds: float
    result: dict[str, Any]
    error_text: str | None


@dataclass
class TaskRunSummary:
    task_name: str
    total_runs: int = 0
    completed_runs: int = 0
    failed_runs: int = 0
    cancelled_runs: int = 0
    retry_runs: int = 0
    last_status: str | None = None
    last_started_at: float | None = None
    last_completed_at: float | None = None
    last_error: str | None = None
    last_result: dict[str, Any] = field(default_factory=dict)
    avg_duration_seconds: float = 0.0
    total_lines_compressed: int = 0


class PostgresTaskQueue:
    def __init__(self, session_manager: SessionManager[DbConnectionLike] | None) -> None:
        self._sessions = session_manager

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
        if self._sessions is None:
            raise RuntimeError("task_queue_unavailable")
        if not task_name or not task_name.strip():
            raise ValueError("task_name is required")

        now = time.time()
        task_payload = data or {}
        task_identifier = task_id or str(uuid4())
        ready_at = now if available_at is None else available_at
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO tasks (
                        id, task_name, workspace_id, data, status, execution_epoch,
                        priority, retries_count, max_retries, created_at, updated_at,
                        available_at, claimed_at, started_at, completed_at, last_error,
                        subprocess_pid, active_request_id, cancellation_requested_at,
                        cancelled_at, cancellation_reason, cancelled_by
                    ) VALUES (%s, %s, %s, %s::jsonb, 'pending', 0, %s, 0, %s, %s, %s, %s, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL)
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
            connection.commit()
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
        task = self.enqueue(
            task_name=task_name,
            data=data,
            workspace_id=workspace_id,
            priority=priority,
            max_retries=max_retries,
            available_at=available_at,
        )
        return task, True

    def claim_next(
        self,
        now: float | None = None,
        workspace_id: str | None | object = _ANY_WORKSPACE,
    ) -> TaskRecord | None:
        if self._sessions is None:
            return None
        claimed_at = time.time() if now is None else now
        clauses = ["status = 'pending'", "available_at <= %s"]
        params: list[object] = [claimed_at]
        if workspace_id is None:
            clauses.append("workspace_id IS NULL")
        elif workspace_id is not _ANY_WORKSPACE:
            clauses.append("workspace_id = %s")
            params.append(workspace_id)

        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"SELECT id FROM tasks WHERE {' AND '.join(clauses)} ORDER BY priority ASC, created_at ASC LIMIT 1 FOR UPDATE SKIP LOCKED",
                    tuple(params),
                )
                row = cursor.fetchone()
                if row is None:
                    connection.commit()
                    return None
                task_id = str(row[0])
                cursor.execute(
                    """
                    UPDATE tasks
                    SET status = 'running',
                        execution_epoch = execution_epoch + 1,
                        updated_at = %s,
                        claimed_at = %s,
                        started_at = COALESCE(started_at, %s),
                        last_error = NULL
                    WHERE id = %s AND status = 'pending'
                    """,
                    (claimed_at, claimed_at, claimed_at, task_id),
                )
                if int(getattr(cursor, "rowcount", 0) or 0) != 1:
                    connection.rollback()
                    return None
            connection.commit()
        return self.get_task(task_id)

    def complete(
        self,
        task_id: str,
        completed_at: float | None = None,
        run_result: dict[str, Any] | None = None,
        execution_epoch: int | None = None,
    ) -> TaskRecord:
        if self._sessions is None:
            raise RuntimeError("task_queue_unavailable")
        now = time.time() if completed_at is None else completed_at
        with self._sessions.open_connection() as connection:
            row = self._get_running_task_row(connection, task_id, execution_epoch=execution_epoch)
            if row is None:
                raise self._running_task_mismatch_error(task_id, execution_epoch=execution_epoch)
            with connection.cursor() as cursor:
                cursor.execute(*self._running_task_update_statement(
                    """
                    UPDATE tasks
                    SET status = 'completed',
                        updated_at = %s,
                        completed_at = %s,
                        last_error = NULL,
                        subprocess_pid = NULL,
                        active_request_id = NULL
                    """,
                    task_id,
                    now,
                    now,
                    execution_epoch=execution_epoch,
                ))
                if int(getattr(cursor, "rowcount", 0) or 0) != 1:
                    connection.rollback()
                    raise self._running_task_mismatch_error(task_id, execution_epoch=execution_epoch)
            self._insert_task_run(
                connection,
                task_id=task_id,
                task_name=str(row[1]),
                workspace_id=None if row[2] is None else str(row[2]),
                status="completed",
                started_at=_coalesce_float(row[12], row[13], now),
                completed_at=now,
                result=run_result or {},
                error_text=None,
            )
            connection.commit()
        return self.get_task(task_id)

    def fail(
        self,
        task_id: str,
        error: str,
        retry_delay_seconds: float = 0.0,
        failed_at: float | None = None,
        execution_epoch: int | None = None,
    ) -> TaskRecord:
        if self._sessions is None:
            raise RuntimeError("task_queue_unavailable")
        now = time.time() if failed_at is None else failed_at
        with self._sessions.open_connection() as connection:
            row = self._get_running_task_row(connection, task_id, execution_epoch=execution_epoch)
            if row is None:
                raise self._running_task_mismatch_error(task_id, execution_epoch=execution_epoch)

            next_retries = _coerce_int(row[7]) + 1
            terminal = next_retries >= _coerce_int(row[8])
            status = "failed" if terminal else "pending"
            available_at = now if terminal else now + retry_delay_seconds
            completed_at = now if terminal else None
            with connection.cursor() as cursor:
                cursor.execute(*self._running_task_update_statement(
                    """
                    UPDATE tasks
                    SET status = %s,
                        retries_count = %s,
                        updated_at = %s,
                        available_at = %s,
                        claimed_at = NULL,
                        completed_at = %s,
                        last_error = %s,
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
                if int(getattr(cursor, "rowcount", 0) or 0) != 1:
                    connection.rollback()
                    raise self._running_task_mismatch_error(task_id, execution_epoch=execution_epoch)
            self._insert_task_run(
                connection,
                task_id=task_id,
                task_name=str(row[1]),
                workspace_id=None if row[2] is None else str(row[2]),
                status="failed" if terminal else "retry",
                started_at=_coalesce_float(row[12], row[13], now),
                completed_at=now,
                result={},
                error_text=error,
            )
            connection.commit()
        return self.get_task(task_id)

    def fail_permanently(
        self,
        task_id: str,
        error: str,
        failed_at: float | None = None,
        execution_epoch: int | None = None,
    ) -> TaskRecord:
        if self._sessions is None:
            raise RuntimeError("task_queue_unavailable")
        now = time.time() if failed_at is None else failed_at
        with self._sessions.open_connection() as connection:
            row = self._get_running_task_row(connection, task_id, execution_epoch=execution_epoch)
            if row is None:
                raise self._running_task_mismatch_error(task_id, execution_epoch=execution_epoch)
            with connection.cursor() as cursor:
                cursor.execute(*self._running_task_update_statement(
                    """
                    UPDATE tasks
                    SET status = 'failed',
                        retries_count = max_retries,
                        updated_at = %s,
                        available_at = %s,
                        claimed_at = NULL,
                        completed_at = %s,
                        last_error = %s,
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
                if int(getattr(cursor, "rowcount", 0) or 0) != 1:
                    connection.rollback()
                    raise self._running_task_mismatch_error(task_id, execution_epoch=execution_epoch)
            self._insert_task_run(
                connection,
                task_id=task_id,
                task_name=str(row[1]),
                workspace_id=None if row[2] is None else str(row[2]),
                status="failed",
                started_at=_coalesce_float(row[12], row[13], now),
                completed_at=now,
                result={},
                error_text=error,
            )
            connection.commit()
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
        if self._sessions is None:
            raise RuntimeError("task_queue_unavailable")
        now = time.time() if updated_at is None else updated_at
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(*self._running_task_update_statement(
                    """
                    UPDATE tasks
                    SET subprocess_pid = %s,
                        active_request_id = %s,
                        updated_at = %s
                    """,
                    task_id,
                    subprocess_pid,
                    request_id,
                    now,
                    execution_epoch=execution_epoch,
                ))
                if int(getattr(cursor, "rowcount", 0) or 0) != 1:
                    connection.rollback()
                    raise self._running_task_mismatch_error(task_id, execution_epoch=execution_epoch)
            connection.commit()
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
        if self._sessions is None:
            raise RuntimeError("task_queue_unavailable")
        now = time.time() if updated_at is None else updated_at
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(*self._running_task_update_statement(
                    """
                    UPDATE tasks
                    SET updated_at = %s
                    """,
                    task_id,
                    now,
                    execution_epoch=execution_epoch,
                ))
                if int(getattr(cursor, "rowcount", 0) or 0) != 1:
                    connection.rollback()
                    raise self._running_task_mismatch_error(task_id, execution_epoch=execution_epoch)
            connection.commit()
        return self.get_task(task_id)

    def request_cancel(
        self,
        task_id: str,
        *,
        cancelled_by: str,
        reason: str,
        requested_at: float | None = None,
    ) -> TaskRecord:
        if self._sessions is None:
            raise RuntimeError("task_queue_unavailable")
        now = time.time() if requested_at is None else requested_at
        with self._sessions.open_connection() as connection:
            row = self._get_task_row(connection, task_id)
            if row is None:
                raise ValueError(f"Task {task_id} was not found")
            status = str(row[4])
            with connection.cursor() as cursor:
                if status == "pending":
                    cursor.execute(
                        """
                        UPDATE tasks
                        SET status = 'cancelled',
                            updated_at = %s,
                            completed_at = %s,
                            cancelled_at = %s,
                            cancellation_reason = %s,
                            cancelled_by = %s,
                            last_error = %s,
                            subprocess_pid = NULL,
                            active_request_id = NULL
                        WHERE id = %s AND status = 'pending'
                        """,
                        (now, now, now, reason, cancelled_by, reason, task_id),
                    )
                    if int(getattr(cursor, "rowcount", 0) or 0) != 1:
                        connection.rollback()
                        raise ValueError(f"Task {task_id} could not be cancelled")
                    self._insert_task_run(
                        connection,
                        task_id=task_id,
                        task_name=str(row[1]),
                        workspace_id=None if row[2] is None else str(row[2]),
                        status="cancelled",
                        started_at=now,
                        completed_at=now,
                        result={},
                        error_text=reason,
                    )
                    connection.commit()
                    return self.get_task(task_id)
                if status != "running":
                    raise ValueError(f"Task {task_id} is not cancellable")
                cursor.execute(
                    """
                    UPDATE tasks
                    SET updated_at = %s,
                        cancellation_requested_at = COALESCE(cancellation_requested_at, %s),
                        cancellation_reason = %s,
                        cancelled_by = %s
                    WHERE id = %s AND status = 'running'
                    """,
                    (now, now, reason, cancelled_by, task_id),
                )
                if int(getattr(cursor, "rowcount", 0) or 0) != 1:
                    connection.rollback()
                    raise ValueError(f"Task {task_id} could not be marked for cancellation")
            connection.commit()
        return self.get_task(task_id)

    def is_cancellation_requested(self, task_id: str) -> bool:
        row = self._get_task_row_from_sessions(task_id, columns="cancellation_requested_at")
        return row is not None and row[0] is not None

    def finalize_cancellation(
        self,
        task_id: str,
        *,
        cancelled_at: float | None = None,
        execution_epoch: int | None = None,
    ) -> TaskRecord:
        if self._sessions is None:
            raise RuntimeError("task_queue_unavailable")
        now = time.time() if cancelled_at is None else cancelled_at
        with self._sessions.open_connection() as connection:
            row = self._get_running_task_row(connection, task_id, execution_epoch=execution_epoch)
            if row is None:
                raise self._running_task_mismatch_error(task_id, execution_epoch=execution_epoch)
            reason = str(row[20] or "cancelled")
            cancelled_by = row[21]
            with connection.cursor() as cursor:
                cursor.execute(*self._running_task_update_statement(
                    """
                    UPDATE tasks
                    SET status = 'cancelled',
                        updated_at = %s,
                        completed_at = %s,
                        cancelled_at = %s,
                        last_error = %s,
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
                if int(getattr(cursor, "rowcount", 0) or 0) != 1:
                    connection.rollback()
                    raise self._running_task_mismatch_error(task_id, execution_epoch=execution_epoch)
            self._insert_task_run(
                connection,
                task_id=task_id,
                task_name=str(row[1]),
                workspace_id=None if row[2] is None else str(row[2]),
                status="cancelled",
                started_at=_coalesce_float(row[12], row[13], now),
                completed_at=now,
                result={"cancelled_by": cancelled_by, "reason": reason},
                error_text=reason,
            )
            connection.commit()
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
            recent_activity_at = max(task.updated_at, task.started_at or task.updated_at, task.claimed_at or task.updated_at)
            is_stale = max(current_time - recent_activity_at, 0.0) >= stale_after_seconds
            if task.subprocess_pid is not None:
                if _is_process_alive(task.subprocess_pid):
                    continue
                if not is_stale:
                    continue
                if task.cancellation_requested_at is not None:
                    recovered.append(self._retry_recovery_transition(lambda: self.finalize_cancellation(task.id, cancelled_at=current_time)))
                else:
                    recovered.append(self._retry_recovery_transition(lambda: self.fail_permanently(task.id, f"Provider subprocess {task.subprocess_pid} exited unexpectedly", failed_at=current_time)))
                continue
            if task.cancellation_requested_at is not None:
                recovered.append(self._retry_recovery_transition(lambda: self.finalize_cancellation(task.id, cancelled_at=current_time)))
                continue
            if is_stale:
                recovered.append(self._retry_recovery_transition(lambda: self.fail_permanently(task.id, "Task was abandoned without an active provider subprocess", failed_at=current_time)))
        return recovered

    def _retry_recovery_transition(self, callback: Callable[[], TaskRecord]) -> TaskRecord:
        for attempt in range(_RECOVERY_LOCK_RETRY_ATTEMPTS):
            try:
                return callback()
            except Exception as exc:
                message = str(exc).lower()
                if ("lock" not in message and "deadlock" not in message) or attempt + 1 >= _RECOVERY_LOCK_RETRY_ATTEMPTS:
                    raise
                time.sleep(_RECOVERY_LOCK_RETRY_DELAY_SECONDS * (attempt + 1))
        raise RuntimeError("recovery transition retries exhausted")

    def get_task(self, task_id: str) -> TaskRecord:
        row = self._get_task_row_from_sessions(task_id)
        if row is None:
            raise ValueError(f"Task {task_id} was not found")
        return self._row_to_record(row)

    def update_pending_task(
        self,
        task_id: str,
        *,
        data: dict[str, Any] | None = None,
        available_at: float | None = None,
        priority: int | None = None,
    ) -> TaskRecord:
        if self._sessions is None:
            raise RuntimeError("task_queue_unavailable")
        with self._sessions.open_connection() as connection:
            row = self._get_task_row(connection, task_id, columns="data, available_at, priority, status")
            if row is None or str(row[3]) != "pending":
                raise ValueError(f"Task {task_id} is not pending")

            next_data = _decode_json_object(row[0]) if data is None else data
            next_available_at = _as_float(row[1]) if available_at is None else available_at
            next_priority = _coerce_int(row[2]) if priority is None else priority
            now = time.time()
            with connection.cursor() as cursor:
                cursor.execute(
                    "UPDATE tasks SET data = %s::jsonb, available_at = %s, priority = %s, updated_at = %s WHERE id = %s AND status = 'pending'",
                    (json.dumps(next_data, sort_keys=True), next_available_at, next_priority, now, task_id),
                )
                if int(getattr(cursor, "rowcount", 0) or 0) != 1:
                    connection.rollback()
                    raise ValueError(f"Task {task_id} is not pending")
            connection.commit()
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
        normalized_values = [value for value in values if isinstance(value, int) and not isinstance(value, bool) and value > 0]
        if not normalized_values:
            return self.get_task(task_id)

        def _merge(task_data: dict[str, Any]) -> bool:
            existing_values = task_data.get(field_name)
            if not isinstance(existing_values, list):
                existing_values = []
            merged_values = sorted({*[int(value) for value in existing_values if isinstance(value, int) and not isinstance(value, bool) and value > 0], *normalized_values})
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
        if self._sessions is None:
            raise RuntimeError("task_queue_unavailable")
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT data FROM tasks WHERE id = %s AND status IN ('pending', 'running') FOR UPDATE",
                    (task_id,),
                )
                row = cursor.fetchone()
                if row is None:
                    connection.rollback()
                    raise ValueError(f"Task {task_id} is not open")
                task_data = _decode_json_object(row[0])
                updated = mutate(task_data)
                if not updated:
                    connection.rollback()
                    return self.get_task(task_id)
                now = time.time()
                cursor.execute(
                    "UPDATE tasks SET data = %s::jsonb, updated_at = %s WHERE id = %s AND status IN ('pending', 'running')",
                    (json.dumps(task_data, sort_keys=True), now, task_id),
                )
                if int(getattr(cursor, "rowcount", 0) or 0) != 1:
                    connection.rollback()
                    raise ValueError(f"Task {task_id} is not open")
            connection.commit()
        return self.get_task(task_id)

    def find_open_task(self, task_name: str, workspace_id: str | None = None) -> TaskRecord | None:
        if self._sessions is None:
            return None
        query = f"SELECT {_TASK_COLUMNS} FROM tasks WHERE task_name = %s AND status IN ('pending', 'running')"
        params: list[object] = [task_name]
        if workspace_id is None:
            query += " AND workspace_id IS NULL"
        else:
            query += " AND workspace_id = %s"
            params.append(workspace_id)
        query += " ORDER BY created_at ASC LIMIT 1"
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, tuple(params))
                row = cursor.fetchone()
        return None if row is None else self._row_to_record(row)

    def find_open_task_any_workspace(self, task_name: str) -> TaskRecord | None:
        if self._sessions is None:
            return None
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"SELECT {_TASK_COLUMNS} FROM tasks WHERE task_name = %s AND status IN ('pending', 'running') ORDER BY created_at ASC LIMIT 1",
                    (task_name,),
                )
                row = cursor.fetchone()
        return None if row is None else self._row_to_record(row)

    def find_open_task_with_data(
        self,
        task_name: str,
        *,
        workspace_id: str | None = None,
        data_fields: dict[str, Any],
    ) -> TaskRecord | None:
        if not data_fields:
            return self.find_open_task(task_name, workspace_id)
        rows = self._list_open_task_rows(task_name, workspace_id=workspace_id)
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
        rows = self._list_open_task_rows(task_name, workspace_id=_ANY_WORKSPACE)
        for row in rows:
            record = self._row_to_record(row)
            if all(record.data.get(key) == value for key, value in data_fields.items()):
                return record
        return None

    def count_by_status(self) -> dict[str, int]:
        if self._sessions is None:
            return {}
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT status, COUNT(*) FROM tasks GROUP BY status", ())
                rows = cursor.fetchall()
        return {str(row[0]): _coerce_int(row[1]) for row in rows}

    def list_tasks(
        self,
        status: str | None = None,
        workspace_id: str | None = None,
        limit: int = 20,
    ) -> list[TaskRecord]:
        if self._sessions is None:
            return []
        clauses: list[str] = []
        params: list[object] = []
        query = f"SELECT {_TASK_COLUMNS} FROM tasks"
        if status is not None:
            clauses.append("status = %s")
            params.append(status)
        if workspace_id is not None:
            clauses.append("workspace_id = %s")
            params.append(workspace_id)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY updated_at DESC, created_at DESC LIMIT %s"
        params.append(limit)
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, tuple(params))
                rows = cursor.fetchall()
        return [self._row_to_record(row) for row in rows]

    def list_task_runs(
        self,
        task_id: str | None = None,
        task_name: str | None = None,
        workspace_id: str | None = None,
        limit: int = 50,
    ) -> list[TaskRunRecord]:
        if self._sessions is None:
            return []
        clauses: list[str] = []
        params: list[object] = []
        query = f"SELECT {_TASK_RUN_COLUMNS} FROM task_runs"
        if task_id is not None:
            clauses.append("task_id = %s")
            params.append(task_id)
        if task_name is not None:
            clauses.append("task_name = %s")
            params.append(task_name)
        if workspace_id is not None:
            clauses.append("workspace_id = %s")
            params.append(workspace_id)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY completed_at DESC, started_at DESC LIMIT %s"
        params.append(limit)
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, tuple(params))
                rows = cursor.fetchall()
        return [self._row_to_task_run(row) for row in rows]

    def get_latest_successful_task_completion(self, task_name: str) -> float | None:
        if self._sessions is None:
            return None
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT MAX(completed_at) FROM task_runs WHERE task_name = %s AND status = 'completed'",
                    (task_name,),
                )
                row = cursor.fetchone()
        if row is None or row[0] is None:
            return None
        return _as_float(row[0])

    def summarize_task_runs(
        self,
        task_names: list[str],
        workspace_id: str | None = None,
    ) -> list[TaskRunSummary]:
        if self._sessions is None:
            return []
        if not task_names:
            return []
        summaries = {name: TaskRunSummary(task_name=name) for name in task_names}
        placeholders = ",".join(["%s"] * len(task_names))
        params: list[object] = [*task_names]
        query = f"SELECT {_TASK_RUN_COLUMNS} FROM task_runs WHERE task_name IN ({placeholders})"
        if workspace_id is not None:
            query += " AND workspace_id = %s"
            params.append(workspace_id)
        query += " ORDER BY completed_at DESC, started_at DESC"
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, tuple(params))
                rows = cursor.fetchall()
        duration_totals: dict[str, float] = {name: 0.0 for name in task_names}
        for row in rows:
            task_name = str(row[2])
            summary = summaries[task_name]
            summary.total_runs += 1
            duration_totals[task_name] += _as_float(row[7])
            if row[4] == "completed":
                summary.completed_runs += 1
            elif row[4] == "failed":
                summary.failed_runs += 1
            elif row[4] == "cancelled":
                summary.cancelled_runs += 1
            else:
                summary.retry_runs += 1
            result = _decode_json_object(row[8])
            summary.total_lines_compressed += _coerce_int(result.get("lines_compressed"), default=0)
            if summary.last_completed_at is None:
                summary.last_status = str(row[4])
                summary.last_started_at = _as_float(row[5])
                summary.last_completed_at = _as_float(row[6])
                summary.last_error = None if row[9] is None else str(row[9])
                summary.last_result = result
        for summary in summaries.values():
            if summary.total_runs > 0:
                summary.avg_duration_seconds = duration_totals[summary.task_name] / summary.total_runs
        return [summaries[name] for name in task_names]

    def _get_running_task_row(self, connection: DbConnectionLike, task_id: str, *, execution_epoch: int | None):
        query = f"SELECT {_TASK_COLUMNS} FROM tasks WHERE id = %s AND status = 'running'"
        params: list[object] = [task_id]
        if execution_epoch is not None:
            query += " AND execution_epoch = %s"
            params.append(execution_epoch)
        with connection.cursor() as cursor:
            cursor.execute(query, tuple(params))
            return cursor.fetchone()

    def _running_task_update_statement(
        self,
        base_query: str,
        task_id: str,
        *params: object,
        execution_epoch: int | None,
    ) -> tuple[str, tuple[object, ...]]:
        query = base_query + "\n                    WHERE id = %s AND status = 'running'"
        query_params: list[object] = [*params, task_id]
        if execution_epoch is not None:
            query += " AND execution_epoch = %s"
            query_params.append(execution_epoch)
        return query, tuple(query_params)

    def _running_task_mismatch_error(self, task_id: str, *, execution_epoch: int | None) -> ValueError:
        if execution_epoch is None:
            return ValueError(f"Task {task_id} is not running")
        return ValueError(f"Task {task_id} is not running for execution epoch {execution_epoch}")

    def _get_task_row_from_sessions(self, task_id: str, *, columns: str = _TASK_COLUMNS):
        if self._sessions is None:
            return None
        with self._sessions.open_connection() as connection:
            return self._get_task_row(connection, task_id, columns=columns)

    def _get_task_row(self, connection: DbConnectionLike, task_id: str, *, columns: str = _TASK_COLUMNS):
        with connection.cursor() as cursor:
            cursor.execute(f"SELECT {columns} FROM tasks WHERE id = %s", (task_id,))
            return cursor.fetchone()

    def _list_open_task_rows(self, task_name: str, *, workspace_id: str | None | object) -> list[tuple[object, ...]]:
        if self._sessions is None:
            return []
        query = f"SELECT {_TASK_COLUMNS} FROM tasks WHERE task_name = %s AND status IN ('pending', 'running')"
        params: list[object] = [task_name]
        if workspace_id is None:
            query += " AND workspace_id IS NULL"
        elif workspace_id is not _ANY_WORKSPACE:
            query += " AND workspace_id = %s"
            params.append(workspace_id)
        query += " ORDER BY created_at ASC"
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, tuple(params))
                return cursor.fetchall()

    def _row_to_record(self, row: tuple[object, ...]) -> TaskRecord:
        return TaskRecord(
            id=str(row[0]),
            task_name=str(row[1]),
            data=_decode_json_object(row[3]),
            workspace_id=None if row[2] is None else str(row[2]),
            status=str(row[4]),
            execution_epoch=_coerce_int(row[5]),
            priority=_coerce_int(row[6]),
            retries_count=_coerce_int(row[7]),
            max_retries=_coerce_int(row[8]),
            created_at=_as_float(row[9]),
            updated_at=_as_float(row[10]),
            available_at=_as_float(row[11]),
            claimed_at=None if row[12] is None else _as_float(row[12]),
            started_at=None if row[13] is None else _as_float(row[13]),
            completed_at=None if row[14] is None else _as_float(row[14]),
            last_error=None if row[15] is None else str(row[15]),
            subprocess_pid=None if row[16] is None else _coerce_int(row[16]),
            active_request_id=None if row[17] is None else str(row[17]),
            cancellation_requested_at=None if row[18] is None else _as_float(row[18]),
            cancelled_at=None if row[19] is None else _as_float(row[19]),
            cancellation_reason=None if row[20] is None else str(row[20]),
            cancelled_by=None if row[21] is None else str(row[21]),
        )

    def _row_to_task_run(self, row: tuple[object, ...]) -> TaskRunRecord:
        return TaskRunRecord(
            id=str(row[0]),
            task_id=str(row[1]),
            task_name=str(row[2]),
            workspace_id=None if row[3] is None else str(row[3]),
            status=str(row[4]),
            started_at=_as_float(row[5]),
            completed_at=_as_float(row[6]),
            duration_seconds=_as_float(row[7]),
            result=_decode_json_object(row[8]),
            error_text=None if row[9] is None else str(row[9]),
        )

    def _insert_task_run(
        self,
        connection: DbConnectionLike,
        *,
        task_id: str,
        task_name: str,
        workspace_id: str | None,
        status: str,
        started_at: float,
        completed_at: float,
        result: dict[str, Any],
        error_text: str | None,
    ) -> None:
        duration_seconds = max(completed_at - started_at, 0.0)
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO task_runs (
                    id, task_id, task_name, workspace_id, status,
                    started_at, completed_at, duration_seconds, result_json, error_text
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)
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
                    json.dumps(result or {}, sort_keys=True),
                    error_text,
                ),
            )


def _decode_json_object(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value:
        decoded = json.loads(value)
        return decoded if isinstance(decoded, dict) else {}
    return {}


def _coalesce_float(*values: object) -> float:
    for value in values:
        if isinstance(value, bool):
            return float(int(value))
        if isinstance(value, (int, float, str)):
            return float(value)
    return time.time()


def _coerce_int(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        return int(value)
    return default


def _as_float(value: object) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"Expected float-compatible value, got {type(value)!r}")


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
