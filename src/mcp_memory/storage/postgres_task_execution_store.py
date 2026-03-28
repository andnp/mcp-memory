from __future__ import annotations

from dataclasses import dataclass
import time

from mcp_memory.storage.session import CursorLike, DbConnectionLike, SessionManager
from mcp_memory.task_execution_store import TaskExecutionAttemptRecord


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


def _coerce_float(value: object) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"Expected float-compatible value, got {type(value)!r}")


@dataclass(frozen=True)
class _StoredAttempt:
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


class PostgresTaskExecutionAttemptRepository:
    def __init__(self, session_manager: SessionManager[DbConnectionLike], *, workspace_id: str | None):
        self._sessions = session_manager
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
        now = time.time() if started_at is None else started_at
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                existing = self._get_attempt_row(cursor, task_id=task_id, execution_epoch=execution_epoch)
                if existing is None:
                    cursor.execute(
                        """
                        INSERT INTO task_execution_attempts (
                            task_id, execution_epoch, workspace_id, task_name, request_id, subprocess_pid,
                            provider_key, provider_name, model_name, status, started_at, last_heartbeat_at,
                            completed_at, error_text, termination_reason
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NULL, NULL, NULL)
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
                else:
                    updated = existing if existing.completed_at is not None else _StoredAttempt(
                        id=existing.id,
                        task_id=existing.task_id,
                        execution_epoch=existing.execution_epoch,
                        workspace_id=existing.workspace_id or self._workspace_id,
                        task_name=existing.task_name or task_name,
                        request_id=request_id or existing.request_id,
                        subprocess_pid=subprocess_pid if subprocess_pid is not None else existing.subprocess_pid,
                        provider_key=existing.provider_key or provider_key,
                        provider_name=existing.provider_name or provider_name,
                        model_name=existing.model_name or model_name,
                        status=status,
                        started_at=min(existing.started_at, now),
                        last_heartbeat_at=max(existing.last_heartbeat_at or now, now),
                        completed_at=existing.completed_at,
                        error_text=existing.error_text,
                        termination_reason=existing.termination_reason,
                    )
                    self._update_attempt_row(cursor, updated)
            connection.commit()
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
        now = time.time() if heartbeat_at is None else heartbeat_at
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                existing = self._get_attempt_row(cursor, task_id=task_id, execution_epoch=execution_epoch)
                if existing is None:
                    raise ValueError(f"Task execution attempt {task_id}@{execution_epoch} was not found")
                if existing.completed_at is None:
                    updated = _StoredAttempt(
                        id=existing.id,
                        task_id=existing.task_id,
                        execution_epoch=existing.execution_epoch,
                        workspace_id=existing.workspace_id,
                        task_name=existing.task_name,
                        request_id=request_id or existing.request_id,
                        subprocess_pid=subprocess_pid if subprocess_pid is not None else existing.subprocess_pid,
                        provider_key=existing.provider_key,
                        provider_name=existing.provider_name,
                        model_name=existing.model_name,
                        status=existing.status,
                        started_at=existing.started_at,
                        last_heartbeat_at=max(existing.last_heartbeat_at or now, now),
                        completed_at=None,
                        error_text=existing.error_text,
                        termination_reason=existing.termination_reason,
                    )
                    self._update_attempt_row(cursor, updated)
            connection.commit()
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
        now = time.time() if completed_at is None else completed_at
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                existing = self._get_attempt_row(cursor, task_id=task_id, execution_epoch=execution_epoch)
                if existing is None:
                    raise ValueError(f"Task execution attempt {task_id}@{execution_epoch} was not found")
                if existing.completed_at is None:
                    updated = _StoredAttempt(
                        id=existing.id,
                        task_id=existing.task_id,
                        execution_epoch=existing.execution_epoch,
                        workspace_id=existing.workspace_id,
                        task_name=existing.task_name,
                        request_id=request_id or existing.request_id,
                        subprocess_pid=subprocess_pid if subprocess_pid is not None else existing.subprocess_pid,
                        provider_key=existing.provider_key,
                        provider_name=existing.provider_name,
                        model_name=existing.model_name,
                        status=status,
                        started_at=existing.started_at,
                        last_heartbeat_at=max(existing.last_heartbeat_at or now, now),
                        completed_at=now,
                        error_text=error_text or existing.error_text,
                        termination_reason=termination_reason or existing.termination_reason,
                    )
                else:
                    updated = _StoredAttempt(
                        id=existing.id,
                        task_id=existing.task_id,
                        execution_epoch=existing.execution_epoch,
                        workspace_id=existing.workspace_id,
                        task_name=existing.task_name,
                        request_id=request_id or existing.request_id,
                        subprocess_pid=subprocess_pid if subprocess_pid is not None else existing.subprocess_pid,
                        provider_key=existing.provider_key,
                        provider_name=existing.provider_name,
                        model_name=existing.model_name,
                        status=existing.status,
                        started_at=existing.started_at,
                        last_heartbeat_at=existing.last_heartbeat_at,
                        completed_at=existing.completed_at,
                        error_text=error_text or existing.error_text,
                        termination_reason=termination_reason or existing.termination_reason,
                    )
                self._update_attempt_row(cursor, updated)
            connection.commit()
        return self.get_attempt(task_id=task_id, execution_epoch=execution_epoch)

    def get_attempt(self, *, task_id: str, execution_epoch: int) -> TaskExecutionAttemptRecord:
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                existing = self._get_attempt_row(cursor, task_id=task_id, execution_epoch=execution_epoch)
                if existing is None:
                    raise ValueError(f"Task execution attempt {task_id}@{execution_epoch} was not found")
                return self._to_public_record(existing)

    def list_attempts(
        self,
        *,
        task_id: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[TaskExecutionAttemptRecord]:
        clauses: list[str] = []
        params: list[object] = []
        if self._workspace_id is not None:
            clauses.append("workspace_id = %s")
            params.append(self._workspace_id)
        if task_id is not None:
            clauses.append("task_id = %s")
            params.append(task_id)
        if status is not None:
            clauses.append("status = %s")
            params.append(status)
        query = "SELECT * FROM task_execution_attempts"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY started_at DESC, id DESC LIMIT %s"
        params.append(limit)
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, tuple(params))
                return [self._to_public_record(self._row_to_attempt(row)) for row in cursor.fetchall()]

    def _get_attempt_row(self, cursor: CursorLike, *, task_id: str, execution_epoch: int) -> _StoredAttempt | None:
        cursor.execute(
            "SELECT * FROM task_execution_attempts WHERE task_id = %s AND execution_epoch = %s LIMIT 1",
            (task_id, execution_epoch),
        )
        row = cursor.fetchone()
        return None if row is None else self._row_to_attempt(row)

    def _update_attempt_row(self, cursor: CursorLike, attempt: _StoredAttempt) -> None:
        cursor.execute(
            """
            UPDATE task_execution_attempts
            SET workspace_id = %s,
                task_name = %s,
                request_id = %s,
                subprocess_pid = %s,
                provider_key = %s,
                provider_name = %s,
                model_name = %s,
                status = %s,
                started_at = %s,
                last_heartbeat_at = %s,
                completed_at = %s,
                error_text = %s,
                termination_reason = %s
            WHERE task_id = %s AND execution_epoch = %s
            """,
            (
                attempt.workspace_id,
                attempt.task_name,
                attempt.request_id,
                attempt.subprocess_pid,
                attempt.provider_key,
                attempt.provider_name,
                attempt.model_name,
                attempt.status,
                attempt.started_at,
                attempt.last_heartbeat_at,
                attempt.completed_at,
                attempt.error_text,
                attempt.termination_reason,
                attempt.task_id,
                attempt.execution_epoch,
            ),
        )

    def _row_to_attempt(self, row: tuple[object, ...]) -> _StoredAttempt:
        return _StoredAttempt(
            id=_coerce_int(row[0]),
            task_id=str(row[1]),
            execution_epoch=_coerce_int(row[2]),
            workspace_id=None if row[3] is None else str(row[3]),
            task_name=None if row[4] is None else str(row[4]),
            request_id=None if row[5] is None else str(row[5]),
            subprocess_pid=None if row[6] is None else _coerce_int(row[6]),
            provider_key=None if row[7] is None else str(row[7]),
            provider_name=None if row[8] is None else str(row[8]),
            model_name=None if row[9] is None else str(row[9]),
            status=str(row[10]),
            started_at=_coerce_float(row[11]),
            last_heartbeat_at=None if row[12] is None else _coerce_float(row[12]),
            completed_at=None if row[13] is None else _coerce_float(row[13]),
            error_text=None if row[14] is None else str(row[14]),
            termination_reason=None if row[15] is None else str(row[15]),
        )

    def _to_public_record(self, attempt: _StoredAttempt) -> TaskExecutionAttemptRecord:
        return TaskExecutionAttemptRecord(
            id=attempt.id,
            task_id=attempt.task_id,
            execution_epoch=attempt.execution_epoch,
            workspace_id=attempt.workspace_id,
            task_name=attempt.task_name,
            request_id=attempt.request_id,
            subprocess_pid=attempt.subprocess_pid,
            provider_key=attempt.provider_key,
            provider_name=attempt.provider_name,
            model_name=attempt.model_name,
            status=attempt.status,
            started_at=attempt.started_at,
            last_heartbeat_at=attempt.last_heartbeat_at,
            completed_at=attempt.completed_at,
            error_text=attempt.error_text,
            termination_reason=attempt.termination_reason,
        )
