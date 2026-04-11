from __future__ import annotations

from collections.abc import Sequence
import json
import logging

import pytest

from mcp_memory.config import LoggingConfig
from mcp_memory.storage.postgres_runtime_log_store import PostgresRuntimeLogRepository, PostgresStructuredLogHandler
from mcp_memory.storage.postgres_task_execution_store import PostgresTaskExecutionAttemptRepository


pytestmark = pytest.mark.small


def _as_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        return int(value)
    raise TypeError(f"Expected int-compatible value, got {type(value)!r}")


def _as_float(value: object) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"Expected float-compatible value, got {type(value)!r}")


class FakeOperationalState:
    def __init__(self) -> None:
        self.runtime_logs: list[dict[str, object]] = []
        self.task_execution_attempts: list[dict[str, object]] = []
        self.next_log_id = 1
        self.next_attempt_id = 1


class FakeCursor:
    def __init__(self, state: FakeOperationalState) -> None:
        self._state = state
        self._result: list[tuple[object, ...]] = []

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def execute(self, query: str, params: tuple[object, ...] | None = None) -> None:
        normalized = " ".join(query.split())
        arguments = tuple(() if params is None else params)
        if normalized.startswith("INSERT INTO runtime_logs"):
            data_json = arguments[6]
            if isinstance(data_json, str):
                data_json = json.loads(data_json)
            self._state.runtime_logs.append(
                {
                    "id": self._state.next_log_id,
                    "workspace_id": arguments[0],
                    "source": arguments[1],
                    "logger_name": arguments[2],
                    "level": arguments[3],
                    "message": arguments[4],
                    "created_at": _as_float(arguments[5]),
                    "data_json": data_json,
                }
            )
            self._state.next_log_id += 1
            self._result = []
            return
        if normalized.startswith("SELECT id, workspace_id, source, logger_name, level, message, created_at, data_json FROM runtime_logs"):
            rows = list(self._state.runtime_logs)
            clauses = normalized.split(" WHERE ", 1)[1].split(" ORDER BY ", 1)[0] if " WHERE " in normalized else ""
            params_list = list(arguments)
            limit = _as_int(params_list.pop())
            rows = self._filter_runtime_logs(rows, clauses, params_list)
            rows.sort(key=lambda row: (_as_float(row["created_at"]), _as_int(row["id"])), reverse=True)
            self._result = [
                (
                    row["id"],
                    row["workspace_id"],
                    row["source"],
                    row["logger_name"],
                    row["level"],
                    row["message"],
                    row["created_at"],
                    row["data_json"],
                )
                for row in rows[:limit]
            ]
            return
        if normalized.startswith("SELECT COUNT(*) FROM runtime_logs"):
            rows = list(self._state.runtime_logs)
            clauses = normalized.split(" WHERE ", 1)[1] if " WHERE " in normalized else ""
            rows = self._filter_runtime_logs(rows, clauses, list(arguments))
            self._result = [(len(rows),)]
            return
        if normalized.startswith("SELECT level, source, COUNT(*) FROM runtime_logs"):
            rows = list(self._state.runtime_logs)
            clauses = normalized.split(" WHERE ", 1)[1].split(" GROUP BY ", 1)[0] if " WHERE " in normalized else ""
            rows = self._filter_runtime_logs(rows, clauses, list(arguments))
            grouped: dict[tuple[str, str], int] = {}
            for row in rows:
                key = (str(row["level"]), str(row["source"]))
                grouped[key] = grouped.get(key, 0) + 1
            self._result = [
                (level, source, count)
                for (level, source), count in sorted(grouped.items())
            ]
            return
        if normalized == "DELETE FROM runtime_logs WHERE id = %s":
            log_id = _as_int(arguments[0])
            self._state.runtime_logs = [row for row in self._state.runtime_logs if _as_int(row["id"]) != log_id]
            self._result = []
            return
        if normalized.startswith("INSERT INTO task_execution_attempts"):
            self._state.task_execution_attempts.append(
                {
                    "id": self._state.next_attempt_id,
                    "task_id": str(arguments[0]),
                    "execution_epoch": _as_int(arguments[1]),
                    "workspace_id": arguments[2],
                    "task_name": arguments[3],
                    "request_id": arguments[4],
                    "subprocess_pid": arguments[5],
                    "provider_key": arguments[6],
                    "provider_name": arguments[7],
                    "model_name": arguments[8],
                    "status": str(arguments[9]),
                    "started_at": _as_float(arguments[10]),
                    "last_heartbeat_at": _as_float(arguments[11]),
                    "completed_at": None,
                    "error_text": None,
                    "termination_reason": None,
                }
            )
            self._state.next_attempt_id += 1
            self._result = []
            return
        if normalized == "SELECT * FROM task_execution_attempts WHERE task_id = %s AND execution_epoch = %s LIMIT 1":
            task_id, execution_epoch = str(arguments[0]), _as_int(arguments[1])
            attempt = next((row for row in self._state.task_execution_attempts if row["task_id"] == task_id and _as_int(row["execution_epoch"]) == execution_epoch), None)
            self._result = [] if attempt is None else [self._attempt_row(attempt)]
            return
        if normalized.startswith("UPDATE task_execution_attempts SET workspace_id = %s"):
            task_id, execution_epoch = str(arguments[-2]), _as_int(arguments[-1])
            attempt = next(row for row in self._state.task_execution_attempts if row["task_id"] == task_id and _as_int(row["execution_epoch"]) == execution_epoch)
            keys = [
                "workspace_id", "task_name", "request_id", "subprocess_pid", "provider_key", "provider_name",
                "model_name", "status", "started_at", "last_heartbeat_at", "completed_at", "error_text", "termination_reason",
            ]
            for key, value in zip(keys, arguments[:-2], strict=False):
                attempt[key] = value
            self._result = []
            return
        if normalized.startswith("SELECT * FROM task_execution_attempts"):
            rows = list(self._state.task_execution_attempts)
            clauses = normalized.split(" WHERE ", 1)[1].split(" ORDER BY ", 1)[0] if " WHERE " in normalized else ""
            params_list = list(arguments)
            limit = _as_int(params_list.pop())
            rows = self._filter_attempts(rows, clauses, params_list)
            rows.sort(key=lambda row: (_as_float(row["started_at"]), _as_int(row["id"])), reverse=True)
            self._result = [self._attempt_row(row) for row in rows[:limit]]
            return
        raise AssertionError(f"Unhandled query: {normalized}")

    def fetchone(self) -> tuple[object, ...] | None:
        return None if not self._result else self._result[0]

    def fetchall(self) -> list[tuple[object, ...]]:
        return list(self._result)

    def executemany(self, query: str, rows: Sequence[tuple[object, ...]]) -> None:
        for row in rows:
            self.execute(query, tuple(row))

    def close(self) -> None:
        return None

    def _filter_runtime_logs(self, rows: list[dict[str, object]], clauses: str, params: list[object]) -> list[dict[str, object]]:
        filtered = rows
        if not clauses:
            return filtered
        for clause in clauses.split(" AND "):
            if clause == "workspace_id = %s":
                value = params.pop(0)
                filtered = [row for row in filtered if row["workspace_id"] == value]
            elif clause == "level = %s":
                value = params.pop(0)
                filtered = [row for row in filtered if row["level"] == value]
            elif clause == "logger_name = %s":
                value = params.pop(0)
                filtered = [row for row in filtered if row["logger_name"] == value]
            elif clause == "source = %s":
                value = params.pop(0)
                filtered = [row for row in filtered if row["source"] == value]
            elif clause == "(message LIKE %s OR logger_name LIKE %s OR source LIKE %s)":
                needle = str(params.pop(0)).strip("%")
                params.pop(0)
                params.pop(0)
                filtered = [row for row in filtered if needle in str(row["message"]) or needle in str(row["logger_name"]) or needle in str(row["source"])]
            elif clause == "created_at >= %s":
                value = _as_float(params.pop(0))
                filtered = [row for row in filtered if _as_float(row["created_at"]) >= value]
            elif clause == "created_at <= %s":
                value = _as_float(params.pop(0))
                filtered = [row for row in filtered if _as_float(row["created_at"]) <= value]
        return filtered

    def _filter_attempts(self, rows: list[dict[str, object]], clauses: str, params: list[object]) -> list[dict[str, object]]:
        filtered = rows
        if not clauses:
            return filtered
        for clause in clauses.split(" AND "):
            if clause == "workspace_id = %s":
                value = params.pop(0)
                filtered = [row for row in filtered if row["workspace_id"] == value]
            elif clause == "task_id = %s":
                value = str(params.pop(0))
                filtered = [row for row in filtered if row["task_id"] == value]
            elif clause == "status = %s":
                value = str(params.pop(0))
                filtered = [row for row in filtered if row["status"] == value]
        return filtered

    def _attempt_row(self, row: dict[str, object]) -> tuple[object, ...]:
        return (
            row["id"], row["task_id"], row["execution_epoch"], row["workspace_id"], row["task_name"], row["request_id"],
            row["subprocess_pid"], row["provider_key"], row["provider_name"], row["model_name"], row["status"], row["started_at"],
            row["last_heartbeat_at"], row["completed_at"], row["error_text"], row["termination_reason"],
        )


class FakeConnection:
    def __init__(self, state: FakeOperationalState) -> None:
        self._state = state

    def cursor(self) -> FakeCursor:
        return FakeCursor(self._state)

    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None


class FakeLease:
    def __init__(self, connection: FakeConnection) -> None:
        self._connection = connection

    def __enter__(self) -> FakeConnection:
        return self._connection

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None:
            self._connection.rollback()
        return False

    def close(self) -> None:
        return None


class FakeSessionManager:
    def __init__(self) -> None:
        self.state = FakeOperationalState()

    def __enter__(self) -> FakeSessionManager:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def open_connection(self) -> FakeLease:
        return FakeLease(FakeConnection(self.state))

    def close(self) -> None:
        return None


class InspectablePostgresStructuredLogHandler(PostgresStructuredLogHandler):
    def __init__(self, *, session_manager: FakeSessionManager, workspace_id: str | None, source: str = "runtime") -> None:
        super().__init__(session_manager=session_manager, workspace_id=workspace_id, source=source)
        self.error_records: list[logging.LogRecord] = []

    def handleError(self, record: logging.LogRecord) -> None:
        self.error_records.append(record)


def test_postgres_runtime_log_repository_prunes_by_age_and_count() -> None:
    session_manager = FakeSessionManager()
    repository = PostgresRuntimeLogRepository(
        session_manager,
        workspace_id="workspace-a",
        config=LoggingConfig(max_runtime_logs=2, max_log_age_days=1, retention_check_interval_seconds=0.01),
    )

    now = 1_000_000.0
    repository.write_log(source="daemon", logger_name="mcp_memory.old", level="INFO", message="too old", created_at=now - 172800, data={})
    repository.write_log(source="daemon", logger_name="mcp_memory.keep1", level="INFO", message="keep one", created_at=now, data={})
    repository.write_log(source="daemon", logger_name="mcp_memory.keep2", level="WARNING", message="keep two", created_at=now + 1, data={})
    repository.write_log(source="daemon", logger_name="mcp_memory.drop", level="ERROR", message="drop by count", created_at=now + 2, data={})

    records = repository.list_logs(limit=10)

    assert [record.message for record in records] == ["drop by count", "keep two"]
    repository.close()


def test_postgres_structured_log_handler_writes_runtime_log() -> None:
    session_manager = FakeSessionManager()
    handler = PostgresStructuredLogHandler(session_manager=session_manager, workspace_id="workspace-a", source="stdio")
    record = logging.LogRecord(
        name="mcp_memory.server",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello %s",
        args=("world",),
        exc_info=None,
    )
    handler.emit(record)
    repository = PostgresRuntimeLogRepository(session_manager, workspace_id="workspace-a")
    logs = repository.list_logs(limit=10)
    assert logs[0].message == "hello world"
    assert logs[0].source == "stdio"
    repository.close()
    handler.close()


def test_postgres_structured_log_handler_emit_after_close_is_noop() -> None:
    session_manager = FakeSessionManager()
    handler = InspectablePostgresStructuredLogHandler(
        session_manager=session_manager,
        workspace_id="workspace-a",
        source="stdio",
    )
    record = logging.LogRecord(
        name="mcp_memory.server",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="late log",
        args=(),
        exc_info=None,
    )

    handler.close()
    handler.emit(record)

    assert handler.error_records == []
    assert session_manager.state.runtime_logs == []


def test_postgres_structured_log_handler_preserves_active_error_handling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_manager = FakeSessionManager()
    handler = InspectablePostgresStructuredLogHandler(
        session_manager=session_manager,
        workspace_id="workspace-a",
        source="stdio",
    )
    record = logging.LogRecord(
        name="mcp_memory.server",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="boom",
        args=(),
        exc_info=None,
    )

    def raise_write_log(
        *,
        source: str,
        logger_name: str,
        level: str,
        message: str,
        created_at: float,
        data: dict[str, object],
    ) -> None:
        raise RuntimeError("repository write failed")

    monkeypatch.setattr(handler._repository, "write_log", raise_write_log)

    handler.emit(record)

    assert handler.error_records == [record]
    handler.close()


def test_postgres_runtime_log_repository_preserves_structured_data_payloads() -> None:
    session_manager = FakeSessionManager()
    repository = PostgresRuntimeLogRepository(session_manager, workspace_id="workspace-a")

    repository.write_log(
        source="memory-tool",
        logger_name="mcp_memory.management.service",
        level="WARNING",
        message="Slow management.search_memories operation",
        created_at=100.0,
        data={
            "tool_name": "management.search_memories",
            "duration_ms": 2345.678,
            "query": "project memory search",
            "result_count": 3,
            "extra": {
                "workspace_id": "workspace-a",
                "filters": ["active", "fact"],
            },
        },
    )

    logs = repository.list_logs(source="memory-tool", limit=10)

    assert len(logs) == 1
    assert logs[0].data == {
        "tool_name": "management.search_memories",
        "duration_ms": 2345.678,
        "query": "project memory search",
        "result_count": 3,
        "extra": {
            "workspace_id": "workspace-a",
            "filters": ["active", "fact"],
        },
    }
    repository.close()


def test_postgres_runtime_log_repository_summarize_logs_aggregates_all_matching_rows() -> None:
    session_manager = FakeSessionManager()
    repository = PostgresRuntimeLogRepository(session_manager, workspace_id=None)

    repository.write_log(source="daemon", logger_name="mcp_memory.policy", level="WARNING", message="warning one", created_at=100.0, data={})
    repository.write_log(source="daemon", logger_name="mcp_memory.policy", level="WARNING", message="warning two", created_at=101.0, data={})
    repository.write_log(source="daemon", logger_name="mcp_memory.policy", level="ERROR", message="error one", created_at=102.0, data={})
    repository.write_log(source="stdio", logger_name="mcp_memory.other", level="INFO", message="info one", created_at=103.0, data={})

    summary = repository.summarize_logs(source="daemon", after=100.0)

    assert summary.total == 3
    assert summary.by_level == {"ERROR": 1, "WARNING": 2}
    assert summary.by_source == {"daemon": 3}
    repository.close()


def test_postgres_task_execution_attempt_repository_records_attempt_lifecycle() -> None:
    session_manager = FakeSessionManager()
    repository = PostgresTaskExecutionAttemptRepository(session_manager, workspace_id="workspace-a")

    started = repository.start_attempt(
        task_id="task-1",
        execution_epoch=1,
        task_name="memory-curator",
        request_id="req-1",
        subprocess_pid=1234,
        provider_key="gemini-cli",
        provider_name="Gemini CLI",
        model_name="gemini-2.5-pro",
        started_at=10.0,
    )
    duplicate_start = repository.start_attempt(
        task_id="task-1",
        execution_epoch=1,
        task_name="memory-curator",
        request_id="req-1",
        subprocess_pid=1234,
        provider_key="gemini-cli",
        provider_name="Gemini CLI",
        model_name="gemini-2.5-pro",
        started_at=12.0,
    )
    heartbeated = repository.heartbeat_attempt(task_id="task-1", execution_epoch=1, heartbeat_at=14.0, request_id="req-1", subprocess_pid=5678)
    finished = repository.finish_attempt(task_id="task-1", execution_epoch=1, status="finished", completed_at=16.0, request_id="req-1", subprocess_pid=5678, termination_reason="provider_finished")
    late_heartbeat = repository.heartbeat_attempt(task_id="task-1", execution_epoch=1, heartbeat_at=99.0, subprocess_pid=9999)
    duplicate_finish = repository.finish_attempt(task_id="task-1", execution_epoch=1, status="error", completed_at=100.0, error_text="late duplicate finish")

    assert started.status == "running"
    assert duplicate_start.started_at == 10.0
    assert duplicate_start.last_heartbeat_at == 12.0
    assert heartbeated.last_heartbeat_at == 14.0
    assert heartbeated.subprocess_pid == 5678
    assert finished.status == "finished"
    assert finished.completed_at == 16.0
    assert late_heartbeat.last_heartbeat_at == 16.0
    assert late_heartbeat.subprocess_pid == 5678
    assert duplicate_finish.status == "finished"
    assert duplicate_finish.completed_at == 16.0
    assert duplicate_finish.error_text == "late duplicate finish"


def test_postgres_task_execution_attempt_repository_reopens_terminal_attempt_for_new_provider_call() -> None:
    session_manager = FakeSessionManager()
    repository = PostgresTaskExecutionAttemptRepository(session_manager, workspace_id="workspace-a")

    repository.start_attempt(
        task_id="task-2",
        execution_epoch=7,
        task_name="ingest-system1",
        request_id="req-1",
        subprocess_pid=1111,
        provider_key="gemini-cli",
        provider_name="Gemini CLI",
        model_name="gemini-2.5-pro",
        started_at=10.0,
    )
    repository.finish_attempt(
        task_id="task-2",
        execution_epoch=7,
        status="error",
        completed_at=12.0,
        request_id="req-1",
        subprocess_pid=1111,
        error_text="Provider subprocess 1111 exited unexpectedly",
        termination_reason="provider_exited",
    )

    reopened = repository.start_attempt(
        task_id="task-2",
        execution_epoch=7,
        task_name="ingest-system1",
        request_id="req-2",
        subprocess_pid=2222,
        provider_key="gemini-cli",
        provider_name="Gemini CLI",
        model_name="gemini-2.5-pro",
        started_at=20.0,
    )

    assert reopened.request_id == "req-2"
    assert reopened.subprocess_pid == 2222
    assert reopened.status == "running"
    assert reopened.started_at == 20.0
    assert reopened.last_heartbeat_at == 20.0
    assert reopened.completed_at is None
    assert reopened.error_text is None
    assert reopened.termination_reason is None


def test_postgres_task_execution_attempt_repository_ignores_duplicate_late_start_for_same_provider_call() -> None:
    session_manager = FakeSessionManager()
    repository = PostgresTaskExecutionAttemptRepository(session_manager, workspace_id="workspace-a")

    repository.start_attempt(
        task_id="task-3",
        execution_epoch=8,
        task_name="ingest-system1",
        request_id="req-1",
        subprocess_pid=1111,
        provider_key="gemini-cli",
        provider_name="Gemini CLI",
        model_name="gemini-2.5-pro",
        started_at=10.0,
    )
    finished = repository.finish_attempt(
        task_id="task-3",
        execution_epoch=8,
        status="error",
        completed_at=12.0,
        request_id="req-1",
        subprocess_pid=1111,
        error_text="Provider subprocess 1111 exited unexpectedly",
        termination_reason="provider_exited",
    )

    duplicate_start = repository.start_attempt(
        task_id="task-3",
        execution_epoch=8,
        task_name="ingest-system1",
        request_id="req-1",
        subprocess_pid=1111,
        provider_key="gemini-cli",
        provider_name="Gemini CLI",
        model_name="gemini-2.5-pro",
        started_at=20.0,
    )

    assert duplicate_start.id == finished.id
    assert duplicate_start.request_id == "req-1"
    assert duplicate_start.subprocess_pid == 1111
    assert duplicate_start.status == "error"
    assert duplicate_start.started_at == 10.0
    assert duplicate_start.last_heartbeat_at == 12.0
    assert duplicate_start.completed_at == 12.0
    assert duplicate_start.error_text == "Provider subprocess 1111 exited unexpectedly"
    assert duplicate_start.termination_reason == "provider_exited"


def test_postgres_task_execution_attempt_repository_lists_workspace_scoped_attempts() -> None:
    session_manager = FakeSessionManager()
    workspace_a = PostgresTaskExecutionAttemptRepository(session_manager, workspace_id="workspace-a")
    workspace_b = PostgresTaskExecutionAttemptRepository(session_manager, workspace_id="workspace-b")

    workspace_a.start_attempt(
        task_id="task-a", execution_epoch=1, task_name="memory-curator", request_id="req-a", subprocess_pid=1001,
        provider_key="gemini-cli", provider_name="Gemini CLI", model_name="gemini-2.5-pro", started_at=10.0,
    )
    workspace_a.finish_attempt(task_id="task-a", execution_epoch=1, status="finished", completed_at=11.0)
    workspace_b.start_attempt(
        task_id="task-b", execution_epoch=1, task_name="memory-deduper", request_id="req-b", subprocess_pid=2002,
        provider_key="claude-code", provider_name="Claude Code", model_name="sonnet", started_at=12.0,
    )

    assert [attempt.task_id for attempt in workspace_a.list_attempts()] == ["task-a"]
    assert [attempt.task_id for attempt in workspace_b.list_attempts(status="running")] == ["task-b"]
    assert workspace_a.list_attempts(task_id="task-a")[0].workspace_id == "workspace-a"


def test_postgres_task_execution_attempt_repository_requires_existing_attempt_for_updates() -> None:
    repository = PostgresTaskExecutionAttemptRepository(FakeSessionManager(), workspace_id="workspace-a")

    with pytest.raises(ValueError, match="missing-task@3"):
        repository.heartbeat_attempt(task_id="missing-task", execution_epoch=3, heartbeat_at=10.0)

    with pytest.raises(ValueError, match="missing-task@3"):
        repository.finish_attempt(task_id="missing-task", execution_epoch=3, status="error", completed_at=11.0)
