from __future__ import annotations

import threading
from typing import cast

import pytest

from mcp_memory.config import LoggingConfig
from mcp_memory.retrieval_telemetry_store import RetrievalTelemetryRepository
from mcp_memory.storage.postgres_runtime_log_store import PostgresRuntimeLogRepository
from mcp_memory.storage.session import DbConnectionLike, SessionManager


pytestmark = pytest.mark.small


def _as_float(value: object) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"Expected float-compatible value, got {type(value)!r}")


class _BlockingRuntimeLogState:
    def __init__(self) -> None:
        self.runtime_logs: list[dict[str, object]] = []
        self.next_log_id = 1


class _BlockingRuntimeLogCursor:
    def __init__(self, state: _BlockingRuntimeLogState, *, seen_write: threading.Event, release_writes: threading.Event) -> None:
        self._state = state
        self._seen_write = seen_write
        self._release_writes = release_writes
        self._result: list[tuple[object, ...]] = []

    def __enter__(self) -> _BlockingRuntimeLogCursor:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def execute(self, query: str, params: tuple[object, ...] | None = None) -> None:
        normalized = " ".join(query.split())
        arguments = tuple(() if params is None else params)
        if normalized.startswith("INSERT INTO runtime_logs"):
            self._seen_write.set()
            self._release_writes.wait(timeout=1.0)
            self._state.runtime_logs.append(
                {
                    "id": self._state.next_log_id,
                    "workspace_id": arguments[0],
                    "source": arguments[1],
                    "logger_name": arguments[2],
                    "level": arguments[3],
                    "message": arguments[4],
                    "created_at": _as_float(arguments[5]),
                    "data_json": arguments[6],
                }
            )
            self._state.next_log_id += 1
            self._result = []
            return
        raise AssertionError(f"Unhandled query: {normalized}")

    def executemany(self, query: str, rows: list[tuple[object, ...]]) -> None:
        for row in rows:
            self.execute(query, row)

    def fetchone(self) -> tuple[object, ...] | None:
        return None if not self._result else self._result[0]

    def fetchall(self) -> list[tuple[object, ...]]:
        return list(self._result)

    def close(self) -> None:
        return None


class _BlockingRuntimeLogConnection:
    def __init__(self, state: _BlockingRuntimeLogState, *, seen_write: threading.Event, release_writes: threading.Event) -> None:
        self._state = state
        self._seen_write = seen_write
        self._release_writes = release_writes

    def cursor(self) -> _BlockingRuntimeLogCursor:
        return _BlockingRuntimeLogCursor(
            self._state,
            seen_write=self._seen_write,
            release_writes=self._release_writes,
        )

    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None


class _BlockingRuntimeLogLease:
    def __init__(self, connection: _BlockingRuntimeLogConnection) -> None:
        self._connection = connection

    def __enter__(self) -> _BlockingRuntimeLogConnection:
        return self._connection

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def close(self) -> None:
        return None


class _BlockingRuntimeLogSessionManager:
    def __init__(self) -> None:
        self.state = _BlockingRuntimeLogState()
        self.seen_write = threading.Event()
        self.release_writes = threading.Event()

    def __enter__(self) -> _BlockingRuntimeLogSessionManager:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def open_connection(self) -> _BlockingRuntimeLogLease:
        return _BlockingRuntimeLogLease(
            _BlockingRuntimeLogConnection(
                self.state,
                seen_write=self.seen_write,
                release_writes=self.release_writes,
            )
        )

    def close(self) -> None:
        return None


class _TelemetryState:
    def __init__(self) -> None:
        self.rows: list[tuple[object, ...]] = []


class _TelemetryCursor:
    def __init__(self, state: _TelemetryState) -> None:
        self._state = state

    def execute(self, query: str, params: tuple[object, ...] | None = None) -> None:
        if params is not None:
            self._state.rows.append(tuple(params))

    def executemany(self, query: str, rows: list[tuple[object, ...]]) -> None:
        self._state.rows.extend(tuple(row) for row in rows)

    def close(self) -> None:
        return None


class _TelemetryConnection:
    def __init__(self, state: _TelemetryState) -> None:
        self._state = state

    def cursor(self) -> _TelemetryCursor:
        return _TelemetryCursor(self._state)

    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None


class _TelemetryLease:
    def __init__(self, connection: _TelemetryConnection) -> None:
        self._connection = connection

    def __enter__(self) -> _TelemetryConnection:
        return self._connection

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def close(self) -> None:
        return None


class _TelemetrySessionManager:
    def __init__(self) -> None:
        self.state = _TelemetryState()

    def __enter__(self) -> _TelemetrySessionManager:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def open_connection(self) -> _TelemetryLease:
        return _TelemetryLease(_TelemetryConnection(self.state))

    def close(self) -> None:
        return None


def test_postgres_runtime_log_repository_write_is_buffered_until_flush() -> None:
    session_manager = _BlockingRuntimeLogSessionManager()
    repository = PostgresRuntimeLogRepository(
        cast(SessionManager[DbConnectionLike], session_manager),
        workspace_id="workspace-a",
        config=LoggingConfig(retention_check_interval_seconds=10_000.0),
    )

    try:
        repository.write_log(
            source="daemon",
            logger_name="mcp_memory.tests",
            level="INFO",
            message="buffered write",
            created_at=100.0,
            data={},
        )

        assert session_manager.seen_write.wait(timeout=1.0) is True
        assert session_manager.state.runtime_logs == []

        session_manager.release_writes.set()
        repository.flush()

        assert [row["message"] for row in session_manager.state.runtime_logs] == ["buffered write"]
    finally:
        session_manager.release_writes.set()
        repository.close()


def test_postgres_retrieval_telemetry_flushes_buffered_rows_on_close() -> None:
    session_manager = _TelemetrySessionManager()
    repository = RetrievalTelemetryRepository(
        cast(object, session_manager),
        workspace_id="workspace-a",
    )

    repository.record_search(
        invocation_id="search-1",
        caller_kind="external",
        query="buffered telemetry",
        surfaced_memory_ids=["memory-1", "memory-2"],
        duration_ms=12.0,
        created_at=100.0,
    )
    repository.record_read(
        invocation_id="read-1",
        caller_kind="external",
        memory_id="memory-1",
        duration_ms=3.0,
        created_at=101.0,
    )

    repository.close()

    assert len(session_manager.state.rows) == 3
    assert [row[3] for row in session_manager.state.rows] == ["search", "search", "read"]
