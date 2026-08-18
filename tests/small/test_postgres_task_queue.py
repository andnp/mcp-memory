from __future__ import annotations

from typing import cast

import pytest

from mcp_memory.core.maintenance_schedule import BACKGROUND_CLEANUP_TASK_NAMES
from mcp_memory.core.ports.tasks import TaskRecord
from mcp_memory.storage.postgres_task_queue import PostgresTaskQueue
from mcp_memory.storage.session import DbConnectionLike, SessionManager

pytestmark = pytest.mark.small


class _FakeCursor:
    def __init__(self, manager: _FakeSessionManager) -> None:
        self._manager = manager
        self._result: tuple[object, ...] | None = None
        self.rowcount = 0

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def execute(self, query: str, params: tuple[object, ...] | None = None) -> None:
        normalized = " ".join(query.split())
        self._manager.queries.append(normalized)
        self.rowcount = 0
        if normalized.startswith("SELECT id, task_name FROM tasks"):
            self._result = self._manager.claim_rows.pop(0) if self._manager.claim_rows else None
        elif normalized.startswith("SELECT pg_advisory_xact_lock"):
            self._manager.advisory_lock_calls += 1
            self._result = None
        elif normalized.startswith("UPDATE tasks SET"):
            self.rowcount = 1
            self._result = None
        else:
            raise AssertionError(f"Unhandled query: {normalized}")

    def fetchone(self) -> tuple[object, ...] | None:
        return self._result


class _FakeConnection:
    def __init__(self, manager: _FakeSessionManager) -> None:
        self._manager = manager

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self._manager)

    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None


class _FakeLease:
    def __init__(self, manager: _FakeSessionManager) -> None:
        self._connection = _FakeConnection(manager)

    def __enter__(self) -> _FakeConnection:
        return self._connection

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class _FakeSessionManager:
    def __init__(self, claim_rows: list[tuple[object, ...] | None]) -> None:
        self.claim_rows = claim_rows
        self.queries: list[str] = []
        self.advisory_lock_calls = 0

    def open_connection(self) -> _FakeLease:
        return _FakeLease(self)


def _queue_with_stubbed_get_task(
    session_manager: _FakeSessionManager,
) -> tuple[PostgresTaskQueue, dict[str, TaskRecord]]:
    queue = PostgresTaskQueue(cast(SessionManager[DbConnectionLike], session_manager))
    returned_tasks: dict[str, TaskRecord] = {}
    queue.get_task = lambda task_id: returned_tasks[task_id]
    return queue, returned_tasks


def test_claim_next_skips_advisory_lock_for_ordinary_task() -> None:
    session_manager = _FakeSessionManager([("ordinary-id", "ordinary-task")])
    queue, returned_tasks = _queue_with_stubbed_get_task(session_manager)
    returned_tasks["ordinary-id"] = cast(TaskRecord, object())

    claimed = queue.claim_next(now=1.0)

    assert claimed is returned_tasks["ordinary-id"]
    assert session_manager.advisory_lock_calls == 0
    assert not any("pg_advisory_xact_lock" in query for query in session_manager.queries)


def test_claim_next_rechecks_cleanup_after_acquiring_advisory_lock() -> None:
    cleanup_name = sorted(BACKGROUND_CLEANUP_TASK_NAMES)[0]
    session_manager = _FakeSessionManager(
        [("cleanup-id", cleanup_name), ("cleanup-id", cleanup_name)]
    )
    queue, returned_tasks = _queue_with_stubbed_get_task(session_manager)
    returned_tasks["cleanup-id"] = cast(TaskRecord, object())

    claimed = queue.claim_next(now=1.0)

    assert claimed is returned_tasks["cleanup-id"]
    assert session_manager.advisory_lock_calls == 1
    assert sum("SELECT id, task_name FROM tasks" in query for query in session_manager.queries) == 2
    assert session_manager.queries[1].startswith("SELECT pg_advisory_xact_lock")
