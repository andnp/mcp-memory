from __future__ import annotations

import json
from collections.abc import Sequence
from typing import cast

import pytest

from mcp_memory.embedding_integrity_event_store import (
    EMBEDDING_INTEGRITY_EVENT_KIND_BLOCKED_FALLBACK_WRITE,
    EMBEDDING_INTEGRITY_EVENT_KIND_SCAN_SUMMARY,
    EmbeddingIntegrityEventRepository,
)
from mcp_memory.storage.postgres_embedding_integrity_event_store import (
    PostgresEmbeddingIntegrityEventRepository,
)
from mcp_memory.storage.session import DbConnectionLike, SessionManager

pytestmark = pytest.mark.small


def _as_int(value: object) -> int:
    assert isinstance(value, bool | int | float | str)
    return int(value)


def _as_float(value: object) -> float:
    assert isinstance(value, bool | int | float | str)
    return float(value)


class _EmbeddingIntegrityEventState:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []
        self.next_id = 1


class _EmbeddingIntegrityEventCursor:
    def __init__(self, state: _EmbeddingIntegrityEventState) -> None:
        self._state = state
        self._result: list[tuple[object, ...]] = []

    def __enter__(self) -> _EmbeddingIntegrityEventCursor:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def execute(self, query: str, params: tuple[object, ...] | None = None) -> None:
        normalized = " ".join(query.split())
        arguments = tuple(() if params is None else params)
        if normalized.startswith("INSERT INTO embedding_integrity_events"):
            self._state.events.append(
                {
                    "id": self._state.next_id,
                    "workspace_id": arguments[0],
                    "event_kind": arguments[1],
                    "model_name": arguments[2],
                    "source_kind": arguments[3],
                    "source_id": arguments[4],
                    "scanned_row_count": arguments[5],
                    "invalid_row_count": arguments[6],
                    "mixed_dimension_group_count": arguments[7],
                    "details_json": arguments[8],
                    "created_at": arguments[9],
                }
            )
            self._state.next_id += 1
            self._result = []
            return
        if normalized.startswith("SELECT COUNT(*) FROM embedding_integrity_events"):
            rows = self._filter_rows(normalized, list(arguments))
            self._result = [(len(rows),)]
            return
        if normalized.startswith("SELECT event_kind, COUNT(*) FROM embedding_integrity_events"):
            rows = self._filter_rows(normalized, list(arguments))
            counts: dict[str, int] = {}
            for row in rows:
                event_kind = str(row["event_kind"])
                counts[event_kind] = counts.get(event_kind, 0) + 1
            self._result = [(event_kind, count) for event_kind, count in sorted(counts.items())]
            return
        if normalized.startswith("SELECT id, workspace_id, event_kind, model_name, source_kind, source_id, scanned_row_count, invalid_row_count, mixed_dimension_group_count, details_json, created_at FROM embedding_integrity_events WHERE"):
            rows = list(self._state.events)
            event_kind = str(arguments[0])
            rows = [row for row in rows if str(row["event_kind"]) == event_kind]
            if "AND workspace_id = %s" in normalized:
                workspace_id = arguments[1]
                rows = [row for row in rows if row["workspace_id"] == workspace_id]
            rows.sort(key=lambda row: (_as_float(row["created_at"]), _as_int(row["id"])), reverse=True)
            self._result = [] if not rows else [self._event_row(rows[0])]
            return
        raise AssertionError(f"Unhandled query: {normalized}")

    def fetchone(self) -> tuple[object, ...] | None:
        return None if not self._result else self._result[0]

    def fetchall(self) -> list[tuple[object, ...]]:
        return list(self._result)

    def executemany(self, query: str, rows: Sequence[tuple[object, ...]]) -> None:
        for row in rows:
            self.execute(query, tuple(row))

    def _filter_rows(self, query: str, params: list[object]) -> list[dict[str, object]]:
        rows = list(self._state.events)
        if " WHERE workspace_id = %s" not in query:
            return rows
        workspace_id = params.pop(0)
        return [row for row in rows if row["workspace_id"] == workspace_id]

    def _event_row(self, row: dict[str, object]) -> tuple[object, ...]:
        return (
            row["id"],
            row["workspace_id"],
            row["event_kind"],
            row["model_name"],
            row["source_kind"],
            row["source_id"],
            row["scanned_row_count"],
            row["invalid_row_count"],
            row["mixed_dimension_group_count"],
            row["details_json"],
            row["created_at"],
        )


class _EmbeddingIntegrityEventConnection:
    def __init__(self, state: _EmbeddingIntegrityEventState) -> None:
        self._state = state

    def cursor(self) -> _EmbeddingIntegrityEventCursor:
        return _EmbeddingIntegrityEventCursor(self._state)

    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None


class _EmbeddingIntegrityEventLease:
    def __init__(self, connection: _EmbeddingIntegrityEventConnection) -> None:
        self._connection = connection

    def __enter__(self) -> _EmbeddingIntegrityEventConnection:
        return self._connection

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None:
            self._connection.rollback()
        return False

    def close(self) -> None:
        return None


class _EmbeddingIntegrityEventSessionManager:
    def __init__(self) -> None:
        self.state = _EmbeddingIntegrityEventState()

    def __enter__(self) -> _EmbeddingIntegrityEventSessionManager:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def open_connection(self) -> _EmbeddingIntegrityEventLease:
        return _EmbeddingIntegrityEventLease(
            _EmbeddingIntegrityEventConnection(self.state)
        )

    def close(self) -> None:
        return None


def test_embedding_integrity_event_repository_records_and_summarizes_events(db_manager) -> None:
    repository = EmbeddingIntegrityEventRepository(db_manager, workspace_id="workspace-a")

    repository.record_event(
        event_kind=EMBEDDING_INTEGRITY_EVENT_KIND_SCAN_SUMMARY,
        model_name="mini-embed",
        scanned_row_count=4,
        invalid_row_count=2,
        mixed_dimension_group_count=1,
        details={"expected_dimension": 2},
        created_at=100.0,
    )
    repository.record_event(
        event_kind=EMBEDDING_INTEGRITY_EVENT_KIND_BLOCKED_FALLBACK_WRITE,
        model_name="hash:sentence-transformers/all-MiniLM-L6-v2",
        source_kind="memory",
        source_id="memory-1",
        details={"reason": "fallback_embedding_persistence_blocked"},
        created_at=101.0,
    )

    summary = repository.summarize_events()
    global_summary = repository.summarize_events(workspace_id=None)
    workspace_summary = repository.summarize_events(workspace_id="workspace-a")

    assert summary.total == 2
    assert summary.by_kind == {
        EMBEDDING_INTEGRITY_EVENT_KIND_BLOCKED_FALLBACK_WRITE: 1,
        EMBEDDING_INTEGRITY_EVENT_KIND_SCAN_SUMMARY: 1,
    }
    assert summary.last_scan is not None
    assert summary.last_scan.model_name == "mini-embed"
    assert summary.last_scan.scanned_row_count == 4
    assert summary.last_scan.invalid_row_count == 2
    assert summary.last_scan.mixed_dimension_group_count == 1
    assert summary.last_scan.details == {"expected_dimension": 2}
    assert summary.last_blocked_fallback_write is not None
    assert summary.last_blocked_fallback_write.model_name == "hash:sentence-transformers/all-MiniLM-L6-v2"
    assert summary.last_blocked_fallback_write.source_kind == "memory"
    assert summary.last_blocked_fallback_write.source_id == "memory-1"
    assert summary.last_scan.workspace_id is None
    assert summary.last_blocked_fallback_write.workspace_id is None
    assert global_summary.total == 2
    assert workspace_summary.total == 0


def test_postgres_embedding_integrity_event_repository_records_and_summarizes_events() -> None:
    session_manager = _EmbeddingIntegrityEventSessionManager()
    repository = PostgresEmbeddingIntegrityEventRepository(
        cast(SessionManager[DbConnectionLike], session_manager),
        workspace_id="workspace-a",
    )

    repository.record_event(
        event_kind=EMBEDDING_INTEGRITY_EVENT_KIND_SCAN_SUMMARY,
        model_name="mini-embed",
        scanned_row_count=4,
        invalid_row_count=2,
        mixed_dimension_group_count=1,
        details={"expected_dimension": 2},
        created_at=100.0,
    )
    repository.record_event(
        event_kind=EMBEDDING_INTEGRITY_EVENT_KIND_BLOCKED_FALLBACK_WRITE,
        model_name="hash:sentence-transformers/all-MiniLM-L6-v2",
        source_kind="memory",
        source_id="memory-1",
        details={"reason": "fallback_embedding_persistence_blocked"},
        created_at=101.0,
    )

    summary = repository.summarize_events()
    global_summary = repository.summarize_events(workspace_id=None)
    workspace_summary = repository.summarize_events(workspace_id="workspace-a")

    assert summary.total == 2
    assert summary.by_kind == {
        EMBEDDING_INTEGRITY_EVENT_KIND_BLOCKED_FALLBACK_WRITE: 1,
        EMBEDDING_INTEGRITY_EVENT_KIND_SCAN_SUMMARY: 1,
    }
    assert summary.last_scan is not None
    assert summary.last_scan.model_name == "mini-embed"
    assert summary.last_scan.scanned_row_count == 4
    assert summary.last_scan.invalid_row_count == 2
    assert summary.last_scan.mixed_dimension_group_count == 1
    assert summary.last_scan.details == {"expected_dimension": 2}
    assert summary.last_blocked_fallback_write is not None
    assert summary.last_blocked_fallback_write.model_name == "hash:sentence-transformers/all-MiniLM-L6-v2"
    assert summary.last_blocked_fallback_write.source_kind == "memory"
    assert summary.last_blocked_fallback_write.source_id == "memory-1"
    assert summary.last_scan.workspace_id is None
    assert summary.last_blocked_fallback_write.workspace_id is None
    assert global_summary.total == 2
    assert workspace_summary.total == 0

    stored_details = session_manager.state.events[0]["details_json"]
    assert isinstance(stored_details, str)
    assert json.loads(stored_details) == {"expected_dimension": 2}
