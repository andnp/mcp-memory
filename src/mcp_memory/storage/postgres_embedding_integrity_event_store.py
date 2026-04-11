from __future__ import annotations

import json
import time
from typing import cast

from mcp_memory.embedding_integrity_event_store import (
    EMBEDDING_INTEGRITY_EVENT_KIND_BLOCKED_FALLBACK_WRITE,
    EMBEDDING_INTEGRITY_EVENT_KIND_SCAN_SUMMARY,
    EmbeddingIntegrityEventRecord,
    EmbeddingIntegrityEventSummary,
)
from mcp_memory.storage.session import DbConnectionLike, SessionManager


_USE_REPOSITORY_WORKSPACE = object()


class PostgresEmbeddingIntegrityEventRepository:
    def __init__(self, session_manager: SessionManager[DbConnectionLike] | None, *, workspace_id: str | None):
        self._sessions = session_manager
        self._workspace_id = workspace_id

    def record_event(
        self,
        *,
        event_kind: str,
        model_name: str | None = None,
        source_kind: str | None = None,
        source_id: str | None = None,
        scanned_row_count: int | None = None,
        invalid_row_count: int | None = None,
        mixed_dimension_group_count: int | None = None,
        details: dict[str, object] | None = None,
        created_at: float | None = None,
    ) -> None:
        if self._sessions is None:
            return
        event_time = time.time() if created_at is None else created_at
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO embedding_integrity_events (
                        workspace_id,
                        event_kind,
                        model_name,
                        source_kind,
                        source_id,
                        scanned_row_count,
                        invalid_row_count,
                        mixed_dimension_group_count,
                        details_json,
                        created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)
                    """,
                    (
                        None,
                        event_kind,
                        model_name,
                        source_kind,
                        source_id,
                        scanned_row_count,
                        invalid_row_count,
                        mixed_dimension_group_count,
                        json.dumps(details or {}, sort_keys=True),
                        event_time,
                    ),
                )
            connection.commit()

    def summarize_events(
        self,
        *,
        workspace_id: str | None | object = _USE_REPOSITORY_WORKSPACE,
    ) -> EmbeddingIntegrityEventSummary:
        if self._sessions is None:
            return EmbeddingIntegrityEventSummary()
        resolved_workspace_id = None if workspace_id is _USE_REPOSITORY_WORKSPACE else cast(str | None, workspace_id)
        where_clause, params = _workspace_where_clause(resolved_workspace_id)
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT COUNT(*) FROM embedding_integrity_events" + where_clause,
                    tuple(params),
                )
                total_row = cursor.fetchone()
                cursor.execute(
                    "SELECT event_kind, COUNT(*) FROM embedding_integrity_events"
                    + where_clause
                    + " GROUP BY event_kind",
                    tuple(params),
                )
                grouped_rows = cursor.fetchall()
            connection.commit()
        by_kind = {
            str(row[0]): _coerce_int(row[1])
            for row in grouped_rows
        }
        return EmbeddingIntegrityEventSummary(
            total=0 if total_row is None else _coerce_int(total_row[0]),
            by_kind=by_kind,
            last_scan=self._fetch_latest_event(
                event_kind=EMBEDDING_INTEGRITY_EVENT_KIND_SCAN_SUMMARY,
                workspace_id=resolved_workspace_id,
            ),
            last_blocked_fallback_write=self._fetch_latest_event(
                event_kind=EMBEDDING_INTEGRITY_EVENT_KIND_BLOCKED_FALLBACK_WRITE,
                workspace_id=resolved_workspace_id,
            ),
        )

    def _fetch_latest_event(
        self,
        *,
        event_kind: str,
        workspace_id: str | None,
    ) -> EmbeddingIntegrityEventRecord | None:
        assert self._sessions is not None
        clauses = ["event_kind = %s"]
        params: list[object] = [event_kind]
        if workspace_id is not None:
            clauses.append("workspace_id = %s")
            params.append(workspace_id)
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id, workspace_id, event_kind, model_name, source_kind, source_id,
                           scanned_row_count, invalid_row_count, mixed_dimension_group_count,
                           details_json, created_at
                    FROM embedding_integrity_events
                    WHERE """
                    + " AND ".join(clauses)
                    + " ORDER BY created_at DESC, id DESC LIMIT 1",
                    tuple(params),
                )
                row = cursor.fetchone()
            connection.commit()
        return None if row is None else _row_to_event(row)


def _workspace_where_clause(workspace_id: str | None) -> tuple[str, list[object]]:
    if workspace_id is None:
        return "", []
    return " WHERE workspace_id = %s", [workspace_id]


def _row_to_event(row: tuple[object, ...]) -> EmbeddingIntegrityEventRecord:
    raw_details = row[9]
    if isinstance(raw_details, str):
        details = json.loads(raw_details) if raw_details.strip() else {}
    elif isinstance(raw_details, dict):
        details = raw_details
    else:
        details = {}
    return EmbeddingIntegrityEventRecord(
        id=_coerce_int(row[0]),
        workspace_id=None if row[1] is None else str(row[1]),
        event_kind=str(row[2]),
        model_name=None if row[3] is None else str(row[3]),
        source_kind=None if row[4] is None else str(row[4]),
        source_id=None if row[5] is None else str(row[5]),
        scanned_row_count=None if row[6] is None else _coerce_int(row[6]),
        invalid_row_count=None if row[7] is None else _coerce_int(row[7]),
        mixed_dimension_group_count=None if row[8] is None else _coerce_int(row[8]),
        created_at=_coerce_float(row[10]),
        details={str(key): value for key, value in details.items()} if isinstance(details, dict) else {},
    )


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
