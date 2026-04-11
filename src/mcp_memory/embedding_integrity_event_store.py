from __future__ import annotations

from dataclasses import dataclass, field
import json
import time
from typing import cast

from mcp_memory.utils.db import DatabaseManager


EMBEDDING_INTEGRITY_EVENT_KIND_SCAN_SUMMARY = "integrity_scan_summary"
EMBEDDING_INTEGRITY_EVENT_KIND_BLOCKED_FALLBACK_WRITE = "blocked_fallback_write"

_USE_REPOSITORY_WORKSPACE = object()


@dataclass(frozen=True)
class EmbeddingIntegrityEventRecord:
    id: int
    workspace_id: str | None
    event_kind: str
    model_name: str | None
    source_kind: str | None
    source_id: str | None
    scanned_row_count: int | None
    invalid_row_count: int | None
    mixed_dimension_group_count: int | None
    created_at: float
    details: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class EmbeddingIntegrityEventSummary:
    total: int = 0
    by_kind: dict[str, int] = field(default_factory=dict)
    last_scan: EmbeddingIntegrityEventRecord | None = None
    last_blocked_fallback_write: EmbeddingIntegrityEventRecord | None = None


class EmbeddingIntegrityEventRepository:
    def __init__(self, db_manager: DatabaseManager | None, *, workspace_id: str | None):
        self._db_manager = db_manager
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
        if self._db_manager is None:
            return
        event_time = time.time() if created_at is None else created_at
        self._db_manager.get_connection().execute(
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
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        self._db_manager.get_connection().commit()

    def summarize_events(
        self,
        *,
        workspace_id: str | None | object = _USE_REPOSITORY_WORKSPACE,
    ) -> EmbeddingIntegrityEventSummary:
        if self._db_manager is None:
            return EmbeddingIntegrityEventSummary()
        resolved_workspace_id = None if workspace_id is _USE_REPOSITORY_WORKSPACE else cast(str | None, workspace_id)
        where_clause, params = _workspace_where_clause(resolved_workspace_id)
        conn = self._db_manager.get_connection()
        total_row = conn.execute(
            "SELECT COUNT(*) AS count FROM embedding_integrity_events" + where_clause,
            params,
        ).fetchone()
        grouped_rows = conn.execute(
            "SELECT event_kind, COUNT(*) AS count FROM embedding_integrity_events"
            + where_clause
            + " GROUP BY event_kind",
            params,
        ).fetchall()
        by_kind = {
            str(row["event_kind"]): int(row["count"])
            for row in grouped_rows
        }
        return EmbeddingIntegrityEventSummary(
            total=0 if total_row is None else int(total_row["count"]),
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
        assert self._db_manager is not None
        clauses = ["event_kind = ?"]
        params: list[object] = [event_kind]
        if workspace_id is not None:
            clauses.append("workspace_id = ?")
            params.append(workspace_id)
        row = self._db_manager.get_connection().execute(
            """
            SELECT id, workspace_id, event_kind, model_name, source_kind, source_id,
                   scanned_row_count, invalid_row_count, mixed_dimension_group_count,
                   details_json, created_at
            FROM embedding_integrity_events
            WHERE """
            + " AND ".join(clauses)
            + " ORDER BY created_at DESC, id DESC LIMIT 1",
            params,
        ).fetchone()
        return None if row is None else _row_to_event(row)


def _decode_details(raw_result: object) -> dict[str, object]:
    if isinstance(raw_result, dict):
        return {
            str(key): value
            for key, value in raw_result.items()
        }
    if not isinstance(raw_result, str) or not raw_result.strip():
        return {}
    try:
        decoded = json.loads(raw_result)
    except ValueError:
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _workspace_where_clause(workspace_id: str | None) -> tuple[str, list[object]]:
    if workspace_id is None:
        return "", []
    return " WHERE workspace_id = ?", [workspace_id]


def _row_to_event(row) -> EmbeddingIntegrityEventRecord:
    return EmbeddingIntegrityEventRecord(
        id=int(row["id"]),
        workspace_id=None if row["workspace_id"] is None else str(row["workspace_id"]),
        event_kind=str(row["event_kind"]),
        model_name=None if row["model_name"] is None else str(row["model_name"]),
        source_kind=None if row["source_kind"] is None else str(row["source_kind"]),
        source_id=None if row["source_id"] is None else str(row["source_id"]),
        scanned_row_count=None if row["scanned_row_count"] is None else int(row["scanned_row_count"]),
        invalid_row_count=None if row["invalid_row_count"] is None else int(row["invalid_row_count"]),
        mixed_dimension_group_count=None if row["mixed_dimension_group_count"] is None else int(row["mixed_dimension_group_count"]),
        created_at=float(row["created_at"]),
        details=_decode_details(row["details_json"]),
    )
