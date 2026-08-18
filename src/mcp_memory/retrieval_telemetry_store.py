from __future__ import annotations

import json
import logging
import sqlite3
import time
from collections.abc import Mapping
from typing import Any

from mcp_memory.storage.buffered_writer import BufferedWriter

logger = logging.getLogger(__name__)
_NONCRITICAL_WRITE_TIMEOUT_SECONDS = 0.1
_STORAGE_BACKEND_UNSET = object()


type _TelemetryRow = tuple[object, ...]


class RetrievalTelemetryRepository:
    def __init__(
        self,
        db_manager: Any = None,
        *,
        workspace_id: str | None,
        storage_backend: str | None | object = _STORAGE_BACKEND_UNSET,
    ) -> None:
        self._db_manager = db_manager
        self._workspace_id = workspace_id
        self._storage_backend = storage_backend
        self._writer = (
            None
            if not self._supports_buffered_writes()
            else BufferedWriter[tuple[object, ...]](
                self._flush_rows,
                name="retrieval-telemetry",
                low_watermark=1,
                high_watermark=256,
                flush_interval_seconds=0.01,
            )
        )

    def record_search(
        self,
        *,
        invocation_id: str,
        caller_kind: str,
        query: str,
        surfaced_memory_ids: list[str],
        graph_provenance: Mapping[str, object] | None = None,
        duration_ms: float | None = None,
        created_at: float | None = None,
    ) -> None:
        if self._db_manager is None:
            return

        event_time = time.time() if created_at is None else created_at
        result_count = len(surfaced_memory_ids)
        rows: list[_TelemetryRow] = []
        if surfaced_memory_ids:
            rows.extend(
                (
                    invocation_id,
                    self._workspace_id,
                    caller_kind,
                    "search",
                    memory_id,
                    query,
                    index,
                    result_count,
                    event_time,
                    duration_ms,
                    _provenance_json(graph_provenance, memory_id),
                )
                for index, memory_id in enumerate(surfaced_memory_ids, start=1)
            )
        else:
            rows.append(
                (
                    invocation_id,
                    self._workspace_id,
                    caller_kind,
                    "search",
                    None,
                    query,
                    None,
                    0,
                    event_time,
                    duration_ms,
                    None,
                )
            )

        self._write_rows(rows)

    def record_read(
        self,
        *,
        invocation_id: str,
        caller_kind: str,
        memory_id: str,
        duration_ms: float | None = None,
        created_at: float | None = None,
    ) -> None:
        if self._db_manager is None:
            return

        event_time = time.time() if created_at is None else created_at
        self._write_rows(
            [
                (
                    invocation_id,
                    self._workspace_id,
                    caller_kind,
                    "read",
                    memory_id,
                    None,
                    None,
                    None,
                    event_time,
                    duration_ms,
                    None,
                )
            ]
        )

    def flush(self) -> None:
        if self._writer is not None:
            self._writer.flush()

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()

    def engagement_stats(self, memory_ids: list[str]) -> dict[str, dict[str, int]]:
        if self._db_manager is None or not memory_ids:
            return {}
        placeholders = ",".join("?" for _ in memory_ids)
        workspace_clause = "workspace_id IS NULL" if self._workspace_id is None else "workspace_id = ?"
        query = (
            "SELECT memory_id, event_kind, created_at FROM memory_tool_events "
            f"WHERE {workspace_clause} AND caller_kind != 'internal' AND memory_id IN ({placeholders}) "
        )
        params: tuple[object, ...] = (
            *((self._workspace_id,) if self._workspace_id is not None else ()),
            *memory_ids,
        )
        try:
            if self._uses_postgres_sessions():
                query = _postgres_placeholder_query(query)
                with self._db_manager.open_connection() as connection:
                    with connection.cursor() as cursor:
                        cursor.execute(query, params)
                        rows = cursor.fetchall()
            else:
                connection = self._db_manager.open_connection(timeout_seconds=_NONCRITICAL_WRITE_TIMEOUT_SECONDS)
                try:
                    rows = connection.execute(query, params).fetchall()
                finally:
                    connection.close()
        except Exception:
            return {}
        stats = {
            memory_id: {"search_count": 0, "read_count": 0, "converted_search_count": 0}
            for memory_id in memory_ids
        }
        search_times: dict[str, list[float]] = {memory_id: [] for memory_id in memory_ids}
        read_times: dict[str, list[float]] = {memory_id: [] for memory_id in memory_ids}
        for memory_id, event_kind, created_at in rows:
            memory_key = str(memory_id)
            if memory_key not in stats:
                continue
            if str(event_kind) == "search":
                search_times[memory_key].append(float(created_at))
            elif str(event_kind) == "read":
                read_times[memory_key].append(float(created_at))
        for memory_id in memory_ids:
            stats[memory_id]["search_count"] = len(search_times[memory_id])
            stats[memory_id]["read_count"] = len(read_times[memory_id])
            stats[memory_id]["converted_search_count"] = sum(
                any(read_time >= search_time for read_time in read_times[memory_id])
                for search_time in search_times[memory_id]
            )
        return stats

    def _write_rows(self, rows: list[_TelemetryRow]) -> None:
        if self._writer is not None:
            self._writer.write_many(rows)
            return
        self._best_effort_sqlite_write(rows)

    def _supports_buffered_writes(self) -> bool:
        return bool(self._db_manager is not None and hasattr(self._db_manager, "open_connection"))

    def _uses_postgres_sessions(self) -> bool:
        if self._storage_backend is not _STORAGE_BACKEND_UNSET:
            return self._storage_backend == "postgres"
        return bool(
            self._db_manager is not None
            and hasattr(self._db_manager, "open_connection")
            and not hasattr(self._db_manager, "get_connection")
        )

    def _flush_rows(self, rows: list[_TelemetryRow]) -> None:
        if self._uses_postgres_sessions():
            self._flush_postgres_rows(rows)
            return
        self._best_effort_sqlite_write(rows)

    def _flush_postgres_rows(self, rows: list[_TelemetryRow]) -> None:
        assert self._db_manager is not None
        try:
            with self._db_manager.open_connection() as connection:
                cursor = connection.cursor()
                try:
                    cursor.executemany(
                        _postgres_placeholder_query(
                            "INSERT INTO memory_tool_events (invocation_id, workspace_id, caller_kind, event_kind, memory_id, query_text, result_rank, result_count, created_at, duration_ms, graph_provenance_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                        ),
                        rows,
                    )
                finally:
                    close = getattr(cursor, "close", None)
                    if callable(close):
                        close()
                connection.commit()
        except Exception as exc:  # pragma: no cover - best-effort Postgres telemetry path
            logger.debug("Skipping retrieval telemetry write due to noncritical Postgres failure: %s", exc)

    def _best_effort_sqlite_write(self, rows: list[_TelemetryRow]) -> None:
        assert self._db_manager is not None
        conn = self._db_manager.open_connection(timeout_seconds=_NONCRITICAL_WRITE_TIMEOUT_SECONDS)
        try:
            with conn:
                conn.executemany(
                    "INSERT INTO memory_tool_events (invocation_id, workspace_id, caller_kind, event_kind, memory_id, query_text, result_rank, result_count, created_at, duration_ms, graph_provenance_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    rows,
                )
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc).lower():
                raise
            logger.debug("Skipping retrieval telemetry write due to SQLite lock contention")
        finally:
            conn.close()


def _postgres_placeholder_query(query: str) -> str:
    return query.replace("?", "%s")


def _provenance_json(
    graph_provenance: Mapping[str, object] | None,
    memory_id: str,
) -> str | None:
    if not graph_provenance:
        return None
    value = graph_provenance.get(memory_id)
    if not isinstance(value, Mapping):
        return None
    return json.dumps(dict(value), ensure_ascii=False, sort_keys=True)
