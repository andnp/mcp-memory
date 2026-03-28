from __future__ import annotations

import logging
import sqlite3
import time
from typing import Any

from mcp_memory.storage.buffered_writer import BufferedWriter


logger = logging.getLogger(__name__)
_NONCRITICAL_WRITE_TIMEOUT_SECONDS = 0.1


class RetrievalTelemetryRepository:
    def __init__(self, db_manager: Any = None, *, workspace_id: str | None) -> None:
        self._db_manager = db_manager
        self._workspace_id = workspace_id
        self._postgres_writer = (
            None
            if not self._uses_postgres_sessions()
            else BufferedWriter[tuple[object, ...]](
                self._flush_postgres_rows,
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
        duration_ms: float | None = None,
        created_at: float | None = None,
    ) -> None:
        if self._db_manager is None:
            return

        event_time = time.time() if created_at is None else created_at
        result_count = len(surfaced_memory_ids)
        rows: list[tuple[str, str | None, str, str, str | None, str | None, int | None, int | None, float, float | None]] = []
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
                )
            ]
        )

    def flush(self) -> None:
        if self._postgres_writer is not None:
            self._postgres_writer.flush()

    def close(self) -> None:
        if self._postgres_writer is not None:
            self._postgres_writer.close()

    def _write_rows(self, rows: list[tuple[object, ...]]) -> None:
        if self._postgres_writer is not None:
            self._postgres_writer.write_many(rows)
            return
        self._best_effort_sqlite_write(rows)

    def _uses_postgres_sessions(self) -> bool:
        return bool(
            self._db_manager is not None
            and hasattr(self._db_manager, "open_connection")
            and not hasattr(self._db_manager, "get_connection")
        )

    def _flush_postgres_rows(self, rows: list[tuple[object, ...]]) -> None:
        assert self._db_manager is not None
        try:
            with self._db_manager.open_connection() as connection:
                cursor = connection.cursor()
                try:
                    cursor.executemany(
                        _postgres_placeholder_query(
                            "INSERT INTO memory_tool_events (invocation_id, workspace_id, caller_kind, event_kind, memory_id, query_text, result_rank, result_count, created_at, duration_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
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

    def _best_effort_sqlite_write(self, rows: list[tuple[object, ...]]) -> None:
        assert self._db_manager is not None
        conn = self._db_manager.open_connection(timeout_seconds=_NONCRITICAL_WRITE_TIMEOUT_SECONDS)
        try:
            with conn:
                conn.executemany(
                    "INSERT INTO memory_tool_events (invocation_id, workspace_id, caller_kind, event_kind, memory_id, query_text, result_rank, result_count, created_at, duration_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
