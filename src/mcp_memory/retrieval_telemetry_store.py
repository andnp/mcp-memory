from __future__ import annotations

import logging
import sqlite3
import time

from mcp_memory.utils.db import DatabaseManager


logger = logging.getLogger(__name__)
_NONCRITICAL_WRITE_TIMEOUT_SECONDS = 0.1


class RetrievalTelemetryRepository:
    def __init__(self, db_manager: DatabaseManager | None, *, workspace_id: str | None) -> None:
        self._db_manager = db_manager
        self._workspace_id = workspace_id

    def record_search(
        self,
        *,
        invocation_id: str,
        caller_kind: str,
        query: str,
        surfaced_memory_ids: list[str],
        created_at: float | None = None,
    ) -> None:
        if self._db_manager is None:
            return

        event_time = time.time() if created_at is None else created_at
        result_count = len(surfaced_memory_ids)
        rows: list[tuple[str, str | None, str, str, str | None, str | None, int | None, int | None, float]] = []
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
                )
            )

        self._best_effort_write(
            lambda conn: conn.executemany(
                "INSERT INTO memory_tool_events (invocation_id, workspace_id, caller_kind, event_kind, memory_id, query_text, result_rank, result_count, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
        )

    def record_read(
        self,
        *,
        invocation_id: str,
        caller_kind: str,
        memory_id: str,
        created_at: float | None = None,
    ) -> None:
        if self._db_manager is None:
            return

        event_time = time.time() if created_at is None else created_at
        self._best_effort_write(
            lambda conn: conn.execute(
                "INSERT INTO memory_tool_events (invocation_id, workspace_id, caller_kind, event_kind, memory_id, query_text, result_rank, result_count, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                ),
            )
        )

    def _best_effort_write(self, operation) -> None:
        assert self._db_manager is not None
        conn = self._db_manager.open_connection(timeout_seconds=_NONCRITICAL_WRITE_TIMEOUT_SECONDS)
        try:
            with conn:
                operation(conn)
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc).lower():
                raise
            logger.debug("Skipping retrieval telemetry write due to SQLite lock contention")
        finally:
            conn.close()