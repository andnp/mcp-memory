from __future__ import annotations

from dataclasses import dataclass
import logging
import threading
import time
from typing import cast

from mcp_memory.config import LoggingConfig, PostgresStorageConfig
from mcp_memory.operational_store_rows import RuntimeLogRecord, RuntimeLogSummary, encode_json_object
from mcp_memory.runtime_log_store import _ALL_WORKSPACES, _AllWorkspacesSentinel, _build_log_filters
from mcp_memory.storage.buffered_writer import BufferedWriter
from mcp_memory.storage.postgres_connection import PostgresConnectionManager
from mcp_memory.storage.session import DbConnectionLike, SessionManager


def _normalize_workspace_id(workspace_id: str | None | _AllWorkspacesSentinel) -> str | None:
    if workspace_id is _ALL_WORKSPACES:
        return None
    if workspace_id is None or isinstance(workspace_id, str):
        return workspace_id
    raise TypeError(f"Expected workspace id, got {type(workspace_id)!r}")


@dataclass(frozen=True)
class _PendingRuntimeLog:
    workspace_id: str | None
    source: str
    logger_name: str
    level: str
    message: str
    created_at: float
    data_json: str


class PostgresRuntimeLogRepository:
    def __init__(
        self,
        session_manager: SessionManager[DbConnectionLike] | None,
        *,
        workspace_id: str | None,
        config: LoggingConfig | None = None,
    ) -> None:
        self._sessions = session_manager
        self._workspace_id = workspace_id
        self._config = config if config is not None else LoggingConfig()
        self._last_retention_at = 0.0
        self._closed = False
        self._state_lock = threading.Lock()
        self._writer = BufferedWriter[_PendingRuntimeLog](
            self._flush_log_batch,
            name="postgres-runtime-logs",
            low_watermark=1,
            high_watermark=128,
            flush_interval_seconds=0.01,
        )

    @property
    def retention_policy(self) -> LoggingConfig:
        return self._config

    def write_log(
        self,
        *,
        source: str,
        logger_name: str,
        level: str,
        message: str,
        created_at: float,
        data: dict[str, object],
    ) -> None:
        if self._sessions is None:
            return
        with self._state_lock:
            if self._closed:
                return
            self._writer.write(
                _PendingRuntimeLog(
                    workspace_id=self._workspace_id,
                    source=source,
                    logger_name=logger_name,
                    level=level,
                    message=message,
                    created_at=created_at,
                    data_json=encode_json_object(data),
                )
            )
        self.apply_retention_policy(now=created_at)

    def flush(self) -> None:
        with self._state_lock:
            if self._closed:
                return
        self._writer.flush()

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
        self._writer.close()

    def list_logs(
        self,
        *,
        workspace_id: str | None | _AllWorkspacesSentinel = _ALL_WORKSPACES,
        level: str | None = None,
        logger_name: str | None = None,
        source: str | None = None,
        query: str | None = None,
        after: float | None = None,
        before: float | None = None,
        limit: int = 50,
    ) -> list[RuntimeLogRecord]:
        if self._sessions is None:
            return []
        self.flush()
        resolved_workspace_id = _normalize_workspace_id(workspace_id)
        where_clause, params = _build_log_filters(
            workspace_id=resolved_workspace_id,
            level=level,
            logger_name=logger_name,
            source=source,
            query=query,
            after=after,
            before=before,
        )
        query_sql = (
            "SELECT id, workspace_id, source, logger_name, level, message, created_at, data_json FROM runtime_logs"
            + where_clause.replace("?", "%s")
            + " ORDER BY created_at DESC, id DESC LIMIT %s"
        )
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query_sql, tuple([*params, limit]))
                return [RuntimeLogRecord.from_postgres_row(row) for row in cursor.fetchall()]

    def summarize_logs(
        self,
        *,
        workspace_id: str | None | _AllWorkspacesSentinel = _ALL_WORKSPACES,
        level: str | None = None,
        logger_name: str | None = None,
        source: str | None = None,
        query: str | None = None,
        after: float | None = None,
        before: float | None = None,
    ) -> RuntimeLogSummary:
        if self._sessions is None:
            return RuntimeLogSummary(total=0)
        self.flush()
        resolved_workspace_id = _normalize_workspace_id(workspace_id)
        where_clause, params = _build_log_filters(
            workspace_id=resolved_workspace_id,
            level=level,
            logger_name=logger_name,
            source=source,
            query=query,
            after=after,
            before=before,
        )
        total_sql = "SELECT COUNT(*) FROM runtime_logs" + where_clause.replace("?", "%s")
        grouped_sql = (
            "SELECT level, source, COUNT(*) FROM runtime_logs"
            + where_clause.replace("?", "%s")
            + " GROUP BY level, source"
        )

        by_level: dict[str, int] = {}
        by_source: dict[str, int] = {}
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(total_sql, tuple(params))
                total_row = cursor.fetchone()
                total = 0 if total_row is None else int(cast(bool | int | float | str, total_row[0]))

                cursor.execute(grouped_sql, tuple(params))
                for row in cursor.fetchall():
                    level_name = str(row[0])
                    source_name = str(row[1])
                    count = int(cast(bool | int | float | str, row[2]))
                    by_level[level_name] = by_level.get(level_name, 0) + count
                    by_source[source_name] = by_source.get(source_name, 0) + count

        return RuntimeLogSummary(total=total, by_level=by_level, by_source=by_source)

    def apply_retention_policy(self, *, now: float | None = None) -> int:
        current_time = time.time() if now is None else now
        if current_time - self._last_retention_at < self._config.retention_check_interval_seconds:
            return 0
        self._last_retention_at = current_time
        return self.prune_logs(
            max_runtime_logs=self._config.max_runtime_logs,
            max_age_days=self._config.max_log_age_days,
            now=current_time,
        )

    def prune_logs(
        self,
        *,
        workspace_id: str | None | _AllWorkspacesSentinel = _ALL_WORKSPACES,
        max_runtime_logs: int | None = None,
        max_age_days: int | None = None,
        now: float | None = None,
    ) -> int:
        if self._sessions is None:
            return 0
        self.flush()
        resolved_workspace_id = _normalize_workspace_id(workspace_id)
        effective_max_runtime_logs = self._config.max_runtime_logs if max_runtime_logs is None else max_runtime_logs
        effective_max_age_days = self._config.max_log_age_days if max_age_days is None else max_age_days
        records = self.list_logs(workspace_id=resolved_workspace_id, limit=10_000)
        doomed_ids: list[int] = []
        if effective_max_age_days > 0:
            cutoff = (time.time() if now is None else now) - (effective_max_age_days * 86400)
            doomed_ids.extend(record.id for record in records if record.created_at < cutoff)
        surviving = [record for record in records if record.id not in doomed_ids]
        if effective_max_runtime_logs > 0 and len(surviving) > effective_max_runtime_logs:
            doomed_ids.extend(record.id for record in surviving[effective_max_runtime_logs:])
        doomed_ids = list(dict.fromkeys(doomed_ids))
        if not doomed_ids:
            return 0
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                for log_id in doomed_ids:
                    cursor.execute("DELETE FROM runtime_logs WHERE id = %s", (log_id,))
            connection.commit()
        return len(doomed_ids)

    def _flush_log_batch(self, batch: list[_PendingRuntimeLog]) -> None:
        if self._sessions is None or not batch:
            return
        rows: list[tuple[object, ...]] = [
            (
                item.workspace_id,
                item.source,
                item.logger_name,
                item.level,
                item.message,
                item.created_at,
                item.data_json,
            )
            for item in batch
        ]
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.executemany(
                    """
                    INSERT INTO runtime_logs (workspace_id, source, logger_name, level, message, created_at, data_json)
                    VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
                    """,
                    rows,
                )
            connection.commit()


class PostgresStructuredLogHandler(logging.Handler):
    def __init__(
        self,
        *,
        session_manager: SessionManager[DbConnectionLike] | None = None,
        config: PostgresStorageConfig | None = None,
        workspace_id: str | None = None,
        source: str = "runtime",
    ) -> None:
        super().__init__()
        if session_manager is None and config is None:
            raise ValueError("session_manager_or_config_required")
        if session_manager is not None:
            self._session_manager = session_manager
            self._owns_session_manager = False
        else:
            assert config is not None
            self._session_manager = PostgresConnectionManager(config)
            self._owns_session_manager = True
        self._repository = PostgresRuntimeLogRepository(
            self._session_manager,
            workspace_id=workspace_id,
        )
        self._source = source
        self._closed = False

    def emit(self, record: logging.LogRecord) -> None:
        if self._closed:
            return
        try:
            from mcp_memory.runtime_logging import _build_log_data

            self._repository.write_log(
                source=self._source,
                logger_name=record.name,
                level=record.levelname,
                message=record.getMessage(),
                created_at=float(record.created),
                data=_build_log_data(record),
            )
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        try:
            self._closed = True
            self._repository.close()
            if self._owns_session_manager:
                self._session_manager.close()
        finally:
            super().close()
