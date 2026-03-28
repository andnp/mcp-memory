from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import time

from mcp_memory.config import LoggingConfig, PostgresStorageConfig
from mcp_memory.runtime_log_store import RuntimeLogRecord, RuntimeLogSummary, _ALL_WORKSPACES, _AllWorkspacesSentinel, _build_log_filters, _decode_log_data
from mcp_memory.storage.postgres_connection import PostgresConnectionManager
from mcp_memory.storage.session import DbConnectionLike, SessionManager


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


def _normalize_workspace_id(workspace_id: str | None | _AllWorkspacesSentinel) -> str | None:
    if workspace_id is _ALL_WORKSPACES:
        return None
    if workspace_id is None or isinstance(workspace_id, str):
        return workspace_id
    raise TypeError(f"Expected workspace id, got {type(workspace_id)!r}")


@dataclass(frozen=True)
class _StoredRuntimeLog:
    id: int
    workspace_id: str | None
    source: str
    logger_name: str
    level: str
    message: str
    created_at: float
    data_json: object


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
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO runtime_logs (workspace_id, source, logger_name, level, message, created_at, data_json)
                    VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
                    """,
                    (
                        self._workspace_id,
                        source,
                        logger_name,
                        level,
                        message,
                        created_at,
                        json.dumps(data, sort_keys=True),
                    ),
                )
            connection.commit()
        self.apply_retention_policy(now=created_at)

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
                return [self._to_public_record(self._row_to_log(row)) for row in cursor.fetchall()]

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
                total = 0 if total_row is None else _coerce_int(total_row[0])

                cursor.execute(grouped_sql, tuple(params))
                for row in cursor.fetchall():
                    level_name = str(row[0])
                    source_name = str(row[1])
                    count = _coerce_int(row[2])
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

    def _row_to_log(self, row: tuple[object, ...]) -> _StoredRuntimeLog:
        return _StoredRuntimeLog(
            id=_coerce_int(row[0]),
            workspace_id=None if row[1] is None else str(row[1]),
            source=str(row[2]),
            logger_name=str(row[3]),
            level=str(row[4]),
            message=str(row[5]),
            created_at=_coerce_float(row[6]),
            data_json=row[7],
        )

    def _to_public_record(self, log_record: _StoredRuntimeLog) -> RuntimeLogRecord:
        return RuntimeLogRecord(
            id=log_record.id,
            created_at=log_record.created_at,
            level=log_record.level,
            logger_name=log_record.logger_name,
            source=log_record.source,
            message=log_record.message,
            data=_decode_log_data(log_record.data_json),
        )


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

    def emit(self, record: logging.LogRecord) -> None:
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
            if self._owns_session_manager:
                self._session_manager.close()
        finally:
            super().close()
