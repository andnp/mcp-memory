from __future__ import annotations

from dataclasses import dataclass, field
import json
import time
from typing import cast

from mcp_memory.config import LoggingConfig
from mcp_memory.utils.db import DatabaseManager


_ALL_WORKSPACES = object()


class _AllWorkspacesSentinel:
    pass


_ALL_WORKSPACES = _AllWorkspacesSentinel()


@dataclass(frozen=True)
class RuntimeLogRecord:
    id: int
    created_at: float
    level: str
    logger_name: str
    source: str
    message: str
    data: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeLogSummary:
    total: int
    by_level: dict[str, int] = field(default_factory=dict)
    by_source: dict[str, int] = field(default_factory=dict)


class RuntimeLogRepository:
    def __init__(
        self,
        db_manager: DatabaseManager | None,
        *,
        workspace_id: str | None,
        config: LoggingConfig | None = None,
    ):
        self._db_manager = db_manager
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
    ):
        if self._db_manager is None:
            return
        conn = self._db_manager.get_connection()
        conn.execute(
            "INSERT INTO runtime_logs (workspace_id, source, logger_name, level, message, created_at, data_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
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
        conn.commit()
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
    ):
        if self._db_manager is None:
            return []
        resolved_workspace_id: str | None = None if workspace_id is _ALL_WORKSPACES else cast(str | None, workspace_id)
        where_clause, params = _build_log_filters(
            workspace_id=resolved_workspace_id,
            level=level,
            logger_name=logger_name,
            source=source,
            query=query,
            after=after,
            before=before,
        )
        rows = self._db_manager.get_connection().execute(
            "SELECT id, created_at, level, logger_name, source, message, data_json FROM runtime_logs"
            + where_clause
            + " ORDER BY created_at DESC, id DESC LIMIT ?",
            [*params, limit],
        ).fetchall()
        return [
            RuntimeLogRecord(
                id=int(row["id"]),
                created_at=float(row["created_at"]),
                level=str(row["level"]),
                logger_name=str(row["logger_name"]),
                source=str(row["source"]),
                message=str(row["message"]),
                data=_decode_log_data(row["data_json"]),
            )
            for row in rows
        ]

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
    ):
        if self._db_manager is None:
            return RuntimeLogSummary(total=0)
        resolved_workspace_id: str | None = None if workspace_id is _ALL_WORKSPACES else cast(str | None, workspace_id)
        where_clause, params = _build_log_filters(
            workspace_id=resolved_workspace_id,
            level=level,
            logger_name=logger_name,
            source=source,
            query=query,
            after=after,
            before=before,
        )
        conn = self._db_manager.get_connection()
        total_row = conn.execute(
            "SELECT COUNT(*) AS count FROM runtime_logs" + where_clause,
            params,
        ).fetchone()
        grouped_rows = conn.execute(
            "SELECT level, source, COUNT(*) AS count FROM runtime_logs"
            + where_clause
            + " GROUP BY level, source",
            params,
        ).fetchall()
        by_level: dict[str, int] = {}
        by_source: dict[str, int] = {}
        for row in grouped_rows:
            level_name = str(row["level"])
            source_name = str(row["source"])
            count = int(row["count"])
            by_level[level_name] = by_level.get(level_name, 0) + count
            by_source[source_name] = by_source.get(source_name, 0) + count
        return RuntimeLogSummary(
            total=0 if total_row is None else int(total_row["count"]),
            by_level=by_level,
            by_source=by_source,
        )

    def apply_retention_policy(self, *, now: float | None = None):
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
    ):
        if self._db_manager is None:
            return 0

        effective_max_runtime_logs = self._config.max_runtime_logs if max_runtime_logs is None else max_runtime_logs
        effective_max_age_days = self._config.max_log_age_days if max_age_days is None else max_age_days
        deleted = 0
        conn = self._db_manager.get_connection()
        resolved_workspace_id: str | None = None if workspace_id is _ALL_WORKSPACES else cast(str | None, workspace_id)
        workspace_clause, workspace_params = _workspace_scope_clause(resolved_workspace_id)

        if effective_max_age_days > 0:
            cutoff = (time.time() if now is None else now) - (effective_max_age_days * 86400)
            deleted += conn.execute(
                "DELETE FROM runtime_logs"
                + workspace_clause
                + (" AND " if workspace_clause else " WHERE ")
                + "created_at < ?",
                [*workspace_params, cutoff],
            ).rowcount

        if effective_max_runtime_logs > 0:
            over_limit_rows = conn.execute(
                "SELECT id FROM runtime_logs"
                + workspace_clause
                + " ORDER BY created_at DESC, id DESC LIMIT -1 OFFSET ?",
                [*workspace_params, effective_max_runtime_logs],
            ).fetchall()
            if over_limit_rows:
                deleted_ids = [int(row["id"]) for row in over_limit_rows]
                placeholders = ",".join("?" for _ in deleted_ids)
                deleted += conn.execute(
                    f"DELETE FROM runtime_logs WHERE id IN ({placeholders})",
                    deleted_ids,
                ).rowcount

        conn.commit()
        return deleted


def _decode_log_data(raw_result: object):
    if not isinstance(raw_result, str) or not raw_result.strip():
        return {}
    try:
        decoded = json.loads(raw_result)
    except ValueError:
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _build_log_filters(
    *,
    workspace_id: str | None,
    level: str | None,
    logger_name: str | None,
    source: str | None,
    query: str | None,
    after: float | None,
    before: float | None,
):
    clauses: list[str] = []
    params: list[object] = []
    if workspace_id is not None:
        clauses.append("workspace_id = ?")
        params.append(workspace_id)
    if level is not None:
        clauses.append("level = ?")
        params.append(level)
    if logger_name is not None:
        clauses.append("logger_name = ?")
        params.append(logger_name)
    if source is not None:
        clauses.append("source = ?")
        params.append(source)
    if query is not None and query.strip():
        needle = f"%{query.strip()}%"
        clauses.append("(message LIKE ? OR logger_name LIKE ? OR source LIKE ?)")
        params.extend([needle, needle, needle])
    if after is not None:
        clauses.append("created_at >= ?")
        params.append(after)
    if before is not None:
        clauses.append("created_at <= ?")
        params.append(before)
    return ("" if not clauses else " WHERE " + " AND ".join(clauses)), params


def _workspace_scope_clause(workspace_id: str | None):
    if workspace_id is None:
        return "", []
    return " WHERE workspace_id = ?", [workspace_id]
