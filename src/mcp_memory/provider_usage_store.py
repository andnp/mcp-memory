from __future__ import annotations

from dataclasses import dataclass
import time

from mcp_memory.utils.db import DatabaseManager


@dataclass(frozen=True)
class ProviderUsageSummary:
    provider_key: str
    provider_name: str
    model_name: str
    calls_last_hour: int
    calls_last_day: int
    failures_last_hour: int
    failures_last_day: int
    avg_duration_last_hour: float
    avg_duration_last_day: float


class ProviderUsageRepository:
    def __init__(self, db_manager: DatabaseManager | None, *, workspace_id: str | None):
        self._db_manager = db_manager
        self._workspace_id = workspace_id

    def record_call(
        self,
        *,
        provider_key: str,
        provider_name: str,
        model_name: str,
        status: str,
        duration_seconds: float,
        created_at: float,
        error_text: str | None,
    ):
        if self._db_manager is None:
            return
        self._db_manager.get_connection().execute(
            "INSERT INTO provider_usage (workspace_id, provider_key, provider_name, model_name, status, duration_seconds, created_at, error_text) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                self._workspace_id,
                provider_key,
                provider_name,
                model_name,
                status,
                duration_seconds,
                created_at,
                error_text,
            ),
        )
        self._db_manager.get_connection().commit()

    def summarize_usage(self, *, now: float | None = None):
        if self._db_manager is None:
            return []
        current_time = time.time() if now is None else now
        last_hour = current_time - 3600
        last_day = current_time - 86400
        params: list[object] = [last_hour, last_day, last_hour, last_day, last_hour, last_day]
        query = (
            "SELECT provider_key, provider_name, model_name, "
            "SUM(CASE WHEN created_at >= ? THEN 1 ELSE 0 END) AS calls_last_hour, "
            "SUM(CASE WHEN created_at >= ? THEN 1 ELSE 0 END) AS calls_last_day, "
            "SUM(CASE WHEN created_at >= ? AND status != 'success' THEN 1 ELSE 0 END) AS failures_last_hour, "
            "SUM(CASE WHEN created_at >= ? AND status != 'success' THEN 1 ELSE 0 END) AS failures_last_day, "
            "AVG(CASE WHEN created_at >= ? THEN duration_seconds END) AS avg_duration_last_hour, "
            "AVG(CASE WHEN created_at >= ? THEN duration_seconds END) AS avg_duration_last_day "
            "FROM provider_usage"
        )
        if self._workspace_id is not None:
            query += " WHERE workspace_id = ?"
            params.append(self._workspace_id)
        query += " GROUP BY provider_key, provider_name, model_name ORDER BY calls_last_day DESC, provider_key ASC"
        rows = self._db_manager.get_connection().execute(query, params).fetchall()
        return [
            ProviderUsageSummary(
                provider_key=str(row["provider_key"]),
                provider_name=str(row["provider_name"]),
                model_name=str(row["model_name"]),
                calls_last_hour=int(row["calls_last_hour"] or 0),
                calls_last_day=int(row["calls_last_day"] or 0),
                failures_last_hour=int(row["failures_last_hour"] or 0),
                failures_last_day=int(row["failures_last_day"] or 0),
                avg_duration_last_hour=float(row["avg_duration_last_hour"] or 0.0),
                avg_duration_last_day=float(row["avg_duration_last_day"] or 0.0),
            )
            for row in rows
        ]