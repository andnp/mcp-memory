from __future__ import annotations

from dataclasses import dataclass
import json
import time

from mcp_memory.utils.db import DatabaseManager


_ALL_WORKSPACES = object()
_UNCHANGED = object()


@dataclass(frozen=True)
class ProviderUsageSummary:
    task_name: str | None
    provider_key: str
    provider_name: str
    model_name: str
    calls_last_hour: int
    calls_last_day: int
    failures_last_hour: int
    failures_last_day: int
    avg_duration_last_hour: float
    avg_duration_last_day: float


@dataclass(frozen=True)
class AIConversationRecord:
    id: int
    request_id: str
    attempt: int
    workspace_id: str | None
    task_name: str | None
    task_id: str | None
    provider_key: str
    provider_name: str
    model_name: str
    subprocess_pid: int | None
    prompt_text: str
    response_text: str
    parsed: dict | None
    status: str
    error_text: str | None
    started_at: float
    completed_at: float
    duration_seconds: float


class ProviderUsageRepository:
    def __init__(self, db_manager: DatabaseManager | None, *, workspace_id: str | None):
        self._db_manager = db_manager
        self._workspace_id = workspace_id

    def record_call(
        self,
        *,
        task_name: str | None,
        task_id: str | None,
        request_id: str | None,
        subprocess_pid: int | None,
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
            "INSERT INTO provider_usage (workspace_id, task_name, task_id, request_id, subprocess_pid, provider_key, provider_name, model_name, status, duration_seconds, created_at, error_text) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                self._workspace_id,
                task_name,
                task_id,
                request_id,
                subprocess_pid,
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

    def record_conversation(
        self,
        *,
        request_id: str,
        attempt: int,
        task_name: str | None,
        task_id: str | None,
        provider_key: str,
        provider_name: str,
        model_name: str,
        subprocess_pid: int | None,
        prompt_text: str,
        response_text: str,
        parsed: dict | None,
        status: str,
        error_text: str | None,
        started_at: float,
        completed_at: float,
    ) -> None:
        if self._db_manager is None:
            return
        self._db_manager.get_connection().execute(
            "INSERT OR REPLACE INTO ai_conversations (request_id, attempt, workspace_id, task_name, task_id, provider_key, provider_name, model_name, subprocess_pid, prompt_text, response_text, parsed_json, status, error_text, started_at, completed_at, duration_seconds) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                request_id,
                attempt,
                self._workspace_id,
                task_name,
                task_id,
                provider_key,
                provider_name,
                model_name,
                subprocess_pid,
                prompt_text,
                response_text,
                None if parsed is None else json.dumps(parsed, sort_keys=True),
                status,
                error_text,
                started_at,
                completed_at,
                max(completed_at - started_at, 0.0),
            ),
        )
        self._db_manager.get_connection().commit()

    def list_conversations(
        self,
        *,
        workspace_id: str | None | object = _ALL_WORKSPACES,
        request_id: str | None = None,
        task_name: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[AIConversationRecord]:
        if self._db_manager is None:
            return []
        clauses: list[str] = []
        params: list[object] = []
        query = "SELECT * FROM ai_conversations"
        active_workspace_id = None if workspace_id is _ALL_WORKSPACES else workspace_id
        if active_workspace_id is not None:
            clauses.append("workspace_id = ?")
            params.append(active_workspace_id)
        if request_id is not None:
            clauses.append("request_id = ?")
            params.append(request_id)
        if task_name is not None:
            clauses.append("task_name = ?")
            params.append(task_name)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY completed_at DESC, id DESC LIMIT ?"
        params.append(limit)
        rows = self._db_manager.get_connection().execute(query, params).fetchall()
        return [self._row_to_conversation(row) for row in rows]

    def get_conversation(self, request_id: str) -> list[AIConversationRecord]:
        return self.list_conversations(request_id=request_id, limit=200)

    def finalize_running_conversation(
        self,
        *,
        request_id: str,
        status: str,
        error_text: str | None,
        completed_at: float | None = None,
        response_text: str | object = _UNCHANGED,
        parsed: dict | None | object = _UNCHANGED,
    ) -> int:
        return self._finalize_running_conversations(
            where_clause="request_id = ?",
            where_params=[request_id],
            status=status,
            error_text=error_text,
            completed_at=completed_at,
            response_text=response_text,
            parsed=parsed,
        )

    def reconcile_running_task_conversations(
        self,
        *,
        task_id: str,
        status: str,
        error_text: str | None,
        completed_at: float | None = None,
    ) -> int:
        return self._finalize_running_conversations(
            where_clause="task_id = ?",
            where_params=[task_id],
            status=status,
            error_text=error_text,
            completed_at=completed_at,
        )

    def touch_running_conversation(
        self,
        *,
        request_id: str,
        completed_at: float | None = None,
        subprocess_pid: int | None | object = _UNCHANGED,
    ) -> int:
        if self._db_manager is None:
            return 0
        heartbeat_at = time.time() if completed_at is None else completed_at
        keep_subprocess_pid = subprocess_pid is _UNCHANGED
        next_subprocess_pid = None if keep_subprocess_pid else subprocess_pid
        cursor = self._db_manager.get_connection().execute(
            (
                "UPDATE ai_conversations "
                "SET completed_at = ?, "
                "duration_seconds = MAX(? - started_at, 0.0), "
                "subprocess_pid = CASE WHEN ? THEN subprocess_pid ELSE ? END "
                "WHERE request_id = ? AND status = 'running'"
            ),
            (
                heartbeat_at,
                heartbeat_at,
                1 if keep_subprocess_pid else 0,
                next_subprocess_pid,
                request_id,
            ),
        )
        self._db_manager.get_connection().commit()
        return int(cursor.rowcount or 0)

    def _finalize_running_conversations(
        self,
        *,
        where_clause: str,
        where_params: list[object],
        status: str,
        error_text: str | None,
        completed_at: float | None,
        response_text: str | object = _UNCHANGED,
        parsed: dict | None | object = _UNCHANGED,
    ) -> int:
        if self._db_manager is None:
            return 0
        finalized_at = time.time() if completed_at is None else completed_at
        keep_response_text = response_text is _UNCHANGED
        next_response_text = "" if keep_response_text else str(response_text)
        keep_parsed = parsed is _UNCHANGED
        next_parsed_json = None
        if not keep_parsed and parsed is not None:
            next_parsed_json = json.dumps(parsed, sort_keys=True)

        cursor = self._db_manager.get_connection().execute(
            (
                "UPDATE ai_conversations "
                "SET status = ?, "
                "error_text = COALESCE(?, error_text), "
                "completed_at = ?, "
                "duration_seconds = MAX(? - started_at, 0.0), "
                "response_text = CASE WHEN ? THEN response_text ELSE ? END, "
                "parsed_json = CASE WHEN ? THEN parsed_json ELSE ? END "
                f"WHERE {where_clause} AND status = 'running'"
            ),
            (
                status,
                error_text,
                finalized_at,
                finalized_at,
                1 if keep_response_text else 0,
                next_response_text,
                1 if keep_parsed else 0,
                next_parsed_json,
                *where_params,
            ),
        )
        self._db_manager.get_connection().commit()
        return int(cursor.rowcount or 0)

    def _row_to_conversation(self, row) -> AIConversationRecord:
        parsed = None
        if isinstance(row["parsed_json"], str) and row["parsed_json"].strip():
            decoded = json.loads(row["parsed_json"])
            if isinstance(decoded, dict):
                parsed = decoded
        return AIConversationRecord(
            id=int(row["id"]),
            request_id=str(row["request_id"]),
            attempt=int(row["attempt"]),
            workspace_id=row["workspace_id"],
            task_name=row["task_name"],
            task_id=row["task_id"],
            provider_key=str(row["provider_key"]),
            provider_name=str(row["provider_name"]),
            model_name=str(row["model_name"]),
            subprocess_pid=None if row["subprocess_pid"] is None else int(row["subprocess_pid"]),
            prompt_text=str(row["prompt_text"]),
            response_text=str(row["response_text"]),
            parsed=parsed,
            status=str(row["status"]),
            error_text=row["error_text"],
            started_at=float(row["started_at"]),
            completed_at=float(row["completed_at"]),
            duration_seconds=float(row["duration_seconds"] or 0.0),
        )

    def summarize_usage(self, *, workspace_id: str | None | object = _ALL_WORKSPACES, now: float | None = None):
        if self._db_manager is None:
            return []
        current_time = time.time() if now is None else now
        last_hour = current_time - 3600
        last_day = current_time - 86400
        params: list[object] = [last_hour, last_day, last_hour, last_day, last_hour, last_day]
        query = (
            "SELECT task_name, provider_key, provider_name, model_name, "
            "SUM(CASE WHEN created_at >= ? THEN 1 ELSE 0 END) AS calls_last_hour, "
            "SUM(CASE WHEN created_at >= ? THEN 1 ELSE 0 END) AS calls_last_day, "
            "SUM(CASE WHEN created_at >= ? AND status != 'success' THEN 1 ELSE 0 END) AS failures_last_hour, "
            "SUM(CASE WHEN created_at >= ? AND status != 'success' THEN 1 ELSE 0 END) AS failures_last_day, "
            "AVG(CASE WHEN created_at >= ? THEN duration_seconds END) AS avg_duration_last_hour, "
            "AVG(CASE WHEN created_at >= ? THEN duration_seconds END) AS avg_duration_last_day "
            "FROM provider_usage"
        )
        active_workspace_id = None if workspace_id is _ALL_WORKSPACES else workspace_id
        if active_workspace_id is not None:
            query += " WHERE workspace_id = ?"
            params.append(active_workspace_id)
        query += " GROUP BY task_name, provider_key, provider_name, model_name ORDER BY calls_last_day DESC, task_name ASC, provider_key ASC"
        rows = self._db_manager.get_connection().execute(query, params).fetchall()
        return [
            ProviderUsageSummary(
                task_name=row["task_name"],
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

    def count_recent_calls(
        self,
        *,
        provider_keys: list[str],
        now: float | None = None,
        window_seconds: float = 86400.0,
    ) -> int:
        if self._db_manager is None or not provider_keys:
            return 0
        current_time = time.time() if now is None else now
        cutoff = current_time - window_seconds
        placeholders = ",".join("?" for _ in provider_keys)
        params: list[object] = [*provider_keys, cutoff]
        query = f"SELECT COUNT(*) AS total FROM provider_usage WHERE provider_key IN ({placeholders}) AND created_at >= ?"
        if self._workspace_id is not None:
            query += " AND workspace_id = ?"
            params.append(self._workspace_id)
        row = self._db_manager.get_connection().execute(query, params).fetchone()
        return 0 if row is None else int(row["total"] or 0)
