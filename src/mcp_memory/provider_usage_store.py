from __future__ import annotations

from collections.abc import Mapping
import time

from mcp_memory.operational_store_rows import (
    AIConversationRecord,
    ProviderAdmissionStateRecord,
    ProviderUsageSample,
    ProviderUsageSummary,
    build_provider_usage_summaries,
    encode_optional_json_object,
)
from mcp_memory.utils.db import DatabaseManager


_ALL_WORKSPACES = object()
_UNCHANGED = object()


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
        reason_category: str | None = None,
        reason_code: str | None = None,
        retry_delay_seconds: float | None = None,
    ):
        if self._db_manager is None:
            return
        self._db_manager.get_connection().execute(
            "INSERT INTO provider_usage (workspace_id, task_name, task_id, request_id, subprocess_pid, provider_key, provider_name, model_name, status, duration_seconds, created_at, error_text, reason_category, reason_code, retry_delay_seconds) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                reason_category,
                reason_code,
                retry_delay_seconds,
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
        parsed: Mapping[str, object] | None,
        status: str,
        error_text: str | None,
        reason_category: str | None = None,
        reason_code: str | None = None,
        retry_delay_seconds: float | None = None,
        started_at: float,
        completed_at: float,
    ) -> None:
        if self._db_manager is None:
            return
        self._db_manager.get_connection().execute(
            "INSERT OR REPLACE INTO ai_conversations (request_id, attempt, workspace_id, task_name, task_id, provider_key, provider_name, model_name, subprocess_pid, prompt_text, response_text, parsed_json, status, error_text, reason_category, reason_code, retry_delay_seconds, started_at, completed_at, duration_seconds) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                encode_optional_json_object(parsed),
                status,
                error_text,
                reason_category,
                reason_code,
                retry_delay_seconds,
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

    def count_conversation_statuses_since(
        self,
        *,
        after: float,
        workspace_id: str | None,
    ) -> dict[str, int]:
        if self._db_manager is None:
            return {}
        query = "SELECT status, COUNT(*) FROM ai_conversations WHERE completed_at >= ?"
        params: list[object] = [after]
        if workspace_id is not None:
            query += " AND workspace_id = ?"
            params.append(workspace_id)
        query += " GROUP BY status"
        rows = self._db_manager.get_connection().execute(query, params).fetchall()
        return {str(row[0]): int(row[1]) for row in rows}

    def finalize_running_conversation(
        self,
        *,
        request_id: str,
        status: str,
        error_text: str | None,
        reason_category: str | None = None,
        reason_code: str | None = None,
        retry_delay_seconds: float | None = None,
        completed_at: float | None = None,
        response_text: str | object = _UNCHANGED,
        parsed: Mapping[str, object] | None | object = _UNCHANGED,
    ) -> int:
        return self._finalize_running_conversations(
            where_clause="request_id = ?",
            where_params=[request_id],
            status=status,
            error_text=error_text,
            reason_category=reason_category,
            reason_code=reason_code,
            retry_delay_seconds=retry_delay_seconds,
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
        reason_category: str | None = None,
        reason_code: str | None = None,
        retry_delay_seconds: float | None = None,
        completed_at: float | None = None,
    ) -> int:
        return self._finalize_running_conversations(
            where_clause="task_id = ?",
            where_params=[task_id],
            status=status,
            error_text=error_text,
            reason_category=reason_category,
            reason_code=reason_code,
            retry_delay_seconds=retry_delay_seconds,
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
        reason_category: str | None,
        reason_code: str | None,
        retry_delay_seconds: float | None,
        completed_at: float | None,
        response_text: str | object = _UNCHANGED,
        parsed: Mapping[str, object] | None | object = _UNCHANGED,
    ) -> int:
        if self._db_manager is None:
            return 0
        finalized_at = time.time() if completed_at is None else completed_at
        keep_response_text = response_text is _UNCHANGED
        next_response_text = "" if keep_response_text else str(response_text)
        keep_parsed = parsed is _UNCHANGED
        next_parsed_json = None
        if not keep_parsed:
            assert parsed is None or isinstance(parsed, Mapping)
            next_parsed_json = encode_optional_json_object(parsed)

        cursor = self._db_manager.get_connection().execute(
            (
                "UPDATE ai_conversations "
                "SET status = ?, "
                "error_text = COALESCE(?, error_text), "
                "reason_category = COALESCE(?, reason_category), "
                "reason_code = COALESCE(?, reason_code), "
                "retry_delay_seconds = COALESCE(?, retry_delay_seconds), "
                "completed_at = ?, "
                "duration_seconds = MAX(? - started_at, 0.0), "
                "response_text = CASE WHEN ? THEN response_text ELSE ? END, "
                "parsed_json = CASE WHEN ? THEN parsed_json ELSE ? END "
                f"WHERE {where_clause} AND status = 'running'"
            ),
            (
                status,
                error_text,
                reason_category,
                reason_code,
                retry_delay_seconds,
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
        return AIConversationRecord.from_sqlite_row(row)

    def summarize_usage(
        self,
        *,
        workspace_id: str | None | object = _ALL_WORKSPACES,
        now: float | None = None,
    ) -> list[ProviderUsageSummary]:
        if self._db_manager is None:
            return []
        current_time = time.time() if now is None else now
        params: list[object] = []
        query = (
            "SELECT task_name, provider_key, provider_name, model_name, status, duration_seconds, created_at, reason_code "
            "FROM provider_usage"
        )
        active_workspace_id = None if workspace_id is _ALL_WORKSPACES else workspace_id
        if active_workspace_id is not None:
            query += " WHERE workspace_id = ?"
            params.append(active_workspace_id)
        rows = self._db_manager.get_connection().execute(query, params).fetchall()
        samples = [ProviderUsageSample.from_sqlite_row(row) for row in rows]
        return build_provider_usage_summaries(
            samples=samples,
            active_states=self.list_active_admission_states(now=current_time),
            current_time=current_time,
        )

    def upsert_admission_state(
        self,
        *,
        provider_key: str,
        model_name: str,
        reason_category: str,
        reason_code: str,
        error_text: str | None,
        retry_delay_seconds: float | None,
        active_until: float,
        updated_at: float,
    ) -> None:
        if self._db_manager is None:
            return
        self._db_manager.get_connection().execute(
            "INSERT OR REPLACE INTO provider_admission_state (provider_key, model_name, reason_category, reason_code, error_text, retry_delay_seconds, active_until, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                provider_key,
                model_name,
                reason_category,
                reason_code,
                error_text,
                retry_delay_seconds,
                active_until,
                updated_at,
            ),
        )
        self._db_manager.get_connection().commit()

    def clear_admission_state(self, *, provider_key: str, model_name: str) -> None:
        if self._db_manager is None:
            return
        self._db_manager.get_connection().execute(
            "DELETE FROM provider_admission_state WHERE provider_key = ? AND model_name = ?",
            (provider_key, model_name),
        )
        self._db_manager.get_connection().commit()

    def get_active_admission_state(
        self,
        *,
        provider_key: str,
        model_name: str,
        now: float | None = None,
    ) -> ProviderAdmissionStateRecord | None:
        states = self.list_active_admission_states(now=now, provider_key=provider_key, model_name=model_name)
        return None if not states else states[0]

    def list_active_admission_states(
        self,
        *,
        now: float | None = None,
        provider_key: str | None = None,
        model_name: str | None = None,
    ) -> list[ProviderAdmissionStateRecord]:
        if self._db_manager is None:
            return []
        current_time = time.time() if now is None else now
        query = "SELECT * FROM provider_admission_state WHERE active_until > ?"
        params: list[object] = [current_time]
        if provider_key is not None:
            query += " AND provider_key = ?"
            params.append(provider_key)
        if model_name is not None:
            query += " AND model_name = ?"
            params.append(model_name)
        query += " ORDER BY active_until DESC, updated_at DESC"
        rows = self._db_manager.get_connection().execute(query, params).fetchall()
        return [ProviderAdmissionStateRecord.from_sqlite_row(row) for row in rows]

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
        query = f"SELECT COUNT(*) AS total FROM provider_usage WHERE provider_key IN ({placeholders}) AND created_at >= ? AND status != 'skipped'"
        if self._workspace_id is not None:
            query += " AND workspace_id = ?"
            params.append(self._workspace_id)
        row = self._db_manager.get_connection().execute(query, params).fetchone()
        return 0 if row is None else int(row["total"] or 0)

    def count_recent_model_calls(
        self,
        *,
        model_names: list[str],
        now: float | None = None,
        window_seconds: float = 600.0,
    ) -> int:
        if self._db_manager is None or not model_names:
            return 0
        current_time = time.time() if now is None else now
        cutoff = current_time - window_seconds
        placeholders = ",".join("?" for _ in model_names)
        params: list[object] = [*model_names, cutoff]
        query = f"SELECT COUNT(*) AS total FROM provider_usage WHERE model_name IN ({placeholders}) AND created_at >= ? AND status != 'skipped'"
        if self._workspace_id is not None:
            query += " AND workspace_id = ?"
            params.append(self._workspace_id)
        row = self._db_manager.get_connection().execute(query, params).fetchone()
        return 0 if row is None else int(row["total"] or 0)
