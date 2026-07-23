from __future__ import annotations

from collections.abc import Mapping
import time
from typing import cast

from mcp_memory.operational_store_rows import (
    AIConversationRecord,
    ProviderAdmissionStateRecord,
    ProviderUsageSample,
    ProviderUsageSummary,
    build_provider_usage_summaries,
    encode_optional_json_object,
)
from mcp_memory.provider_usage_store import (
    _ALL_WORKSPACES,
    _UNCHANGED,
)
from mcp_memory.storage.postgres_store_support import optional_connection
from mcp_memory.storage.session import DbConnectionLike, SessionManager


class PostgresProviderUsageRepository:
    def __init__(self, session_manager: SessionManager[DbConnectionLike] | None, *, workspace_id: str | None):
        self._sessions = session_manager
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
    ) -> None:
        with optional_connection(self._sessions) as connection:
            if connection is None:
                return
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO provider_usage (
                        workspace_id, task_name, task_id, request_id, subprocess_pid,
                        provider_key, provider_name, model_name, status, duration_seconds,
                        created_at, error_text, reason_category, reason_code, retry_delay_seconds
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
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
            connection.commit()

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
        with optional_connection(self._sessions) as connection:
            if connection is None:
                return
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO ai_conversations (
                        request_id, attempt, workspace_id, task_name, task_id, provider_key,
                        provider_name, model_name, subprocess_pid, prompt_text, response_text,
                        parsed_json, status, error_text, reason_category, reason_code,
                        retry_delay_seconds, started_at, completed_at, duration_seconds
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s,
                        %s::jsonb, %s, %s, %s, %s,
                        %s, %s, %s, %s
                    )
                    ON CONFLICT (request_id, attempt)
                    DO UPDATE SET
                        workspace_id = EXCLUDED.workspace_id,
                        task_name = EXCLUDED.task_name,
                        task_id = EXCLUDED.task_id,
                        provider_key = EXCLUDED.provider_key,
                        provider_name = EXCLUDED.provider_name,
                        model_name = EXCLUDED.model_name,
                        subprocess_pid = EXCLUDED.subprocess_pid,
                        prompt_text = EXCLUDED.prompt_text,
                        response_text = EXCLUDED.response_text,
                        parsed_json = EXCLUDED.parsed_json,
                        status = EXCLUDED.status,
                        error_text = EXCLUDED.error_text,
                        reason_category = EXCLUDED.reason_category,
                        reason_code = EXCLUDED.reason_code,
                        retry_delay_seconds = EXCLUDED.retry_delay_seconds,
                        started_at = EXCLUDED.started_at,
                        completed_at = EXCLUDED.completed_at,
                        duration_seconds = EXCLUDED.duration_seconds
                    """,
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
            connection.commit()

    def list_conversations(
        self,
        *,
        workspace_id: str | None | object = _ALL_WORKSPACES,
        request_id: str | None = None,
        task_name: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[AIConversationRecord]:
        clauses: list[str] = []
        params: list[object] = []
        active_workspace_id = None if workspace_id is _ALL_WORKSPACES else workspace_id
        if active_workspace_id is not None:
            clauses.append("workspace_id = %s")
            params.append(active_workspace_id)
        if request_id is not None:
            clauses.append("request_id = %s")
            params.append(request_id)
        if task_name is not None:
            clauses.append("task_name = %s")
            params.append(task_name)
        if status is not None:
            clauses.append("status = %s")
            params.append(status)
        query = (
            "SELECT id, request_id, attempt, workspace_id, task_name, task_id, provider_key, provider_name, model_name, "
            "subprocess_pid, prompt_text, response_text, parsed_json, status, error_text, reason_category, reason_code, "
            "retry_delay_seconds, started_at, completed_at, duration_seconds FROM ai_conversations"
        )
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY completed_at DESC, id DESC LIMIT %s"
        params.append(limit)
        with optional_connection(self._sessions) as connection:
            if connection is None:
                return []
            with connection.cursor() as cursor:
                cursor.execute(query, tuple(params))
                return [self._row_to_conversation(row) for row in cursor.fetchall()]

    def get_conversation(self, request_id: str) -> list[AIConversationRecord]:
        return self.list_conversations(request_id=request_id, limit=200)

    def count_conversation_statuses_since(
        self,
        *,
        after: float,
        workspace_id: str | None,
    ) -> dict[str, int]:
        query = "SELECT status, COUNT(*) FROM ai_conversations WHERE completed_at >= %s"
        params: list[object] = [after]
        if workspace_id is not None:
            query += " AND workspace_id = %s"
            params.append(workspace_id)
        query += " GROUP BY status"
        with optional_connection(self._sessions) as connection:
            if connection is None:
                return {}
            with connection.cursor() as cursor:
                cursor.execute(query, tuple(params))
                return {str(row[0]): int(row[1]) for row in cursor.fetchall()}

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
            where_clause="request_id = %s",
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
            where_clause="task_id = %s",
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
        heartbeat_at = time.time() if completed_at is None else completed_at
        keep_subprocess_pid = subprocess_pid is _UNCHANGED
        next_subprocess_pid = None if keep_subprocess_pid else subprocess_pid
        with optional_connection(self._sessions) as connection:
            if connection is None:
                return 0
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE ai_conversations
                    SET completed_at = %s,
                        duration_seconds = GREATEST(%s - started_at, 0.0),
                        subprocess_pid = CASE WHEN %s THEN subprocess_pid ELSE %s END
                    WHERE request_id = %s AND status = 'running'
                    """,
                    (
                        heartbeat_at,
                        heartbeat_at,
                        keep_subprocess_pid,
                        next_subprocess_pid,
                        request_id,
                    ),
                )
                rowcount = int(getattr(cursor, "rowcount", 0) or 0)
            connection.commit()
        return rowcount

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
        finalized_at = time.time() if completed_at is None else completed_at
        keep_response_text = response_text is _UNCHANGED
        next_response_text = "" if keep_response_text else str(response_text)
        keep_parsed = parsed is _UNCHANGED
        next_parsed_json = None
        if not keep_parsed:
            assert parsed is None or isinstance(parsed, Mapping)
            next_parsed_json = encode_optional_json_object(parsed)
        with optional_connection(self._sessions) as connection:
            if connection is None:
                return 0
            with connection.cursor() as cursor:
                cursor.execute(
                    (
                        "UPDATE ai_conversations "
                        "SET status = %s, "
                        "error_text = COALESCE(%s, error_text), "
                        "reason_category = COALESCE(%s, reason_category), "
                        "reason_code = COALESCE(%s, reason_code), "
                        "retry_delay_seconds = COALESCE(%s, retry_delay_seconds), "
                        "completed_at = %s, "
                        "duration_seconds = GREATEST(%s - started_at, 0.0), "
                        "response_text = CASE WHEN %s THEN response_text ELSE %s END, "
                        "parsed_json = CASE WHEN %s THEN parsed_json ELSE %s::jsonb END "
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
                        keep_response_text,
                        next_response_text,
                        keep_parsed,
                        next_parsed_json,
                        *where_params,
                    ),
                )
                rowcount = int(getattr(cursor, "rowcount", 0) or 0)
            connection.commit()
        return rowcount

    def summarize_usage(
        self,
        *,
        workspace_id: str | None | object = _ALL_WORKSPACES,
        now: float | None = None,
    ) -> list[ProviderUsageSummary]:
        current_time = time.time() if now is None else now
        active_workspace_id = None if workspace_id is _ALL_WORKSPACES else workspace_id
        query = (
            "SELECT task_name, provider_key, provider_name, model_name, status, duration_seconds, created_at, reason_code "
            "FROM provider_usage"
        )
        params: list[object] = []
        if active_workspace_id is not None:
            query += " WHERE workspace_id = %s"
            params.append(active_workspace_id)
        with optional_connection(self._sessions) as connection:
            if connection is None:
                return []
            with connection.cursor() as cursor:
                cursor.execute(query, tuple(params))
                rows = cursor.fetchall()
        samples = [ProviderUsageSample.from_postgres_row(row) for row in rows]
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
        with optional_connection(self._sessions) as connection:
            if connection is None:
                return
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO provider_admission_state (
                        provider_key, model_name, reason_category, reason_code,
                        error_text, retry_delay_seconds, active_until, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (provider_key, model_name)
                    DO UPDATE SET
                        reason_category = EXCLUDED.reason_category,
                        reason_code = EXCLUDED.reason_code,
                        error_text = EXCLUDED.error_text,
                        retry_delay_seconds = EXCLUDED.retry_delay_seconds,
                        active_until = EXCLUDED.active_until,
                        updated_at = EXCLUDED.updated_at
                    """,
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
            connection.commit()

    def clear_admission_state(self, *, provider_key: str, model_name: str) -> None:
        with optional_connection(self._sessions) as connection:
            if connection is None:
                return
            with connection.cursor() as cursor:
                cursor.execute(
                    "DELETE FROM provider_admission_state WHERE provider_key = %s AND model_name = %s",
                    (provider_key, model_name),
                )
            connection.commit()

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
        current_time = time.time() if now is None else now
        query = (
            "SELECT provider_key, model_name, reason_category, reason_code, error_text, retry_delay_seconds, active_until, updated_at "
            "FROM provider_admission_state WHERE active_until > %s"
        )
        params: list[object] = [current_time]
        if provider_key is not None:
            query += " AND provider_key = %s"
            params.append(provider_key)
        if model_name is not None:
            query += " AND model_name = %s"
            params.append(model_name)
        query += " ORDER BY active_until DESC, updated_at DESC"
        with optional_connection(self._sessions) as connection:
            if connection is None:
                return []
            with connection.cursor() as cursor:
                cursor.execute(query, tuple(params))
                rows = cursor.fetchall()
        return [ProviderAdmissionStateRecord.from_postgres_row(row) for row in rows]

    def count_recent_calls(
        self,
        *,
        provider_keys: list[str],
        now: float | None = None,
        window_seconds: float = 86400.0,
    ) -> int:
        if not provider_keys:
            return 0
        current_time = time.time() if now is None else now
        cutoff = current_time - window_seconds
        placeholders = ",".join(["%s"] * len(provider_keys))
        params: list[object] = [*provider_keys, cutoff]
        query = (
            f"SELECT COUNT(*) FROM provider_usage WHERE provider_key IN ({placeholders}) "
            "AND created_at >= %s AND status != 'skipped'"
        )
        if self._workspace_id is not None:
            query += " AND workspace_id = %s"
            params.append(self._workspace_id)
        with optional_connection(self._sessions) as connection:
            if connection is None:
                return 0
            with connection.cursor() as cursor:
                cursor.execute(query, tuple(params))
                row = cursor.fetchone()
        return 0 if row is None else int(cast(bool | int | float | str, row[0]))

    def count_recent_model_calls(
        self,
        *,
        model_names: list[str],
        now: float | None = None,
        window_seconds: float = 600.0,
    ) -> int:
        if not model_names:
            return 0
        current_time = time.time() if now is None else now
        cutoff = current_time - window_seconds
        placeholders = ",".join(["%s"] * len(model_names))
        params: list[object] = [*model_names, cutoff]
        query = (
            f"SELECT COUNT(*) FROM provider_usage WHERE model_name IN ({placeholders}) "
            "AND created_at >= %s AND status != 'skipped'"
        )
        if self._workspace_id is not None:
            query += " AND workspace_id = %s"
            params.append(self._workspace_id)
        with optional_connection(self._sessions) as connection:
            if connection is None:
                return 0
            with connection.cursor() as cursor:
                cursor.execute(query, tuple(params))
                row = cursor.fetchone()
        return 0 if row is None else int(cast(bool | int | float | str, row[0]))

    def _row_to_conversation(self, row: tuple[object, ...]) -> AIConversationRecord:
        return AIConversationRecord.from_postgres_row(row)
