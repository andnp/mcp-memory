from __future__ import annotations

import json
import time

from mcp_memory.provider_usage_store import (
    AIConversationRecord,
    ProviderAdmissionStateRecord,
    ProviderUsageSummary,
    _ALL_WORKSPACES,
    _UNCHANGED,
    _mean,
    _top_reason,
)
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
        if self._sessions is None:
            return
        with self._sessions.open_connection() as connection:
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
        parsed: dict | None,
        status: str,
        error_text: str | None,
        reason_category: str | None = None,
        reason_code: str | None = None,
        retry_delay_seconds: float | None = None,
        started_at: float,
        completed_at: float,
    ) -> None:
        if self._sessions is None:
            return
        with self._sessions.open_connection() as connection:
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
                        json.dumps(parsed, sort_keys=True) if parsed is not None else None,
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
        if self._sessions is None:
            return []
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
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, tuple(params))
                return [self._row_to_conversation(row) for row in cursor.fetchall()]

    def get_conversation(self, request_id: str) -> list[AIConversationRecord]:
        return self.list_conversations(request_id=request_id, limit=200)

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
        parsed: dict | None | object = _UNCHANGED,
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
        if self._sessions is None:
            return 0
        heartbeat_at = time.time() if completed_at is None else completed_at
        keep_subprocess_pid = subprocess_pid is _UNCHANGED
        next_subprocess_pid = None if keep_subprocess_pid else subprocess_pid
        with self._sessions.open_connection() as connection:
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
        parsed: dict | None | object = _UNCHANGED,
    ) -> int:
        if self._sessions is None:
            return 0
        finalized_at = time.time() if completed_at is None else completed_at
        keep_response_text = response_text is _UNCHANGED
        next_response_text = "" if keep_response_text else str(response_text)
        keep_parsed = parsed is _UNCHANGED
        next_parsed_json = None
        if not keep_parsed and parsed is not None:
            next_parsed_json = json.dumps(parsed, sort_keys=True)
        with self._sessions.open_connection() as connection:
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

    def summarize_usage(self, *, workspace_id: str | None | object = _ALL_WORKSPACES, now: float | None = None):
        if self._sessions is None:
            return []
        current_time = time.time() if now is None else now
        last_hour = current_time - 3600
        last_day = current_time - 86400
        active_workspace_id = None if workspace_id is _ALL_WORKSPACES else workspace_id
        query = (
            "SELECT task_name, provider_key, provider_name, model_name, status, duration_seconds, created_at, reason_code "
            "FROM provider_usage"
        )
        params: list[object] = []
        if active_workspace_id is not None:
            query += " WHERE workspace_id = %s"
            params.append(active_workspace_id)
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, tuple(params))
                rows = cursor.fetchall()
        active_states = {
            (state.provider_key, state.model_name): state
            for state in self.list_active_admission_states(now=current_time)
        }
        aggregate: dict[tuple[str | None, str, str, str], dict[str, object]] = {}
        for row in rows:
            key = (
                None if row[0] is None else str(row[0]),
                str(row[1]),
                str(row[2]),
                str(row[3]),
            )
            bucket = aggregate.setdefault(
                key,
                {
                    "calls_last_hour": 0,
                    "calls_last_day": 0,
                    "failures_last_hour": 0,
                    "failures_last_day": 0,
                    "skips_last_hour": 0,
                    "skips_last_day": 0,
                    "duration_last_hour": [],
                    "duration_last_day": [],
                    "failure_reasons": {},
                    "skip_reasons": {},
                },
            )
            created_at = _coerce_float(row[6])
            status = str(row[4])
            reason_code = None if row[7] is None else str(row[7])
            in_last_hour = created_at >= last_hour
            in_last_day = created_at >= last_day
            executed = status != "skipped"
            if executed:
                if in_last_day:
                    bucket["calls_last_day"] = int(bucket["calls_last_day"]) + 1
                    cast_list = bucket["duration_last_day"]
                    assert isinstance(cast_list, list)
                    cast_list.append(_coerce_float(row[5]))
                if in_last_hour:
                    bucket["calls_last_hour"] = int(bucket["calls_last_hour"]) + 1
                    cast_list = bucket["duration_last_hour"]
                    assert isinstance(cast_list, list)
                    cast_list.append(_coerce_float(row[5]))
                if status != "success" and in_last_day:
                    bucket["failures_last_day"] = int(bucket["failures_last_day"]) + 1
                    failure_key = reason_code or "unclassified_error"
                    failure_reasons = bucket["failure_reasons"]
                    assert isinstance(failure_reasons, dict)
                    failure_reasons[failure_key] = int(failure_reasons.get(failure_key, 0)) + 1
                    if in_last_hour:
                        bucket["failures_last_hour"] = int(bucket["failures_last_hour"]) + 1
            else:
                if in_last_day:
                    bucket["skips_last_day"] = int(bucket["skips_last_day"]) + 1
                    skip_key = reason_code or "unclassified_skip"
                    skip_reasons = bucket["skip_reasons"]
                    assert isinstance(skip_reasons, dict)
                    skip_reasons[skip_key] = int(skip_reasons.get(skip_key, 0)) + 1
                if in_last_hour:
                    bucket["skips_last_hour"] = int(bucket["skips_last_hour"]) + 1
        summaries: list[ProviderUsageSummary] = []
        for (task_name, provider_key, provider_name, model_name), bucket in aggregate.items():
            active_state = active_states.get((provider_key, model_name))
            duration_last_hour = bucket["duration_last_hour"]
            duration_last_day = bucket["duration_last_day"]
            failure_reasons = bucket["failure_reasons"]
            skip_reasons = bucket["skip_reasons"]
            assert isinstance(duration_last_hour, list)
            assert isinstance(duration_last_day, list)
            assert isinstance(failure_reasons, dict)
            assert isinstance(skip_reasons, dict)
            summaries.append(
                ProviderUsageSummary(
                    task_name=task_name,
                    provider_key=provider_key,
                    provider_name=provider_name,
                    model_name=model_name,
                    calls_last_hour=int(bucket["calls_last_hour"]),
                    calls_last_day=int(bucket["calls_last_day"]),
                    failures_last_hour=int(bucket["failures_last_hour"]),
                    failures_last_day=int(bucket["failures_last_day"]),
                    skips_last_hour=int(bucket["skips_last_hour"]),
                    skips_last_day=int(bucket["skips_last_day"]),
                    avg_duration_last_hour=_mean(duration_last_hour),
                    avg_duration_last_day=_mean(duration_last_day),
                    top_failure_reason_last_day=_top_reason(failure_reasons),
                    top_skip_reason_last_day=_top_reason(skip_reasons),
                    active_admission_reason=None if active_state is None else active_state.reason_code,
                    active_admission_category=None if active_state is None else active_state.reason_category,
                    active_retry_delay_seconds=None if active_state is None else max(active_state.active_until - current_time, 0.0),
                )
            )
        summaries.sort(key=lambda item: (-item.calls_last_day, -item.skips_last_day, item.task_name or "", item.provider_key))
        return summaries

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
        if self._sessions is None:
            return
        with self._sessions.open_connection() as connection:
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
        if self._sessions is None:
            return
        with self._sessions.open_connection() as connection:
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
        if self._sessions is None:
            return []
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
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, tuple(params))
                rows = cursor.fetchall()
        return [
            ProviderAdmissionStateRecord(
                provider_key=str(row[0]),
                model_name=str(row[1]),
                reason_category=str(row[2]),
                reason_code=str(row[3]),
                error_text=None if row[4] is None else str(row[4]),
                retry_delay_seconds=None if row[5] is None else _coerce_float(row[5]),
                active_until=_coerce_float(row[6]),
                updated_at=_coerce_float(row[7]),
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
        if self._sessions is None or not provider_keys:
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
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, tuple(params))
                row = cursor.fetchone()
        return 0 if row is None else _coerce_int(row[0])

    def count_recent_model_calls(
        self,
        *,
        model_names: list[str],
        now: float | None = None,
        window_seconds: float = 600.0,
    ) -> int:
        if self._sessions is None or not model_names:
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
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, tuple(params))
                row = cursor.fetchone()
        return 0 if row is None else _coerce_int(row[0])

    def _row_to_conversation(self, row: tuple[object, ...]) -> AIConversationRecord:
        parsed = None
        if row[12] is not None:
            if isinstance(row[12], str):
                decoded = json.loads(row[12]) if row[12].strip() else None
            else:
                decoded = row[12]
            if isinstance(decoded, dict):
                parsed = {str(key): value for key, value in decoded.items()}
        return AIConversationRecord(
            id=_coerce_int(row[0]),
            request_id=str(row[1]),
            attempt=_coerce_int(row[2]),
            workspace_id=None if row[3] is None else str(row[3]),
            task_name=None if row[4] is None else str(row[4]),
            task_id=None if row[5] is None else str(row[5]),
            provider_key=str(row[6]),
            provider_name=str(row[7]),
            model_name=str(row[8]),
            subprocess_pid=None if row[9] is None else _coerce_int(row[9]),
            prompt_text=str(row[10]),
            response_text=str(row[11]),
            parsed=parsed,
            status=str(row[13]),
            error_text=None if row[14] is None else str(row[14]),
            reason_category=None if row[15] is None else str(row[15]),
            reason_code=None if row[16] is None else str(row[16]),
            retry_delay_seconds=None if row[17] is None else _coerce_float(row[17]),
            started_at=_coerce_float(row[18]),
            completed_at=_coerce_float(row[19]),
            duration_seconds=_coerce_float(row[20]),
        )