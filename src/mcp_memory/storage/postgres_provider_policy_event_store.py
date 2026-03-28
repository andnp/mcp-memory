from __future__ import annotations

import json
import time

from mcp_memory.storage.session import DbConnectionLike, SessionManager


class PostgresProviderPolicyEventRepository:
    def __init__(self, session_manager: SessionManager[DbConnectionLike] | None, *, workspace_id: str | None):
        self._sessions = session_manager
        self._workspace_id = workspace_id

    def record_event(
        self,
        *,
        task_name: str,
        task_id: str | None,
        event_kind: str,
        warning_kind: str | None = None,
        provider_key: str | None = None,
        provider_name: str | None = None,
        model_name: str | None = None,
        route_key: str | None = None,
        candidate_routes: list[str] | None = None,
        reason_category: str | None = None,
        reason_code: str | None = None,
        retry_delay_seconds: float | None = None,
        warning_suppressed: bool = False,
        created_at: float | None = None,
    ) -> None:
        if self._sessions is None:
            return
        event_time = time.time() if created_at is None else created_at
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO provider_policy_events (
                        workspace_id, task_name, task_id, event_kind, warning_kind,
                        provider_key, provider_name, model_name, route_key,
                        candidate_routes_json, reason_category, reason_code,
                        retry_delay_seconds, warning_suppressed, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s)
                    """,
                    (
                        self._workspace_id,
                        task_name,
                        task_id,
                        event_kind,
                        warning_kind,
                        provider_key,
                        provider_name,
                        model_name,
                        route_key,
                        json.dumps(candidate_routes or [], sort_keys=True),
                        reason_category,
                        reason_code,
                        retry_delay_seconds,
                        warning_suppressed,
                        event_time,
                    ),
                )
            connection.commit()
