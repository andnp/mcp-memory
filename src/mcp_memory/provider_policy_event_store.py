from __future__ import annotations

import json
import time
from dataclasses import dataclass

from mcp_memory.utils.db import DatabaseManager


@dataclass(frozen=True)
class ProviderPolicyEventRecord:
    id: int
    workspace_id: str | None
    task_name: str
    task_id: str | None
    event_kind: str
    warning_kind: str | None
    provider_key: str | None
    provider_name: str | None
    model_name: str | None
    route_key: str | None
    candidate_routes: list[str]
    reason_category: str | None
    reason_code: str | None
    retry_delay_seconds: float | None
    warning_suppressed: bool
    created_at: float


class ProviderPolicyEventRepository:
    def __init__(self, db_manager: DatabaseManager | None, *, workspace_id: str | None):
        self._db_manager = db_manager
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
        if self._db_manager is None:
            return
        event_time = time.time() if created_at is None else created_at
        self._db_manager.get_connection().execute(
            "INSERT INTO provider_policy_events (workspace_id, task_name, task_id, event_kind, warning_kind, provider_key, provider_name, model_name, route_key, candidate_routes_json, reason_category, reason_code, retry_delay_seconds, warning_suppressed, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                1 if warning_suppressed else 0,
                event_time,
            ),
        )
        self._db_manager.get_connection().commit()
