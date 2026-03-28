from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
import json
from typing import Any, cast
import time

from mcp_memory.core.task_handlers import MAINTENANCE_TASK_NAMES
from mcp_memory.management.models import QueueDiagnosticPayload


def _uses_sqlite_connection_api(db_manager) -> bool:
    return hasattr(db_manager, "get_connection")


def _adapt_query_for_backend(query: str, db_manager) -> str:
    if _uses_sqlite_connection_api(db_manager):
        return query
    return (
        query.replace("CHAR(10)", "CHR(10)")
        .replace("GROUP_CONCAT(DISTINCT workspace_id)", "STRING_AGG(DISTINCT workspace_id, ',')")
        .replace("GROUP_CONCAT(DISTINCT tags.name)", "STRING_AGG(DISTINCT tags.name, ',')")
        .replace("?", "%s")
    )


def _row_to_mapping(row: object, columns: list[str] | None = None) -> dict[str, object]:
    if isinstance(row, dict):
        return row
    if isinstance(row, Mapping):
        return {str(key): value for key, value in row.items()}
    row_keys = getattr(row, "keys", None)
    if callable(row_keys):
        row_like = cast(Any, row)
        return {str(key): row_like[key] for key in cast(Any, row_keys)()}
    if isinstance(row, tuple) and columns is not None:
        row_values = cast(tuple[object, ...], row)
        return dict(zip(columns, row_values, strict=False))
    raise TypeError(f"Unsupported row type: {type(row)!r}")


def _fetchall_rows(db_manager, query: str, params: Sequence[object] | None = None) -> list[dict[str, object]]:
    if db_manager is None:
        return []
    effective_params = list(params or [])
    adapted_query = _adapt_query_for_backend(query, db_manager)
    if _uses_sqlite_connection_api(db_manager):
        conn = db_manager.get_connection()
        rows = conn.execute(adapted_query, effective_params).fetchall()
        return [_row_to_mapping(row) for row in rows]
    with db_manager.open_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(adapted_query, tuple(effective_params))
            rows = cursor.fetchall()
            columns = [column.name for column in cursor.description] if cursor.description is not None else []
            return [_row_to_mapping(row, columns) for row in rows]


def _fetchone_row(db_manager, query: str, params: Sequence[object] | None = None) -> dict[str, object] | None:
    rows = _fetchall_rows(db_manager, query, params)
    return rows[0] if rows else None


def fetch_memory_count_rows(db_manager, workspace_id: str | None):
    if db_manager is None:
        return []
    params: list[object] = []
    if workspace_id is None:
        query = (
            "SELECT memories.type, memories.status, COUNT(*) AS count "
            "FROM memories"
        )
    else:
        query = (
            "SELECT memories.type, memories.status, COUNT(*) AS count "
            "FROM memories JOIN memory_workspaces ON memory_workspaces.memory_id = memories.id "
            "WHERE memory_workspaces.workspace_id = ?"
        )
        params.append(workspace_id)
    query += " GROUP BY memories.type, memories.status"
    return _fetchall_rows(db_manager, query, params)


def fetch_task_count_rows(db_manager, workspace_id: str | None):
    if db_manager is None:
        return []
    query = "SELECT status, COUNT(*) AS count FROM tasks"
    params: list[object] = []
    if workspace_id is not None:
        query += " WHERE workspace_id = ?"
        params.append(workspace_id)
    query += " GROUP BY status"
    return _fetchall_rows(db_manager, query, params)


def fetch_running_task_attempt_rows(db_manager, workspace_id: str | None):
    if db_manager is None:
        return []
    query = (
        "SELECT tasks.id AS task_id, tasks.workspace_id, tasks.updated_at, tasks.started_at, tasks.claimed_at, "
        "tasks.execution_epoch, attempt.status AS attempt_status, attempt.started_at AS attempt_started_at, "
        "attempt.last_heartbeat_at, attempt.subprocess_pid AS attempt_subprocess_pid "
        "FROM tasks "
        "LEFT JOIN task_execution_attempts AS attempt "
        "ON attempt.task_id = tasks.id AND attempt.execution_epoch = tasks.execution_epoch "
        "WHERE tasks.status = 'running'"
    )
    params: list[object] = []

    if workspace_id is not None:
        query += " AND tasks.workspace_id = ?"
        params.append(workspace_id)
    query += " ORDER BY tasks.updated_at DESC, tasks.id DESC"
    return _fetchall_rows(db_manager, query, params)


def fetch_memory_metrics_row(db_manager, workspace_id: str | None):
    if db_manager is None:
        return None
    params: list[object] = []
    if workspace_id is None:
        query = (
            "SELECT "
            "COUNT(*) AS total_memories, "
            "COALESCE(SUM(CASE WHEN memories.content = '' THEN 0 ELSE 1 + LENGTH(memories.content) - LENGTH(REPLACE(memories.content, CHAR(10), '')) END), 0) AS total_memory_lines, "
            "COALESCE(SUM(CASE WHEN memories.summary IS NULL OR memories.summary = '' THEN 0 ELSE 1 + LENGTH(memories.summary) - LENGTH(REPLACE(memories.summary, CHAR(10), '')) END), 0) AS total_summary_lines "
            "FROM memories"
        )
    else:
        query = (
            "SELECT "
            "COUNT(*) AS total_memories, "
            "COALESCE(SUM(CASE WHEN memories.content = '' THEN 0 ELSE 1 + LENGTH(memories.content) - LENGTH(REPLACE(memories.content, CHAR(10), '')) END), 0) AS total_memory_lines, "
            "COALESCE(SUM(CASE WHEN memories.summary IS NULL OR memories.summary = '' THEN 0 ELSE 1 + LENGTH(memories.summary) - LENGTH(REPLACE(memories.summary, CHAR(10), '')) END), 0) AS total_summary_lines "
            "FROM memories JOIN memory_workspaces ON memory_workspaces.memory_id = memories.id "
            "WHERE memory_workspaces.workspace_id = ?"
        )
        params.append(workspace_id)
    return _fetchone_row(db_manager, query, params)


def fetch_pending_journal_metrics_row(db_manager, workspace_id: str | None):
    if db_manager is None:
        return None
    query = (
        "SELECT "
        "COUNT(*) AS thought_buffer_entries, "
        "COALESCE(SUM(CASE WHEN content = '' THEN 0 ELSE 1 + LENGTH(content) - LENGTH(REPLACE(content, CHAR(10), '')) END), 0) AS thought_buffer_lines "
        "FROM system1_journal WHERE status = 'pending'"
    )
    params: list[object] = []
    if workspace_id is not None:
        query += " AND workspace_id = ?"
        params.append(workspace_id)
    return _fetchone_row(db_manager, query, params)


def list_task_run_rows_since(db_manager, *, cutoff: float, workspace_id: str | None):
    if db_manager is None:
        return []
    query = "SELECT status, completed_at, duration_seconds, result_json FROM task_runs WHERE completed_at >= ?"
    params: list[object] = [cutoff]
    if workspace_id is not None:
        query += " AND workspace_id = ?"
        params.append(workspace_id)
    return _fetchall_rows(db_manager, query, params)


def list_maintenance_task_run_rows_since(db_manager, *, cutoff: float, workspace_id: str | None):
    if db_manager is None:
        return []
    placeholders = ",".join("?" for _ in MAINTENANCE_TASK_NAMES)
    query = (
        f"SELECT task_id, task_name, status, completed_at, duration_seconds, result_json, error_text "
        f"FROM task_runs WHERE task_name IN ({placeholders}) AND completed_at >= ?"
    )
    params: list[object] = [*MAINTENANCE_TASK_NAMES, cutoff]
    if workspace_id is not None:
        query += " AND workspace_id = ?"
        params.append(workspace_id)
    query += " ORDER BY completed_at DESC, started_at DESC"
    return _fetchall_rows(db_manager, query, params)


def list_provider_usage_rows_since(db_manager, *, cutoff: float, workspace_id: str | None):
    if db_manager is None:
        return []
    query = (
        "SELECT task_name, provider_key, provider_name, model_name, status, duration_seconds, created_at, reason_category, reason_code, retry_delay_seconds "
        "FROM provider_usage WHERE created_at >= ?"
    )
    params: list[object] = [cutoff]
    if workspace_id is not None:
        query += " AND workspace_id = ?"
        params.append(workspace_id)
    return _fetchall_rows(db_manager, query, params)


def list_ai_conversation_rows_since(db_manager, *, cutoff: float, upper_bound: float, workspace_id: str | None):
    if db_manager is None:
        return []
    query = (
        "SELECT provider_key, completed_at, response_text, parsed_json "
        "FROM ai_conversations WHERE completed_at >= ? AND completed_at <= ?"
    )
    params: list[object] = [cutoff, upper_bound]
    if workspace_id is not None:
        query += " AND workspace_id = ?"
        params.append(workspace_id)
    return _fetchall_rows(db_manager, query, params)


def summarize_copilot_premium_requests(db_manager, *, workspace_id: str | None, now: float | None = None) -> dict[str, int]:
    if db_manager is None:
        return {
            "copilot_premium_requests_today": 0,
            "copilot_premium_requests_last_day": 0,
        }

    current_time = time.time() if now is None else now
    last_day_cutoff = current_time - 86_400
    today_start = datetime.fromtimestamp(current_time, UTC).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    query_cutoff = min(last_day_cutoff, today_start)

    premium_requests_today = 0
    premium_requests_last_day = 0
    for row in list_ai_conversation_rows_since(
        db_manager,
        cutoff=query_cutoff,
        upper_bound=current_time,
        workspace_id=workspace_id,
    ):
        provider_key = str(row["provider_key"] or "")
        if not provider_key.startswith("copilot"):
            continue

        premium_requests = extract_copilot_premium_requests(
            response_text=_optional_string(row.get("response_text")),
            parsed_json=_optional_string(row.get("parsed_json")),
        )
        if premium_requests <= 0:
            continue

        completed_at = _coerce_float(row.get("completed_at"))
        if completed_at >= last_day_cutoff:
            premium_requests_last_day += premium_requests
        if completed_at >= today_start:
            premium_requests_today += premium_requests

    return {
        "copilot_premium_requests_today": premium_requests_today,
        "copilot_premium_requests_last_day": premium_requests_last_day,
    }


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _coerce_float(value: object) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"Expected float-compatible value, got {type(value)!r}")


def extract_copilot_premium_requests(*, response_text: str | None, parsed_json: str | None) -> int:
    premium_requests: list[int] = []
    premium_requests.extend(_premium_requests_from_maybe_json(parsed_json))
    premium_requests.extend(_premium_requests_from_maybe_json(response_text))
    return max(premium_requests, default=0)


def _premium_requests_from_maybe_json(value: str | None) -> list[int]:
    if not isinstance(value, str):
        return []
    text = value.strip()
    if not text:
        return []

    premium_requests: list[int] = []
    parsed_text = _try_json_loads(text)
    if parsed_text is not None:
        premium_requests.extend(_collect_premium_requests(parsed_text))
        return premium_requests

    for line in text.splitlines():
        parsed_line = _try_json_loads(line.strip())
        if parsed_line is not None:
            premium_requests.extend(_collect_premium_requests(parsed_line))
    return premium_requests


def _try_json_loads(text: str) -> Any | None:
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _collect_premium_requests(payload: Any) -> list[int]:
    results: list[int] = []
    if isinstance(payload, dict):
        premium_value = payload.get("premiumRequests")
        if isinstance(premium_value, bool):
            premium_value = None
        if isinstance(premium_value, (int, float)):
            results.append(int(premium_value))
        for value in payload.values():
            results.extend(_collect_premium_requests(value))
    elif isinstance(payload, list):
        for item in payload:
            results.extend(_collect_premium_requests(item))
    return results


def list_runtime_log_rows_since(
    db_manager,
    *,
    cutoff: float,
    workspace_id: str | None,
    logger_name: str | None = None,
    level: str | None = None,
):
    if db_manager is None:
        return []
    query = (
        "SELECT logger_name, level, message, created_at, data_json "
        "FROM runtime_logs WHERE created_at >= ?"
    )
    params: list[object] = [cutoff]
    if workspace_id is not None:
        query += " AND workspace_id = ?"
        params.append(workspace_id)
    if logger_name is not None:
        query += " AND logger_name = ?"
        params.append(logger_name)
    if level is not None:
        query += " AND level = ?"
        params.append(level)
    return _fetchall_rows(db_manager, query, params)


def list_provider_policy_event_rows_since(db_manager, *, cutoff: float, workspace_id: str | None):
    if db_manager is None:
        return []
    query = (
        "SELECT task_name, task_id, event_kind, warning_kind, provider_key, provider_name, model_name, route_key, candidate_routes_json, reason_category, reason_code, retry_delay_seconds, warning_suppressed, created_at "
        "FROM provider_policy_events WHERE created_at >= ?"
    )
    params: list[object] = [cutoff]
    if workspace_id is not None:
        query += " AND workspace_id = ?"
        params.append(workspace_id)
    return _fetchall_rows(db_manager, query, params)


def list_memory_tool_event_rows_since(db_manager, *, cutoff: float, workspace_id: str | None):
    if db_manager is None:
        return []
    query = (
        "SELECT invocation_id, workspace_id, caller_kind, event_kind, memory_id, query_text, result_rank, result_count, duration_ms, created_at "
        "FROM memory_tool_events WHERE created_at >= ?"
    )
    params: list[object] = [cutoff]
    if workspace_id is not None:
        query += " AND workspace_id = ?"
        params.append(workspace_id)
    query += " ORDER BY created_at DESC, id DESC"
    return _fetchall_rows(db_manager, query, params)


def list_scoped_memory_rows(db_manager, workspace_id: str | None):
    if db_manager is None:
        return []
    query = (
        "SELECT memories.id, memories.title, memories.summary, memories.type, memories.status, memories.created_at, memories.updated_at, memories.metadata, "
        "memories.last_accessed_at, memories.last_surfaced_at, LENGTH(COALESCE(memories.content, '')) AS content_bytes, "
        "COALESCE(workspace_agg.workspace_ids, '') AS workspace_ids_csv, COALESCE(tag_agg.tags, '') AS tags_csv "
        "FROM memories "
        "LEFT JOIN ("
        "SELECT memory_id, GROUP_CONCAT(DISTINCT workspace_id) AS workspace_ids "
        "FROM memory_workspaces GROUP BY memory_id"
        ") workspace_agg ON workspace_agg.memory_id = memories.id "
        "LEFT JOIN ("
        "SELECT memory_tags.memory_id, GROUP_CONCAT(DISTINCT tags.name) AS tags "
        "FROM memory_tags JOIN tags ON tags.id = memory_tags.tag_id GROUP BY memory_tags.memory_id"
        ") tag_agg ON tag_agg.memory_id = memories.id"
    )
    params: list[object] = []
    if workspace_id is not None:
        query += (
            " WHERE EXISTS (SELECT 1 FROM memory_workspaces WHERE memory_workspaces.memory_id = memories.id "
            "AND memory_workspaces.workspace_id = ?)"
        )
        params.append(workspace_id)
    return _fetchall_rows(db_manager, query, params)


def list_scoped_link_rows(db_manager, workspace_id: str | None, memory_ids: set[str]):
    if db_manager is None or not memory_ids:
        return []
    if workspace_id is None:
        return _fetchall_rows(db_manager, "SELECT source_id, target_id, type FROM links")

    placeholders = ",".join("?" for _ in memory_ids)
    params = [*memory_ids, *memory_ids]
    query = (
        f"SELECT source_id, target_id, type FROM links WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})"
    )
    return _fetchall_rows(db_manager, query, params)


def build_queue_diagnostics(task_queue, workspace_id: str | None, limit: int = 8, now: float | None = None) -> list[QueueDiagnosticPayload]:
    current_time = now if now is not None else time.time()
    pending_tasks = task_queue.list_tasks(
        status="pending",
        workspace_id=workspace_id,
        limit=200,
    )
    ordered = sorted(
        pending_tasks,
        key=lambda task: (
            0 if task.available_at <= current_time else 1,
            task.priority,
            task.available_at,
            task.created_at,
        ),
    )
    diagnostics: list[QueueDiagnosticPayload] = []
    for task in ordered[:limit]:
        runnable = task.available_at <= current_time
        diagnostics.append(
            QueueDiagnosticPayload(
                task_id=task.id,
                task_name=task.task_name,
                workspace_id=task.workspace_id,
                priority=task.priority,
                pending_state="runnable" if runnable else "scheduled",
                trigger=task.data.get("trigger") if isinstance(task.data.get("trigger"), str) else None,
                created_at=task.created_at,
                available_at=task.available_at,
                age_seconds=max(current_time - task.created_at, 0.0),
                ready_in_seconds=0.0 if runnable else max(task.available_at - current_time, 0.0),
                overdue_seconds=max(current_time - task.available_at, 0.0) if runnable else 0.0,
            )
        )
    return diagnostics


def row_int(row: Any, key: str) -> int:
    if row is None:
        return 0
    value = row[key]
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        return int(value)
    raise TypeError(f"Expected int-compatible value for {key!r}, got {type(value)!r}")


def split_csv_values(value: str | None) -> list[str]:
    if not value:
        return []
    return [item for item in value.split(",") if item]
