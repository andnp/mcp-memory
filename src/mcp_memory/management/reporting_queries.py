from __future__ import annotations

from typing import Any
import time

from mcp_memory.core.task_handlers import MAINTENANCE_TASK_NAMES
from mcp_memory.management.models import QueueDiagnosticPayload


def fetch_memory_count_rows(db_manager, workspace_id: str | None):
    if db_manager is None:
        return []
    conn = db_manager.get_connection()
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
    return conn.execute(query, params).fetchall()


def fetch_task_count_rows(db_manager, workspace_id: str | None):
    if db_manager is None:
        return []
    conn = db_manager.get_connection()
    query = "SELECT status, COUNT(*) AS count FROM tasks"
    params: list[object] = []
    if workspace_id is not None:
        query += " WHERE workspace_id = ?"
        params.append(workspace_id)
    query += " GROUP BY status"
    return conn.execute(query, params).fetchall()


def fetch_running_task_attempt_rows(db_manager, workspace_id: str | None):
    if db_manager is None:
        return []
    conn = db_manager.get_connection()
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
    return conn.execute(query, params).fetchall()


def fetch_memory_metrics_row(db_manager, workspace_id: str | None):
    if db_manager is None:
        return None
    conn = db_manager.get_connection()
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
    return conn.execute(query, params).fetchone()


def fetch_pending_journal_metrics_row(db_manager, workspace_id: str | None):
    if db_manager is None:
        return None
    conn = db_manager.get_connection()
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
    return conn.execute(query, params).fetchone()


def list_task_run_rows_since(db_manager, *, cutoff: float, workspace_id: str | None):
    if db_manager is None:
        return []
    conn = db_manager.get_connection()
    query = "SELECT status, completed_at, duration_seconds, result_json FROM task_runs WHERE completed_at >= ?"
    params: list[object] = [cutoff]
    if workspace_id is not None:
        query += " AND workspace_id = ?"
        params.append(workspace_id)
    return conn.execute(query, params).fetchall()


def list_maintenance_task_run_rows_since(db_manager, *, cutoff: float, workspace_id: str | None):
    if db_manager is None:
        return []
    conn = db_manager.get_connection()
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
    return conn.execute(query, params).fetchall()


def list_provider_usage_rows_since(db_manager, *, cutoff: float, workspace_id: str | None):
    if db_manager is None:
        return []
    conn = db_manager.get_connection()
    query = (
        "SELECT task_name, provider_key, provider_name, model_name, status, duration_seconds, created_at, reason_category, reason_code, retry_delay_seconds "
        "FROM provider_usage WHERE created_at >= ?"
    )
    params: list[object] = [cutoff]
    if workspace_id is not None:
        query += " AND workspace_id = ?"
        params.append(workspace_id)
    return conn.execute(query, params).fetchall()


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
    conn = db_manager.get_connection()
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
    return conn.execute(query, params).fetchall()


def list_provider_policy_event_rows_since(db_manager, *, cutoff: float, workspace_id: str | None):
    if db_manager is None:
        return []
    conn = db_manager.get_connection()
    query = (
        "SELECT task_name, task_id, event_kind, warning_kind, provider_key, provider_name, model_name, route_key, candidate_routes_json, reason_category, reason_code, retry_delay_seconds, warning_suppressed, created_at "
        "FROM provider_policy_events WHERE created_at >= ?"
    )
    params: list[object] = [cutoff]
    if workspace_id is not None:
        query += " AND workspace_id = ?"
        params.append(workspace_id)
    return conn.execute(query, params).fetchall()


def list_memory_tool_event_rows_since(db_manager, *, cutoff: float, workspace_id: str | None):
    if db_manager is None:
        return []
    conn = db_manager.get_connection()
    query = (
        "SELECT invocation_id, workspace_id, caller_kind, event_kind, memory_id, query_text, result_rank, result_count, created_at "
        "FROM memory_tool_events WHERE created_at >= ?"
    )
    params: list[object] = [cutoff]
    if workspace_id is not None:
        query += " AND workspace_id = ?"
        params.append(workspace_id)
    query += " ORDER BY created_at DESC, id DESC"
    return conn.execute(query, params).fetchall()


def list_scoped_memory_rows(db_manager, workspace_id: str | None):
    if db_manager is None:
        return []
    conn = db_manager.get_connection()
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
    return conn.execute(query, params).fetchall()


def list_scoped_link_rows(db_manager, workspace_id: str | None, memory_ids: set[str]):
    if db_manager is None or not memory_ids:
        return []
    conn = db_manager.get_connection()
    if workspace_id is None:
        return conn.execute("SELECT source_id, target_id, type FROM links").fetchall()

    placeholders = ",".join("?" for _ in memory_ids)
    params = [*memory_ids, *memory_ids]
    query = (
        f"SELECT source_id, target_id, type FROM links WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})"
    )
    return conn.execute(query, params).fetchall()


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
    return 0 if row is None else int(row[key])


def split_csv_values(value: str | None) -> list[str]:
    if not value:
        return []
    return [item for item in value.split(",") if item]
