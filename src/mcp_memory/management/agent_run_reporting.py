from __future__ import annotations

import time

from mcp_memory.core.task_results import TaskRunResult
from mcp_memory.core.task_handlers import TRIGGERABLE_BACKGROUND_TASK_NAMES
from mcp_memory.management.models import (
    AgentRunHistoryPayload,
    AgentRunPayload,
    IngestAuditPayload,
    RunResultMetadataPayload,
)
from mcp_memory.management.query_runner import ManagementQueryRunner
from mcp_memory.management.reporting_rows import AgentRunHistoryRow, adapt_agent_run_history_row
from mcp_memory.management.result_views import (
    JsonObject,
    TaskResultSource,
    coerce_task_result_view,
    decode_task_result_payload,
)


def build_agent_runs(task_queue, workspace_id: str | None) -> list[AgentRunPayload]:
    now = time.time()
    running_tasks = task_queue.list_tasks(
        status="running",
        workspace_id=workspace_id,
        limit=200,
    )
    pending_tasks = task_queue.list_tasks(
        status="pending",
        workspace_id=workspace_id,
        limit=200,
    )
    running_by_name: dict[str, int] = {}
    for task in running_tasks:
        running_by_name[task.task_name] = running_by_name.get(task.task_name, 0) + 1

    next_pending_by_name: dict[str, float] = {}
    for task in pending_tasks:
        current = next_pending_by_name.get(task.task_name)
        if current is None or task.available_at < current:
            next_pending_by_name[task.task_name] = task.available_at

    summaries = task_queue.summarize_task_runs(
        list(TRIGGERABLE_BACKGROUND_TASK_NAMES),
        workspace_id=workspace_id,
    )
    payloads: list[AgentRunPayload] = []
    for summary in summaries:
        last_result = summary.last_result
        result_view = coerce_task_result_view(last_result)
        seconds_since_last_completion = None
        if summary.last_completed_at is not None:
            seconds_since_last_completion = max(now - summary.last_completed_at, 0.0)
        next_available_at = next_pending_by_name.get(summary.task_name)
        seconds_until_next_run = None
        if next_available_at is not None:
            seconds_until_next_run = max(next_available_at - now, 0.0)
        payloads.append(
            AgentRunPayload(
                task_name=summary.task_name,
                running_count=running_by_name.get(summary.task_name, 0),
                total_runs=summary.total_runs,
                completed_runs=summary.completed_runs,
                failed_runs=summary.failed_runs,
                cancelled_runs=summary.cancelled_runs,
                retry_runs=summary.retry_runs,
                avg_duration_seconds=summary.avg_duration_seconds,
                total_lines_compressed=summary.total_lines_compressed,
                last_status=summary.last_status,
                last_completed_at=summary.last_completed_at,
                seconds_since_last_completion=seconds_since_last_completion,
                last_error=summary.last_error,
                last_result_summary=last_result.summary if isinstance(last_result, TaskRunResult) else result_view.summary,
                last_result_metadata=result_view.metadata_copy(),
                last_ingest_audit=result_view.compact_ingest_audit(),
                next_available_at=next_available_at,
                seconds_until_next_run=seconds_until_next_run,
            )
        )
    return payloads


def build_recent_agent_runs(
    db_manager,
    workspace_id: str | None,
    limit: int = 20,
    *,
    detail_level: str = "compact",
) -> list[AgentRunHistoryPayload]:
    rows = [] if db_manager is None else _fetch_recent_agent_run_rows(db_manager, workspace_id=workspace_id, limit=limit)
    payloads: list[AgentRunHistoryPayload] = []
    for row in rows:
        payloads.append(
            build_agent_run_history_payload(
                task_id=row.task_id,
                task_name=row.task_name,
                status=row.status,
                started_at=row.started_at,
                completed_at=row.completed_at,
                duration_seconds=row.duration_seconds,
                error_text=row.error_text,
                result=row.result,
                detail_level=detail_level,
            )
        )
    return payloads


def _fetch_recent_agent_run_rows(db_manager, *, workspace_id: str | None, limit: int) -> list[AgentRunHistoryRow]:
    placeholders = ",".join("?" for _ in TRIGGERABLE_BACKGROUND_TASK_NAMES)
    query = (
        f"SELECT * FROM task_runs WHERE task_name IN ({placeholders})"
        + (" AND workspace_id = ?" if workspace_id is not None else "")
        + " ORDER BY completed_at DESC, started_at DESC LIMIT ?"
    )
    params: list[object] = [
        *TRIGGERABLE_BACKGROUND_TASK_NAMES,
        *([workspace_id] if workspace_id is not None else []),
        limit,
    ]
    rows = ManagementQueryRunner(db_manager).fetchall(query, params)
    return [adapt_agent_run_history_row(row) for row in rows]


def build_agent_run_history_payload(
    *,
    task_id: str,
    task_name: str,
    status: str,
    started_at: float,
    completed_at: float,
    duration_seconds: float,
    error_text: str | None,
    result: TaskResultSource,
    detail_level: str = "compact",
) -> AgentRunHistoryPayload:
    include_full_result = detail_level == "full"
    result_view = coerce_task_result_view(result)
    result_summary = result.summary if isinstance(result, TaskRunResult) else result_view.summary
    return AgentRunHistoryPayload(
        task_id=task_id,
        task_name=task_name,
        status=status,
        started_at=started_at,
        completed_at=completed_at,
        duration_seconds=duration_seconds,
        error_text=error_text,
        result_summary=result_summary,
        result_metadata=result_view.metadata_copy(),
        ingest_audit=result_view.full_ingest_audit() if include_full_result else result_view.compact_ingest_audit(),
        result=result_view.raw_payload if include_full_result else None,
    )


def format_result_summary(result: TaskResultSource) -> str | None:
    if isinstance(result, TaskRunResult):
        return result.summary
    return coerce_task_result_view(result).summary


def extract_run_result_metadata(result: TaskResultSource) -> RunResultMetadataPayload:
    return coerce_task_result_view(result).metadata_copy()


def extract_ingest_audit(
    result: TaskResultSource,
    *,
    include_entries: bool = False,
) -> IngestAuditPayload:
    result_view = coerce_task_result_view(result)
    return result_view.full_ingest_audit() if include_entries else result_view.compact_ingest_audit()


def decode_run_result(raw_result: object) -> JsonObject:
    return decode_task_result_payload(raw_result)
