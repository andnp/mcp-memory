from __future__ import annotations

import json
import time
from typing import Any

from mcp_memory.core.task_handlers import TRIGGERABLE_BACKGROUND_TASK_NAMES
from mcp_memory.management.models import (
    AgentRunHistoryPayload,
    AgentRunPayload,
    IngestAuditPayload,
    IngestEntryDispositionPayload,
    RunResultMetadataPayload,
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
                last_result_summary=format_result_summary(summary.last_result),
                last_result_metadata=extract_run_result_metadata(summary.last_result),
                last_ingest_audit=extract_ingest_audit(summary.last_result),
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
    rows = [] if db_manager is None else db_manager.get_connection().execute(
        f"SELECT * FROM task_runs WHERE task_name IN ({','.join('?' for _ in TRIGGERABLE_BACKGROUND_TASK_NAMES)})"
        + (" AND workspace_id = ?" if workspace_id is not None else "")
        + " ORDER BY completed_at DESC, started_at DESC LIMIT ?",
        [
            *TRIGGERABLE_BACKGROUND_TASK_NAMES,
            *([workspace_id] if workspace_id is not None else []),
            limit,
        ],
    ).fetchall()
    payloads: list[AgentRunHistoryPayload] = []
    for row in rows:
        result = decode_run_result(row["result_json"])
        payloads.append(
            build_agent_run_history_payload(
                task_id=str(row["task_id"]),
                task_name=str(row["task_name"]),
                status=str(row["status"]),
                started_at=float(row["started_at"]),
                completed_at=float(row["completed_at"]),
                duration_seconds=float(row["duration_seconds"] or 0.0),
                error_text=row["error_text"],
                result=result,
                detail_level=detail_level,
            )
        )
    return payloads


def build_agent_run_history_payload(
    *,
    task_id: str,
    task_name: str,
    status: str,
    started_at: float,
    completed_at: float,
    duration_seconds: float,
    error_text: str | None,
    result: dict[str, object],
    detail_level: str = "compact",
) -> AgentRunHistoryPayload:
    include_full_result = detail_level == "full"
    return AgentRunHistoryPayload(
        task_id=task_id,
        task_name=task_name,
        status=status,
        started_at=started_at,
        completed_at=completed_at,
        duration_seconds=duration_seconds,
        error_text=error_text,
        result_summary=format_result_summary(result),
        result_metadata=extract_run_result_metadata(result),
        ingest_audit=extract_ingest_audit(result, include_entries=include_full_result),
        result=result if include_full_result else None,
    )


def format_result_summary(result: dict[str, object]) -> str | None:
    if not result:
        return None
    preferred_keys = (
        "created",
        "merged",
        "updated",
        "archived",
        "absorbed_observations",
        "degraded",
        "restored",
        "deleted_tasks",
        "deleted_journal_entries",
        "claimed_entry_ids",
        "recoverable_entry_ids",
        "deleted_entry_ids",
        "released_entry_ids",
        "meaningful_actions",
        "processed_entry_ids",
        "created_memory_ids",
        "lines_compressed",
        "requested_strategy",
        "strategy_used",
        "strategy_fallback_reason",
        "candidate_count",
        "sampled_memory_ids",
        "requested_grouping_strategy",
        "grouping_strategy_used",
        "grouping_fallback_reason",
        "group_count",
    )
    formatted_parts: list[str] = []
    for key in preferred_keys:
        if key not in result:
            continue
        value = result[key]
        if isinstance(value, list):
            formatted_parts.append(f"{key}={len(value)}")
        else:
            formatted_parts.append(f"{key}={value}")

    if formatted_parts:
        return ", ".join(formatted_parts)

    for key in sorted(result):
        value = result[key]
        if isinstance(value, (str, int, float, bool)):
            formatted_parts.append(f"{key}={value}")
    return ", ".join(formatted_parts) if formatted_parts else None


def extract_run_result_metadata(result: dict[str, object]) -> RunResultMetadataPayload:
    sampled_memory_ids = result.get("sampled_memory_ids")
    return RunResultMetadataPayload(
        requested_strategy=_coerce_str(result.get("requested_strategy")),
        strategy_used=_coerce_str(result.get("strategy_used")),
        strategy_fallback_reason=_coerce_str(result.get("strategy_fallback_reason")),
        candidate_count=_coerce_int(result.get("candidate_count")),
        sampled_memory_ids=[str(item) for item in sampled_memory_ids] if isinstance(sampled_memory_ids, list) else [],
        requested_grouping_strategy=_coerce_str(result.get("requested_grouping_strategy")),
        grouping_strategy_used=_coerce_str(result.get("grouping_strategy_used")),
        grouping_fallback_reason=_coerce_str(result.get("grouping_fallback_reason")),
        group_count=_coerce_int(result.get("group_count")),
    )


def extract_ingest_audit(
    result: dict[str, object],
    *,
    include_entries: bool = False,
) -> IngestAuditPayload:
    if not _contains_ingest_audit(result):
        return IngestAuditPayload()

    created_memory_ids = _coerce_str_list(result.get("created_memory_ids"))
    touched_memory_ids = _coerce_str_list(result.get("touched_memory_ids"))
    appended_memory_ids = _coerce_str_list(result.get("appended_memory_ids"))
    matched_memory_ids = _coerce_str_list(result.get("matched_memory_ids"))
    provider_reported_touched_memory_ids = _coerce_str_list(result.get("provider_reported_touched_memory_ids"))
    provider_reported_matched_memory_ids = _coerce_str_list(result.get("provider_reported_matched_memory_ids"))
    entry_dispositions = _coerce_ingest_entry_dispositions(result.get("entry_dispositions")) if include_entries else []
    provider_reported_entry_outcomes = (
        _coerce_ingest_entry_dispositions(result.get("provider_reported_entry_outcomes")) if include_entries else []
    )
    processed_count = _result_list_count(result, "processed_entry_ids")
    handled_count = processed_count or _result_list_count(result, "recoverable_entry_ids")
    return IngestAuditPayload(
        claimed_count=_result_list_count(result, "claimed_entry_ids"),
        handled_count=handled_count,
        released_count=_result_list_count(result, "released_entry_ids"),
        deleted_count=_result_list_count(result, "deleted_entry_ids"),
        processed_count=processed_count,
        meaningful_actions=_coerce_int(result.get("meaningful_actions")) or 0,
        created_count=len(created_memory_ids),
        touched_count=len(touched_memory_ids),
        appended_count=len(appended_memory_ids),
        matched_count=len(matched_memory_ids),
        tool_calls_executed=_coerce_int(result.get("tool_calls_executed")),
        mutations=_coerce_int(result.get("mutations")),
        provider_reported_tool_calls=_coerce_int(result.get("provider_reported_tool_calls")),
        provider_reported_mutations=_coerce_int(result.get("provider_reported_mutations")),
        created_memory_ids=created_memory_ids,
        touched_memory_ids=touched_memory_ids,
        appended_memory_ids=appended_memory_ids,
        matched_memory_ids=matched_memory_ids,
        provider_reported_touched_memory_ids=provider_reported_touched_memory_ids,
        provider_reported_matched_memory_ids=provider_reported_matched_memory_ids,
        entry_dispositions=entry_dispositions,
        provider_reported_entry_outcomes=provider_reported_entry_outcomes,
    )


def _coerce_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _coerce_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _coerce_str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str) and item]


def _coerce_ingest_entry_dispositions(value: object) -> list[IngestEntryDispositionPayload]:
    if not isinstance(value, list):
        return []
    payloads: list[IngestEntryDispositionPayload] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        entry_id = item.get("entry_id")
        disposition = item.get("disposition")
        if not isinstance(entry_id, int) or isinstance(entry_id, bool) or not isinstance(disposition, str):
            continue
        payloads.append(
            IngestEntryDispositionPayload(
                entry_id=entry_id,
                disposition=disposition,
                finalization_status=_coerce_str(item.get("finalization_status")),
                memory_id=_coerce_str(item.get("memory_id")),
                memory_title=_coerce_str(item.get("memory_title")),
                reason=_coerce_str(item.get("reason")),
            )
        )
    return payloads


def _contains_ingest_audit(result: dict[str, Any]) -> bool:
    ingest_keys = {
        "claimed_entry_ids",
        "released_entry_ids",
        "deleted_entry_ids",
        "processed_entry_ids",
        "entry_dispositions",
        "touched_memory_ids",
        "appended_memory_ids",
        "matched_memory_ids",
        "provider_reported_tool_calls",
        "provider_reported_mutations",
        "provider_reported_entry_outcomes",
    }
    return any(key in result for key in ingest_keys)


def _result_list_count(result: dict[str, Any], key: str) -> int:
    value = result.get(key)
    return len(value) if isinstance(value, list) else 0


def decode_run_result(raw_result: object) -> dict[str, object]:
    if not isinstance(raw_result, str) or not raw_result.strip():
        return {}
    try:
        decoded = json.loads(raw_result)
    except ValueError:
        return {}
    return decoded if isinstance(decoded, dict) else {}
