from __future__ import annotations

from typing import Any, TypedDict

from mcp_memory.context import ApplicationContext, TaskQueueContext
from mcp_memory.mcp.internal_ingest_keys import (
    INGEST_ENTRY_DISPOSITIONS_TASK_DATA_KEY,
    INGEST_HANDLED_ENTRY_IDS_TASK_DATA_KEY,
    INGEST_TOUCHED_MEMORY_IDS_TASK_DATA_KEY,
    INGEST_TOOL_INVOCATIONS_TASK_DATA_KEY,
)


class IngestRecordedToolUsage(TypedDict):
    tool_calls_executed: int
    mutations: int
    tool_names_used: list[str]


class IngestRecordedRunMetadata(TypedDict):
    handled_entry_ids: list[int]
    entry_dispositions: list[dict[str, Any]]
    touched_memory_ids: list[str]
    tool_usage: IngestRecordedToolUsage


def _normalize_ingest_entry_ids(arguments: dict[str, Any], field_name: str) -> list[int]:
    raw_entry_ids = arguments.get(field_name)
    if not isinstance(raw_entry_ids, list) or not raw_entry_ids:
        raise ValueError(f"{field_name} must contain at least one entry id")

    normalized: list[int] = []
    seen: set[int] = set()
    for item in raw_entry_ids:
        value: int | None = None
        if isinstance(item, bool):
            value = None
        elif isinstance(item, int):
            value = item
        elif isinstance(item, str) and item.strip().isdigit():
            value = int(item.strip())
        if value is None or value <= 0 or value in seen:
            if value is None or value <= 0:
                raise ValueError(f"{field_name} must contain only positive integer ids")
            continue
        seen.add(value)
        normalized.append(value)
    if not normalized:
        raise ValueError(f"{field_name} must contain at least one entry id")
    return normalized


def _journal_entry_payload(entry) -> dict[str, Any]:
    return {
        "id": entry.id,
        "content": entry.content,
        "workspace_id": entry.workspace_id,
        "timestamp": entry.timestamp,
        "status": entry.status,
    }


def _reset_recorded_ingest_handled_entry_ids(ctx: TaskQueueContext, task_id: str) -> None:
    if ctx.task_queue is None:
        return
    try:
        ctx.task_queue.clear_running_task_data_keys(
            task_id,
            field_names=[
                INGEST_HANDLED_ENTRY_IDS_TASK_DATA_KEY,
                INGEST_ENTRY_DISPOSITIONS_TASK_DATA_KEY,
                INGEST_TOUCHED_MEMORY_IDS_TASK_DATA_KEY,
                INGEST_TOOL_INVOCATIONS_TASK_DATA_KEY,
            ],
        )
    except ValueError:
        return


def _recorded_ingest_run_metadata(ctx: TaskQueueContext, task_id: str) -> IngestRecordedRunMetadata:
    return {
        "handled_entry_ids": _recorded_ingest_handled_entry_ids(ctx, task_id),
        "entry_dispositions": _recorded_ingest_entry_dispositions(ctx, task_id),
        "touched_memory_ids": _recorded_ingest_touched_memory_ids(ctx, task_id),
        "tool_usage": _recorded_ingest_tool_usage(ctx, task_id),
    }


def _recorded_ingest_tool_usage(ctx: TaskQueueContext, task_id: str) -> IngestRecordedToolUsage:
    empty_usage: IngestRecordedToolUsage = {
        "tool_calls_executed": 0,
        "mutations": 0,
        "tool_names_used": [],
    }
    if ctx.task_queue is None:
        return empty_usage
    try:
        task = ctx.task_queue.get_task(task_id)
    except ValueError:
        return empty_usage

    raw_invocations = task.data.get(INGEST_TOOL_INVOCATIONS_TASK_DATA_KEY)
    if not isinstance(raw_invocations, list):
        return empty_usage

    normalized_invocations: list[dict[str, Any]] = []
    for item in raw_invocations:
        if not isinstance(item, dict):
            continue
        tool_name = item.get("tool_name")
        if not isinstance(tool_name, str) or not tool_name.strip():
            continue
        normalized_invocations.append(
            {
                "tool_name": tool_name.strip(),
                "mutation": bool(item.get("mutation")),
            }
        )

    return {
        "tool_calls_executed": len(normalized_invocations),
        "mutations": sum(1 for item in normalized_invocations if item["mutation"]),
        "tool_names_used": sorted({item["tool_name"] for item in normalized_invocations}),
    }


def _recorded_ingest_handled_entry_ids(ctx: TaskQueueContext, task_id: str) -> list[int]:
    if ctx.task_queue is None:
        return []
    try:
        task = ctx.task_queue.get_task(task_id)
    except ValueError:
        return []
    raw_entry_ids = task.data.get(INGEST_HANDLED_ENTRY_IDS_TASK_DATA_KEY)
    if not isinstance(raw_entry_ids, list):
        return []
    return [
        entry_id
        for entry_id in raw_entry_ids
        if isinstance(entry_id, int) and not isinstance(entry_id, bool) and entry_id > 0
    ]


def _recorded_ingest_entry_dispositions(ctx: TaskQueueContext, task_id: str) -> list[dict[str, Any]]:
    if ctx.task_queue is None:
        return []
    try:
        task = ctx.task_queue.get_task(task_id)
    except ValueError:
        return []
    raw_entry_dispositions = task.data.get(INGEST_ENTRY_DISPOSITIONS_TASK_DATA_KEY)
    if not isinstance(raw_entry_dispositions, list):
        return []

    normalized: list[dict[str, Any]] = []
    for item in raw_entry_dispositions:
        if not isinstance(item, dict):
            continue
        entry_id = item.get("entry_id")
        if not isinstance(entry_id, int) or isinstance(entry_id, bool) or entry_id <= 0:
            continue
        disposition = item.get("disposition")
        if not isinstance(disposition, str) or not disposition.strip():
            continue
        normalized.append(
            {
                "entry_id": entry_id,
                "disposition": disposition.strip(),
                **({"memory_id": item["memory_id"].strip()} if isinstance(item.get("memory_id"), str) and item["memory_id"].strip() else {}),
                **({"memory_title": item["memory_title"].strip()} if isinstance(item.get("memory_title"), str) and item["memory_title"].strip() else {}),
                **({"reason": item["reason"].strip()} if isinstance(item.get("reason"), str) and item["reason"].strip() else {}),
            }
        )
    return normalized


def _recorded_ingest_touched_memory_ids(ctx: TaskQueueContext, task_id: str) -> list[str]:
    if ctx.task_queue is None:
        return []
    try:
        task = ctx.task_queue.get_task(task_id)
    except ValueError:
        return []
    raw_memory_ids = task.data.get(INGEST_TOUCHED_MEMORY_IDS_TASK_DATA_KEY)
    if not isinstance(raw_memory_ids, list):
        return []
    return sorted(
        {
            memory_id.strip()
            for item in raw_memory_ids
            for memory_id in [item if isinstance(item, str) else item.get("memory_id") if isinstance(item, dict) else None]
            if isinstance(memory_id, str) and memory_id.strip()
        }
    )


def _record_successful_ingest_entry_ids(ctx: TaskQueueContext, *, task_id: str, entry_ids: list[int]) -> None:
    if ctx.task_queue is None:
        return
    try:
        ctx.task_queue.extend_running_task_data_int_list(
            task_id,
            field_name=INGEST_HANDLED_ENTRY_IDS_TASK_DATA_KEY,
            values=entry_ids,
        )
    except ValueError:
        return


def _record_successful_ingest_entry_dispositions(
    ctx: TaskQueueContext,
    *,
    task_id: str,
    entry_dispositions: list[dict[str, Any]],
) -> None:
    if ctx.task_queue is None:
        return
    try:
        ctx.task_queue.extend_running_task_data_object_list(
            task_id,
            field_name=INGEST_ENTRY_DISPOSITIONS_TASK_DATA_KEY,
            values=entry_dispositions,
        )
    except ValueError:
        return


def _record_ingest_tool_invocation(
    ctx: TaskQueueContext,
    *,
    task_id: str,
    tool_name: str,
    mutation: bool,
) -> None:
    if ctx.task_queue is None:
        return
    try:
        ctx.task_queue.extend_running_task_data_object_list(
            task_id,
            field_name=INGEST_TOOL_INVOCATIONS_TASK_DATA_KEY,
            values=[
                {
                    "tool_name": tool_name,
                    "mutation": mutation,
                }
            ],
        )
    except ValueError:
        return


def _record_touched_memory_ids(ctx: TaskQueueContext, *, task_id: str, memory_ids: list[str]) -> None:
    if ctx.task_queue is None:
        return
    normalized_memory_ids = [memory_id for memory_id in memory_ids if isinstance(memory_id, str) and memory_id.strip()]
    if not normalized_memory_ids:
        return
    try:
        ctx.task_queue.extend_running_task_data_object_list(
            task_id,
            field_name=INGEST_TOUCHED_MEMORY_IDS_TASK_DATA_KEY,
            values=[{"memory_id": memory_id.strip()} for memory_id in normalized_memory_ids],
        )
    except ValueError:
        return


def _finalize_claimed_ingest_entries(
    ctx: ApplicationContext,
    *,
    task_id: str,
    handled_entry_ids: list[int],
) -> tuple[list[int], list[int], list[int]]:
    if ctx.journal is None:
        return [], [], []
    claimed_ids = ctx.journal.get_claimed_entry_ids(task_id)
    claimed_entry_id_set = set(claimed_ids)
    handled_claimed_entry_ids = [entry_id for entry_id in handled_entry_ids if entry_id in claimed_entry_id_set]
    recoverable_ids = ctx.journal.move_claimed_entry_ids_to_recoverable(task_id, handled_claimed_entry_ids)
    recoverable_entry_id_set = set(recoverable_ids)
    released_ids = ctx.journal.release_claimed_entry_ids(
        task_id,
        [entry_id for entry_id in claimed_ids if entry_id not in recoverable_entry_id_set],
    )
    return claimed_ids, recoverable_ids, released_ids