from __future__ import annotations

from typing import Any

from mcp_memory.context import ApplicationContext
from mcp_memory.core.system1_scheduling import resolve_pending_workspace_id
from mcp_memory.mcp.internal_service_support import (
    _append_content,
    _enqueue_summary_task,
    _merge_memory_metadata,
    _normalize_tags,
)
from mcp_memory.mcp.validation import (
    optional_object,
    optional_positive_int,
    optional_string,
    require_string,
    string_list,
    validate_ingest_mutation_payload,
)
from mcp_memory.serialization import memory_record_payload


INGEST_HANDLED_ENTRY_IDS_TASK_DATA_KEY = "ingest_handled_entry_ids"
INGEST_ENTRY_DISPOSITIONS_TASK_DATA_KEY = "ingest_entry_dispositions"
INGEST_TOOL_INVOCATIONS_TASK_DATA_KEY = "ingest_tool_invocations"


def internal_get_next_ingest_batch_service(ctx: ApplicationContext, arguments: dict) -> dict:
    if ctx.journal is None:
        return {"status": "error", "error": "journal_not_initialized"}

    from mcp_memory.core.task_handlers.ingest import _build_ingest_groups, _resolve_grouping_strategy

    task_id = require_string(arguments, "task_id")
    journal_workspace_id = resolve_pending_workspace_id(
        ctx.journal,
        optional_string(arguments, "workspace_id") or ctx.workspace_id,
    )
    workspace_id = optional_string(arguments, "workspace_id") or ctx.workspace_id or "workspace-unknown"
    batch_size = optional_positive_int(arguments, "batch_size", 20)
    grouping_strategy_requested = optional_string(arguments, "grouping_strategy")
    grouping_strategy_used, grouping_fallback_reason = _resolve_grouping_strategy(
        ctx,
        requested_strategy=grouping_strategy_requested,
    )

    entries = ctx.journal.claim_pending(
        task_id=task_id,
        limit=batch_size,
        workspace_id=journal_workspace_id,
    )
    groups = _build_ingest_groups(
        ctx,
        entries,
        workspace_id,
        task_id=task_id,
        grouping_strategy=grouping_strategy_used,
    )
    pending_remaining = ctx.journal.count_by_status(workspace_id=journal_workspace_id).get("pending", 0)
    _record_ingest_tool_invocation(
        ctx,
        task_id=task_id,
        tool_name="internal_get_next_ingest_batch",
        mutation=False,
    )
    return {
        "status": "ok",
        "task_id": task_id,
        "requested_grouping_strategy": grouping_strategy_requested,
        "grouping_strategy_used": grouping_strategy_used,
        "grouping_fallback_reason": grouping_fallback_reason,
        "claimed_entry_ids": [entry.id for entry in entries],
        "pending_remaining": pending_remaining,
        "has_more": pending_remaining > 0,
        "group_count": len(groups),
        "groups": [
            {
                "group_index": index,
                "entries": [_journal_entry_payload(entry) for entry in group],
            }
            for index, group in enumerate(groups)
        ],
    }


def internal_append_to_existing_memory_for_ingest_service(ctx: ApplicationContext, arguments: dict) -> dict:
    if ctx.repository is None:
        return {"status": "error", "error": "repository_not_initialized"}

    try:
        memory_id = require_string(arguments, "memory_id")
        content = require_string(arguments, "content")
        task_id = require_string(arguments, "task_id")
        entry_ids = _normalize_ingest_entry_ids(arguments, "entry_ids")
        validate_ingest_mutation_payload(entry_ids=entry_ids, content=content)
    except (TypeError, ValueError) as exc:
        return {"status": "error", "error": "invalid_ingest_payload", "detail": str(exc)}
    record = ctx.repository.get_memory(memory_id)
    if record is None:
        return {"status": "error", "error": "memory_not_found"}
    if record.status != "active":
        return {"status": "error", "error": "memory_not_active"}

    tags = string_list(arguments, "tags") if "tags" in arguments else []
    workspace_ids = string_list(arguments, "workspace_ids") if "workspace_ids" in arguments else []
    metadata_override = optional_object(arguments, "metadata") or {}
    merged_metadata = _merge_memory_metadata(
        record.metadata,
        {
            **metadata_override,
            "appended_entry_ids": entry_ids,
            "ingest_task_id": task_id,
        },
    )
    updated = ctx.repository.update_memory(
        memory_id,
        content=_append_content(record.content, content),
        summary=optional_string(arguments, "summary") if "summary" in arguments else None,
        tags=_normalize_tags([*record.tags, *tags, "system1-appended"]),
        workspace_ids=sorted({*record.workspace_ids, *workspace_ids}),
        metadata=merged_metadata,
    )
    if updated is None:
        return {"status": "error", "error": "memory_not_found"}
    _record_successful_ingest_entry_ids(ctx, task_id=task_id, entry_ids=entry_ids)
    _record_successful_ingest_entry_dispositions(
        ctx,
        task_id=task_id,
        entry_dispositions=[
            {
                "entry_id": entry_id,
                "disposition": "appended",
                "memory_id": updated.id,
                "memory_title": updated.title,
            }
            for entry_id in entry_ids
        ],
    )
    _record_ingest_tool_invocation(
        ctx,
        task_id=task_id,
        tool_name="internal_ingest_append_memory",
        mutation=True,
    )
    return {"status": "ok", "record": memory_record_payload(updated), "handled_entry_ids": entry_ids}


def internal_create_memory_record_for_ingest_service(ctx: ApplicationContext, arguments: dict) -> dict:
    if ctx.repository is None:
        return {"status": "error", "error": "repository_not_initialized"}

    try:
        task_id = require_string(arguments, "task_id")
        entry_ids = _normalize_ingest_entry_ids(arguments, "entry_ids")
        title = require_string(arguments, "title")
        content = require_string(arguments, "content")
        validate_ingest_mutation_payload(entry_ids=entry_ids, content=content, title=title)
    except (TypeError, ValueError) as exc:
        return {"status": "error", "error": "invalid_ingest_payload", "detail": str(exc)}
    workspace_ids = string_list(arguments, "workspace_ids") or ([ctx.workspace_id] if ctx.workspace_id is not None else ["workspace-unknown"])
    metadata_override = optional_object(arguments, "metadata") or {}
    record = ctx.repository.create_memory(
        title=title,
        content=content,
        summary=optional_string(arguments, "summary"),
        memory_type=optional_string(arguments, "memory_type") or "observation",
        status=optional_string(arguments, "status") or "active",
        workspace_ids=workspace_ids,
        tags=_normalize_tags([*string_list(arguments, "tags"), "auto-ingested", "system1"]),
        metadata={
            **metadata_override,
            "source_entry_ids": entry_ids,
            "ingest_task_id": task_id,
        },
    )
    assert record is not None
    _enqueue_summary_task(ctx, record.id, list(record.workspace_ids))
    _record_successful_ingest_entry_ids(ctx, task_id=task_id, entry_ids=entry_ids)
    _record_successful_ingest_entry_dispositions(
        ctx,
        task_id=task_id,
        entry_dispositions=[
            {
                "entry_id": entry_id,
                "disposition": "created",
                "memory_id": record.id,
                "memory_title": record.title,
            }
            for entry_id in entry_ids
        ],
    )
    _record_ingest_tool_invocation(
        ctx,
        task_id=task_id,
        tool_name="internal_ingest_create_memory",
        mutation=True,
    )
    return {"status": "ok", "record": memory_record_payload(record), "handled_entry_ids": entry_ids}


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


def _record_successful_ingest_entry_ids(ctx: ApplicationContext, *, task_id: str, entry_ids: list[int]) -> None:
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
    ctx: ApplicationContext,
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
    ctx: ApplicationContext,
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