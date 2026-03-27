from __future__ import annotations

from mcp_memory.context import ApplicationContext
from mcp_memory.core.ingest_claim_lifecycle import (
    _journal_entry_payload,
    _normalize_ingest_entry_ids,
    _record_ingest_tool_invocation,
    _record_successful_ingest_entry_dispositions,
    _record_successful_ingest_entry_ids,
    _record_touched_memory_ids,
)
from mcp_memory.core.ingest_provenance import (
    build_ingest_appended_metadata,
    build_ingest_created_metadata,
)
from mcp_memory.core.system1_scheduling import resolve_pending_workspace_id
from mcp_memory.mcp.internal_service_support import (
    _append_content,
    _memory_write_quality_error,
    _memory_write_quality_warnings,
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

def internal_get_next_ingest_batch_service(ctx: ApplicationContext, arguments: dict) -> dict:
    if ctx.journal is None:
        return {"status": "error", "error": "journal_not_initialized"}

    from mcp_memory.core.task_handlers.ingest import _build_ingest_groups, _resolve_grouping_strategy

    task_id = require_string(arguments, "task_id")
    requested_workspace_id = optional_string(arguments, "workspace_id")
    journal_workspace_id = resolve_pending_workspace_id(
        ctx.journal,
        requested_workspace_id,
    )
    workspace_id = requested_workspace_id or ctx.workspace_id or "workspace-unknown"
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

    summary = optional_string(arguments, "summary") if "summary" in arguments else record.summary
    quality_error = _memory_write_quality_error(
        title=record.title,
        content=content,
        summary=optional_string(arguments, "summary"),
    )
    if quality_error is not None:
        return {"status": "error", "error": "low_value_memory_rejected", "detail": quality_error}

    tags = string_list(arguments, "tags") if "tags" in arguments else []
    workspace_ids = string_list(arguments, "workspace_ids") if "workspace_ids" in arguments else []
    metadata_override = optional_object(arguments, "metadata") or {}
    merged_content = _append_content(record.content, content)
    merged_tags = _normalize_tags([*record.tags, *tags])
    warnings = _memory_write_quality_warnings(
        summary=summary,
        memory_type=record.type,
        tags=merged_tags,
        content=merged_content,
    )
    merged_metadata = _merge_memory_metadata(
        record.metadata,
        metadata_override,
    )
    merged_metadata = build_ingest_appended_metadata(
        task_id=task_id,
        entry_ids=entry_ids,
        metadata=merged_metadata,
    )
    updated = ctx.repository.update_memory(
        memory_id,
        content=merged_content,
        summary=optional_string(arguments, "summary") if "summary" in arguments else None,
        tags=merged_tags,
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
    _record_touched_memory_ids(ctx, task_id=task_id, memory_ids=[updated.id])
    payload: dict[str, object] = {"status": "ok", "record": memory_record_payload(updated), "handled_entry_ids": entry_ids}
    if warnings:
        payload["warnings"] = warnings
    return payload


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
    summary = optional_string(arguments, "summary")
    memory_type = optional_string(arguments, "memory_type") or "observation"
    tags = _normalize_tags([*string_list(arguments, "tags")])
    quality_error = _memory_write_quality_error(title=title, content=content, summary=summary)
    if quality_error is not None:
        return {"status": "error", "error": "low_value_memory_rejected", "detail": quality_error}
    warnings = _memory_write_quality_warnings(summary=summary, memory_type=memory_type, tags=tags, content=content)
    record = ctx.repository.create_memory(
        title=title,
        content=content,
        summary=summary,
        memory_type=memory_type,
        status=optional_string(arguments, "status") or "active",
        workspace_ids=workspace_ids,
        tags=tags,
        metadata=build_ingest_created_metadata(
            task_id=task_id,
            entry_ids=entry_ids,
            metadata=metadata_override,
        ),
    )
    assert record is not None
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
    _record_touched_memory_ids(ctx, task_id=task_id, memory_ids=[record.id])
    payload: dict[str, object] = {"status": "ok", "record": memory_record_payload(record), "handled_entry_ids": entry_ids}
    if warnings:
        payload["warnings"] = warnings
    return payload
