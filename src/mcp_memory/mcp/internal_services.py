from __future__ import annotations

from typing import Any
from uuid import uuid4

from mcp_memory.context import ApplicationContext
from mcp_memory.mcp.internal_ingest_keys import (
    INGEST_ENTRY_DISPOSITIONS_TASK_DATA_KEY,
    INGEST_HANDLED_ENTRY_IDS_TASK_DATA_KEY,
    INGEST_TOOL_INVOCATIONS_TASK_DATA_KEY,
)
from mcp_memory.mcp.internal_ingest_services import (
    internal_append_to_existing_memory_for_ingest_service,
    internal_create_memory_record_for_ingest_service,
    internal_get_next_ingest_batch_service,
)
from mcp_memory.mcp.internal_work_item_services import (
    resolve_internal_workspace_scope,
)
from mcp_memory.mcp.internal_service_support import (
    _append_content,
    _enqueue_summary_task,
    _memory_write_quality_error,
    _memory_write_quality_warnings,
    _merge_memory_metadata,
    _normalize_tags,
)
from mcp_memory.mcp.services import read_memory_record_service, search_memory_records_service
from mcp_memory.mcp.validation import optional_object, optional_positive_int, optional_string, optional_bool, require_string, string_list
from mcp_memory.serialization import compact_memory_record_payload, memory_record_payload


__all__ = [
    "INGEST_HANDLED_ENTRY_IDS_TASK_DATA_KEY",
    "INGEST_ENTRY_DISPOSITIONS_TASK_DATA_KEY",
    "INGEST_TOOL_INVOCATIONS_TASK_DATA_KEY",
    "internal_get_next_ingest_batch_service",
    "internal_append_to_existing_memory_for_ingest_service",
    "internal_create_memory_record_for_ingest_service",
]


def internal_search_memory_records_service(ctx: ApplicationContext, arguments: dict) -> dict:
    return search_memory_records_service(ctx, arguments, caller_kind="internal")


def internal_read_memory_record_service(ctx: ApplicationContext, arguments: dict) -> dict:
    return read_memory_record_service(ctx, arguments, caller_kind="internal")


def internal_list_memory_records_service(ctx: ApplicationContext, arguments: dict) -> dict:
    if ctx.repository is None:
        return {"status": "error", "error": "repository_not_initialized"}
    records = ctx.repository.list_memories(
        workspace_id=resolve_internal_workspace_scope(ctx, arguments),
        memory_type=optional_string(arguments, "memory_type"),
        status=optional_string(arguments, "status"),
        limit=optional_positive_int(arguments, "limit", 10),
    )
    return {
        "status": "ok",
        "records": [compact_memory_record_payload(record).model_dump() for record in records],
    }


def internal_task_complete_service(ctx: ApplicationContext, arguments: dict) -> dict:
    summary = optional_string(arguments, "summary")
    task_id = optional_string(arguments, "task_id")
    task_name = optional_string(arguments, "task_name")
    return {
        "status": "ok",
        "task_id": task_id,
        "task_name": task_name,
        "summary": summary,
        "completion_recorded": True,
    }


def internal_get_next_dedup_batch_service(ctx: ApplicationContext, arguments: dict) -> dict:
    if ctx.repository is None:
        return {"status": "error", "error": "repository_not_initialized"}

    from mcp_memory.core.task_handlers.constants import DEDUPLICATOR_TASK_NAME
    from mcp_memory.core.task_handlers.deduplicator_support import select_deduplicator_seed_batch

    task_id = optional_string(arguments, "task_id") or f"{DEDUPLICATOR_TASK_NAME}:internal"
    strategy = optional_string(arguments, "strategy")
    workspace_id = resolve_internal_workspace_scope(ctx, arguments)
    limit = optional_positive_int(arguments, "limit", 100)
    candidates = [
        record
        for record in ctx.repository.list_memories(
            workspace_id=workspace_id,
            status="active",
            limit=limit,
        )
        if not ctx.repository.has_incoming_link(record.id, "SUPERSEDES")
    ]
    seed_batch = select_deduplicator_seed_batch(
        ctx,
        candidates,
        task_id=task_id,
        strategy=strategy,
    )
    return {
        "status": "ok",
        "requested_strategy": seed_batch.requested_strategy,
        "strategy": seed_batch.strategy_used,
        "strategy_used": seed_batch.strategy_used,
        "strategy_fallback_reason": seed_batch.strategy_fallback_reason,
        "candidate_count": seed_batch.candidate_count,
        "has_more": len(candidates) > len(seed_batch.records),
        "sampled_memory_ids": [record.id for record in seed_batch.records],
        "records": [compact_memory_record_payload(record).model_dump() for record in seed_batch.records],
    }


def internal_get_next_curator_batch_service(ctx: ApplicationContext, arguments: dict) -> dict:
    if ctx.repository is None:
        return {"status": "error", "error": "repository_not_initialized"}

    from mcp_memory.core.task_handlers.constants import CURATOR_TASK_NAME
    from mcp_memory.core.task_handlers.curator_support import (
        CURATOR_MAX_SEED_RECORDS,
        select_curator_seed_batch,
    )
    from mcp_memory.core.tasks import TaskRecord

    task_id = optional_string(arguments, "task_id") or f"{CURATOR_TASK_NAME}:internal"
    strategy = optional_string(arguments, "strategy")
    workspace_id = resolve_internal_workspace_scope(ctx, arguments)
    limit = optional_positive_int(arguments, "limit", CURATOR_MAX_SEED_RECORDS)
    exclude_memory_ids = set(string_list(arguments, "exclude_memory_ids"))
    task_data = {"strategy": strategy} if strategy else {}
    if workspace_id is not None:
        task_data["workspace_id"] = workspace_id
    task = TaskRecord(
        id=task_id,
        task_name=CURATOR_TASK_NAME,
        data=task_data,
        workspace_id=workspace_id,
        status="running",
        priority=100,
        retries_count=0,
        max_retries=3,
        created_at=0.0,
        updated_at=0.0,
        available_at=0.0,
        claimed_at=0.0,
        started_at=0.0,
        completed_at=None,
        last_error=None,
    )
    seed_batch = select_curator_seed_batch(
        ctx,
        task,
        seed_limit=limit,
        exclude_memory_ids=exclude_memory_ids,
    )
    return {
        "status": "ok",
        "requested_strategy": seed_batch.requested_strategy,
        "strategy": seed_batch.strategy_used,
        "strategy_used": seed_batch.strategy_used,
        "strategy_fallback_reason": seed_batch.strategy_fallback_reason,
        "candidate_count": seed_batch.candidate_count,
        "excluded_memory_ids": sorted(exclude_memory_ids),
        "has_more": seed_batch.candidate_count > len(seed_batch.records),
        "sampled_memory_ids": [record.id for record in seed_batch.records],
        "records": [compact_memory_record_payload(record).model_dump() for record in seed_batch.records],
    }


def internal_append_memory_content_service(ctx: ApplicationContext, arguments: dict) -> dict:
    if ctx.repository is None:
        return {"status": "error", "error": "repository_not_initialized"}
    memory_id = require_string(arguments, "memory_id")
    content = require_string(arguments, "content")
    record = ctx.repository.get_memory(memory_id)
    if record is None:
        return {"status": "error", "error": "memory_not_found"}

    quality_error = _memory_write_quality_error(
        title=record.title,
        content=content,
        summary=optional_string(arguments, "summary"),
    )
    if quality_error is not None:
        return {"status": "error", "error": "low_value_memory_rejected", "detail": quality_error}

    merged_content = _append_content(record.content, content)
    tags = arguments.get("tags")
    merged_tags = record.tags
    if isinstance(tags, list):
        merged_tags = _normalize_tags([*record.tags, *[str(tag) for tag in tags]])
    merged_summary = optional_string(arguments, "summary") if "summary" in arguments else record.summary
    warnings = _memory_write_quality_warnings(
        summary=merged_summary,
        memory_type=record.type,
        tags=merged_tags,
        content=merged_content,
    )
    workspace_ids = string_list(arguments, "workspace_ids") if "workspace_ids" in arguments else None
    metadata_override = optional_object(arguments, "metadata") if "metadata" in arguments else None
    merged_metadata = _merge_memory_metadata(record.metadata, metadata_override)
    updated = ctx.repository.update_memory(
        memory_id,
        content=merged_content,
        summary=optional_string(arguments, "summary") if "summary" in arguments else None,
        tags=merged_tags,
        workspace_ids=sorted({*record.workspace_ids, *workspace_ids}) if workspace_ids else None,
        metadata=merged_metadata,
    )
    if updated is None:
        return {"status": "error", "error": "memory_not_found"}
    payload = {"status": "ok", "record": memory_record_payload(updated)}
    if warnings:
        payload["warnings"] = warnings
    return payload


def internal_archive_memory_record_service(ctx: ApplicationContext, arguments: dict) -> dict:
    if ctx.repository is None:
        return {"status": "error", "error": "repository_not_initialized"}
    memory_id = require_string(arguments, "memory_id")
    updated = ctx.repository.update_memory(memory_id, status="archived")
    if updated is None:
        return {"status": "error", "error": "memory_not_found"}
    return {"status": "ok", "record": memory_record_payload(updated)}


def internal_merge_memory_into_canonical_service(ctx: ApplicationContext, arguments: dict) -> dict:
    if ctx.repository is None:
        return {"status": "error", "error": "repository_not_initialized"}
    canonical_memory_id = require_string(arguments, "canonical_memory_id")
    source_memory_id = require_string(arguments, "source_memory_id")
    canonical = ctx.repository.get_memory(canonical_memory_id)
    source = ctx.repository.get_memory(source_memory_id)
    if canonical is None or source is None:
        return {"status": "error", "error": "memory_not_found"}

    merged_title = optional_string(arguments, "title") or canonical.title
    merged_content = optional_string(arguments, "content") or _append_content(canonical.content, source.summary or source.content)
    merged_summary = optional_string(arguments, "summary")
    tags = string_list(arguments, "tags") if "tags" in arguments else None
    workspace_ids = string_list(arguments, "workspace_ids") if "workspace_ids" in arguments else None
    metadata_override = optional_object(arguments, "metadata") or {}
    merged_metadata = dict(canonical.metadata)
    if metadata_override:
        merged_metadata.update(metadata_override)
    merged_source_ids = merged_metadata.get("merged_source_ids", [])
    if not isinstance(merged_source_ids, list):
        merged_source_ids = []
    merged_metadata["merged_source_ids"] = sorted({*map(str, merged_source_ids), source.id})

    updated = ctx.repository.update_memory(
        canonical.id,
        title=merged_title,
        content=merged_content,
        summary=merged_summary,
        tags=_normalize_tags(tags) if tags is not None else _normalize_tags([*canonical.tags, *source.tags]),
        workspace_ids=list(workspace_ids) if workspace_ids else sorted({*canonical.workspace_ids, *source.workspace_ids}),
        metadata=merged_metadata,
    )
    assert updated is not None
    ctx.repository.add_link(
        updated.id,
        source.id,
        "SUPERSEDES",
        optional_string(arguments, "link_context") or "Merged into canonical memory by internal maintenance tools.",
    )
    archived = ctx.repository.update_memory(source.id, status="archived")
    return {
        "status": "ok",
        "canonical": memory_record_payload(updated),
        "archived": None if archived is None else memory_record_payload(archived),
    }


def internal_split_memory_record_service(ctx: ApplicationContext, arguments: dict) -> dict:
    if ctx.repository is None:
        return {"status": "error", "error": "repository_not_initialized"}

    memory_id = require_string(arguments, "memory_id")
    original = ctx.repository.get_memory(memory_id)
    if original is None:
        return {"status": "error", "error": "memory_not_found"}

    raw_parts = arguments.get("parts")
    if not isinstance(raw_parts, list) or len(raw_parts) < 2:
        return {"status": "error", "error": "split_parts_required"}

    link_type = optional_string(arguments, "link_type") or "DEPENDS_ON"
    link_context = optional_string(arguments, "link_context") or "Derived from an oversized memory split by internal maintenance tools."
    archive_original = optional_bool(arguments, "archive_original")
    split_group_id = str(uuid4())
    split_part_count = len(raw_parts)

    normalized_parts: list[dict[str, Any]] = []
    for index, raw_part in enumerate(raw_parts, start=1):
        if not isinstance(raw_part, dict):
            return {"status": "error", "error": "split_part_invalid"}
        try:
            title = require_string(raw_part, "title")
            content = require_string(raw_part, "content")
        except ValueError:
            return {"status": "error", "error": "split_part_invalid"}
        normalized_parts.append(
            {
                "title": title,
                "content": content,
                "summary": optional_string(raw_part, "summary"),
                "memory_type": optional_string(raw_part, "memory_type") or original.type,
                "status": optional_string(raw_part, "status") or "active",
                "workspace_ids": string_list(raw_part, "workspace_ids") or list(original.workspace_ids),
                "tags": _normalize_tags([*original.tags, *string_list(raw_part, "tags")]),
                "metadata": {
                    "split_from_memory_id": original.id,
                    "split_from_memory_title": original.title,
                    "split_group_id": split_group_id,
                    "split_part_index": index,
                    "split_part_count": split_part_count,
                    **(optional_object(raw_part, "metadata") or {}),
                },
            }
        )

    created_records = []
    try:
        for part in normalized_parts:
            created = ctx.repository.create_memory(
                title=str(part["title"]),
                content=str(part["content"]),
                summary=part["summary"] if isinstance(part["summary"], str) else None,
                memory_type=str(part["memory_type"]),
                status=str(part["status"]),
                workspace_ids=list(part["workspace_ids"]),
                tags=list(part["tags"]),
                metadata=optional_object(part, "metadata") or {},
            )
            assert created is not None
            created_records.append(created)
            ctx.repository.add_link(created.id, original.id, link_type, link_context)
    except Exception:
        for created in created_records:
            ctx.repository.delete_memory(created.id)
        raise

    child_memory_ids = [record.id for record in created_records]
    refreshed_created_records = []
    for created in created_records:
        sibling_memory_ids = [record_id for record_id in child_memory_ids if record_id != created.id]
        refreshed = ctx.repository.update_memory(
            created.id,
            metadata=_merge_memory_metadata(
                created.metadata,
                {
                    "split_child_memory_ids": child_memory_ids,
                    "split_sibling_memory_ids": sibling_memory_ids,
                },
            ),
        )
        refreshed_created_records.append(refreshed or created)
    created_records = refreshed_created_records

    original_metadata = _merge_memory_metadata(
        original.metadata,
        {
            "split_group_id": split_group_id,
            "split_child_memory_ids": child_memory_ids,
            "split_child_count": len(child_memory_ids),
        },
    )

    archived = None
    refreshed_original = None
    if archive_original:
        archived = ctx.repository.update_memory(original.id, status="archived", metadata=original_metadata)
        refreshed_original = archived
    else:
        refreshed_original = ctx.repository.update_memory(original.id, metadata=original_metadata)

    return {
        "status": "ok",
        "original": memory_record_payload(refreshed_original or original),
        "created": [memory_record_payload(record) for record in created_records],
        "archived": None if archived is None else memory_record_payload(archived),
    }


def internal_create_memory_record_service(ctx: ApplicationContext, arguments: dict) -> dict:
    if ctx.repository is None:
        return {"status": "error", "error": "repository_not_initialized"}

    workspace_ids = string_list(arguments, "workspace_ids") or ([ctx.workspace_id] if ctx.workspace_id is not None else [])
    title = require_string(arguments, "title")
    content = require_string(arguments, "content")
    summary = optional_string(arguments, "summary")
    memory_type = optional_string(arguments, "memory_type") or "observation"
    tags = string_list(arguments, "tags")
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
        metadata=optional_object(arguments, "metadata"),
    )
    assert record is not None
    if optional_bool(arguments, "enqueue_summary_task"):
        _enqueue_summary_task(ctx, record.id, list(record.workspace_ids))
    payload = {"status": "ok", "record": memory_record_payload(record)}
    if warnings:
        payload["warnings"] = warnings
    return payload


def internal_update_memory_record_service(ctx: ApplicationContext, arguments: dict) -> dict:
    if ctx.repository is None:
        return {"status": "error", "error": "repository_not_initialized"}

    memory_id = require_string(arguments, "memory_id")
    record = ctx.repository.get_memory(memory_id)
    if record is None:
        return {"status": "error", "error": "memory_not_found"}

    updatable_fields = [
        "title",
        "content",
        "summary",
        "memory_type",
        "status",
        "workspace_ids",
        "tags",
        "metadata",
    ]
    if not any(field in arguments for field in updatable_fields):
        return {"status": "error", "error": "no_updates_requested"}

    requested_title = optional_string(arguments, "title") if "title" in arguments else None
    requested_content = optional_string(arguments, "content") if "content" in arguments else None
    requested_memory_type = optional_string(arguments, "memory_type") if "memory_type" in arguments else None
    title = requested_title if requested_title is not None else record.title
    content = requested_content if requested_content is not None else record.content
    summary = optional_string(arguments, "summary") if "summary" in arguments else record.summary
    memory_type = requested_memory_type if requested_memory_type is not None else record.type
    tags = string_list(arguments, "tags") if "tags" in arguments else record.tags

    quality_error = _memory_write_quality_error(title=title, content=content, summary=summary)
    if quality_error is not None:
        return {"status": "error", "error": "low_value_memory_rejected", "detail": quality_error}
    warnings = _memory_write_quality_warnings(
        summary=summary,
        memory_type=memory_type,
        tags=tags,
        content=content,
    )

    updated = ctx.repository.update_memory(
        memory_id,
        title=optional_string(arguments, "title"),
        content=optional_string(arguments, "content"),
        summary=optional_string(arguments, "summary"),
        memory_type=optional_string(arguments, "memory_type"),
        status=optional_string(arguments, "status"),
        workspace_ids=string_list(arguments, "workspace_ids") if "workspace_ids" in arguments else None,
        tags=string_list(arguments, "tags") if "tags" in arguments else None,
        metadata=optional_object(arguments, "metadata") if "metadata" in arguments else None,
    )
    if updated is None:
        return {"status": "error", "error": "memory_not_found"}
    payload = {"status": "ok", "record": memory_record_payload(updated)}
    if warnings:
        payload["warnings"] = warnings
    return payload


def internal_delete_memory_record_service(ctx: ApplicationContext, arguments: dict) -> dict:
    if ctx.repository is None:
        return {"status": "error", "error": "repository_not_initialized"}

    memory_id = require_string(arguments, "memory_id")
    confirm = optional_bool(arguments, "confirm")
    if not confirm:
        return {"status": "error", "error": "delete_not_confirmed"}

    record = ctx.repository.get_memory(memory_id)
    if record is None:
        return {"status": "error", "error": "memory_not_found"}
    if record.status != "archived":
        return {"status": "error", "error": "memory_not_archived"}

    deleted = ctx.repository.delete_memory(memory_id)
    if deleted is None:
        return {"status": "error", "error": "memory_not_found"}
    return {"status": "ok", "deleted": memory_record_payload(deleted)}


def internal_create_memory_link_service(ctx: ApplicationContext, arguments: dict) -> dict:
    if ctx.repository is None:
        return {"status": "error", "error": "repository_not_initialized"}

    source_id = require_string(arguments, "source_id")
    target_id = require_string(arguments, "target_id")
    link_type = require_string(arguments, "link_type")
    context = optional_string(arguments, "context") or ""
    source = ctx.repository.get_memory(source_id)
    target = None if target_id.startswith("ext:") else ctx.repository.get_memory(target_id)
    if source is None or (target is None and not target_id.startswith("ext:")):
        return {"status": "error", "error": "memory_not_found"}
    link = ctx.repository.add_link(source_id, target_id, link_type, context)
    return {
        "status": "ok",
        "link": {
            "source_id": link.source_id,
            "target_id": link.target_id,
            "link_type": link.link_type,
            "context": link.context,
        },
    }


def internal_delete_memory_link_service(ctx: ApplicationContext, arguments: dict) -> dict:
    if ctx.repository is None:
        return {"status": "error", "error": "repository_not_initialized"}

    deleted = ctx.repository.remove_link(
        require_string(arguments, "source_id"),
        require_string(arguments, "target_id"),
        require_string(arguments, "link_type"),
    )
    if not deleted:
        return {"status": "error", "error": "link_not_found"}
    return {"status": "ok"}
