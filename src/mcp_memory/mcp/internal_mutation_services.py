from __future__ import annotations

from typing import Any, cast
from uuid import uuid4

from mcp_memory.context import ApplicationContext
from mcp_memory.core.ingest_claim_lifecycle import _record_ingest_tool_invocation, _record_touched_memory_ids
from mcp_memory.mcp.internal_service_support import (
    _append_content,
    _enqueue_summary_task,
    _memory_write_quality_error,
    _memory_write_quality_warnings,
    _merge_memory_metadata,
    _normalize_tags,
)
from mcp_memory.mcp.validation import optional_object, optional_string, optional_bool, require_string, string_list
from mcp_memory.serialization import memory_record_payload


def _maybe_record_ingest_tool_invocation(ctx: ApplicationContext, arguments: dict[str, Any], *, tool_name: str) -> None:
    task_id = optional_string(arguments, "task_id")
    if task_id is None:
        return
    _record_ingest_tool_invocation(ctx, task_id=task_id, tool_name=tool_name, mutation=True)


def _maybe_record_ingest_touched_memory_ids(
    ctx: ApplicationContext,
    arguments: dict[str, Any],
    *,
    memory_ids: list[str],
) -> None:
    task_id = optional_string(arguments, "task_id")
    if task_id is None:
        return
    _record_touched_memory_ids(ctx, task_id=task_id, memory_ids=memory_ids)


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
    _maybe_record_ingest_tool_invocation(ctx, arguments, tool_name="internal_append_memory_content")
    _maybe_record_ingest_touched_memory_ids(ctx, arguments, memory_ids=[updated.id])
    payload: dict[str, Any] = {"status": "ok", "record": memory_record_payload(updated)}
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
    _maybe_record_ingest_tool_invocation(ctx, arguments, tool_name="internal_archive_memory_record")
    _maybe_record_ingest_touched_memory_ids(ctx, arguments, memory_ids=[updated.id])
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
    _maybe_record_ingest_tool_invocation(ctx, arguments, tool_name="internal_merge_memory_into_canonical")
    _maybe_record_ingest_touched_memory_ids(
        ctx,
        arguments,
        memory_ids=[updated.id, source.id],
    )
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
        raw_part_dict = cast(dict[str, Any], raw_part)
        try:
            title = require_string(raw_part_dict, "title")
            content = require_string(raw_part_dict, "content")
        except ValueError:
            return {"status": "error", "error": "split_part_invalid"}
        normalized_parts.append(
            {
                "title": title,
                "content": content,
                "summary": optional_string(raw_part_dict, "summary"),
                "memory_type": optional_string(raw_part_dict, "memory_type") or original.type,
                "status": optional_string(raw_part_dict, "status") or "active",
                "workspace_ids": string_list(raw_part_dict, "workspace_ids") or list(original.workspace_ids),
                "tags": _normalize_tags([*original.tags, *string_list(raw_part_dict, "tags")]),
                "metadata": {
                    "split_from_memory_id": original.id,
                    "split_from_memory_title": original.title,
                    "split_group_id": split_group_id,
                    "split_part_index": index,
                    "split_part_count": split_part_count,
                    **(optional_object(raw_part_dict, "metadata") or {}),
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

    _maybe_record_ingest_tool_invocation(ctx, arguments, tool_name="internal_split_memory_record")
    _maybe_record_ingest_touched_memory_ids(
        ctx,
        arguments,
        memory_ids=[original.id, *child_memory_ids],
    )

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
    _maybe_record_ingest_tool_invocation(ctx, arguments, tool_name="internal_create_memory_record")
    _maybe_record_ingest_touched_memory_ids(ctx, arguments, memory_ids=[record.id])
    payload: dict[str, Any] = {"status": "ok", "record": memory_record_payload(record)}
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
    _maybe_record_ingest_tool_invocation(ctx, arguments, tool_name="internal_update_memory_record")
    _maybe_record_ingest_touched_memory_ids(ctx, arguments, memory_ids=[updated.id])
    payload: dict[str, Any] = {"status": "ok", "record": memory_record_payload(updated)}
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
    _maybe_record_ingest_tool_invocation(ctx, arguments, tool_name="internal_delete_memory_record")
    _maybe_record_ingest_touched_memory_ids(ctx, arguments, memory_ids=[deleted.id])
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
    _maybe_record_ingest_tool_invocation(ctx, arguments, tool_name="internal_create_memory_link")
    _maybe_record_ingest_touched_memory_ids(
        ctx,
        arguments,
        memory_ids=[source_id, *( [] if target_id.startswith("ext:") else [target_id])],
    )
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
    _maybe_record_ingest_tool_invocation(ctx, arguments, tool_name="internal_delete_memory_link")
    _maybe_record_ingest_touched_memory_ids(
        ctx,
        arguments,
        memory_ids=[
            require_string(arguments, "source_id"),
            *([] if require_string(arguments, "target_id").startswith("ext:") else [require_string(arguments, "target_id")]),
        ],
    )
    return {"status": "ok"}
