from __future__ import annotations

import sqlite3
from typing import Any
from uuid import uuid4

from mcp_memory.context import ApplicationContext
from mcp_memory.core.system1_scheduling import resolve_pending_workspace_id
from mcp_memory.mcp.services import read_memory_record_service, search_memory_records_service
from mcp_memory.mcp.validation import optional_object, optional_positive_int, optional_string, optional_bool, require_string, string_list
from mcp_memory.serialization import compact_memory_record_payload, memory_record_payload


def internal_search_memory_records_service(ctx: ApplicationContext, arguments: dict) -> dict:
    return search_memory_records_service(ctx, arguments)


def internal_read_memory_record_service(ctx: ApplicationContext, arguments: dict) -> dict:
    return read_memory_record_service(ctx, arguments)


def internal_list_memory_records_service(ctx: ApplicationContext, arguments: dict) -> dict:
    if ctx.repository is None:
        return {"status": "error", "error": "repository_not_initialized"}
    records = ctx.repository.list_memories(
        workspace_id=optional_string(arguments, "workspace_id") or ctx.workspace_id,
        memory_type=optional_string(arguments, "memory_type"),
        status=optional_string(arguments, "status"),
        limit=optional_positive_int(arguments, "limit", 10),
    )
    return {
        "status": "ok",
        "records": [compact_memory_record_payload(record).model_dump() for record in records],
    }


def internal_get_next_dedup_batch_service(ctx: ApplicationContext, arguments: dict) -> dict:
    if ctx.repository is None:
        return {"status": "error", "error": "repository_not_initialized"}

    from mcp_memory.core.task_handlers.constants import DEDUPLICATOR_TASK_NAME
    from mcp_memory.core.task_handlers.maintenance import _select_deduplicator_seed_batch

    task_id = optional_string(arguments, "task_id") or f"{DEDUPLICATOR_TASK_NAME}:internal"
    strategy = optional_string(arguments, "strategy")
    workspace_id = optional_string(arguments, "workspace_id") or ctx.workspace_id
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
    seed_batch = _select_deduplicator_seed_batch(
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

    memory_id = require_string(arguments, "memory_id")
    content = require_string(arguments, "content")
    task_id = require_string(arguments, "task_id")
    entry_ids = _normalize_ingest_entry_ids(arguments, "entry_ids")
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
    return {"status": "ok", "record": memory_record_payload(updated)}


def internal_create_memory_record_for_ingest_service(ctx: ApplicationContext, arguments: dict) -> dict:
    if ctx.repository is None:
        return {"status": "error", "error": "repository_not_initialized"}

    task_id = require_string(arguments, "task_id")
    entry_ids = _normalize_ingest_entry_ids(arguments, "entry_ids")
    workspace_ids = string_list(arguments, "workspace_ids") or ([ctx.workspace_id] if ctx.workspace_id is not None else ["workspace-unknown"])
    metadata_override = optional_object(arguments, "metadata") or {}
    record = ctx.repository.create_memory(
        title=require_string(arguments, "title"),
        content=require_string(arguments, "content"),
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
    return {"status": "ok", "record": memory_record_payload(record)}


def internal_append_memory_content_service(ctx: ApplicationContext, arguments: dict) -> dict:
    if ctx.repository is None:
        return {"status": "error", "error": "repository_not_initialized"}
    memory_id = require_string(arguments, "memory_id")
    content = require_string(arguments, "content")
    record = ctx.repository.get_memory(memory_id)
    if record is None:
        return {"status": "error", "error": "memory_not_found"}

    merged_content = _append_content(record.content, content)
    tags = arguments.get("tags")
    merged_tags = record.tags
    if isinstance(tags, list):
        merged_tags = _normalize_tags([*record.tags, *[str(tag) for tag in tags]])
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
    return {"status": "ok", "record": memory_record_payload(updated)}


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
    record = ctx.repository.create_memory(
        title=require_string(arguments, "title"),
        content=require_string(arguments, "content"),
        summary=optional_string(arguments, "summary"),
        memory_type=optional_string(arguments, "memory_type") or "observation",
        status=optional_string(arguments, "status") or "active",
        workspace_ids=workspace_ids,
        tags=string_list(arguments, "tags"),
        metadata=optional_object(arguments, "metadata"),
    )
    assert record is not None
    if optional_bool(arguments, "enqueue_summary_task"):
        _enqueue_summary_task(ctx, record.id, list(record.workspace_ids))
    return {"status": "ok", "record": memory_record_payload(record)}


def internal_update_memory_record_service(ctx: ApplicationContext, arguments: dict) -> dict:
    if ctx.repository is None:
        return {"status": "error", "error": "repository_not_initialized"}

    memory_id = require_string(arguments, "memory_id")
    if ctx.repository.get_memory(memory_id) is None:
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
    return {"status": "ok", "record": memory_record_payload(updated)}


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


def _append_content(existing: str, addition: str) -> str:
    normalized_addition = addition.strip()
    if not normalized_addition or normalized_addition in existing:
        return existing
    return f"{existing.rstrip()}\n\n{normalized_addition}".strip()


def _normalize_tags(tags: list[str]) -> list[str]:
    normalized = []
    seen: set[str] = set()
    for tag in tags:
        value = tag.strip().lower().replace("_", "-")
        if not value or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    return sorted(normalized)


def _merge_memory_metadata(existing: dict[str, Any], override: dict[str, object] | None) -> dict[str, object] | None:
    if override is None:
        return None
    merged: dict[str, object] = dict(existing)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, list) and isinstance(value, list):
            merged[key] = _merge_metadata_lists(current, value)
        else:
            merged[key] = value
    return merged


def _merge_metadata_lists(existing: list[Any], override: list[Any]) -> list[Any]:
    int_values: set[int] = set()
    if all(isinstance(item, int) and not isinstance(item, bool) for item in [*existing, *override]):
        for item in [*existing, *override]:
            int_values.add(int(item))
        return sorted(int_values)
    str_values: set[str] = set()
    if all(isinstance(item, str) for item in [*existing, *override]):
        for item in [*existing, *override]:
            stripped = str(item).strip()
            if stripped:
                str_values.add(stripped)
        return sorted(str_values)
    return list(override)


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


def _enqueue_summary_task(ctx: ApplicationContext, memory_id: str, workspace_ids: list[str]) -> None:
    if ctx.task_queue is None:
        return
    from mcp_memory.core.task_handlers.constants import SUMMARIZE_MEMORY_PRIORITY, SUMMARIZE_MEMORY_TASK_NAME

    workspace_id = workspace_ids[0] if workspace_ids else ctx.workspace_id
    try:
        ctx.task_queue.enqueue(
            task_name=SUMMARIZE_MEMORY_TASK_NAME,
            task_id=f"{SUMMARIZE_MEMORY_TASK_NAME}:{memory_id}",
            workspace_id=workspace_id,
            data={"memory_id": memory_id},
            priority=SUMMARIZE_MEMORY_PRIORITY,
        )
    except sqlite3.IntegrityError:
        return


def _journal_entry_payload(entry) -> dict[str, Any]:
    return {
        "id": entry.id,
        "content": entry.content,
        "workspace_id": entry.workspace_id,
        "timestamp": entry.timestamp,
        "status": entry.status,
    }
