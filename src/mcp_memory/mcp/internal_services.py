from __future__ import annotations

from typing import Any

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

    from mcp_memory.core.task_handlers.maintenance import _select_deduplicator_seed_records

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
    seed_records = _select_deduplicator_seed_records(candidates)
    return {
        "status": "ok",
        "strategy": "deduplicator_seed_records",
        "candidate_count": len(candidates),
        "has_more": len(candidates) > len(seed_records),
        "records": [compact_memory_record_payload(record).model_dump() for record in seed_records],
    }


def internal_get_next_ingest_batch_service(ctx: ApplicationContext, arguments: dict) -> dict:
    if ctx.journal is None:
        return {"status": "error", "error": "journal_not_initialized"}

    from mcp_memory.core.task_handlers.ingest import _build_ingest_groups

    task_id = require_string(arguments, "task_id")
    journal_workspace_id = resolve_pending_workspace_id(
        ctx.journal,
        optional_string(arguments, "workspace_id") or ctx.workspace_id,
    )
    workspace_id = optional_string(arguments, "workspace_id") or ctx.workspace_id or "workspace-unknown"
    batch_size = optional_positive_int(arguments, "batch_size", 20)

    entries = ctx.journal.claim_pending(
        task_id=task_id,
        limit=batch_size,
        workspace_id=journal_workspace_id,
    )
    groups = _build_ingest_groups(ctx, entries, workspace_id)
    pending_remaining = ctx.journal.count_by_status(workspace_id=journal_workspace_id).get("pending", 0)
    return {
        "status": "ok",
        "task_id": task_id,
        "claimed_entry_ids": [entry.id for entry in entries],
        "pending_remaining": pending_remaining,
        "has_more": pending_remaining > 0,
        "groups": [
            {
                "group_index": index,
                "entries": [_journal_entry_payload(entry) for entry in group],
            }
            for index, group in enumerate(groups)
        ],
    }


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
    updated = ctx.repository.update_memory(memory_id, content=merged_content, tags=merged_tags)
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

    merged_content = _append_content(canonical.content, source.summary or source.content)
    updated = ctx.repository.update_memory(
        canonical.id,
        content=merged_content,
        tags=_normalize_tags([*canonical.tags, *source.tags]),
        workspace_ids=sorted({*canonical.workspace_ids, *source.workspace_ids}),
    )
    assert updated is not None
    ctx.repository.add_link(updated.id, source.id, "SUPERSEDES", "Merged into canonical memory by internal maintenance tools.")
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

    normalized_parts: list[dict[str, Any]] = []
    for raw_part in raw_parts:
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

    archived = None
    if archive_original:
        archived = ctx.repository.update_memory(original.id, status="archived")

    return {
        "status": "ok",
        "original": memory_record_payload(original),
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


def _journal_entry_payload(entry) -> dict[str, Any]:
    return {
        "id": entry.id,
        "content": entry.content,
        "workspace_id": entry.workspace_id,
        "timestamp": entry.timestamp,
        "status": entry.status,
    }