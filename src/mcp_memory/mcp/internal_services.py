from __future__ import annotations

from mcp_memory.context import ApplicationContext
from mcp_memory.mcp.services import read_memory_record_service, search_memory_records_service
from mcp_memory.mcp.validation import optional_positive_int, optional_string, require_string
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
        "records": [compact_memory_record_payload(record) for record in records],
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