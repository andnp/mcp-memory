from __future__ import annotations

from pathlib import Path
from typing import Any

from mcp_memory.context import ApplicationContext
from mcp_memory.core.agent_runtime import (
    SYSTEM1_INGEST_TASK_NAME,
    SYSTEM1_INGEST_THRESHOLD,
)
from mcp_memory.relational.importer import import_markdown_memory


def _require_string(arguments: dict[str, Any], field_name: str) -> str:
    value = arguments.get(field_name)
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} is required")
    return normalized


def _optional_string(arguments: dict[str, Any], field_name: str) -> str | None:
    value = arguments.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    return normalized or None


def _optional_positive_int(arguments: dict[str, Any], field_name: str, default: int) -> int:
    value = arguments.get(field_name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < 1:
        raise ValueError(f"{field_name} must be at least 1")
    return value


def _string_list(arguments: dict[str, Any], field_name: str, required: bool = False) -> list[str]:
    value = arguments.get(field_name)
    if value is None:
        if required:
            raise ValueError(f"{field_name} is required")
        return []
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list of strings")
    normalized_values: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise TypeError(f"{field_name} must be a list of strings")
        normalized = item.strip()
        if normalized:
            normalized_values.append(normalized)
    if required and not normalized_values:
        raise ValueError(f"{field_name} is required")
    return normalized_values


def _optional_object(arguments: dict[str, Any], field_name: str) -> dict[str, object] | None:
    value = arguments.get(field_name)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be an object")
    return dict(value)


def _optional_bool(arguments: dict[str, Any], field_name: str, default: bool = False) -> bool:
    value = arguments.get(field_name, default)
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a boolean")
    return value


def record_to_payload(record) -> dict:
    return {
        "id": record.id,
        "title": record.title,
        "content": record.content,
        "summary": record.summary,
        "type": record.type,
        "status": record.status,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "access_score": record.access_score,
        "last_accessed_at": record.last_accessed_at,
        "last_surfaced_at": record.last_surfaced_at,
        "metadata": record.metadata,
        "workspace_ids": record.workspace_ids,
        "tags": record.tags,
    }


def record_thought_service(ctx: ApplicationContext, arguments: dict) -> dict:
    if ctx.journal is None:
        return {"status": "error", "error": "journal_not_initialized"}

    content = _require_string(arguments, "content")

    entry = ctx.journal.record(content, workspace_id=ctx.workspace_id)
    payload = {
        "status": "recorded",
        "entry": entry.to_dict(),
    }

    if ctx.task_queue is not None:
        pending_count = ctx.journal.count_by_status().get("pending", 0)
        if pending_count >= SYSTEM1_INGEST_THRESHOLD:
            workspace_id = ctx.workspace_id
            ingest_task, created = ctx.task_queue.enqueue_unique(
                task_name=SYSTEM1_INGEST_TASK_NAME,
                workspace_id=workspace_id,
                data={
                    "workspace_id": workspace_id,
                    "trigger": "system1_threshold",
                    "pending_count": pending_count,
                },
            )
            payload["ingest_task"] = {
                "id": ingest_task.id,
                "status": ingest_task.status,
                "task_name": ingest_task.task_name,
                "workspace_id": ingest_task.workspace_id,
                "created": created,
            }

    return payload


def get_pending_thoughts_service(ctx: ApplicationContext, arguments: dict) -> dict:
    if ctx.journal is None:
        return {"status": "error", "error": "journal_not_initialized"}

    limit = _optional_positive_int(arguments, "limit", 10)
    return {
        "status": "ok",
        "entries": [entry.to_dict() for entry in ctx.journal.get_pending(limit=limit)],
    }


def get_memory_stats_service(ctx: ApplicationContext) -> dict:
    if ctx.db_manager is None:
        return {"status": "error", "error": "db_not_initialized"}

    conn = ctx.db_manager.get_connection()
    document_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    relational_count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    return {
        "status": "ok",
        "workspace_id": ctx.workspace_id,
        "workspace_root": str(ctx.workspace_root) if ctx.workspace_root is not None else None,
        "ai_provider": ctx.config.ai.provider if ctx.config is not None else None,
        "ai_model": ctx.config.ai.model if ctx.config is not None else None,
        "memory_path": str(ctx.memory_path) if ctx.memory_path is not None else None,
        "documents": document_count,
        "relational_memories": relational_count,
        "journal": ctx.journal.count_by_status() if ctx.journal is not None else {},
    }


def create_memory_record_service(ctx: ApplicationContext, arguments: dict) -> dict:
    if ctx.repository is None:
        return {"status": "error", "error": "repository_not_initialized"}

    title = _require_string(arguments, "title")
    content = _require_string(arguments, "content")
    workspace_ids = _string_list(arguments, "workspace_ids")
    if not workspace_ids and ctx.workspace_id is not None:
        workspace_ids = [ctx.workspace_id]
    if not workspace_ids:
        raise ValueError("workspace_ids are required")

    record = ctx.repository.create_memory(
        title=title,
        content=content,
        workspace_ids=workspace_ids,
        tags=_string_list(arguments, "tags"),
        summary=_optional_string(arguments, "summary"),
        memory_type=_optional_string(arguments, "memory_type") or "journal",
        status=_optional_string(arguments, "status") or "active",
        metadata=_optional_object(arguments, "metadata") or {},
    )
    return {"status": "created", "record": record_to_payload(record)}


def get_memory_record_service(ctx: ApplicationContext, arguments: dict) -> dict:
    if ctx.repository is None:
        return {"status": "error", "error": "repository_not_initialized"}

    memory_id = _require_string(arguments, "memory_id")
    record = ctx.repository.get_memory(memory_id)
    if record is None:
        return {"status": "error", "error": "memory_not_found"}
    return {"status": "ok", "record": record_to_payload(record)}


def list_memory_records_service(ctx: ApplicationContext, arguments: dict) -> dict:
    if ctx.repository is None:
        return {"status": "error", "error": "repository_not_initialized"}

    records = ctx.repository.list_memories(
        workspace_id=_optional_string(arguments, "workspace_id"),
        memory_type=_optional_string(arguments, "memory_type"),
        status=_optional_string(arguments, "status"),
        limit=_optional_positive_int(arguments, "limit", 100),
    )
    return {
        "status": "ok",
        "records": [record_to_payload(record) for record in records],
    }


def search_memory_records_service(ctx: ApplicationContext, arguments: dict) -> dict:
    if ctx.relational_search is None:
        return {"status": "error", "error": "relational_search_not_initialized"}

    query = _require_string(arguments, "query")

    results = ctx.relational_search.search_memories(
        query=query,
        workspace_id=_optional_string(arguments, "workspace_id"),
        limit=_optional_positive_int(arguments, "limit", 5),
        memory_type=_optional_string(arguments, "memory_type"),
        status=_optional_string(arguments, "status"),
        include_superseded=_optional_bool(arguments, "include_superseded", False),
    )
    return {
        "status": "ok",
        "results": [
            {
                "memory_id": result.memory_id,
                "title": result.title,
                "summary": result.summary,
                "memory_type": result.memory_type,
                "status": result.status,
                "tags": result.tags,
                "workspace_ids": result.workspace_ids,
                "score": result.score,
            }
            for result in results
        ],
    }


def read_memory_record_service(ctx: ApplicationContext, arguments: dict) -> dict:
    if ctx.relational_search is None:
        return {"status": "error", "error": "relational_search_not_initialized"}

    memory_id = _require_string(arguments, "memory_id")

    result = ctx.relational_search.read_memory(memory_id)
    if result is None:
        return {"status": "error", "error": "memory_not_found"}

    return {
        "status": "ok",
        "record": record_to_payload(result.record),
        "relationships": {
            direction: [
                {
                    "source_id": link.source_id,
                    "target_id": link.target_id,
                    "link_type": link.link_type,
                    "context": link.context,
                }
                for link in links
            ]
            for direction, links in result.relationships.items()
        },
        "superseded": [record_to_payload(record) for record in result.superseded],
    }


def import_markdown_memory_file_service(ctx: ApplicationContext, arguments: dict) -> dict:
    if ctx.repository is None:
        return {"status": "error", "error": "repository_not_initialized"}

    file_path = _require_string(arguments, "file_path")

    workspace_ids = _string_list(arguments, "workspace_ids")
    if not workspace_ids:
        if ctx.workspace_id:
            workspace_ids = [ctx.workspace_id]
        else:
            return {
                "status": "error",
                "error": "workspace_ids are required when workspace_id is unavailable",
            }

    import_path = Path(file_path)
    if not import_path.exists():
        raise FileNotFoundError(f"markdown memory file not found: {file_path}")

    imported = import_markdown_memory(ctx.repository, import_path, workspace_ids)
    return {
        "status": "imported",
        "record": record_to_payload(imported),
    }