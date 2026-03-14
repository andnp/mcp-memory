from __future__ import annotations

from mcp_memory.context import ApplicationContext


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

    content = str(arguments.get("content", "")).strip()
    if not content:
        return {"status": "error", "error": "content is required"}

    entry = ctx.journal.record(content)
    return {
        "status": "recorded",
        "entry": entry.to_dict(),
    }


def get_pending_thoughts_service(ctx: ApplicationContext, arguments: dict) -> dict:
    if ctx.journal is None:
        return {"status": "error", "error": "journal_not_initialized"}

    limit = int(arguments.get("limit", 10))
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
        "project": ctx.project_name,
        "memory_path": str(ctx.memory_path) if ctx.memory_path is not None else None,
        "documents": document_count,
        "relational_memories": relational_count,
        "journal": ctx.journal.count_by_status() if ctx.journal is not None else {},
    }


def create_memory_record_service(ctx: ApplicationContext, arguments: dict) -> dict:
    if ctx.repository is None:
        return {"status": "error", "error": "repository_not_initialized"}

    title = str(arguments.get("title", "")).strip()
    content = str(arguments.get("content", "")).strip()
    workspace_ids = [str(value) for value in arguments.get("workspace_ids", [])]
    if not title or not content or not workspace_ids:
        return {
            "status": "error",
            "error": "title, content, and workspace_ids are required",
        }

    record = ctx.repository.create_memory(
        title=title,
        content=content,
        workspace_ids=workspace_ids,
        tags=[str(value) for value in arguments.get("tags", [])],
        summary=arguments.get("summary"),
        memory_type=str(arguments.get("memory_type", "journal")),
        status=str(arguments.get("status", "active")),
        metadata=dict(arguments.get("metadata", {})),
    )
    return {"status": "created", "record": record_to_payload(record)}


def get_memory_record_service(ctx: ApplicationContext, arguments: dict) -> dict:
    if ctx.repository is None:
        return {"status": "error", "error": "repository_not_initialized"}

    memory_id = str(arguments.get("memory_id", "")).strip()
    record = ctx.repository.get_memory(memory_id)
    if record is None:
        return {"status": "error", "error": "memory_not_found"}
    return {"status": "ok", "record": record_to_payload(record)}


def list_memory_records_service(ctx: ApplicationContext, arguments: dict) -> dict:
    if ctx.repository is None:
        return {"status": "error", "error": "repository_not_initialized"}

    records = ctx.repository.list_memories(
        workspace_id=arguments.get("workspace_id"),
        memory_type=arguments.get("memory_type"),
        status=arguments.get("status"),
        limit=int(arguments.get("limit", 100)),
    )
    return {
        "status": "ok",
        "records": [record_to_payload(record) for record in records],
    }