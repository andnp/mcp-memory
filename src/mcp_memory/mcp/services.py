from __future__ import annotations

from pathlib import Path

from mcp_memory.context import ApplicationContext
from mcp_memory.core.importer import import_markdown_memory


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


def search_memory_records_service(ctx: ApplicationContext, arguments: dict) -> dict:
    if ctx.relational_search is None:
        return {"status": "error", "error": "relational_search_not_initialized"}

    query = str(arguments.get("query", "")).strip()
    if not query:
        return {"status": "error", "error": "query is required"}

    results = ctx.relational_search.search_memories(
        query=query,
        workspace_id=arguments.get("workspace_id"),
        limit=int(arguments.get("limit", 5)),
        memory_type=arguments.get("memory_type"),
        status=arguments.get("status"),
        include_superseded=bool(arguments.get("include_superseded", False)),
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

    memory_id = str(arguments.get("memory_id", "")).strip()
    if not memory_id:
        return {"status": "error", "error": "memory_id is required"}

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

    file_path = str(arguments.get("file_path", "")).strip()
    if not file_path:
        return {"status": "error", "error": "file_path is required"}

    workspace_ids = [
        str(value).strip()
        for value in arguments.get("workspace_ids", [])
        if str(value).strip()
    ]
    if not workspace_ids:
        if ctx.project_name:
            workspace_ids = [ctx.project_name]
        else:
            return {
                "status": "error",
                "error": "workspace_ids are required when project_name is unavailable",
            }

    imported = import_markdown_memory(ctx.repository, Path(file_path), workspace_ids)
    return {
        "status": "imported",
        "record": record_to_payload(imported),
    }