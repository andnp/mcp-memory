import json

from mcp.types import TextContent

from mcp_memory.mcp.runtime import MCPRuntime

def _text_response(payload: dict) -> list[TextContent]:
    return [TextContent(type="text", text=json.dumps(payload, sort_keys=True))]


def _record_to_payload(record) -> dict:
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


async def call_memory_tool(ctx, name: str, arguments: dict) -> list[TextContent]:
    if not isinstance(ctx, MCPRuntime):
        return _text_response(
            {
                "status": "error",
                "error": "runtime_not_initialized",
                "tool": name,
            }
        )

    if name == "record_thought":
        content = str(arguments.get("content", "")).strip()
        if not content:
            return _text_response(
                {
                    "status": "error",
                    "error": "content is required",
                }
            )

        entry = ctx.journal.record(content)
        payload = {
            "status": "recorded",
            "entry": entry.to_dict(),
        }
        return _text_response(payload)

    if name == "get_pending_thoughts":
        limit = int(arguments.get("limit", 10))
        payload = {
            "status": "ok",
            "entries": [entry.to_dict() for entry in ctx.journal.get_pending(limit=limit)],
        }
        return _text_response(payload)

    if name == "get_memory_stats":
        conn = ctx.db_manager.get_connection()
        document_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        relational_count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        payload = {
            "status": "ok",
            "project": ctx.project_name,
            "memory_path": str(ctx.memory_path),
            "documents": document_count,
            "relational_memories": relational_count,
            "journal": ctx.journal.count_by_status(),
        }
        return _text_response(payload)

    if name == "create_memory_record":
        title = str(arguments.get("title", "")).strip()
        content = str(arguments.get("content", "")).strip()
        workspace_ids = [str(value) for value in arguments.get("workspace_ids", [])]
        if not title or not content or not workspace_ids:
            return _text_response(
                {
                    "status": "error",
                    "error": "title, content, and workspace_ids are required",
                }
            )

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
        return _text_response({"status": "created", "record": _record_to_payload(record)})

    if name == "get_memory_record":
        memory_id = str(arguments.get("memory_id", "")).strip()
        record = ctx.repository.get_memory(memory_id)
        if record is None:
            return _text_response({"status": "error", "error": "memory_not_found"})
        return _text_response({"status": "ok", "record": _record_to_payload(record)})

    if name == "list_memory_records":
        records = ctx.repository.list_memories(
            workspace_id=arguments.get("workspace_id"),
            memory_type=arguments.get("memory_type"),
            status=arguments.get("status"),
            limit=int(arguments.get("limit", 100)),
        )
        return _text_response(
            {
                "status": "ok",
                "records": [_record_to_payload(record) for record in records],
            }
        )

    return _text_response(
        {
            "status": "error",
            "error": "unknown_tool",
            "tool": name,
            "arguments": arguments,
        }
    )