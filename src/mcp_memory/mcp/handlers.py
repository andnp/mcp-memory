import json

from mcp.types import TextContent

from mcp_memory.mcp.runtime import MCPRuntime

def _text_response(payload: dict) -> list[TextContent]:
    return [TextContent(type="text", text=json.dumps(payload, sort_keys=True))]


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
        payload = {
            "status": "ok",
            "project": ctx.project_name,
            "memory_path": str(ctx.memory_path),
            "documents": document_count,
            "journal": ctx.journal.count_by_status(),
        }
        return _text_response(payload)

    return _text_response(
        {
            "status": "error",
            "error": "unknown_tool",
            "tool": name,
            "arguments": arguments,
        }
    )