import json

from mcp.types import TextContent

from mcp_memory.context import ApplicationContext
from mcp_memory.mcp.services import (
    create_memory_record_service,
    get_memory_record_service,
    get_memory_stats_service,
    get_pending_thoughts_service,
    import_markdown_memory_file_service,
    list_memory_records_service,
    read_memory_record_service,
    record_thought_service,
    search_memory_records_service,
)

def _text_response(payload: dict) -> list[TextContent]:
    return [TextContent(type="text", text=json.dumps(payload, sort_keys=True))]

async def call_memory_tool(ctx, name: str, arguments: dict) -> list[TextContent]:
    if not isinstance(ctx, ApplicationContext):
        return _text_response(
            {
                "status": "error",
                "error": "runtime_not_initialized",
                "tool": name,
            }
        )

    if name == "record_thought":
        return _text_response(record_thought_service(ctx, arguments))

    if name == "get_pending_thoughts":
        return _text_response(get_pending_thoughts_service(ctx, arguments))

    if name == "get_memory_stats":
        return _text_response(get_memory_stats_service(ctx))

    if name == "create_memory_record":
        return _text_response(create_memory_record_service(ctx, arguments))

    if name == "get_memory_record":
        return _text_response(get_memory_record_service(ctx, arguments))

    if name == "list_memory_records":
        return _text_response(list_memory_records_service(ctx, arguments))

    if name == "search_memory_records":
        return _text_response(search_memory_records_service(ctx, arguments))

    if name == "read_memory_record":
        return _text_response(read_memory_record_service(ctx, arguments))

    if name == "import_markdown_memory_file":
        return _text_response(import_markdown_memory_file_service(ctx, arguments))

    return _text_response(
        {
            "status": "error",
            "error": "unknown_tool",
            "tool": name,
            "arguments": arguments,
        }
    )