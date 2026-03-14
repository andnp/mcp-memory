from __future__ import annotations

import json
from collections.abc import Callable

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


ToolService = Callable[[ApplicationContext, dict], dict]


def text_response(payload: dict) -> list[TextContent]:
    return [TextContent(type="text", text=json.dumps(payload, sort_keys=True))]


def call_service(service: ToolService, ctx: ApplicationContext, arguments: dict) -> list[TextContent]:
    try:
        return text_response(service(ctx, arguments))
    except FileNotFoundError as exc:
        return text_response(
            {"status": "error", "error": "file_not_found", "detail": str(exc)}
        )
    except (TypeError, ValueError) as exc:
        return text_response(
            {"status": "error", "error": "invalid_arguments", "detail": str(exc)}
        )


def tool_services() -> dict[str, ToolService]:
    return {
        "record_thought": record_thought_service,
        "get_pending_thoughts": get_pending_thoughts_service,
        "create_memory_record": create_memory_record_service,
        "get_memory_record": get_memory_record_service,
        "list_memory_records": list_memory_records_service,
        "search_memory_records": search_memory_records_service,
        "read_memory_record": read_memory_record_service,
        "import_markdown_memory_file": import_markdown_memory_file_service,
    }


async def dispatch_memory_tool(
    ctx: object,
    name: str,
    arguments: dict,
) -> list[TextContent]:
    if not isinstance(ctx, ApplicationContext):
        return text_response(
            {
                "status": "error",
                "error": "runtime_not_initialized",
                "tool": name,
            }
        )

    if name == "get_memory_stats":
        return text_response(get_memory_stats_service(ctx))

    service = tool_services().get(name)
    if service is None:
        return text_response(
            {
                "status": "error",
                "error": "unknown_tool",
                "tool": name,
                "arguments": arguments,
            }
        )

    return call_service(service, ctx, arguments)