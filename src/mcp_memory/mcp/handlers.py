import json

from mcp.types import TextContent


async def call_memory_tool(ctx, name: str, arguments: dict) -> list[TextContent]:
    payload = {
        "status": "not_implemented",
        "tool": name,
        "arguments": arguments,
    }
    return [TextContent(type="text", text=json.dumps(payload, sort_keys=True))]