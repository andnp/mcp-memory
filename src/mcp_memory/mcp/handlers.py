from mcp.types import TextContent

from mcp_memory.mcp.transport import dispatch_memory_tool


async def call_memory_tool(ctx, name: str, arguments: dict) -> list[TextContent]:
    return await dispatch_memory_tool(ctx, name, arguments)