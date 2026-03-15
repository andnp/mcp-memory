from mcp.types import TextContent

from mcp_memory.mcp.transport import dispatch_internal_memory_tool, dispatch_memory_tool


async def call_memory_tool(ctx, name: str, arguments: dict) -> list[TextContent]:
    return await dispatch_memory_tool(ctx, name, arguments)


async def call_internal_memory_tool(ctx, name: str, arguments: dict) -> list[TextContent]:
    return await dispatch_internal_memory_tool(ctx, name, arguments)