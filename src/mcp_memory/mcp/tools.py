from mcp.types import Tool


def get_memory_tools() -> list[Tool]:
    return [
        Tool(
            name="record_thought",
            description="Record a raw system-1 thought in the local memory journal.",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                },
                "required": ["content"],
            },
        ),
        Tool(
            name="get_pending_thoughts",
            description="List pending journal thoughts awaiting consolidation.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "minimum": 1},
                },
            },
        ),
        Tool(
            name="get_memory_stats",
            description="Return basic runtime and journal statistics for the memory store.",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
    ]