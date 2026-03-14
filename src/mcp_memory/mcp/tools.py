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
        Tool(
            name="create_memory_record",
            description="Create a relational memory record in the runtime database.",
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "content": {"type": "string"},
                    "workspace_ids": {"type": "array", "items": {"type": "string"}},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "summary": {"type": "string"},
                    "memory_type": {"type": "string"},
                    "status": {"type": "string"},
                    "metadata": {"type": "object"},
                },
                "required": ["title", "content", "workspace_ids"],
            },
        ),
        Tool(
            name="get_memory_record",
            description="Fetch a relational memory record by ID.",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string"},
                },
                "required": ["memory_id"],
            },
        ),
        Tool(
            name="list_memory_records",
            description="List relational memory records with optional filters.",
            inputSchema={
                "type": "object",
                "properties": {
                    "workspace_id": {"type": "string"},
                    "memory_type": {"type": "string"},
                    "status": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1},
                },
            },
        ),
        Tool(
            name="search_memory_records",
            description="Search relational memory records with summary-first results.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "workspace_id": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1},
                    "memory_type": {"type": "string"},
                    "status": {"type": "string"},
                    "include_superseded": {"type": "boolean"},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="read_memory_record",
            description="Read a relational memory record with relationships and superseded breadcrumbs.",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string"},
                },
                "required": ["memory_id"],
            },
        ),
        Tool(
            name="import_markdown_memory_file",
            description="Import a legacy markdown memory file into the relational repository.",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "workspace_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["file_path"],
            },
        ),
    ]