from mcp.types import Tool


def get_internal_maintenance_tools() -> list[Tool]:
    return [
        Tool(
            name="internal_search_memory_records",
            description="Search memory records for maintenance and organization tasks.",
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
            name="internal_read_memory_record",
            description="Read one memory record with relationships and superseded breadcrumbs.",
            inputSchema={
                "type": "object",
                "properties": {"memory_id": {"type": "string"}},
                "required": ["memory_id"],
            },
        ),
        Tool(
            name="internal_list_memory_records",
            description="List recent memory records for maintenance work.",
            inputSchema={
                "type": "object",
                "properties": {
                    "workspace_id": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1},
                    "memory_type": {"type": "string"},
                    "status": {"type": "string"},
                },
            },
        ),
        Tool(
            name="internal_append_memory_content",
            description="Append new content into an existing memory record and optionally update tags.",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string"},
                    "content": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["memory_id", "content"],
            },
        ),
        Tool(
            name="internal_archive_memory_record",
            description="Archive a memory record without deleting it.",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string"},
                },
                "required": ["memory_id"],
            },
        ),
        Tool(
            name="internal_merge_memory_into_canonical",
            description="Merge a source memory into a canonical memory, then archive the source and link lineage.",
            inputSchema={
                "type": "object",
                "properties": {
                    "canonical_memory_id": {"type": "string"},
                    "source_memory_id": {"type": "string"},
                },
                "required": ["canonical_memory_id", "source_memory_id"],
            },
        ),
    ]