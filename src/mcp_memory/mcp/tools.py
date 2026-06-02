from mcp.types import Tool


def get_memory_tools() -> list[Tool]:
    return [
        Tool(
            name="record_thought",
            description=(
                "Capture a durable working memory such as a finding, decision, anomaly, hypothesis, trade-off, "
                "or reusable next step. Prefer one self-contained thought with the conclusion, brief supporting "
                "evidence, and why it matters. Avoid routine play-by-play or status-only updates with no durable takeaway. "
                "Returns minimal status confirmation to save tokens."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                },
                "required": ["content"],
            },
        ),
        Tool(
            name="search_memory_records",
            description=(
                "Search relational memory records with summary-first results. "
                "Results use compact format (memory_id, title, summary) to minimize tokens. "
                "Use summaries to choose promising memory_id values, then read with read_memory_record. "
                "Omit `limit` unless you need a strict fixed cap; when omitted, search may return an adaptive number of high-confidence results."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Optional strict cap. Omit this unless you specifically need a fixed result count; omitted requests may return an adaptive number of high-confidence matches.",
                    },
                    "memory_type": {"type": "string"},
                    "status": {"type": "string"},
                    "include_superseded": {"type": "boolean"},
                    "debug": {"type": "boolean"},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="read_memory_record",
            description=(
                "Read one memory record. Returns minimal fields (id, title, content) by default to save tokens. "
                "Request optional extras (relationships, superseded, metadata) only when needed."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string"},
                    "include_relationships": {
                        "type": "boolean",
                        "description": "Include incoming/outgoing relationship edges. Defaults to false to save tokens.",
                    },
                    "include_superseded": {
                        "type": "boolean",
                        "description": "Include superseded record breadcrumbs. Defaults to false to save tokens.",
                    },
                    "include_metadata": {
                        "type": "boolean",
                        "description": "Include maintenance metadata and workspace IDs. Defaults to false to save tokens.",
                    },
                },
                "required": ["memory_id"],
            },
        ),
    ]
