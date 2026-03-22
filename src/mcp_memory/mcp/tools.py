from mcp.types import Tool


def get_memory_tools() -> list[Tool]:
    return [
        Tool(
            name="record_thought",
            description=(
                "Capture a durable working memory such as a finding, decision, anomaly, hypothesis, trade-off, "
                "or reusable next step. Prefer one self-contained thought with the conclusion, brief supporting "
                "evidence, and why it matters. Avoid routine play-by-play or status-only updates with no durable takeaway."
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
                "Use the returned summaries to identify promising memories, then follow up with read_memory_record for full context."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1},
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
            description="Read a relational memory record with relationships and superseded breadcrumbs.",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string"},
                },
                "required": ["memory_id"],
            },
        ),
    ]
