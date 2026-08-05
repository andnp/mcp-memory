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
            input_schema={
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
                "Results use compact format (memory_ref, title, summary) to minimize tokens. "
                "Use summaries to choose promising memory_ref values, then read with read_memory_record "
                "or read_memory_records. "
                "Omit `limit` unless you need a strict fixed cap; when omitted, search may return an adaptive number of high-confidence results."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Optional strict cap. Omit this unless you specifically need a fixed result count; omitted requests may return an adaptive number of high-confidence matches.",
                    },
                    "workspace_id": {
                        "type": "string",
                        "description": "Optional explicit workspace filter. Omit for global search; the caller workspace only influences ranking.",
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
            name="read_memory_records",
            description=(
                "Read up to 20 memory records by memory_refs or legacy UUIDs in one call. "
                "The legacy memory_ids field is also accepted. "
                "Returns compact records plus a missing list; optional relationships, superseded breadcrumbs, "
                "and metadata can be requested when needed."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "memory_refs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": 20,
                    },
                    "memory_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": 20,
                    },
                    "include_relationships": {"type": "boolean"},
                    "include_superseded": {"type": "boolean"},
                    "include_metadata": {"type": "boolean"},
                    "summary_only": {"type": "boolean"},
                    "content_offset": {"type": "integer", "minimum": 0},
                    "content_limit": {"type": "integer", "minimum": 1},
                },
                "anyOf": [
                    {"required": ["memory_refs"]},
                    {"required": ["memory_ids"]},
                ],
            },
        ),
        Tool(
            name="read_memory_record",
            description=(
                "Read one memory record using its memory_ref or legacy UUID. Returns minimal fields "
                "(memory_ref, title, content) by default to save tokens. Request optional extras "
                "(relationships, superseded, metadata) only when needed."
            ),
            input_schema={
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
                    "summary_only": {
                        "type": "boolean",
                        "description": "Return a bounded summary projection without content.",
                    },
                    "content_offset": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Optional character offset for a bounded content chunk.",
                    },
                    "content_limit": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Optional maximum characters in the returned content chunk.",
                    },
                },
                "required": ["memory_id"],
            },
        ),
    ]
