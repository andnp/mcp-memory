from mcp.types import Tool

SKILL_REVIEW_READ_ONLY_SCOPE = "skill_review_read_only"
SKILL_REVIEW_READ_ONLY_TOOLS = frozenset(
    {
        "get_skill_review_ledger",
        "search_memory_records",
        "read_memory_records",
        "read_memory_record",
    }
)
SKILL_REVIEW_WRITER_SCOPE = "skill_review_writer"
SKILL_REVIEW_WRITER_TOOLS = frozenset({"commit_skill_review"})
RETRIEVAL_MODES = ("keyword", "semantic", "hybrid")


def allowed_memory_tool_names(tool_scope: str | None = None) -> frozenset[str] | None:
    """Return the tool names permitted by an optional stdio scope."""
    if tool_scope is None:
        return None
    if tool_scope == SKILL_REVIEW_READ_ONLY_SCOPE:
        return SKILL_REVIEW_READ_ONLY_TOOLS
    if tool_scope == SKILL_REVIEW_WRITER_SCOPE:
        return SKILL_REVIEW_WRITER_TOOLS
    if tool_scope not in {SKILL_REVIEW_READ_ONLY_SCOPE, SKILL_REVIEW_WRITER_SCOPE}:
        raise ValueError(f"unknown_memory_tool_scope: {tool_scope}")
    return frozenset()


def get_memory_tools(tool_scope: str | None = None, *, include_writer_tools: bool = False) -> list[Tool]:
    """Describe public memory tools, optionally restricted to a read-only scope."""
    tools = [
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
            name="record_skill_observation",
            description=(
                "Record a reusable skill-improvement observation with explicit evidence, scope, and privacy classification."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "summary": {"type": "string"},
                    "content": {"type": "string"},
                    "skill": {"type": "string"},
                    "observation_kind": {"type": "string"},
                    "privacy_classification": {"type": "string"},
                    "workspace_id": {"type": "string"},
                },
                "required": [
                    "title",
                    "summary",
                    "content",
                    "skill",
                    "observation_kind",
                    "privacy_classification",
                ],
            },
        ),
        Tool(
            name="resolve_skill_observation",
            description="Mark one skill observation as actioned, deferred, or verified with a concise resolution note.",
            input_schema={
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string"},
                    "resolution": {"type": "string", "enum": ["actioned", "deferred", "verified"]},
                    "note": {"type": "string"},
                },
                "required": ["memory_id", "resolution", "note"],
            },
        ),
        Tool(
            name="get_skill_review_ledger",
            description=(
                "Return a deterministic, summary-first page of every open skill observation for one workspace. "
                "Use the returned next_page_token until it is null; page tokens are bound to the snapshot_id."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "protocol_version": {"type": "integer", "const": 1},
                    "workspace_id": {"type": "string"},
                    "page_size": {"type": "integer", "minimum": 1, "maximum": 100},
                    "page_token": {"type": "string"},
                },
                "required": ["protocol_version", "workspace_id"],
            },
        ),
        Tool(
            name="commit_skill_review",
            description="Commit an evidence-backed batch of skill observation dispositions atomically.",
            input_schema={
                "type": "object",
                "properties": {
                    "protocol_version": {"type": "integer", "const": 2},
                    "review_run_id": {"type": "string"},
                    "workspace_id": {"type": "string"},
                    "evidence": {
                        "type": "object",
                        "properties": {
                            "skills": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                            "source_hashes": {"type": "object"},
                            "deployment_status": {"type": "string", "const": "passed"},
                            "deployment_receipt_hash": {"type": "string"},
                            "ledger_snapshot_id": {"type": "string"},
                        },
                        "required": [
                            "skills",
                            "source_hashes",
                            "deployment_status",
                            "deployment_receipt_hash",
                            "ledger_snapshot_id",
                        ],
                    },
                    "dispositions": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "properties": {
                                "memory_id": {"type": "string"},
                                "outcome": {
                                    "type": "string",
                                    "enum": ["done", "bad"],
                                },
                                "note": {"type": "string"},
                            },
                            "required": ["memory_id", "outcome", "note"],
                        },
                    },
                },
                "required": [
                    "protocol_version",
                    "review_run_id",
                    "workspace_id",
                    "evidence",
                    "dispositions",
                ],
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
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Return records that have every listed tag.",
                    },
                    "include_superseded": {"type": "boolean"},
                    "debug": {"type": "boolean"},
                    "retrieval_mode": {
                        "type": "string",
                        "enum": list(RETRIEVAL_MODES),
                        "default": "hybrid",
                        "description": "Optional retrieval strategy. Defaults to hybrid.",
                    },
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
    allowed = allowed_memory_tool_names(tool_scope)
    if allowed is None and not include_writer_tools:
        return [tool for tool in tools if tool.name not in SKILL_REVIEW_WRITER_TOOLS]
    return [tool for tool in tools if allowed is None or tool.name in allowed]
