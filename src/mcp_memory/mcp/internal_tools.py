from mcp.types import Tool


def get_internal_maintenance_tools() -> list[Tool]:
    return [
        Tool(
            name="internal_search_memory_records",
            description=(
                "Search memory records for maintenance and organization tasks. "
                "Use the returned summaries to identify promising memories, then follow up with internal_read_memory_record for full context."
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
            name="internal_get_next_dedup_batch",
            description="Return the next scheduler-selected deduplication batch of active candidate memories for agentic maintenance work.",
            inputSchema={
                "type": "object",
                "properties": {
                    "workspace_id": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1},
                },
            },
        ),
        Tool(
            name="internal_get_next_ingest_batch",
            description="Claim the next pending System 1 journal entries for one task and return grouped ingest batches.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "workspace_id": {"type": "string"},
                    "batch_size": {"type": "integer", "minimum": 1},
                },
                "required": ["task_id"],
            },
        ),
        Tool(
            name="internal_append_to_existing_memory_for_ingest",
            description="Append ingest content into an existing active memory while preserving ingest lineage metadata, workspace_ids, and system1-appended tagging.",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string"},
                    "content": {"type": "string"},
                    "task_id": {"type": "string"},
                    "entry_ids": {"type": "array", "minItems": 1, "items": {"type": ["integer", "string"]}},
                    "workspace_ids": {"type": "array", "items": {"type": "string"}},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "metadata": {"type": "object"},
                },
                "required": ["memory_id", "content", "task_id", "entry_ids"],
            },
        ),
        Tool(
            name="internal_create_memory_record_for_ingest",
            description="Create a new memory record for ingest while preserving source_entry_ids, ingest_task_id, default ingest tags, workspace_ids, and summary-task enqueueing.",
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "content": {"type": "string"},
                    "task_id": {"type": "string"},
                    "entry_ids": {"type": "array", "minItems": 1, "items": {"type": ["integer", "string"]}},
                    "summary": {"type": "string"},
                    "memory_type": {"type": "string"},
                    "status": {"type": "string"},
                    "workspace_ids": {"type": "array", "items": {"type": "string"}},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "metadata": {"type": "object"},
                },
                "required": ["title", "content", "task_id", "entry_ids"],
            },
        ),
        Tool(
            name="internal_append_memory_content",
            description="Append new content into an existing memory record and optionally update tags, workspace_ids, and metadata.",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string"},
                    "content": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "workspace_ids": {"type": "array", "items": {"type": "string"}},
                    "metadata": {"type": "object"},
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
            description="Merge a source memory into a canonical memory, optionally rewriting canonical fields, then archive the source and link lineage.",
            inputSchema={
                "type": "object",
                "properties": {
                    "canonical_memory_id": {"type": "string"},
                    "source_memory_id": {"type": "string"},
                    "title": {"type": "string"},
                    "content": {"type": "string"},
                    "summary": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "workspace_ids": {"type": "array", "items": {"type": "string"}},
                    "metadata": {"type": "object"},
                    "link_context": {"type": "string"},
                },
                "required": ["canonical_memory_id", "source_memory_id"],
            },
        ),
        Tool(
            name="internal_split_memory_record",
            description="Split one oversized memory into multiple focused child memories, link them back to the original, and optionally archive the original once the split succeeds.",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string"},
                    "parts": {
                        "type": "array",
                        "minItems": 2,
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "content": {"type": "string"},
                                "summary": {"type": "string"},
                                "memory_type": {"type": "string"},
                                "status": {"type": "string"},
                                "workspace_ids": {"type": "array", "items": {"type": "string"}},
                                "tags": {"type": "array", "items": {"type": "string"}},
                                "metadata": {"type": "object"},
                            },
                            "required": ["title", "content"],
                        },
                    },
                    "link_type": {"type": "string"},
                    "link_context": {"type": "string"},
                    "archive_original": {"type": "boolean"},
                },
                "required": ["memory_id", "parts"],
            },
        ),
        Tool(
            name="internal_create_memory_record",
            description="Create a new memory record for maintenance and cleanup workflows, optionally enqueueing follow-up summarization.",
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "content": {"type": "string"},
                    "summary": {"type": "string"},
                    "memory_type": {"type": "string"},
                    "status": {"type": "string"},
                    "workspace_ids": {"type": "array", "items": {"type": "string"}},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "metadata": {"type": "object"},
                    "enqueue_summary_task": {"type": "boolean"},
                },
                "required": ["title", "content"],
            },
        ),
        Tool(
            name="internal_update_memory_record",
            description="Rewrite or otherwise update an existing memory record, including title, content, type, status, tags, and metadata.",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string"},
                    "title": {"type": "string"},
                    "content": {"type": "string"},
                    "summary": {"type": "string"},
                    "memory_type": {"type": "string"},
                    "status": {"type": "string"},
                    "workspace_ids": {"type": "array", "items": {"type": "string"}},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "metadata": {"type": "object"},
                },
                "required": ["memory_id"],
            },
        ),
        Tool(
            name="internal_delete_memory_record",
            description="Permanently delete an archived memory record and its associated links.",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string"},
                    "confirm": {"type": "boolean"},
                },
                "required": ["memory_id", "confirm"],
            },
        ),
        Tool(
            name="internal_create_memory_link",
            description="Create a typed link between two memory records.",
            inputSchema={
                "type": "object",
                "properties": {
                    "source_id": {"type": "string"},
                    "target_id": {"type": "string"},
                    "link_type": {"type": "string"},
                    "context": {"type": "string"},
                },
                "required": ["source_id", "target_id", "link_type"],
            },
        ),
        Tool(
            name="internal_delete_memory_link",
            description="Delete a typed link between two memory records.",
            inputSchema={
                "type": "object",
                "properties": {
                    "source_id": {"type": "string"},
                    "target_id": {"type": "string"},
                    "link_type": {"type": "string"},
                },
                "required": ["source_id", "target_id", "link_type"],
            },
        ),
    ]
