from mcp.types import Tool


def get_internal_maintenance_tools() -> list[Tool]:
    return [
        Tool(
            name="internal_search_memory_records",
            description=(
                "Search memory records for maintenance and organization tasks. "
                "Use summaries to choose promising memory_id values, then read only those records. "
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
            name="internal_read_memory_record",
            description=(
                "Read one memory record for maintenance. Internal calls include relationships and superseded "
                "breadcrumbs by default; pass false for compact reads. Metadata remains opt-in."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string"},
                    "include_relationships": {"type": "boolean"},
                    "include_superseded": {"type": "boolean"},
                    "include_metadata": {"type": "boolean"},
                },
                "required": ["memory_id"],
            },
        ),
        Tool(
            name="internal_peek_record",
            description="Read one authoritative memory record for maintenance without user access updates or shared-cache reads.",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string"},
                    "include_metadata": {"type": "boolean"},
                },
                "required": ["memory_id"],
            },
        ),
        Tool(
            name="internal_maintenance_search",
            description="Search authoritative memory records for maintenance with a bounded result budget.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                    "memory_type": {"type": "string"},
                    "status": {"type": "string"},
                    "include_superseded": {"type": "boolean"},
                    "include_metadata": {"type": "boolean"},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="internal_list_relationships",
            description="List a bounded incoming, outgoing, or combined relationship slice for maintenance.",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string"},
                    "direction": {"type": "string", "enum": ["incoming", "outgoing", "both"]},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                },
                "required": ["memory_id"],
            },
        ),
        Tool(
            name="internal_bounded_adjacency",
            description="Read a bounded one-hop neighborhood for maintenance without mutation or work-item lifecycle access.",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string"},
                    "direction": {"type": "string", "enum": ["incoming", "outgoing", "both"]},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                    "include_metadata": {"type": "boolean"},
                },
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
            name="task_complete",
            description=(
                "Record a lightweight completion marker for the current maintenance task without mutating memories. "
                "Use this instead of creating journal or memory records for routine completion/status traces."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "task_name": {"type": "string"},
                    "summary": {"type": "string"},
                },
            },
        ),
        Tool(
            name="internal_task_complete",
            description=(
                "Record a lightweight completion marker for the current maintenance task without mutating memories. "
                "Use this instead of creating journal or memory records for routine completion/status traces."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "task_name": {"type": "string"},
                    "summary": {"type": "string"},
                },
            },
        ),
        Tool(
            name="internal_get_next_dedup_batch",
            description="Return the next scheduler-selected deduplication batch of active candidate memories for agentic maintenance work.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "workspace_id": {"type": "string"},
                    "strategy": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1},
                },
            },
        ),
        Tool(
            name="internal_get_next_curator_batch",
            description="Return the next curator-ranked maintenance batch of active candidate memories, optionally excluding records already reviewed in this run.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "workspace_id": {"type": "string"},
                    "strategy": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1},
                    "exclude_memory_ids": {"type": "array", "items": {"type": "string"}},
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
                    "grouping_strategy": {"type": "string"},
                },
                "required": ["task_id"],
            },
        ),
        Tool(
            name="internal_get_work_batch",
            description="Claim the next durable work-item batch for one family and execution lane. Call it repeatedly within one run to safely process more queued work from that same family.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "family_key": {"type": "string"},
                    "execution_lane": {"type": "string"},
                    "workspace_id": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1},
                    "lease_ttl_seconds": {"type": "integer", "minimum": 1},
                },
                "required": ["task_id", "family_key", "execution_lane"],
            },
        ),
        Tool(
            name="internal_get_compatible_work_batch",
            description="Claim the next durable work-item batch from a compatible multi-family group. Use this to widen one run across families that share the same safety model.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "compatibility_group": {"type": "string"},
                    "allowed_families": {"type": "array", "items": {"type": "string"}},
                    "execution_lane": {"type": "string"},
                    "workspace_id": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1},
                    "lease_ttl_seconds": {"type": "integer", "minimum": 1},
                },
                "required": ["task_id", "compatibility_group", "execution_lane"],
            },
        ),
        Tool(
            name="internal_heartbeat_work_item",
            description="Extend the lease for one claimed durable work item owned by the current task.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "work_item_id": {"type": "string"},
                    "lease_ttl_seconds": {"type": "integer", "minimum": 1},
                },
                "required": ["task_id", "work_item_id"],
            },
        ),
        Tool(
            name="internal_complete_work_item",
            description="Mark one claimed durable work item as completed and clear its lease.",
            inputSchema={
                "type": "object",
                "properties": {
                    "work_item_id": {"type": "string"},
                },
                "required": ["work_item_id"],
            },
        ),
        Tool(
            name="internal_defer_work_item",
            description="Defer one claimed durable work item with an error/reason and retry delay.",
            inputSchema={
                "type": "object",
                "properties": {
                    "work_item_id": {"type": "string"},
                    "error": {"type": "string"},
                    "retry_delay_seconds": {"type": "integer", "minimum": 0},
                },
                "required": ["work_item_id", "error"],
            },
        ),
        Tool(
            name="internal_release_work_item",
            description="Release one claimed durable work item back to pending without marking it complete.",
            inputSchema={
                "type": "object",
                "properties": {
                    "work_item_id": {"type": "string"},
                },
                "required": ["work_item_id"],
            },
        ),
        Tool(
            name="internal_ingest_append_memory",
            description="Append ingest content into an existing active memory while preserving ingest lineage metadata, workspace_ids, and system1-appended tagging.",
            inputSchema={
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string"},
                    "content": {"type": "string"},
                    "summary": {"type": "string"},
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
            name="internal_ingest_create_memory",
            description=(
                "Create a new durable memory record for ingest while preserving source_entry_ids, ingest_task_id, "
                "default ingest tags, workspace_ids, and summary-task enqueueing. Do not use this for routine "
                "status or completion markers."
            ),
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
                    "summary": {"type": "string"},
                    "task_id": {"type": "string"},
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
                    "task_id": {"type": "string"},
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
                    "task_id": {"type": "string"},
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
                    "task_id": {"type": "string"},
                },
                "required": ["memory_id", "parts"],
            },
        ),
        Tool(
            name="internal_create_memory_record",
            description=(
                "Create a new durable memory record for maintenance and cleanup workflows, optionally enqueueing "
                "follow-up summarization. Do not use this for routine completion, counters, or status-only traces."
            ),
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
                    "task_id": {"type": "string"},
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
                    "task_id": {"type": "string"},
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
                    "task_id": {"type": "string"},
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
                    "task_id": {"type": "string"},
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
                    "task_id": {"type": "string"},
                },
                "required": ["source_id", "target_id", "link_type"],
            },
        ),
    ]
