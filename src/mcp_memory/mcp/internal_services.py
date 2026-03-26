from __future__ import annotations

from mcp_memory.context import ApplicationContext
from mcp_memory.mcp.internal_ingest_keys import (
    INGEST_ENTRY_DISPOSITIONS_TASK_DATA_KEY,
    INGEST_HANDLED_ENTRY_IDS_TASK_DATA_KEY,
    INGEST_TOOL_INVOCATIONS_TASK_DATA_KEY,
)
from mcp_memory.mcp.internal_ingest_services import (
    internal_append_to_existing_memory_for_ingest_service,
    internal_create_memory_record_for_ingest_service,
    internal_get_next_ingest_batch_service,
)
from mcp_memory.mcp.services import read_memory_record_service, search_memory_records_service
from mcp_memory.mcp.validation import optional_string


__all__ = [
    "INGEST_HANDLED_ENTRY_IDS_TASK_DATA_KEY",
    "INGEST_ENTRY_DISPOSITIONS_TASK_DATA_KEY",
    "INGEST_TOOL_INVOCATIONS_TASK_DATA_KEY",
    "internal_get_next_ingest_batch_service",
    "internal_append_to_existing_memory_for_ingest_service",
    "internal_create_memory_record_for_ingest_service",
]


def internal_search_memory_records_service(ctx: ApplicationContext, arguments: dict) -> dict:
    return search_memory_records_service(ctx, arguments, caller_kind="internal")


def internal_read_memory_record_service(ctx: ApplicationContext, arguments: dict) -> dict:
    return read_memory_record_service(ctx, arguments, caller_kind="internal")


def internal_task_complete_service(ctx: ApplicationContext, arguments: dict) -> dict:
    summary = optional_string(arguments, "summary")
    task_id = optional_string(arguments, "task_id")
    task_name = optional_string(arguments, "task_name")
    return {
        "status": "ok",
        "task_id": task_id,
        "task_name": task_name,
        "summary": summary,
        "completion_recorded": True,
    }
