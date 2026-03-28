from __future__ import annotations

import asyncio
import json
from collections.abc import Callable

from mcp.types import TextContent

from mcp_memory.context import ApplicationContext


ToolService = Callable[[ApplicationContext, dict], dict]


def text_response(payload: dict) -> list[TextContent]:
    return [TextContent(type="text", text=json.dumps(payload, sort_keys=True))]


def _call_service_sync(service: ToolService, ctx: ApplicationContext, arguments: dict) -> list[TextContent]:
    try:
        return text_response(service(ctx, arguments))
    except FileNotFoundError as exc:
        return text_response(
            {"status": "error", "error": "file_not_found", "detail": str(exc)}
        )
    except (TypeError, ValueError) as exc:
        return text_response(
            {"status": "error", "error": "invalid_arguments", "detail": str(exc)}
        )


async def call_service(service: ToolService, ctx: ApplicationContext, arguments: dict) -> list[TextContent]:
    return await asyncio.to_thread(_call_service_sync, service, ctx, arguments)


def tool_services() -> dict[str, ToolService]:
    from mcp_memory.mcp.services import (
        read_memory_record_service,
        record_thought_service,
        search_memory_records_service,
    )

    return {
        "record_thought": record_thought_service,
        "search_memory_records": search_memory_records_service,
        "read_memory_record": read_memory_record_service,
    }


def internal_tool_services() -> dict[str, ToolService]:
    from mcp_memory.mcp.internal_batch_services import (
        internal_get_next_curator_batch_service,
        internal_get_next_dedup_batch_service,
        internal_list_memory_records_service,
    )
    from mcp_memory.mcp.internal_mutation_services import (
        internal_append_memory_content_service,
        internal_archive_memory_record_service,
        internal_create_memory_link_service,
        internal_create_memory_record_service,
        internal_delete_memory_link_service,
        internal_delete_memory_record_service,
        internal_merge_memory_into_canonical_service,
        internal_split_memory_record_service,
        internal_update_memory_record_service,
    )
    from mcp_memory.mcp.internal_services import (
        internal_append_to_existing_memory_for_ingest_service,
        internal_create_memory_record_for_ingest_service,
        internal_get_next_ingest_batch_service,
        internal_read_memory_record_service,
        internal_search_memory_records_service,
        internal_task_complete_service,
    )
    from mcp_memory.mcp.internal_work_item_services import (
        internal_complete_work_item_service,
        internal_defer_work_item_service,
        internal_get_compatible_work_batch_service,
        internal_get_work_batch_service,
        internal_heartbeat_work_item_service,
        internal_release_work_item_service,
    )

    return {
        "internal_search_memory_records": internal_search_memory_records_service,
        "internal_read_memory_record": internal_read_memory_record_service,
        "internal_list_memory_records": internal_list_memory_records_service,
        "task_complete": internal_task_complete_service,
        "internal_task_complete": internal_task_complete_service,
        "internal_get_next_dedup_batch": internal_get_next_dedup_batch_service,
        "internal_get_next_curator_batch": internal_get_next_curator_batch_service,
        "internal_get_next_ingest_batch": internal_get_next_ingest_batch_service,
        "internal_get_work_batch": internal_get_work_batch_service,
        "internal_get_compatible_work_batch": internal_get_compatible_work_batch_service,
        "internal_heartbeat_work_item": internal_heartbeat_work_item_service,
        "internal_complete_work_item": internal_complete_work_item_service,
        "internal_defer_work_item": internal_defer_work_item_service,
        "internal_release_work_item": internal_release_work_item_service,
        "internal_ingest_append_memory": internal_append_to_existing_memory_for_ingest_service,
        "internal_ingest_create_memory": internal_create_memory_record_for_ingest_service,
        "internal_append_to_existing_memory_for_ingest": internal_append_to_existing_memory_for_ingest_service,
        "internal_create_memory_record_for_ingest": internal_create_memory_record_for_ingest_service,
        "internal_append_memory_content": internal_append_memory_content_service,
        "internal_archive_memory_record": internal_archive_memory_record_service,
        "internal_merge_memory_into_canonical": internal_merge_memory_into_canonical_service,
        "internal_split_memory_record": internal_split_memory_record_service,
        "internal_create_memory_record": internal_create_memory_record_service,
        "internal_update_memory_record": internal_update_memory_record_service,
        "internal_delete_memory_record": internal_delete_memory_record_service,
        "internal_create_memory_link": internal_create_memory_link_service,
        "internal_delete_memory_link": internal_delete_memory_link_service,
    }


async def dispatch_memory_tool(
    ctx: object,
    name: str,
    arguments: dict,
) -> list[TextContent]:
    if not isinstance(ctx, ApplicationContext):
        return text_response(
            {
                "status": "error",
                "error": "runtime_not_initialized",
                "tool": name,
            }
        )

    service = tool_services().get(name)
    if service is None:
        return text_response(
            {
                "status": "error",
                "error": "unknown_tool",
                "tool": name,
                "arguments": arguments,
            }
        )

    return await call_service(service, ctx, arguments)


async def dispatch_internal_memory_tool(
    ctx: object,
    name: str,
    arguments: dict,
) -> list[TextContent]:
    if not isinstance(ctx, ApplicationContext):
        return text_response(
            {
                "status": "error",
                "error": "runtime_not_initialized",
                "tool": name,
            }
        )

    service = internal_tool_services().get(name)
    if service is None:
        return text_response(
            {
                "status": "error",
                "error": "unknown_tool",
                "tool": name,
                "arguments": arguments,
            }
        )

    return await call_service(service, ctx, arguments)
