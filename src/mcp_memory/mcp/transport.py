from __future__ import annotations

import asyncio
import json
from collections.abc import Callable

from mcp.types import TextContent

from mcp_memory.context import ApplicationContext
from mcp_memory.internal_tool_call_tracking import InternalToolCallTracker


ToolService = Callable[[ApplicationContext, dict], dict]
ToolServiceResolver = Callable[[], dict[str, ToolService]]
ToolSuccessRecorder = Callable[[ApplicationContext, str, dict], None]


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


def _runtime_not_initialized_response(name: str) -> list[TextContent]:
    return text_response(
        {
            "status": "error",
            "error": "runtime_not_initialized",
            "tool": name,
        }
    )


def _unknown_tool_response(name: str, arguments: dict) -> list[TextContent]:
    return text_response(
        {
            "status": "error",
            "error": "unknown_tool",
            "tool": name,
            "arguments": arguments,
        }
    )


async def _dispatch_tool(
    ctx: object,
    name: str,
    arguments: dict,
    *,
    service_resolver: ToolServiceResolver,
    on_success: ToolSuccessRecorder | None = None,
) -> list[TextContent]:
    if not isinstance(ctx, ApplicationContext):
        return _runtime_not_initialized_response(name)

    service = service_resolver().get(name)
    if service is None:
        return _unknown_tool_response(name, arguments)

    response = await call_service(service, ctx, arguments)
    if on_success is not None:
        on_success(ctx, name, arguments)
    return response


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
    return await _dispatch_tool(
        ctx,
        name,
        arguments,
        service_resolver=tool_services,
    )


async def dispatch_internal_memory_tool(
    ctx: object,
    name: str,
    arguments: dict,
) -> list[TextContent]:
    return await _dispatch_tool(
        ctx,
        name,
        arguments,
        service_resolver=internal_tool_services,
        on_success=_record_internal_tool_call,
    )


def _record_internal_tool_call(ctx: ApplicationContext, name: str, arguments: dict[str, object]) -> None:
    tracker = getattr(ctx, "internal_tool_call_tracker", None)
    if not isinstance(tracker, InternalToolCallTracker):
        return
    raw_task_id = arguments.get("task_id")
    task_id = raw_task_id if isinstance(raw_task_id, str) else None
    tracker.record_call(
        name,
        task_id=task_id,
        session_id=getattr(ctx, "session_id", None),
    )
