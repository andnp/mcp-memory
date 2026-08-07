from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any, cast
import inspect

from mcp.types import TextContent

from mcp_memory.context import ApplicationContext
from mcp_memory.internal_tool_call_tracking import InternalToolCallTracker


ToolService = Callable[..., Any]
ToolServiceResolver = Callable[[], dict[str, ToolService]]
ToolSuccessRecorder = Callable[[ApplicationContext, str, dict, list[TextContent]], None]


def text_response(payload: dict) -> list[TextContent]:
    return [TextContent(type="text", text=json.dumps(payload, sort_keys=True))]


def _compact_public_success_payload(payload: dict) -> dict:
    if payload.get("status") != "ok":
        return payload
    return {key: value for key, value in payload.items() if key != "status"}


def _call_service_sync(
    service: Callable[..., dict],
    ctx: ApplicationContext,
    arguments: dict,
    *,
    compact_success: bool = False,
) -> list[TextContent]:
    try:
        payload = service(ctx, arguments)
        return text_response(
            _compact_public_success_payload(payload) if compact_success else payload
        )
    except FileNotFoundError as exc:
        return text_response(
            {"status": "error", "error": "file_not_found", "detail": str(exc)}
        )
    except (TypeError, ValueError) as exc:
        return text_response(
            {"status": "error", "error": "invalid_arguments", "detail": str(exc)}
        )


async def call_service(
    service: ToolService,
    ctx: ApplicationContext,
    arguments: dict,
    *,
    compact_success: bool = False,
) -> list[TextContent]:
    if inspect.iscoroutinefunction(service):
        try:
            payload = await service(ctx, arguments)
            return text_response(
                _compact_public_success_payload(payload)
                if compact_success
                else payload
            )
        except FileNotFoundError as exc:
            return text_response(
                {"status": "error", "error": "file_not_found", "detail": str(exc)}
            )
        except (TypeError, ValueError) as exc:
            return text_response(
                {"status": "error", "error": "invalid_arguments", "detail": str(exc)}
            )
    return await asyncio.to_thread(
        _call_service_sync,
        cast(Callable[..., dict], service),
        ctx,
        arguments,
        compact_success=compact_success,
    )


def _runtime_not_initialized_response(name: str) -> list[TextContent]:
    return text_response(
        {
            "status": "error",
            "error": "runtime_not_initialized",
            "tool": name,
        }
    )


def _unknown_tool_response(name: str) -> list[TextContent]:
    return text_response(
        {
            "status": "error",
            "error": "unknown_tool",
            "tool": name,
        }
    )


async def _dispatch_tool(
    ctx: object,
    name: str,
    arguments: dict,
    *,
    service_resolver: ToolServiceResolver,
    on_success: ToolSuccessRecorder | None = None,
    compact_success: bool = False,
) -> list[TextContent]:
    if not isinstance(ctx, ApplicationContext):
        return _runtime_not_initialized_response(name)

    service = service_resolver().get(name)
    if service is None:
        return _unknown_tool_response(name)

    response = await call_service(
        service,
        ctx,
        arguments,
        compact_success=compact_success,
    )
    if on_success is not None:
        on_success(ctx, name, arguments, response)
    return response


def tool_services() -> dict[str, ToolService]:
    from mcp_memory.mcp.services import (
        read_memory_record_service,
        read_memory_records_service,
        record_thought_service,
        search_memory_records_async_service,
    )

    return {
        "record_thought": record_thought_service,
        "search_memory_records": search_memory_records_async_service,
        "read_memory_record": read_memory_record_service,
        "read_memory_records": read_memory_records_service,
    }


def internal_tool_services() -> dict[str, ToolService]:
    from mcp_memory.mcp.internal_batch_services import (
        internal_get_next_curator_batch_service,
        internal_get_next_dedup_batch_service,
        internal_list_memory_records_service,
    )
    from mcp_memory.mcp.internal_read_services import (
        internal_bounded_adjacency_service,
        internal_list_relationships_service,
        internal_maintenance_search_service,
        internal_peek_record_service,
        internal_read_memory_record_service,
        internal_read_memory_records_service,
        internal_search_memory_records_async_service,
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
    from mcp_memory.mcp.internal_ingest_services import (
        internal_append_to_existing_memory_for_ingest_service,
        internal_create_memory_record_for_ingest_service,
        internal_get_next_ingest_batch_service,
    )
    from mcp_memory.mcp.internal_task_services import (
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
        "internal_search_memory_records": internal_search_memory_records_async_service,
        "internal_read_memory_record": internal_read_memory_record_service,
        "internal_read_memory_records": internal_read_memory_records_service,
        "internal_peek_record": internal_peek_record_service,
        "internal_maintenance_search": internal_maintenance_search_service,
        "internal_list_relationships": internal_list_relationships_service,
        "internal_bounded_adjacency": internal_bounded_adjacency_service,
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
        compact_success=True,
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


def _record_internal_tool_call(
    ctx: ApplicationContext,
    name: str,
    arguments: dict[str, object],
    response: list[TextContent],
) -> None:
    tracker = getattr(ctx, "internal_tool_call_tracker", None)
    if not isinstance(tracker, InternalToolCallTracker):
        return
    raw_task_id = arguments.get("task_id")
    task_id = raw_task_id if isinstance(raw_task_id, str) else None
    tracker.record_call(
        name,
        task_id=task_id,
        session_id=getattr(ctx, "session_id", None),
        success=_internal_tool_response_succeeded(response),
        arguments=arguments,
    )


def _internal_tool_response_succeeded(response: list[TextContent]) -> bool:
    for content in response:
        if content.type != "text":
            continue
        try:
            payload = json.loads(content.text)
        except (TypeError, ValueError):
            return True
        return not (isinstance(payload, dict) and payload.get("status") == "error")
    return True
