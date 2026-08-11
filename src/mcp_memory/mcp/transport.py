from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any, cast
import inspect

from mcp.types import TextContent

from mcp_memory.context import ApplicationContext
from mcp_memory.core.curator_evidence import (
    begin_tool_call,
    current_curator_execution,
    finish_tool_call,
)
from mcp_memory.core.direct_mutation_evidence import (
    DirectMutationEvidence,
    entity_deltas_for_payload,
    reconcile_direct_mutation_evidence,
)
from mcp_memory.internal_tool_call_tracking import internal_tool_is_mutating
from mcp_memory.internal_tool_call_tracking import InternalToolCallTracker
from mcp_memory.mcp.internal_search_contract import INTERNAL_SEARCH_TOOL_NAME


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

    execution_token = None
    direct_evidence: DirectMutationEvidence | None = None
    if on_success is not None:
        raw_task_id = arguments.get("task_id")
        task_id = raw_task_id if isinstance(raw_task_id, str) else None
        raw_epoch = arguments.get("execution_epoch")
        execution_epoch = raw_epoch if isinstance(raw_epoch, int) and not isinstance(raw_epoch, bool) else None
        if current_curator_execution() is not None or task_id is not None:
            execution_token = begin_tool_call(
                task_id=task_id,
                execution_epoch=execution_epoch,
                session_id=getattr(ctx, "session_id", None),
            )
        if internal_tool_is_mutating(name) and getattr(ctx, "direct_mutation_evidence", None) is not None:
            current = current_curator_execution()
            direct_evidence = DirectMutationEvidence.start(
                task_id=task_id if task_id is not None else getattr(current, "task_id", None),
                execution_epoch=execution_epoch if execution_epoch is not None else getattr(current, "execution_epoch", None),
                session_id=getattr(current, "session_id", None) or getattr(ctx, "session_id", None),
                call_id=getattr(current, "call_id", None),
                sequence=getattr(current, "sequence", None),
                tool_name=name,
                arguments=arguments,
            )
            direct_evidence = direct_evidence.finish(
                payload={"before_entities": _before_entity_snapshots(ctx, arguments)},
                ledger_entry={},
            )
    try:
        response = await call_service(
            service,
            ctx,
            arguments,
            compact_success=compact_success,
        )
        if on_success is not None:
            on_success(ctx, name, arguments, response)
        return response
    finally:
        if direct_evidence is not None:
            try:
                _finalize_direct_mutation_evidence(ctx, direct_evidence, arguments, locals().get("response"))
            except Exception:
                tracker = getattr(ctx, "internal_tool_call_tracker", None)
                recorder = getattr(tracker, "record_runtime_error", None)
                if callable(recorder):
                    recorder(
                        "direct_mutation_evidence_persistence_failed",
                        task_id=direct_evidence.task_id,
                        session_id=getattr(ctx, "session_id", None),
                    )
                raise
        if execution_token is not None:
            finish_tool_call(execution_token)


def tool_services() -> dict[str, ToolService]:
    from mcp_memory.mcp.services import (
        commit_skill_review_service,
        skill_review_ledger_service,
        read_memory_record_service,
        read_memory_records_service,
        record_skill_observation_service,
        record_thought_service,
        resolve_skill_observation_service,
        search_memory_records_async_service,
    )

    return {
        "record_thought": record_thought_service,
        "record_skill_observation": record_skill_observation_service,
        "resolve_skill_observation": resolve_skill_observation_service,
        "commit_skill_review": commit_skill_review_service,
        "get_skill_review_ledger": skill_review_ledger_service,
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
        INTERNAL_SEARCH_TOOL_NAME: internal_search_memory_records_async_service,
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


def _finalize_direct_mutation_evidence(
    ctx: ApplicationContext,
    evidence: DirectMutationEvidence,
    arguments: dict[str, object],
    response: object,
) -> None:
    payload = _response_payload(response)
    tracker = getattr(ctx, "internal_tool_call_tracker", None)
    ledger_entry: dict[str, object] = {}
    if isinstance(tracker, InternalToolCallTracker) and evidence.task_id is not None:
        snapshot = tracker.snapshot_task(evidence.task_id)
        if snapshot is not None:
            ledger_entry = next(
                (dict(item) for item in reversed(snapshot.tool_call_ledger)
                 if item.get("call_id") == evidence.call_id),
                {},
            )
    before_entities = evidence.payload.get("before_entities")
    deltas = entity_deltas_for_payload(
        payload,
        arguments,
        repository=getattr(ctx, "repository", None),
        before_entities=before_entities if isinstance(before_entities, dict) else None,
    )
    semantic = _semantic_postcondition(ctx, payload, deltas)
    finalized = reconcile_direct_mutation_evidence(
        evidence.finish(payload=payload, ledger_entry=ledger_entry, deltas=deltas),
        ledger_entry=ledger_entry,
        payload=payload,
        semantic_postcondition=semantic,
    )
    store = getattr(ctx, "direct_mutation_evidence", None)
    saver = getattr(store, "save", None)
    if callable(saver):
        saver(finalized)


def _response_payload(response: object) -> dict[str, object]:
    if not isinstance(response, list):
        return {"status": "error", "error": "missing_response"}
    for content in response:
        if content.type == "text":
            try:
                value = json.loads(content.text)
            except (TypeError, ValueError):
                return {"status": "error", "error": "malformed_response"}
            return value if isinstance(value, dict) else {"status": "error", "error": "malformed_response"}
    return {"status": "error", "error": "missing_response"}


def _semantic_postcondition(
    ctx: ApplicationContext,
    payload: dict[str, object],
    deltas: tuple[object, ...],
) -> bool | None:
    if payload.get("status", "ok") == "error":
        return False
    if not deltas:
        return False
    repository = getattr(ctx, "repository", None)
    return True if repository is not None else None


def _before_entity_snapshots(ctx: ApplicationContext, arguments: dict[str, object]) -> dict[str, dict[str, object]]:
    repository = getattr(ctx, "repository", None)
    getter = getattr(repository, "get_memory", None)
    if not callable(getter):
        return {}
    values: dict[str, dict[str, object]] = {}
    for key in ("memory_id", "source_memory_id", "canonical_memory_id", "source_id", "target_id"):
        value = arguments.get(key)
        if not isinstance(value, str) or value.startswith("ext:"):
            continue
        record = getter(value)
        if record is not None:
            values[value] = {"id": value, "updated_at": getattr(record, "updated_at", None), "status": getattr(record, "status", None)}
    return values
