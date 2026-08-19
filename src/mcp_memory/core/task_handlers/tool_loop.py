from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable

from mcp_memory.context import ApplicationContext
from mcp_memory.core.ports.tool_dispatch import ToolDefinition, ToolDispatchPort
from mcp_memory.toolkit.bounded_tool_loop import ToolSpec, run_bounded_tool_loop


@dataclass(slots=True)
class InternalToolLoopResult:
    response: dict[str, Any]
    tool_calls_executed: int = 0
    successful_tool_calls: int = 0
    mutating_tool_calls: int = 0
    tool_names_used: list[str] = field(default_factory=list)
    handled_entry_ids: list[int] = field(default_factory=list)


async def run_internal_tool_loop(
    ctx: ApplicationContext,
    provider: Any,
    *,
    prompt: str,
    allowed_tool_names: list[str],
    max_rounds: int = 4,
    max_tool_calls_per_round: int = 5,
    unsupported_no_tool_response_error: Callable[[dict[str, Any], int], str | None] | None = None,
) -> InternalToolLoopResult:
    tool_dispatch = ctx.tool_dispatch
    if tool_dispatch is None:
        raise RuntimeError("internal_tool_dispatch_unavailable")
    allowed_tools = {
        tool.name: _compact_tool_spec(tool)
        for tool in tool_dispatch.available_tools().values()
        if tool.name in allowed_tool_names
    }

    mutating_calls = 0
    handled_entry_ids: list[int] = []

    def _track_tool_call_result(name: str, arguments: dict[str, Any], result: dict[str, Any]) -> None:
        nonlocal mutating_calls
        if result.get("status") != "ok":
            return
        if _is_mutating_tool_name(name):
            mutating_calls += 1
        handled_entry_ids.extend(_handled_ingest_entry_ids(name, arguments, result))

    async def _dispatch(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return await _dispatch_internal_tool(tool_dispatch, ctx, name, arguments)

    result = await run_bounded_tool_loop(
        provider,
        _dispatch,
        prompt=prompt,
        allowed_tools=allowed_tools,
        max_rounds=max_rounds,
        max_tool_calls_per_round=max_tool_calls_per_round,
        unsupported_no_tool_response_error=unsupported_no_tool_response_error,
        on_tool_call_result=_track_tool_call_result,
    )

    return InternalToolLoopResult(
        response=result.response,
        tool_calls_executed=result.tool_calls_executed,
        successful_tool_calls=result.successful_tool_calls,
        mutating_tool_calls=mutating_calls,
        tool_names_used=result.tool_names_used,
        handled_entry_ids=sorted(set(handled_entry_ids)),
    )


async def _dispatch_internal_tool(
    tool_dispatch: ToolDispatchPort[ApplicationContext],
    ctx: ApplicationContext,
    name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    response = await tool_dispatch.dispatch(ctx, name, arguments)
    if not response:
        return {"status": "error", "error": "empty_tool_response", "tool": name}

    first_part = response[0]
    text = first_part.text
    if not isinstance(text, str):
        return {"status": "error", "error": "invalid_tool_response", "tool": name}

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {
            "status": "error",
            "error": "invalid_tool_response",
            "tool": name,
            "detail": "tool did not return valid JSON",
        }

    if not isinstance(payload, dict):
        return {
            "status": "error",
            "error": "invalid_tool_response",
            "tool": name,
            "detail": "tool returned a non-object payload",
        }
    return payload


def _compact_tool_spec(tool: ToolDefinition) -> ToolSpec:
    schema = tool.input_schema
    raw_properties = schema.get("properties", {})
    properties: Mapping[object, object] = raw_properties if isinstance(raw_properties, Mapping) else {}
    raw_required = schema.get("required", [])
    required: list[object] = raw_required if isinstance(raw_required, list) else []
    property_names = [str(name) for name in properties]
    required_names = [name for name in property_names if name in {str(item) for item in required}]
    return ToolSpec(name=tool.name, required_fields=required_names)


def _is_mutating_tool_name(name: str) -> bool:
    return name.startswith(
        (
            "internal_append_",
            "internal_archive_",
            "internal_create_",
            "internal_delete_",
            "internal_merge_",
            "internal_split_",
            "internal_update_",
        )
    )


def _handled_ingest_entry_ids(name: str, arguments: dict[str, Any], result: dict[str, Any]) -> list[int]:
    if name not in {"internal_ingest_append_memory", "internal_ingest_create_memory"}:
        return []
    if result.get("status") != "ok":
        return []
    raw_entry_ids = result.get("handled_entry_ids", arguments.get("entry_ids"))
    if not isinstance(raw_entry_ids, list):
        return []
    return [
        entry_id
        for entry_id in raw_entry_ids
        if isinstance(entry_id, int) and not isinstance(entry_id, bool) and entry_id > 0
    ]
