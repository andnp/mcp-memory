from __future__ import annotations

import json
from dataclasses import dataclass, field
from inspect import isawaitable
from typing import Any, cast

from mcp_memory.context import ApplicationContext
from mcp_memory.mcp.internal_tools import get_internal_maintenance_tools
from mcp_memory.mcp.transport import internal_tool_services


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
) -> InternalToolLoopResult:
    services = internal_tool_services()
    allowed_tools = {
        tool.name: tool
        for tool in get_internal_maintenance_tools()
        if tool.name in allowed_tool_names
    }
    transcript: list[dict[str, Any]] = []
    executed_calls = 0
    successful_calls = 0
    mutating_calls = 0
    tool_names_used: list[str] = []
    handled_entry_ids: list[int] = []
    last_invalid_positive_response: dict[str, Any] | None = None

    for _ in range(max_rounds):
        response = provider.ask(_build_prompt(prompt, allowed_tools, transcript))
        if isawaitable(response):
            response = await response
        if not isinstance(response, dict):
            raise ValueError("provider returned invalid response")

        tool_calls = _normalize_tool_calls(response.get("tool_calls"))
        if not tool_calls:
            reported_actions_taken = _reported_positive_actions_taken(response)
            if reported_actions_taken is not None and executed_calls <= 0:
                last_invalid_positive_response = response
                transcript.append(
                    {
                        "validation_error": (
                            "Do not claim positive actions without first issuing tool_calls. "
                            "If no tools were used, return actions_taken=0 and a no-op summary."
                        ),
                        "invalid_response": response,
                    }
                )
                continue
            return InternalToolLoopResult(
                response=response,
                tool_calls_executed=executed_calls,
                successful_tool_calls=successful_calls,
                mutating_tool_calls=mutating_calls,
                tool_names_used=tool_names_used,
                handled_entry_ids=sorted(set(handled_entry_ids)),
            )

        round_calls = tool_calls[:max_tool_calls_per_round]
        round_results: list[dict[str, Any]] = []
        for tool_call in round_calls:
            name = tool_call["name"]
            arguments = tool_call["arguments"]
            if name not in allowed_tools or name not in services:
                round_results.append(
                    {
                        "name": name,
                        "arguments": arguments,
                        "result": {"status": "error", "error": "tool_not_allowed"},
                    }
                )
                continue

            result = services[name](ctx, arguments)
            executed_calls += 1
            tool_names_used.append(name)
            if result.get("status") == "ok":
                successful_calls += 1
                if _is_mutating_tool_name(name):
                    mutating_calls += 1
                handled_entry_ids.extend(_handled_ingest_entry_ids(name, arguments, result))
            round_results.append({"name": name, "arguments": arguments, "result": result})

        transcript.append({"tool_calls": round_calls, "tool_results": round_results})

    if last_invalid_positive_response is not None and executed_calls <= 0:
        return InternalToolLoopResult(
            response=last_invalid_positive_response,
            tool_calls_executed=executed_calls,
            successful_tool_calls=successful_calls,
            mutating_tool_calls=mutating_calls,
            tool_names_used=tool_names_used,
            handled_entry_ids=sorted(set(handled_entry_ids)),
        )

    return InternalToolLoopResult(
        response={
            "status": "error",
            "error": "tool_loop_exhausted",
            "transcript_rounds": len(transcript),
        },
        tool_calls_executed=executed_calls,
        successful_tool_calls=successful_calls,
        mutating_tool_calls=mutating_calls,
        tool_names_used=tool_names_used,
        handled_entry_ids=sorted(set(handled_entry_ids)),
    )


def _normalize_tool_calls(raw_tool_calls: object) -> list[dict[str, Any]]:
    if not isinstance(raw_tool_calls, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in raw_tool_calls:
        if not isinstance(item, dict):
            continue
        item_dict = cast(dict[str, Any], item)
        name = item_dict.get("name")
        arguments = item_dict.get("arguments", {})
        if not isinstance(name, str) or not name.strip() or not isinstance(arguments, dict):
            continue
        normalized.append({"name": name.strip(), "arguments": dict(arguments)})
    return normalized


def _build_prompt(
    base_prompt: str,
    allowed_tools: dict[str, Any],
    transcript: list[dict[str, Any]],
) -> str:
    tool_specs = [
        _compact_tool_spec(tool)
        for tool in allowed_tools.values()
    ]
    parts = [
        base_prompt.strip(),
        "Available internal tools:",
        json.dumps(tool_specs, sort_keys=True),
        "Respond with `tool_calls` JSON or final JSON.",
        "Never claim positive actions without prior `tool_calls`. If no tools were used, report `actions_taken: 0` and a no-op summary.",
    ]
    if transcript:
        parts.extend(
            [
                "Previous tool interaction transcript:",
                json.dumps(transcript, sort_keys=True),
            ]
        )
    return "\n\n".join(parts)


def _compact_tool_spec(tool: Any) -> dict[str, Any]:
    schema = tool.inputSchema if isinstance(tool.inputSchema, dict) else {}
    properties = schema.get("properties", {}) if isinstance(schema.get("properties", {}), dict) else {}
    required = schema.get("required", []) if isinstance(schema.get("required", []), list) else []
    property_names = [str(name) for name in properties]
    required_names = [name for name in property_names if name in {str(item) for item in required}]
    return {
        "name": tool.name,
        "required_fields": required_names,
    }


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


def _reported_positive_actions_taken(response: dict[str, Any]) -> int | None:
    actions_taken = response.get("actions_taken")
    if not isinstance(actions_taken, int):
        return None
    if actions_taken <= 0:
        return None
    return actions_taken


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
