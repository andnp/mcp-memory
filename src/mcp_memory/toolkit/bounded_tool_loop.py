"""Generic bounded agentic tool-loop with an anti-hallucination guard.

Provider-agnostic ReAct-style loop: caps rounds and calls-per-round, enforces
a tool allowlist, builds a running transcript, and detects a model claiming
positive actions (``actions_taken > 0``) without ever issuing a tool call,
forcing a correction round instead of accepting the unverified claim.

Domain-specific tool discovery, dispatch, and result bookkeeping are injected
by the caller; this module has no knowledge of any particular tool set.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from inspect import isawaitable
from typing import Any, cast


@dataclass(slots=True)
class ToolSpec:
    """Minimal tool description surfaced to the provider prompt."""

    name: str
    required_fields: list[str] = field(default_factory=list)


ToolDispatch = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]
ToolCallObserver = Callable[[str, dict[str, Any], dict[str, Any]], None]


@dataclass(slots=True)
class BoundedToolLoopResult:
    response: dict[str, Any]
    tool_calls_executed: int = 0
    successful_tool_calls: int = 0
    tool_names_used: list[str] = field(default_factory=list)
    transcript_rounds: int = 0


async def run_bounded_tool_loop(
    provider: Any,
    dispatch: ToolDispatch,
    *,
    prompt: str,
    allowed_tools: dict[str, ToolSpec],
    max_rounds: int = 4,
    max_tool_calls_per_round: int = 5,
    unsupported_no_tool_response_error: Callable[[dict[str, Any], int], str | None] | None = None,
    on_tool_call_result: ToolCallObserver | None = None,
) -> BoundedToolLoopResult:
    transcript: list[dict[str, Any]] = []
    executed_calls = 0
    successful_calls = 0
    tool_names_used: list[str] = []
    last_invalid_positive_response: dict[str, Any] | None = None

    for _ in range(max_rounds):
        response = provider.ask(_build_prompt(prompt, allowed_tools, transcript))
        if isawaitable(response):
            response = await response
        if not isinstance(response, dict):
            raise ValueError("provider returned invalid response")

        tool_calls = _normalize_tool_calls(response.get("tool_calls"))
        if not tool_calls:
            validation_error = _default_unsupported_no_tool_response_error(response, executed_calls)
            if validation_error is None and unsupported_no_tool_response_error is not None:
                validation_error = unsupported_no_tool_response_error(response, executed_calls)
            if validation_error is not None:
                last_invalid_positive_response = response
                transcript.append(
                    {
                        "validation_error": validation_error,
                        "invalid_response": response,
                    }
                )
                continue
            return BoundedToolLoopResult(
                response=response,
                tool_calls_executed=executed_calls,
                successful_tool_calls=successful_calls,
                tool_names_used=tool_names_used,
                transcript_rounds=len(transcript),
            )

        round_calls = tool_calls[:max_tool_calls_per_round]
        round_results: list[dict[str, Any]] = []
        for tool_call in round_calls:
            name = tool_call["name"]
            arguments = tool_call["arguments"]
            if name not in allowed_tools:
                round_results.append(
                    {
                        "name": name,
                        "arguments": arguments,
                        "result": {"status": "error", "error": "tool_not_allowed"},
                    }
                )
                continue

            result = await dispatch(name, arguments)
            executed_calls += 1
            tool_names_used.append(name)
            if result.get("status") == "ok":
                successful_calls += 1
            if on_tool_call_result is not None:
                on_tool_call_result(name, arguments, result)
            round_results.append({"name": name, "arguments": arguments, "result": result})

        transcript.append({"tool_calls": round_calls, "tool_results": round_results})

    if last_invalid_positive_response is not None and executed_calls <= 0:
        return BoundedToolLoopResult(
            response=last_invalid_positive_response,
            tool_calls_executed=executed_calls,
            successful_tool_calls=successful_calls,
            tool_names_used=tool_names_used,
            transcript_rounds=len(transcript),
        )

    return BoundedToolLoopResult(
        response={
            "status": "error",
            "error": "tool_loop_exhausted",
            "transcript_rounds": len(transcript),
        },
        tool_calls_executed=executed_calls,
        successful_tool_calls=successful_calls,
        tool_names_used=tool_names_used,
        transcript_rounds=len(transcript),
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
    allowed_tools: dict[str, ToolSpec],
    transcript: list[dict[str, Any]],
) -> str:
    tool_specs = [
        {"name": tool.name, "required_fields": tool.required_fields}
        for tool in allowed_tools.values()
    ]
    parts = [
        base_prompt.strip(),
        "Available internal tools:",
        json.dumps(tool_specs, sort_keys=True, separators=(",", ":")),
        "Return `tool_calls` JSON or final JSON.",
        "Never claim positive actions without prior `tool_calls`; "
        "otherwise return `actions_taken: 0` and a no-op summary.",
    ]
    if transcript:
        parts.extend(
            [
                "Previous tool interaction transcript:",
                json.dumps(transcript, sort_keys=True, separators=(",", ":")),
            ]
        )
    return "\n\n".join(parts)


def _reported_positive_actions_taken(response: dict[str, Any]) -> int | None:
    actions_taken = response.get("actions_taken")
    if not isinstance(actions_taken, int):
        return None
    if actions_taken <= 0:
        return None
    return actions_taken


def _default_unsupported_no_tool_response_error(
    response: dict[str, Any], executed_calls: int
) -> str | None:
    if executed_calls > 0:
        return None
    reported_actions_taken = _reported_positive_actions_taken(response)
    if reported_actions_taken is None:
        return None
    return (
        "Do not claim positive actions without first issuing tool_calls. "
        "If no tools were used, return actions_taken=0 and a no-op summary."
    )
