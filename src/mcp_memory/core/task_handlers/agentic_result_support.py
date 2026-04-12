from __future__ import annotations

import json
from typing import Any, cast


def coerce_non_negative_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(value, 0)
    return 0


def coerce_text_summary(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def extract_embedded_json_object(text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        json_start = text.find("{")
        json_end = text.rfind("}") + 1
        if json_start < 0 or json_end <= json_start:
            return None
        try:
            parsed = json.loads(text[json_start:json_end])
        except json.JSONDecodeError:
            return None
    if isinstance(parsed, dict):
        return parsed
    return None


def extract_tool_payload(parsed: object) -> dict[str, Any]:
    if not isinstance(parsed, dict):
        return {}
    raw_tool_stats = parsed.get("stats")
    if not isinstance(raw_tool_stats, dict):
        return {}
    raw_tool_payload = raw_tool_stats.get("tools")
    if not isinstance(raw_tool_payload, dict):
        return {}
    return cast(dict[str, Any], raw_tool_payload)


def extract_tool_counts(parsed: object) -> dict[str, Any]:
    tool_payload = extract_tool_payload(parsed)
    raw_tool_counts = tool_payload.get("byName")
    if not isinstance(raw_tool_counts, dict):
        return {}
    return cast(dict[str, Any], raw_tool_counts)


def extract_agentic_tool_names(value: object) -> list[str]:
    if not isinstance(value, dict):
        return []
    return sorted(str(name) for name, payload in value.items() if isinstance(name, str) and isinstance(payload, dict))


def count_mutating_agentic_tool_calls(
    value: object,
    *,
    read_only_tool_names: set[str],
) -> int:
    if not isinstance(value, dict):
        return 0
    total = 0
    for name, payload in value.items():
        if not isinstance(name, str) or name in read_only_tool_names or not isinstance(payload, dict):
            continue
        payload_dict = cast(dict[str, Any], payload)
        total += coerce_non_negative_int(payload_dict.get("count"))
    return total


def build_tool_usage_summary(
    parsed: object,
    *,
    read_only_tool_names: set[str],
) -> dict[str, Any]:
    tool_payload = extract_tool_payload(parsed)
    tool_counts = extract_tool_counts(parsed)
    return {
        "tool_calls_executed": coerce_non_negative_int(tool_payload.get("totalCalls")),
        "mutations": count_mutating_agentic_tool_calls(
            tool_counts,
            read_only_tool_names=read_only_tool_names,
        ),
        "tool_names_used": extract_agentic_tool_names(tool_counts),
    }
