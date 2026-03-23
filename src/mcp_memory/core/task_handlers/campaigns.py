from __future__ import annotations

from typing import Any

from mcp_memory.work_item_store import compatibility_group_families


def campaign_metadata(
    *,
    compatibility_group: str,
    origin_family: str,
    execution_lane: str,
    allowed_families: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    family_keys = compatibility_group_families(
        compatibility_group,
        allowed_families=list(allowed_families) if allowed_families is not None else None,
    )
    return {
        "campaign_key": compatibility_group,
        "campaign_origin_family": origin_family,
        "campaign_family_keys": list(family_keys),
        "campaign_continuation_supported": True,
        "compatibility_group": compatibility_group,
        "work_item_execution_lane": execution_lane,
    }


def campaign_family_keys(
    compatibility_group: str,
    *,
    allowed_families: list[str] | tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    return compatibility_group_families(
        compatibility_group,
        allowed_families=list(allowed_families) if allowed_families is not None else None,
    )


def count_named_tool_calls(tool_names_used: list[str], *, tool_name: str) -> int:
    suffix = f"_{tool_name}"
    return sum(1 for used_name in tool_names_used if used_name == tool_name or used_name.endswith(suffix))


def count_named_tool_calls_from_stats(tool_counts: object, *, tool_name: str) -> int:
    if not isinstance(tool_counts, dict):
        return 0
    total = 0
    suffix = f"_{tool_name}"
    for used_name, payload in tool_counts.items():
        if not isinstance(used_name, str) or not isinstance(payload, dict):
            continue
        if used_name != tool_name and not used_name.endswith(suffix):
            continue
        count = payload.get("count")
        if isinstance(count, int) and not isinstance(count, bool) and count > 0:
            total += count
    return total