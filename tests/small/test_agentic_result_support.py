from __future__ import annotations

from mcp_memory.core.task_handlers.agentic_result_support import (
    build_tool_usage_summary,
    coerce_non_negative_int,
    extract_embedded_json_object,
)


pytestmark = __import__("pytest").mark.small


def test_build_tool_usage_summary_counts_mutations_and_tool_names() -> None:
    parsed = {
        "stats": {
            "tools": {
                "totalCalls": 4,
                "byName": {
                    "read_only_tool": {"count": 2},
                    "mutating_tool": {"count": 2},
                },
            }
        }
    }

    payload = build_tool_usage_summary(parsed, read_only_tool_names={"read_only_tool"})

    assert payload == {
        "tool_calls_executed": 4,
        "mutations": 2,
        "tool_names_used": ["mutating_tool", "read_only_tool"],
    }


def test_agentic_result_support_handles_embedded_json_and_non_negative_ints() -> None:
    assert extract_embedded_json_object('prefix {"summary": "done"} suffix') == {"summary": "done"}
    assert coerce_non_negative_int(True) == 0
    assert coerce_non_negative_int(-3) == 0
    assert coerce_non_negative_int(5) == 5
