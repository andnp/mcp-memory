from __future__ import annotations

from mcp_memory.mcp.validation import (
    optional_bool,
    optional_positive_int,
    optional_string,
    require_string,
)


def parse_record_thought_arguments(arguments: dict) -> str:
    return require_string(arguments, "content")


def parse_search_arguments(arguments: dict) -> dict:
    return {
        "query": require_string(arguments, "query"),
        "limit": optional_positive_int(arguments, "limit", 5),
        "adaptive_limit": "limit" not in arguments,
        "memory_type": optional_string(arguments, "memory_type"),
        "status": optional_string(arguments, "status"),
        "include_superseded": optional_bool(arguments, "include_superseded", False),
        "debug": optional_bool(arguments, "debug", False),
    }


def parse_read_arguments(arguments: dict, *, caller_kind: str) -> dict:
    return {
        "memory_id": require_string(arguments, "memory_id"),
        "include_relationships": optional_bool(
            arguments, "include_relationships", caller_kind == "internal"
        ),
        "include_superseded": optional_bool(
            arguments, "include_superseded", caller_kind == "internal"
        ),
        "include_metadata": optional_bool(arguments, "include_metadata", False),
    }
