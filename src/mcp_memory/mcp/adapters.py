from __future__ import annotations

from mcp_memory.mcp.validation import (
    optional_bool,
    optional_nonnegative_int,
    optional_positive_int,
    optional_string,
    require_string,
    string_list,
)


def parse_record_thought_arguments(arguments: dict) -> str:
    return require_string(arguments, "content")


def parse_search_arguments(arguments: dict) -> dict:
    return {
        "query": require_string(arguments, "query"),
        "limit": optional_positive_int(arguments, "limit", 5),
        "adaptive_limit": "limit" not in arguments,
        "workspace_id": optional_string(arguments, "workspace_id"),
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
        "summary_only": optional_bool(arguments, "summary_only", False),
        "content_offset": optional_nonnegative_int(arguments, "content_offset", 0),
        "content_limit": (
            optional_positive_int(arguments, "content_limit", 4_000)
            if arguments.get("content_limit") is not None
            else None
        ),
    }


def parse_batch_read_arguments(arguments: dict, *, caller_kind: str) -> dict:
    field_name = "memory_refs" if "memory_refs" in arguments else "memory_ids"
    memory_ids = list(dict.fromkeys(string_list(arguments, field_name, required=True)))
    maximum = 20 if caller_kind == "external" else 50
    if len(memory_ids) > maximum:
        raise ValueError(f"memory_ids must contain at most {maximum} identifiers")
    return {
        "memory_ids": memory_ids,
        "include_relationships": optional_bool(
            arguments, "include_relationships", caller_kind == "internal"
        ),
        "include_superseded": optional_bool(
            arguments, "include_superseded", caller_kind == "internal"
        ),
        "include_metadata": optional_bool(arguments, "include_metadata", False),
        "summary_only": optional_bool(arguments, "summary_only", False),
        "content_offset": optional_nonnegative_int(arguments, "content_offset", 0),
        "content_limit": (
            optional_positive_int(arguments, "content_limit", 4_000)
            if arguments.get("content_limit") is not None
            else None
        ),
    }
