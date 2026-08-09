from __future__ import annotations

from mcp_memory.mcp.validation import (
    optional_bool,
    optional_nonnegative_int,
    optional_positive_int,
    optional_string,
    require_string,
    string_list,
)
from mcp_memory.application.skill_review_contract import (
    SkillReviewCommitRequest,
    SkillReviewLedgerRequest,
    parse_skill_review_commit_request,
    parse_skill_review_ledger_request,
)


def parse_record_thought_arguments(arguments: dict) -> str:
    return require_string(arguments, "content")


def parse_skill_observation_arguments(arguments: dict) -> dict[str, object]:
    return {
        "title": require_string(arguments, "title"),
        "summary": require_string(arguments, "summary"),
        "content": require_string(arguments, "content"),
        "skill": require_string(arguments, "skill"),
        "observation_kind": require_string(arguments, "observation_kind"),
        "privacy_classification": require_string(arguments, "privacy_classification"),
        "workspace_id": optional_string(arguments, "workspace_id"),
    }


def parse_resolve_skill_observation_arguments(arguments: dict) -> dict[str, object]:
    resolution = require_string(arguments, "resolution")
    if resolution not in {"actioned", "deferred", "verified"}:
        raise ValueError("resolution must be actioned, deferred, or verified")
    return {
        "memory_id": require_string(arguments, "memory_id"),
        "resolution": resolution,
        "note": require_string(arguments, "note"),
    }


def parse_commit_skill_review_arguments(arguments: dict) -> SkillReviewCommitRequest:
    """Parse the versioned batch disposition request at the MCP boundary."""
    return parse_skill_review_commit_request(arguments)


def parse_skill_review_ledger_arguments(arguments: dict) -> SkillReviewLedgerRequest:
    """Parse a versioned read-only ledger page request."""
    return parse_skill_review_ledger_request(arguments)


def parse_search_arguments(arguments: dict) -> dict:
    return {
        "query": require_string(arguments, "query"),
        "limit": optional_positive_int(arguments, "limit", 5),
        "adaptive_limit": "limit" not in arguments,
        "workspace_id": optional_string(arguments, "workspace_id"),
        "memory_type": optional_string(arguments, "memory_type"),
        "status": optional_string(arguments, "status"),
        "tags": tuple(string_list(arguments, "tags")),
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
