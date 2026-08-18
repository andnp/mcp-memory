from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from mcp_memory.context import ApplicationContext
from mcp_memory.internal_tool_call_tracking import (
    InternalToolCallSnapshot,
    internal_tool_is_mutating,
)

_MAX_LEDGER_VALIDATION_ISSUES = 8


def reset_agentic_tool_tracking(
    ctx: ApplicationContext,
    task_id: str,
    *,
    execution_epoch: int = 0,
) -> None:
    tracker = getattr(ctx, "internal_tool_call_tracker", None)
    reset_task = getattr(tracker, "reset_task", None)
    if callable(reset_task):
        reset_task(task_id, session_id=ctx.session_id, execution_epoch=execution_epoch)


def snapshot_agentic_tool_tracking(ctx: ApplicationContext, task_id: str) -> Any:
    tracker = getattr(ctx, "internal_tool_call_tracker", None)
    snapshot_task = getattr(tracker, "snapshot_task", None)
    if callable(snapshot_task):
        return snapshot_task(task_id)
    return None


def finalize_agentic_tool_tracking(ctx: ApplicationContext, task_id: str) -> Any:
    tracker = getattr(ctx, "internal_tool_call_tracker", None)
    finalize_task = getattr(tracker, "finalize_task", None)
    if callable(finalize_task):
        return finalize_task(task_id)
    return None


def validate_agentic_tool_tracking_snapshot(
    snapshot: InternalToolCallSnapshot | None,
    *,
    allowed_tool_names: Iterable[str],
) -> dict[str, Any]:
    issues: list[dict[str, str]] = []

    def add_issue(code: str, message: str) -> None:
        if len(issues) < _MAX_LEDGER_VALIDATION_ISSUES:
            issues.append({"code": code, "message": message})

    if snapshot is None:
        return {"valid": False, "issues": [{"code": "snapshot_missing", "message": "snapshot is missing"}]}

    ledger = getattr(snapshot, "tool_call_ledger", None)
    total_calls = getattr(snapshot, "total_calls", None)
    if not isinstance(ledger, list):
        ledger = []
        add_issue("ledger_invalid", "ledger is not a list")
    if len(ledger) != total_calls:
        add_issue("ledger_length_mismatch", "ledger length does not equal total_calls")

    allowed = set(allowed_tool_names)
    successful_mutations = 0
    for expected_sequence, entry in enumerate(ledger, start=1):
        if not isinstance(entry, Mapping):
            add_issue("entry_invalid", f"ledger entry {expected_sequence} is not an object")
            continue
        sequence = entry.get("sequence")
        if isinstance(sequence, bool) or sequence != expected_sequence:
            add_issue("sequence_invalid", f"ledger entry {expected_sequence} has an invalid sequence")
        tool_name = entry.get("tool_name")
        if not isinstance(tool_name, str) or tool_name not in allowed:
            add_issue("tool_not_allowed", f"ledger entry {expected_sequence} uses a disallowed tool")
        status = entry.get("status")
        if status not in {"success", "error"}:
            add_issue("status_invalid", f"ledger entry {expected_sequence} has an invalid status")
        kind = entry.get("kind")
        if kind not in {"read", "mutation"}:
            add_issue("kind_invalid", f"ledger entry {expected_sequence} has an invalid kind")
        expected_kind = "mutation" if isinstance(tool_name, str) and internal_tool_is_mutating(tool_name) else "read"
        if kind in {"read", "mutation"} and kind != expected_kind:
            add_issue("kind_mismatch", f"ledger entry {expected_sequence} has a mismatched kind")
        if status == "success" and expected_kind == "mutation":
            successful_mutations += 1

    if successful_mutations != getattr(snapshot, "mutating_calls", None):
        add_issue("mutation_count_mismatch", "mutating_calls does not equal successful mutations")
    return {"valid": not issues, "issues": issues}


def prefer_deterministic_agentic_counts(
    result: dict[str, Any],
    *,
    deterministic_counts: Any,
) -> dict[str, Any]:
    if deterministic_counts is None:
        return result
    normalized = dict(result)
    normalized["tool_calls_executed"] = int(getattr(deterministic_counts, "total_calls", 0))
    normalized["mutations"] = int(getattr(deterministic_counts, "mutating_calls", 0))
    normalized["tool_names_used"] = list(getattr(deterministic_counts, "tool_names_used", []))
    return normalized
