from __future__ import annotations

from typing import Any

from mcp_memory.context import ApplicationContext
from mcp_memory.core.system1_scheduling import resolve_pending_workspace_id
from mcp_memory.core.task_handlers.ingest_batch_support import build_ingest_groups
from mcp_memory.core.ingest_claim_lifecycle import _journal_entry_payload, _record_ingest_tool_invocation


FIFO_GROUPING_STRATEGY = "fifo"
SEMANTIC_SEEDED_GROUPING_STRATEGY = "semantic-seeded"
LEXICAL_SEEDED_GROUPING_STRATEGY = "lexical-seeded"
INGEST_GROUPING_STRATEGIES = (
    FIFO_GROUPING_STRATEGY,
    SEMANTIC_SEEDED_GROUPING_STRATEGY,
    LEXICAL_SEEDED_GROUPING_STRATEGY,
)


def _requested_grouping_strategy(values: dict[str, Any]) -> str | None:
    raw_value = values.get("grouping_strategy")
    if isinstance(raw_value, str) and raw_value.strip():
        return raw_value.strip()
    return None


def _resolve_grouping_strategy(
    ctx: ApplicationContext,
    *,
    requested_strategy: str | None,
) -> tuple[str, str | None]:
    embedder = getattr(ctx, "embedder", None)
    vector_store = getattr(ctx, "vector_store", None)
    default_strategy = (
        SEMANTIC_SEEDED_GROUPING_STRATEGY
        if embedder is not None and vector_store is not None
        else LEXICAL_SEEDED_GROUPING_STRATEGY
    )
    if requested_strategy is None:
        return default_strategy, None
    if requested_strategy not in INGEST_GROUPING_STRATEGIES:
        return default_strategy, "unknown_requested_grouping_strategy"
    if requested_strategy == SEMANTIC_SEEDED_GROUPING_STRATEGY and (embedder is None or vector_store is None):
        return LEXICAL_SEEDED_GROUPING_STRATEGY, "semantic_grouping_unavailable"
    return requested_strategy, None


def build_next_ingest_batch_payload(ctx: ApplicationContext, arguments: dict[str, Any]) -> dict[str, Any]:
    if ctx.journal is None:
        return {"status": "error", "error": "journal_not_initialized"}

    from mcp_memory.mcp.validation import optional_positive_int, optional_string, require_string

    task_id = require_string(arguments, "task_id")
    requested_workspace_id = optional_string(arguments, "workspace_id")
    journal_workspace_id = resolve_pending_workspace_id(
        ctx.journal,
        requested_workspace_id,
    )
    workspace_id = requested_workspace_id or ctx.workspace_id or "workspace-unknown"
    batch_size = optional_positive_int(arguments, "batch_size", 20)
    grouping_strategy_requested = optional_string(arguments, "grouping_strategy")
    grouping_strategy_used, grouping_fallback_reason = _resolve_grouping_strategy(
        ctx,
        requested_strategy=grouping_strategy_requested,
    )

    entries = ctx.journal.claim_pending(
        task_id=task_id,
        limit=batch_size,
        workspace_id=journal_workspace_id,
    )
    groups = build_ingest_groups(
        ctx,
        entries,
        workspace_id,
        task_id=task_id,
        grouping_strategy=grouping_strategy_used,
    )
    pending_remaining = ctx.journal.count_by_status(workspace_id=journal_workspace_id).get("pending", 0)
    _record_ingest_tool_invocation(
        ctx,
        task_id=task_id,
        tool_name="internal_get_next_ingest_batch",
        mutation=False,
    )
    return {
        "status": "ok",
        "task_id": task_id,
        "requested_grouping_strategy": grouping_strategy_requested,
        "grouping_strategy_used": grouping_strategy_used,
        "grouping_fallback_reason": grouping_fallback_reason,
        "claimed_entry_ids": [entry.id for entry in entries],
        "pending_remaining": pending_remaining,
        "has_more": pending_remaining > 0,
        "group_count": len(groups),
        "groups": [
            {
                "group_index": index,
                "entries": [_journal_entry_payload(entry) for entry in group],
            }
            for index, group in enumerate(groups)
        ],
    }