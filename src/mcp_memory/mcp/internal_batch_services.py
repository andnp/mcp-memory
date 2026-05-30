from __future__ import annotations

from mcp_memory.context import ApplicationContext
from mcp_memory.mcp.internal_work_item_services import resolve_internal_workspace_scope
from mcp_memory.mcp.validation import optional_positive_int, optional_string, string_list
from mcp_memory.serialization import compact_memory_record_payload


def internal_list_memory_records_service(ctx: ApplicationContext, arguments: dict) -> dict:
    if ctx.repository is None:
        return {"status": "error", "error": "repository_not_initialized"}
    records = ctx.repository.list_memories(
        workspace_id=resolve_internal_workspace_scope(ctx, arguments),
        memory_type=optional_string(arguments, "memory_type"),
        status=optional_string(arguments, "status"),
        limit=optional_positive_int(arguments, "limit", 10),
    )
    return {
        "status": "ok",
        "records": [compact_memory_record_payload(record).model_dump() for record in records],
    }


def internal_get_next_dedup_batch_service(ctx: ApplicationContext, arguments: dict) -> dict:
    if ctx.repository is None:
        return {"status": "error", "error": "repository_not_initialized"}

    from mcp_memory.core.task_handlers.constants import DEDUPLICATOR_TASK_NAME
    from mcp_memory.core.task_handlers.deduplicator_handlers import select_deduplicator_seed_batch

    task_id = optional_string(arguments, "task_id") or f"{DEDUPLICATOR_TASK_NAME}:internal"
    strategy = optional_string(arguments, "strategy")
    workspace_id = resolve_internal_workspace_scope(ctx, arguments)
    limit = optional_positive_int(arguments, "limit", 100)
    candidates = [
        record
        for record in ctx.repository.list_memories(
            workspace_id=workspace_id,
            status="active",
            limit=limit,
        )
        if not ctx.repository.has_incoming_link(record.id, "SUPERSEDES")
    ]
    seed_batch = select_deduplicator_seed_batch(
        ctx,
        candidates,
        task_id=task_id,
        strategy=strategy,
    )
    return {
        "status": "ok",
        "requested_strategy": seed_batch.requested_strategy,
        "strategy": seed_batch.strategy_used,
        "strategy_used": seed_batch.strategy_used,
        "strategy_fallback_reason": seed_batch.strategy_fallback_reason,
        "candidate_count": seed_batch.candidate_count,
        "has_more": len(candidates) > len(seed_batch.records),
        "sampled_memory_ids": [record.id for record in seed_batch.records],
        "records": [compact_memory_record_payload(record).model_dump() for record in seed_batch.records],
    }


def internal_get_next_curator_batch_service(ctx: ApplicationContext, arguments: dict) -> dict:
    if ctx.repository is None:
        return {"status": "error", "error": "repository_not_initialized"}

    from mcp_memory.core.task_handlers.constants import CURATOR_TASK_NAME
    from mcp_memory.core.task_handlers.curator_support import (
        CURATOR_MAX_SEED_RECORDS,
        select_curator_seed_batch,
    )
    from mcp_memory.core.tasks import TaskRecord

    task_id = optional_string(arguments, "task_id") or f"{CURATOR_TASK_NAME}:internal"
    strategy = optional_string(arguments, "strategy")
    workspace_id = resolve_internal_workspace_scope(ctx, arguments)
    limit = optional_positive_int(arguments, "limit", CURATOR_MAX_SEED_RECORDS)
    exclude_memory_ids = set(string_list(arguments, "exclude_memory_ids"))
    task_data = {"strategy": strategy} if strategy else {}
    if workspace_id is not None:
        task_data["workspace_id"] = workspace_id
    task = TaskRecord(
        id=task_id,
        task_name=CURATOR_TASK_NAME,
        data=task_data,
        workspace_id=workspace_id,
        status="running",
        priority=100,
        retries_count=0,
        max_retries=3,
        created_at=0.0,
        updated_at=0.0,
        available_at=0.0,
        claimed_at=0.0,
        started_at=0.0,
        completed_at=None,
        last_error=None,
    )
    seed_batch = select_curator_seed_batch(
        ctx,
        task,
        seed_limit=limit,
        exclude_memory_ids=exclude_memory_ids,
    )
    return {
        "status": "ok",
        "requested_strategy": seed_batch.requested_strategy,
        "strategy": seed_batch.strategy_used,
        "strategy_used": seed_batch.strategy_used,
        "strategy_fallback_reason": seed_batch.strategy_fallback_reason,
        "candidate_count": seed_batch.candidate_count,
        "strategy_selection_mode": seed_batch.strategy_selection_mode,
        "strategy_selection_reason": seed_batch.strategy_selection_reason,
        "strategy_selection_scores": seed_batch.strategy_selection_scores,
        "excluded_memory_ids": sorted(exclude_memory_ids),
        "has_more": seed_batch.candidate_count > len(seed_batch.records),
        "sampled_memory_ids": [record.id for record in seed_batch.records],
        "records": [compact_memory_record_payload(record).model_dump() for record in seed_batch.records],
    }
