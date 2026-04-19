from __future__ import annotations

from typing import Any

from mcp_memory.context import ApplicationContext
import mcp_memory.core.task_handlers.deduplicator_merge as _deduplicator_merge
import mcp_memory.core.task_handlers.deduplicator_support as _deduplicator_support
from mcp_memory.core.task_handlers.constants import DEFAULT_AGENT_SCAN_LIMIT
from mcp_memory.core.task_handlers.maintenance_framework import requested_sampling_strategy, sampling_payload
from mcp_memory.core.task_handlers.maintenance_housekeeping import _resolve_workspace_id
from mcp_memory.core.task_handlers.maintenance_work_items import (
    enqueue_review_work_item,
    work_item_result_metadata,
)
from mcp_memory.core.tasks import TaskRecord
from mcp_memory.work_item_store import (
    EXECUTION_LANE_AGENTIC,
    WORK_FAMILY_MEMORY_DEDUP_REVIEW,
)


async def handle_deduplicator_task(
    ctx: ApplicationContext,
    task: TaskRecord,
    provider: Any = None,
) -> dict[str, Any]:
    if ctx.repository is None:
        return {"merged": 0, "archived": 0, "absorbed_observations": 0}
    del provider

    workspace_id = _resolve_workspace_id(ctx, task)
    candidates = [
        record
        for record in ctx.repository.list_memories(
            workspace_id=workspace_id,
            status="active",
            limit=int(task.data.get("limit", DEFAULT_AGENT_SCAN_LIMIT)),
        )
        if not ctx.repository.has_incoming_link(record.id, "SUPERSEDES")
    ]
    seed_batch = _deduplicator_support.select_deduplicator_seed_batch(
        ctx,
        candidates,
        task_id=task.id,
        strategy=requested_sampling_strategy(task),
    )
    seed_records = seed_batch.records
    facts = [record for record in seed_records if record.type == "fact"]
    if not facts:
        return sampling_payload(
            seed_batch,
            sampled_records=seed_records,
            seed_records=seed_records,
            merged=0,
            archived=0,
            absorbed_observations=0,
        )

    deterministic_result = await _deduplicator_merge.run_deterministic_deduplicator_pass(
        ctx,
        task,
        candidates,
        seed_records,
    )

    if _deduplicator_result_has_effective_change(deterministic_result):
        return sampling_payload(
            seed_batch,
            sampled_records=seed_records,
            seed_records=seed_records,
            merged=deterministic_result["merged"],
            archived=deterministic_result["archived"],
            absorbed_observations=deterministic_result["absorbed_observations"],
        )

    seeded_review = _seed_dedup_review_from_seed_batch(
        ctx,
        task=task,
        workspace_id=workspace_id,
        seed_batch=seed_batch,
        seed_records=seed_records,
        candidates=candidates,
    )

    return sampling_payload(
        seed_batch,
        sampled_records=seed_records,
        seed_records=seeded_review["packet_records"],
        extra=seeded_review["metadata"],
        merged=deterministic_result["merged"],
        archived=deterministic_result["archived"],
        absorbed_observations=deterministic_result["absorbed_observations"],
        seeded_work_item_count=seeded_review["seeded_work_item_count"],
    )


def _enqueue_dedup_review_work_item(
    ctx: ApplicationContext,
    *,
    task: TaskRecord,
    workspace_id: str | None,
    seed_records: list[Any],
    candidates: list[Any],
    strategy_used: str | None,
    candidate_count: int,
) -> tuple[Any, bool]:
    support_records = _deduplicator_support.select_deduplicator_support_records(seed_records, candidates)
    return enqueue_review_work_item(
        ctx,
        task=task,
        family_key=WORK_FAMILY_MEMORY_DEDUP_REVIEW,
        execution_lane=EXECUTION_LANE_AGENTIC,
        workspace_id=workspace_id,
        idempotency_prefix="memory_dedup_review",
        payload_memory_ids_key="seed_memory_ids",
        memory_ids=[record.id for record in seed_records],
        strategy_used=strategy_used,
        candidate_count=candidate_count,
        extra_payload={
            "support_memory_ids": [record.id for record in support_records],
            "packet_record_count": len(seed_records) + len(support_records),
        },
    )


def _deduplicator_result_has_effective_change(result: dict[str, Any]) -> bool:
    return any(
        isinstance(result.get(metric), int) and result.get(metric, 0) > 0
        for metric in ("merged", "archived", "absorbed_observations")
    )


def _seed_dedup_review_from_seed_batch(
    ctx: ApplicationContext,
    *,
    task: TaskRecord,
    workspace_id: str | None,
    seed_batch: Any,
    seed_records: list[Any],
    candidates: list[Any],
) -> dict[str, Any]:
    created_work_item, created = _enqueue_dedup_review_work_item(
        ctx,
        task=task,
        workspace_id=workspace_id,
        seed_records=seed_records,
        candidates=candidates,
        strategy_used=seed_batch.strategy_used,
        candidate_count=seed_batch.candidate_count,
    )
    packet_records = seed_records + _deduplicator_support.select_deduplicator_support_records(seed_records, candidates)
    seeded_work_item_count = 1 if created else 0
    return {
        "packet_records": packet_records,
        "seeded_work_item_count": seeded_work_item_count,
        "metadata": work_item_result_metadata(
            family_key=WORK_FAMILY_MEMORY_DEDUP_REVIEW,
            execution_lane=EXECUTION_LANE_AGENTIC,
            seed_source="frontier_seed",
            seed_records=packet_records,
            created_work_item=created_work_item if created else None,
        ),
    }
