from __future__ import annotations

from typing import Any, cast

from mcp_memory.context import ApplicationContext, TaskRuntimeContext
from mcp_memory.core.curation_shadow import (
    run_curator_verified_campaign,
)
from mcp_memory.core.task_handlers.campaigns import campaign_metadata
import mcp_memory.core.task_handlers.curator_support as _curator_support
from mcp_memory.core.task_handlers.maintenance_framework import sampling_payload
from mcp_memory.core.task_handlers.maintenance_work_items import (
    complete_work_item,
    work_item_result_metadata,
)
from mcp_memory.core.task_handlers.curator_support import (
    CuratorCandidateRequest,
    acquire_curator_candidates,
)
from mcp_memory.core.ports.tasks import TaskRecord
from mcp_memory.core.ports.work_items import (
    COMPATIBILITY_GROUP_STRUCTURAL_REVIEW,
    EXECUTION_LANE_AGENTIC,
    WORK_FAMILY_MEMORY_CURATION_REVIEW,
    compatibility_group_families,
)

async def handle_memory_curator_task(
    ctx: ApplicationContext | TaskRuntimeContext,
    task: TaskRecord,
    provider: Any = None,
) -> dict[str, Any]:
    ctx = cast(ApplicationContext, ctx)
    if ctx.repository is None:
        return {"summary": None, "tool_calls_executed": 0, "mutations": 0}
    if provider is None:
        return {"summary": None, "tool_calls_executed": 0, "mutations": 0, "reason": "provider_not_configured"}

    claimed_review_items = _claim_curator_review_work_batch(ctx, task=task, limit=1)
    claimed_review_item = claimed_review_items[0] if claimed_review_items else None
    if claimed_review_item is not None:
        seed_records = _curator_support.review_seed_records(ctx, claimed_review_item.payload)
        if seed_records:
            seed_batch = _curator_support.review_sampling_batch(claimed_review_item.payload, seed_records)
        else:
            complete_work_item(ctx, claimed_review_item.id)
            claimed_review_item = None
            seed_batch = acquire_curator_candidates(
                ctx,
                CuratorCandidateRequest(
                    task_id=task.id,
                    workspace_id=task.workspace_id,
                    requested_strategy=task.data.get("strategy"),
                ),
            )
            seed_records = seed_batch.records
    else:
        seed_batch = acquire_curator_candidates(
            ctx,
            CuratorCandidateRequest(
                task_id=task.id,
                workspace_id=task.workspace_id,
                requested_strategy=task.data.get("strategy"),
            ),
        )
        seed_records = seed_batch.records

    sampled_records = seed_records
    if claimed_review_item is None:
        support_records = _curator_support.select_curator_support_records(ctx, task, sampled_records)
        seed_records = sampled_records + support_records

    claimed_family_key = (
        claimed_review_item.family_key
        if claimed_review_item is not None
        else WORK_FAMILY_MEMORY_CURATION_REVIEW
    )
    work_item_metadata = work_item_result_metadata(
        family_key=claimed_family_key,
        execution_lane=EXECUTION_LANE_AGENTIC,
        seed_source="claimed_review_work_item" if claimed_review_item is not None else "direct_sampling",
        seed_records=seed_records,
        claimed_work_item=claimed_review_item,
    )
    work_item_metadata.update(
        campaign_metadata(
            compatibility_group=COMPATIBILITY_GROUP_STRUCTURAL_REVIEW,
            origin_family=claimed_family_key,
            execution_lane=EXECUTION_LANE_AGENTIC,
        )
    )

    if not seed_records:
        return sampling_payload(
            seed_batch,
            sampled_records=sampled_records,
            seed_records=seed_records,
            extra=work_item_metadata,
            summary=None,
            tool_calls_executed=0,
            mutations=0,
            claimed_work_item_count=0,
            reason="no_seed_records",
        )

    return await run_curator_verified_campaign(
        ctx,
        task,
        provider=provider,
        seed_batch=seed_batch,
        sampled_records=sampled_records,
        seed_records=seed_records,
        claimed_work_item=claimed_review_item,
        work_item_metadata=work_item_metadata,
    )


def _claim_curator_review_work_batch(
    ctx: ApplicationContext,
    *,
    task: TaskRecord,
    limit: int,
) -> list[Any]:
    work_items = getattr(ctx, "work_items", None)
    if work_items is None or limit < 1:
        return []
    return work_items.claim_compatible_batch(
        family_keys=compatibility_group_families(COMPATIBILITY_GROUP_STRUCTURAL_REVIEW),
        execution_lane=EXECUTION_LANE_AGENTIC,
        lease_owner=task.id,
        limit=limit,
        workspace_id=task.workspace_id,
    )
