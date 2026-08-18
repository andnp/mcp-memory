from __future__ import annotations

import asyncio
from typing import Any, cast

import mcp_memory.core.task_handlers.curator_support as _curator_support
from mcp_memory.context import ApplicationContext, TaskRuntimeContext
from mcp_memory.core.curation_models import (
    campaign_hypothesis_from_payload,
)
from mcp_memory.core.curation_shadow import (
    run_curator_direct_mcp,
)
from mcp_memory.core.curation_validation import CurationMutationBudget
from mcp_memory.core.ports.tasks import TaskRecord
from mcp_memory.core.ports.work_items import (
    COMPATIBILITY_GROUP_STRUCTURAL_REVIEW,
    EXECUTION_LANE_AGENTIC,
    WORK_FAMILY_MEMORY_CURATION_REVIEW,
    compatibility_group_families,
)
from mcp_memory.core.task_handlers.campaigns import campaign_metadata
from mcp_memory.core.task_handlers.curator_seed_resolution import (
    CuratorSeedResolution,
    CuratorSeedResolutionState,
    resolve_curator_seed_packet,
)
from mcp_memory.core.task_handlers.curator_support import (
    CuratorCandidateRequest,
    acquire_curator_candidates,
)
from mcp_memory.core.task_handlers.maintenance_framework import sampling_payload
from mcp_memory.core.task_handlers.maintenance_work_items import (
    complete_work_item,
    defer_work_item,
    work_item_result_metadata,
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
        return {
            "summary": None,
            "tool_calls_executed": 0,
            "mutations": 0,
            "reason": "provider_not_configured",
            "failure_details": {
                "reason_code": "provider_route_unavailable",
                "reason_category": "routing",
                "provider_key": None,
                "provider_profile": None,
                "model": None,
                "retry_count": 0,
                "another_route_available": False,
            },
        }

    claimed_review_items = await asyncio.to_thread(
        _claim_curator_review_work_batch,
        ctx,
        task=task,
        limit=1,
    )
    claimed_review_item = claimed_review_items[0] if claimed_review_items else None
    campaign_hypothesis = campaign_hypothesis_from_payload(
        task.data
        if task.data.get("campaign_hypothesis") is not None or claimed_review_item is None
        else claimed_review_item.payload
    )
    seed_resolution = (
        resolve_curator_seed_packet(ctx, claimed_review_item.payload)
        if claimed_review_item is not None
        else None
    )
    if seed_resolution is not None and claimed_review_item is not None:
        resolution_metadata = seed_resolution.metadata()
        if seed_resolution.state == CuratorSeedResolutionState.TEMPORARILY_UNAVAILABLE:
            defer_work_item(
                ctx,
                claimed_review_item.id,
                error=seed_resolution.reason or "seed_resolution_temporarily_unavailable",
                retry_delay_seconds=60.0,
            )
            return {
                **resolution_metadata,
                "summary": None,
                "tool_calls_executed": 0,
                "mutations": 0,
                "claimed_work_item_count": 0,
                "reason": "seed_resolution_temporarily_unavailable",
            }
        if seed_resolution.state == CuratorSeedResolutionState.MALFORMED:
            _quarantine_curator_review_item(ctx, claimed_review_item, seed_resolution)
            claimed_review_item = None
    seed_batch, sampled_records, seed_records, claimed_review_item = await asyncio.to_thread(
        _prepare_curator_seed_context,
        ctx,
        task,
        claimed_review_item,
        campaign_hypothesis,
        seed_resolution,
    )

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
        campaign_hypothesis=campaign_hypothesis,
    )
    if seed_resolution is not None:
        work_item_metadata.update(seed_resolution.metadata())
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
            curation_no_op_reason="no_candidates",
            reason="no_seed_records",
        )

    return await run_curator_direct_mcp(
        ctx,
        task,
        provider=provider,
        mutation_budget=_mutation_budget_override(task),
        campaign_hypothesis=campaign_hypothesis,
        seed_batch=seed_batch,
        sampled_records=sampled_records,
        seed_records=seed_records,
        claimed_work_item=claimed_review_item,
        work_item_metadata=work_item_metadata,
    )


def _mutation_budget_override(task: TaskRecord) -> CurationMutationBudget | None:
    raw_limit = task.data.get("max_accepted_mutations")
    if raw_limit is None:
        return None
    if isinstance(raw_limit, bool) or not isinstance(raw_limit, int) or raw_limit < 0:
        raise ValueError("max_accepted_mutations must be a non-negative integer")
    return CurationMutationBudget(max_accepted_mutations=raw_limit)


def _prepare_curator_seed_context(
    ctx: ApplicationContext,
    task: TaskRecord,
    claimed_review_item: Any,
    campaign_hypothesis: Any,
    seed_resolution: CuratorSeedResolution | None = None,
) -> tuple[Any, list[Any], list[Any], Any]:
    if claimed_review_item is not None:
        seed_records = (
            list(seed_resolution.records)
            if seed_resolution is not None
            else _curator_support.review_seed_records(ctx, claimed_review_item.payload)
        )
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
                    campaign_hypothesis=campaign_hypothesis,
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
                campaign_hypothesis=campaign_hypothesis,
            ),
        )
        seed_records = seed_batch.records

    sampled_records = seed_records
    if claimed_review_item is None:
        support_records = _curator_support.select_curator_support_records(ctx, task, sampled_records)
        seed_records = sampled_records + support_records
    return seed_batch, sampled_records, seed_records, claimed_review_item


def _quarantine_curator_review_item(
    ctx: ApplicationContext,
    claimed_review_item: Any,
    resolution: CuratorSeedResolution,
) -> None:
    work_items = getattr(ctx, "work_items", None)
    quarantine_item = getattr(work_items, "quarantine_item", None)
    error = resolution.reason or "malformed_seed_packet"
    if callable(quarantine_item):
        quarantine_item(claimed_review_item.id, error=error)
        return
    defer_work_item(
        ctx,
        claimed_review_item.id,
        error=f"quarantined:{error}",
        retry_delay_seconds=365 * 24 * 60 * 60,
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
