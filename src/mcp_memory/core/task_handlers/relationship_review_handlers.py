from __future__ import annotations

from typing import Any

from mcp_memory.context import ApplicationContext
from mcp_memory.core.task_handlers.constants import DEFAULT_AGENT_SCAN_LIMIT
from mcp_memory.core.task_handlers.maintenance_framework import (
    sample_maintenance_candidates,
    sampling_payload,
)
from mcp_memory.core.task_handlers.maintenance_housekeeping import _resolve_workspace_id
from mcp_memory.core.task_handlers.maintenance_work_items import (
    run_claimed_review_work_item,
    run_sparse_frontier_review_task,
)
import mcp_memory.core.task_handlers.relationship_proposals as _relationship_proposals
import mcp_memory.core.task_handlers.relationship_review_support as _relationship_review_support
from mcp_memory.core.tasks import TaskRecord
from mcp_memory.work_item_store import (
    WORK_FAMILY_CONFLICT_REVIEW,
    WORK_FAMILY_GRAPH_LINK_REVIEW,
)


async def handle_graph_linker_task(
    ctx: ApplicationContext,
    task: TaskRecord,
    provider: Any = None,
) -> dict[str, Any]:
    if ctx.repository is None:
        return {"created": 0}

    if provider is not None:
        claimed_review_result = await run_claimed_review_work_item(
            ctx,
            task=task,
            family_key=WORK_FAMILY_GRAPH_LINK_REVIEW,
            load_candidates=_relationship_review_support.graph_link_review_candidates,
            propose_pairs=lambda review_candidates: _relationship_proposals.propose_graph_links(
                ctx,
                review_candidates,
                provider,
            ),
            apply_pairs=lambda proposed_pairs: _relationship_review_support.apply_graph_link_proposals(ctx, proposed_pairs),
        )
        if claimed_review_result is not None:
            return claimed_review_result

    sampled_batch = _sample_graph_link_candidates(ctx, task)
    candidates = sampled_batch.records
    if len(candidates) < 2:
        return sampling_payload(sampled_batch, sampled_records=candidates, created=0)

    proposed_pairs = await _relationship_proposals.propose_graph_links(ctx, candidates, provider)
    created = _relationship_review_support.apply_graph_link_proposals(ctx, proposed_pairs)
    return sampling_payload(sampled_batch, sampled_records=candidates, created=created)


async def handle_graph_link_discovery_task(
    ctx: ApplicationContext,
    task: TaskRecord,
) -> dict[str, Any]:
    if ctx.repository is None:
        return {"created": 0, "seeded_work_item_count": 0}

    workspace_id = _resolve_workspace_id(ctx, task)
    sampled_batch = _sample_graph_link_candidates(ctx, task, workspace_id=workspace_id)
    candidates = sampled_batch.records
    if len(candidates) < 2:
        return sampling_payload(sampled_batch, sampled_records=candidates, created=0, seeded_work_item_count=0)
    return await run_sparse_frontier_review_task(
        ctx,
        task=task,
        sampled_batch=sampled_batch,
        candidates=candidates,
        family_key=WORK_FAMILY_GRAPH_LINK_REVIEW,
        propose_pairs=lambda review_candidates: _relationship_proposals.propose_graph_links(
            ctx,
            review_candidates,
            provider=None,
        ),
        apply_pairs=lambda proposed_pairs: _relationship_review_support.apply_graph_link_proposals(ctx, proposed_pairs),
        should_seed=lambda proposed_pairs, review_candidates: (
            len(proposed_pairs) < _relationship_proposals.GRAPH_LINKER_FALLBACK_LINK_TARGET
            and len(review_candidates) > _relationship_proposals.GRAPH_LINKER_AI_MIN_CANDIDATES
        ),
        enqueue_review_work_item=lambda review_candidates: _relationship_review_support.enqueue_graph_link_review_work_item(
            ctx,
            task=task,
            workspace_id=workspace_id,
            candidates=review_candidates,
            strategy_used=sampled_batch.strategy_used,
        ),
    )


async def handle_conflict_detector_task(
    ctx: ApplicationContext,
    task: TaskRecord,
    provider: Any = None,
) -> dict[str, Any]:
    if ctx.repository is None:
        return {"created": 0}

    if provider is not None:
        claimed_review_result = await run_claimed_review_work_item(
            ctx,
            task=task,
            family_key=WORK_FAMILY_CONFLICT_REVIEW,
            load_candidates=_relationship_review_support.conflict_review_candidates,
            propose_pairs=lambda review_candidates: _relationship_proposals.propose_conflicts(
                ctx,
                review_candidates,
                provider,
            ),
            apply_pairs=lambda proposed_pairs: _relationship_review_support.apply_conflict_proposals(ctx, proposed_pairs),
        )
        if claimed_review_result is not None:
            return claimed_review_result

    sampled_batch = _sample_conflict_candidates(ctx, task)
    candidates = sampled_batch.records
    if len(candidates) < 2:
        return sampling_payload(sampled_batch, sampled_records=candidates, created=0)

    proposed_pairs = await _relationship_proposals.propose_conflicts(ctx, candidates, provider)
    created = _relationship_review_support.apply_conflict_proposals(ctx, proposed_pairs)
    return sampling_payload(sampled_batch, sampled_records=candidates, created=created)


async def handle_conflict_screening_task(
    ctx: ApplicationContext,
    task: TaskRecord,
) -> dict[str, Any]:
    if ctx.repository is None:
        return {"created": 0, "seeded_work_item_count": 0}

    workspace_id = _resolve_workspace_id(ctx, task)
    sampled_batch = _sample_conflict_candidates(ctx, task, workspace_id=workspace_id)
    candidates = sampled_batch.records
    if len(candidates) < 2:
        return sampling_payload(sampled_batch, sampled_records=candidates, created=0, seeded_work_item_count=0)

    return await run_sparse_frontier_review_task(
        ctx,
        task=task,
        sampled_batch=sampled_batch,
        candidates=candidates,
        family_key=WORK_FAMILY_CONFLICT_REVIEW,
        propose_pairs=lambda review_candidates: _relationship_proposals.propose_conflicts(
            ctx,
            review_candidates,
            provider=None,
        ),
        apply_pairs=lambda proposed_pairs: _relationship_review_support.apply_conflict_proposals(ctx, proposed_pairs),
        should_seed=lambda proposed_pairs, review_candidates: (
            not proposed_pairs and len(review_candidates) > _relationship_proposals.CONFLICT_DETECTOR_AI_MIN_CANDIDATES
        ),
        enqueue_review_work_item=lambda review_candidates: _relationship_review_support.enqueue_conflict_review_work_item(
            ctx,
            task=task,
            workspace_id=workspace_id,
            candidates=review_candidates,
            strategy_used=sampled_batch.strategy_used,
        ),
    )


def _sample_graph_link_candidates(
    ctx: ApplicationContext,
    task: TaskRecord,
    *,
    workspace_id: str | None = None,
):
    return _sample_relationship_review_candidates(
        ctx,
        task,
        workspace_id=workspace_id,
        allowed_types=None,
        allowed_strategies=_relationship_review_support.GRAPH_LINKER_ALLOWED_STRATEGIES,
        strategy_weights=_relationship_review_support.GRAPH_LINKER_STRATEGY_WEIGHTS,
    )


def _sample_conflict_candidates(
    ctx: ApplicationContext,
    task: TaskRecord,
    *,
    workspace_id: str | None = None,
):
    return _sample_relationship_review_candidates(
        ctx,
        task,
        workspace_id=workspace_id,
        allowed_types={"fact", "plan"},
        allowed_strategies=_relationship_review_support.CONFLICT_DETECTOR_ALLOWED_STRATEGIES,
        strategy_weights=_relationship_review_support.CONFLICT_DETECTOR_STRATEGY_WEIGHTS,
    )


def _sample_relationship_review_candidates(
    ctx: ApplicationContext,
    task: TaskRecord,
    *,
    workspace_id: str | None,
    allowed_types: set[str] | None,
    allowed_strategies: tuple[str, ...],
    strategy_weights: dict[str, int],
):
    assert ctx.repository is not None
    review_workspace_id = workspace_id if workspace_id is not None else _resolve_workspace_id(ctx, task)
    all_candidates = ctx.repository.list_memories(
        workspace_id=review_workspace_id,
        status="active",
        limit=int(task.data.get("limit", DEFAULT_AGENT_SCAN_LIMIT)),
    )
    if allowed_types is not None:
        all_candidates = [record for record in all_candidates if record.type in allowed_types]
    return sample_maintenance_candidates(
        ctx,
        task,
        all_candidates,
        allowed_strategies=allowed_strategies,
        strategy_weights=strategy_weights,
        limit=min(len(all_candidates), DEFAULT_AGENT_SCAN_LIMIT),
    )
