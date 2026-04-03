from __future__ import annotations

from dataclasses import replace
from typing import Any

from mcp_memory.context import ApplicationContext
from mcp_memory.core.task_handlers.constants import (
    CONFLICT_DETECTOR_TASK_NAME,
    DEFAULT_AGENT_SCAN_LIMIT,
    GRAPH_LINKER_TASK_NAME,
)
from mcp_memory.core.task_handlers.maintenance_framework import (
    sample_maintenance_candidates,
    sampling_payload,
    support_counts_for_candidates,
)
from mcp_memory.core.task_handlers.maintenance_housekeeping import _resolve_workspace_id
from mcp_memory.core.task_handlers.maintenance_work_items import (
    run_claimed_review_work_item,
    run_sparse_frontier_review_task,
    work_item_result_metadata,
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

    workspace_id = _resolve_workspace_id(ctx, task)
    sampled_batch = _sample_graph_link_candidates(ctx, task, workspace_id=workspace_id)
    candidates = sampled_batch.records
    if len(candidates) < 2:
        return sampling_payload(sampled_batch, sampled_records=candidates, created=0)

    if provider is None:
        return await _run_graph_link_sparse_frontier_task(
            ctx,
            task=task,
            workspace_id=workspace_id,
            sampled_batch=sampled_batch,
            candidates=candidates,
        )

    proposed_pairs = await _relationship_proposals.propose_graph_links(ctx, candidates, provider)
    created = _relationship_review_support.apply_graph_link_proposals(ctx, proposed_pairs)
    created_work_item = None
    seeded_work_item_count = 0
    if _should_seed_graph_link_review(proposed_pairs, candidates):
        created_work_item, created_item = _relationship_review_support.enqueue_graph_link_review_work_item(
            ctx,
            task=task,
            workspace_id=workspace_id,
            candidates=candidates,
            strategy_used=sampled_batch.strategy_used,
        )
        seeded_work_item_count = 1 if created_item else 0

    return sampling_payload(
        sampled_batch,
        sampled_records=candidates,
        extra=work_item_result_metadata(
            family_key=WORK_FAMILY_GRAPH_LINK_REVIEW,
            execution_lane="agentic",
            seed_source="frontier_seed",
            seed_records=candidates,
            created_work_item=created_work_item if seeded_work_item_count else None,
        ),
        created=created,
        seeded_work_item_count=seeded_work_item_count,
    )


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

    return await _run_graph_link_sparse_frontier_task(
        ctx,
        task=task,
        workspace_id=workspace_id,
        sampled_batch=sampled_batch,
        candidates=candidates,
    )


async def _run_graph_link_sparse_frontier_task(
    ctx: ApplicationContext,
    *,
    task: TaskRecord,
    workspace_id: str | None,
    sampled_batch: Any,
    candidates: list[Any],
) -> dict[str, Any]:
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
        should_seed=_should_seed_graph_link_review,
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

    workspace_id = _resolve_workspace_id(ctx, task)
    sampled_batch = _sample_conflict_candidates(ctx, task, workspace_id=workspace_id)
    candidates = sampled_batch.records
    if len(candidates) < 2:
        return sampling_payload(sampled_batch, sampled_records=candidates, created=0)

    if provider is None:
        return await _run_conflict_sparse_frontier_task(
            ctx,
            task=task,
            workspace_id=workspace_id,
            sampled_batch=sampled_batch,
            candidates=candidates,
        )

    proposed_pairs = await _relationship_proposals.propose_conflicts(ctx, candidates, provider)
    created = _relationship_review_support.apply_conflict_proposals(ctx, proposed_pairs)
    created_work_item = None
    seeded_work_item_count = 0
    if _should_seed_conflict_review(proposed_pairs, candidates):
        created_work_item, created_item = _relationship_review_support.enqueue_conflict_review_work_item(
            ctx,
            task=task,
            workspace_id=workspace_id,
            candidates=candidates,
            strategy_used=sampled_batch.strategy_used,
        )
        seeded_work_item_count = 1 if created_item else 0

    return sampling_payload(
        sampled_batch,
        sampled_records=candidates,
        extra=work_item_result_metadata(
            family_key=WORK_FAMILY_CONFLICT_REVIEW,
            execution_lane="agentic",
            seed_source="frontier_seed",
            seed_records=candidates,
            created_work_item=created_work_item if seeded_work_item_count else None,
        ),
        created=created,
        seeded_work_item_count=seeded_work_item_count,
    )


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

    return await _run_conflict_sparse_frontier_task(
        ctx,
        task=task,
        workspace_id=workspace_id,
        sampled_batch=sampled_batch,
        candidates=candidates,
    )


async def _run_conflict_sparse_frontier_task(
    ctx: ApplicationContext,
    *,
    task: TaskRecord,
    workspace_id: str | None,
    sampled_batch: Any,
    candidates: list[Any],
) -> dict[str, Any]:
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
        should_seed=_should_seed_conflict_review,
        enqueue_review_work_item=lambda review_candidates: _relationship_review_support.enqueue_conflict_review_work_item(
            ctx,
            task=task,
            workspace_id=workspace_id,
            candidates=review_candidates,
            strategy_used=sampled_batch.strategy_used,
        ),
    )


def _should_seed_conflict_review(proposed_pairs: Any, review_candidates: list[Any]) -> bool:
    return not proposed_pairs and len(review_candidates) > _relationship_proposals.CONFLICT_DETECTOR_AI_MIN_CANDIDATES


def _should_seed_graph_link_review(proposed_pairs: Any, review_candidates: list[Any]) -> bool:
    return (
        len(proposed_pairs) < _relationship_proposals.GRAPH_LINKER_FALLBACK_LINK_TARGET
        and len(review_candidates) > _relationship_proposals.GRAPH_LINKER_AI_MIN_CANDIDATES
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
        canonical_task_name=GRAPH_LINKER_TASK_NAME,
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
        canonical_task_name=CONFLICT_DETECTOR_TASK_NAME,
        allowed_types={"fact", "plan"},
        allowed_strategies=_relationship_review_support.CONFLICT_DETECTOR_ALLOWED_STRATEGIES,
        strategy_weights=_relationship_review_support.CONFLICT_DETECTOR_STRATEGY_WEIGHTS,
    )


def _sample_relationship_review_candidates(
    ctx: ApplicationContext,
    task: TaskRecord,
    *,
    workspace_id: str | None,
    canonical_task_name: str,
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
    sampling_task = task if task.task_name == canonical_task_name else replace(task, task_name=canonical_task_name)
    return sample_maintenance_candidates(
        ctx,
        sampling_task,
        all_candidates,
        allowed_strategies=allowed_strategies,
        strategy_weights=strategy_weights,
        limit=min(len(all_candidates), DEFAULT_AGENT_SCAN_LIMIT),
        support_counts=support_counts_for_candidates(ctx, all_candidates),
    )
