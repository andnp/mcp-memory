from __future__ import annotations

from typing import Any

from mcp_memory.context import ApplicationContext
from mcp_memory.core.sampling import (
    BOUNDED_NOISE_STRATEGY,
    CONFLICT_FRONTIER_STRATEGY,
    GRAPH_BRIDGE_STRATEGY,
    NEVER_SURFACED_STRATEGY,
    SEMANTIC_STRATEGY,
)
from mcp_memory.core.task_handlers.maintenance_work_items import (
    enqueue_review_work_item,
    payload_memory_records,
)
from mcp_memory.core.tasks import TaskRecord
from mcp_memory.work_item_store import (
    EXECUTION_LANE_AGENTIC,
    WORK_FAMILY_CONFLICT_REVIEW,
    WORK_FAMILY_GRAPH_LINK_REVIEW,
)

GRAPH_LINKER_ALLOWED_STRATEGIES = (
    SEMANTIC_STRATEGY,
    GRAPH_BRIDGE_STRATEGY,
    BOUNDED_NOISE_STRATEGY,
)
GRAPH_LINKER_STRATEGY_WEIGHTS = {
    SEMANTIC_STRATEGY: 3,
    GRAPH_BRIDGE_STRATEGY: 3,
    BOUNDED_NOISE_STRATEGY: 1,
}
CONFLICT_DETECTOR_ALLOWED_STRATEGIES = (
    SEMANTIC_STRATEGY,
    CONFLICT_FRONTIER_STRATEGY,
    NEVER_SURFACED_STRATEGY,
)
CONFLICT_DETECTOR_STRATEGY_WEIGHTS = {
    SEMANTIC_STRATEGY: 3,
    CONFLICT_FRONTIER_STRATEGY: 3,
    NEVER_SURFACED_STRATEGY: 1,
}


def apply_graph_link_proposals(
    ctx: ApplicationContext,
    proposed_pairs: list[tuple[str, str, str, str]],
) -> int:
    if ctx.repository is None:
        return 0
    created = 0
    for source_id, target_id, link_type, context in proposed_pairs:
        if source_id == target_id or _has_link(ctx, source_id, target_id, link_type):
            continue
        ctx.repository.add_link(source_id, target_id, link_type, context)
        created += 1
    return created



def apply_conflict_proposals(
    ctx: ApplicationContext,
    proposed_pairs: list[tuple[str, str, str]],
) -> int:
    if ctx.repository is None:
        return 0
    created = 0
    for left_id, right_id, context in proposed_pairs:
        if left_id == right_id:
            continue
        if not _has_link(ctx, left_id, right_id, "CONTRADICTS"):
            ctx.repository.add_link(left_id, right_id, "CONTRADICTS", context)
            created += 1
        if not _has_link(ctx, right_id, left_id, "CONTRADICTS"):
            ctx.repository.add_link(right_id, left_id, "CONTRADICTS", context)
            created += 1
    return created



def enqueue_graph_link_review_work_item(
    ctx: ApplicationContext,
    *,
    task: TaskRecord,
    workspace_id: str | None,
    candidates: list[Any],
    strategy_used: str | None,
) -> tuple[Any, bool]:
    return enqueue_review_work_item(
        ctx,
        task=task,
        family_key=WORK_FAMILY_GRAPH_LINK_REVIEW,
        execution_lane=EXECUTION_LANE_AGENTIC,
        workspace_id=workspace_id,
        idempotency_prefix="graph_link_review",
        payload_memory_ids_key="candidate_memory_ids",
        memory_ids=[record.id for record in candidates],
        strategy_used=strategy_used,
        candidate_count=len(candidates),
    )



def enqueue_conflict_review_work_item(
    ctx: ApplicationContext,
    *,
    task: TaskRecord,
    workspace_id: str | None,
    candidates: list[Any],
    strategy_used: str | None,
) -> tuple[Any, bool]:
    return enqueue_review_work_item(
        ctx,
        task=task,
        family_key=WORK_FAMILY_CONFLICT_REVIEW,
        execution_lane=EXECUTION_LANE_AGENTIC,
        workspace_id=workspace_id,
        idempotency_prefix="conflict_review",
        payload_memory_ids_key="candidate_memory_ids",
        memory_ids=[record.id for record in candidates],
        strategy_used=strategy_used,
        candidate_count=len(candidates),
    )



def graph_link_review_candidates(ctx: ApplicationContext, payload: dict[str, Any]) -> list[Any]:
    return payload_memory_records(ctx, payload, payload_memory_ids_key="candidate_memory_ids")



def conflict_review_candidates(ctx: ApplicationContext, payload: dict[str, Any]) -> list[Any]:
    return payload_memory_records(
        ctx,
        payload,
        payload_memory_ids_key="candidate_memory_ids",
        allowed_types={"fact", "plan"},
    )



def _has_link(ctx: ApplicationContext, source_id: str, target_id: str, link_type: str) -> bool:
    assert ctx.repository is not None
    return any(
        link.target_id == target_id
        for link in ctx.repository.get_links(source_id, direction="outgoing", link_type=link_type)
    )
