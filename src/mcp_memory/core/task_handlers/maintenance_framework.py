from __future__ import annotations

from typing import Any

from mcp_memory.context import ApplicationContext
from mcp_memory.core.sampling import RouletteProvider, SamplingBatch
from mcp_memory.management.task_sampling_summary import build_selection_strategy_priority_feedback
from mcp_memory.toolkit.maintenance_batch_selector import MaintenanceBatchSelector

SELECTION_PRIORITY_RECENT_RUN_LIMIT = 100


def requested_sampling_strategy(task: Any) -> str | None:
    raw_value = getattr(task, "requested_strategy", None)
    if raw_value is None:
        data = getattr(task, "data", {})
        raw_value = data.get("strategy")
    return raw_value.strip() if isinstance(raw_value, str) and raw_value.strip() else None


def support_counts_for_candidates(ctx: ApplicationContext, candidates: list) -> dict[str, int]:
    assert ctx.repository is not None
    return {record.id: ctx.repository.count_incoming_links(record.id) for record in candidates}


def sample_maintenance_candidates(
    ctx: ApplicationContext,
    task: Any,
    candidates: list,
    *,
    allowed_strategies: tuple[str, ...],
    limit: int,
    support_counts: dict[str, int] | None = None,
) -> SamplingBatch:
    requested_strategy = requested_sampling_strategy(task)
    task_name = task.task_name
    task_id = task.task_id if hasattr(task, "task_id") else task.id

    def empty_batch(strategy: str | None) -> SamplingBatch:
        return SamplingBatch(
            requested_strategy=strategy,
            strategy_used=strategy or "none",
            strategy_fallback_reason=None,
            candidate_count=0,
            records=[],
        )

    selector = MaintenanceBatchSelector(
        requested_strategy=requested_strategy,
        candidates=candidates,
        batch_provider_factory=lambda strategy_prior_scores: RouletteProvider(
            task_name=task_name,
            task_id=task_id,
            candidates=candidates,
            support_counts=support_counts or support_counts_for_candidates(ctx, candidates),
            strategy_prior_scores=strategy_prior_scores,
        ),
        priority_feedback_source=lambda: _selection_strategy_priority_feedback(
            ctx,
            task_name=task_name,
            allowed_strategies=allowed_strategies,
        ),
        empty_batch_factory=empty_batch,
    )
    return selector.select(allowed_strategies=allowed_strategies, limit=limit)


def _selection_strategy_priority_feedback(
    ctx: ApplicationContext,
    *,
    task_name: str,
    allowed_strategies: tuple[str, ...],
) -> dict[str, tuple[float, str]]:
    from mcp_memory.management.agent_run_reporting import build_recent_agent_runs

    recent_runs = build_recent_agent_runs(
        ctx.db_manager,
        ctx.workspace_id,
        limit=SELECTION_PRIORITY_RECENT_RUN_LIMIT,
        detail_level="compact",
    )
    if not recent_runs:
        return {}
    return build_selection_strategy_priority_feedback(
        recent_runs,
        task_name=task_name,
        allowed_strategies=allowed_strategies,
    )

def sampling_payload(
    batch: SamplingBatch,
    *,
    sampled_records: list | None = None,
    seed_records: list | None = None,
    extra: dict[str, Any] | None = None,
    **result_metrics: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "requested_strategy": batch.requested_strategy,
        "strategy_used": batch.strategy_used,
        "strategy_fallback_reason": batch.strategy_fallback_reason,
        "candidate_count": batch.candidate_count,
    }
    if batch.strategy_selection_mode is not None:
        payload["strategy_selection_mode"] = batch.strategy_selection_mode
    if batch.strategy_selection_reason is not None:
        payload["strategy_selection_reason"] = batch.strategy_selection_reason
    if batch.strategy_selection_scores is not None:
        payload["strategy_selection_scores"] = batch.strategy_selection_scores
    if batch.selector_feature_snapshot is not None:
        payload["selector_feature_snapshot"] = batch.selector_feature_snapshot
    if batch.sampler_priority_score is not None:
        payload["sampler_priority_score"] = batch.sampler_priority_score
    if batch.sampler_priority_explanation is not None:
        payload["sampler_priority_explanation"] = batch.sampler_priority_explanation
    if sampled_records is not None:
        payload["sampled_memory_ids"] = [record.id for record in sampled_records]
    if seed_records is not None:
        payload["seed_memory_ids"] = [record.id for record in seed_records]
    if extra:
        payload.update(extra)
    if result_metrics:
        payload.update(result_metrics)
    return payload
