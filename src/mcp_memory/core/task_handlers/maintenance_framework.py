from __future__ import annotations

from typing import Any

from mcp_memory.context import ApplicationContext
from mcp_memory.core.sampling import RouletteProvider, SamplingBatch
from mcp_memory.core.tasks import TaskRecord


def requested_sampling_strategy(task: TaskRecord) -> str | None:
    raw_value = task.data.get("strategy")
    if isinstance(raw_value, str) and raw_value.strip():
        return raw_value.strip()
    return None


def support_counts_for_candidates(ctx: ApplicationContext, candidates: list) -> dict[str, int]:
    assert ctx.repository is not None
    return {record.id: ctx.repository.count_incoming_links(record.id) for record in candidates}


def sample_maintenance_candidates(
    ctx: ApplicationContext,
    task: TaskRecord,
    candidates: list,
    *,
    allowed_strategies: tuple[str, ...],
    strategy_weights: dict[str, int],
    limit: int,
    support_counts: dict[str, int] | None = None,
) -> SamplingBatch:
    requested_strategy = requested_sampling_strategy(task)
    if not candidates:
        return SamplingBatch(
            requested_strategy=requested_strategy,
            strategy_used=requested_strategy or "none",
            strategy_fallback_reason=None,
            candidate_count=0,
            records=[],
        )

    return RouletteProvider(
        task_name=task.task_name,
        task_id=task.id,
        candidates=candidates,
        support_counts=support_counts or support_counts_for_candidates(ctx, candidates),
    ).get_batch(
        strategy=requested_strategy,
        allowed_strategies=allowed_strategies,
        strategy_weights=strategy_weights,
        limit=limit,
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
    if sampled_records is not None:
        payload["sampled_memory_ids"] = [record.id for record in sampled_records]
    if seed_records is not None:
        payload["seed_memory_ids"] = [record.id for record in seed_records]
    if extra:
        payload.update(extra)
    if result_metrics:
        payload.update(result_metrics)
    return payload
