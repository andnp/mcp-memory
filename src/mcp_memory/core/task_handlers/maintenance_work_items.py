from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from mcp_memory.context import ApplicationContext
from mcp_memory.core.task_handlers.maintenance_framework import sampling_payload
from mcp_memory.core.sampling import SamplingBatch
from mcp_memory.core.tasks import TaskRecord


def claim_work_batch(
    ctx: ApplicationContext,
    *,
    task: TaskRecord,
    family_key: str,
    execution_lane: str,
    limit: int,
) -> list[Any]:
    work_items = getattr(ctx, "work_items", None)
    if work_items is None or limit < 1:
        return []
    return work_items.claim_batch(
        family_key=family_key,
        execution_lane=execution_lane,
        lease_owner=task.id,
        limit=limit,
        workspace_id=task.workspace_id,
    )


def enqueue_review_work_item(
    ctx: ApplicationContext,
    *,
    task: TaskRecord,
    family_key: str,
    execution_lane: str,
    workspace_id: str | None,
    idempotency_prefix: str,
    payload_memory_ids_key: str,
    memory_ids: list[str],
    strategy_used: str | None,
    candidate_count: int | None = None,
) -> tuple[Any, bool]:
    work_items = getattr(ctx, "work_items", None)
    if work_items is None:
        raise ValueError("work_items repository is not configured")
    sorted_memory_ids = sorted(memory_ids)
    payload: dict[str, Any] = {
        "workspace_id": workspace_id,
        payload_memory_ids_key: sorted_memory_ids,
        "strategy_used": strategy_used,
    }
    if candidate_count is not None:
        payload["candidate_count"] = candidate_count
    return work_items.enqueue_unique(
        family_key=family_key,
        execution_lane=execution_lane,
        workspace_id=workspace_id,
        priority=task.priority,
        idempotency_key=f"{idempotency_prefix}:{'|'.join(sorted_memory_ids)}",
        payload=payload,
    )


def payload_memory_records(
    ctx: ApplicationContext,
    payload: dict[str, Any],
    *,
    payload_memory_ids_key: str,
    allowed_types: set[str] | None = None,
) -> list[Any]:
    if ctx.repository is None:
        return []
    memory_ids = payload.get(payload_memory_ids_key)
    if not isinstance(memory_ids, list):
        return []
    records: list[Any] = []
    for memory_id in memory_ids:
        if not isinstance(memory_id, str):
            continue
        record = ctx.repository.get_memory(memory_id)
        if record is None or record.status != "active":
            continue
        if allowed_types is not None and record.type not in allowed_types:
            continue
        records.append(record)
    return records


def sampling_batch_from_work_payload(payload: dict[str, Any], records: list[Any]) -> SamplingBatch:
    requested_strategy = payload.get("strategy_used")
    if not isinstance(requested_strategy, str):
        requested_strategy = None
    candidate_count = payload.get("candidate_count")
    if not isinstance(candidate_count, int):
        candidate_count = len(records)
    return SamplingBatch(
        requested_strategy=requested_strategy,
        strategy_used=requested_strategy or "none",
        strategy_fallback_reason=None,
        candidate_count=candidate_count,
        records=records,
    )


def work_item_result_metadata(
    *,
    family_key: str,
    execution_lane: str,
    seed_source: str,
    seed_records: list[Any],
    claimed_work_item: Any | None = None,
    created_work_item: Any | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "work_item_family": family_key,
        "work_item_execution_lane": execution_lane,
        "seed_source": seed_source,
        "seed_record_count": len(seed_records),
    }
    if claimed_work_item is not None:
        metadata["claimed_work_item_id"] = claimed_work_item.id
    if created_work_item is not None:
        metadata["created_work_item_id"] = created_work_item.id
    return metadata


def complete_work_item(ctx: ApplicationContext, work_item_id: str) -> None:
    work_items = getattr(ctx, "work_items", None)
    if work_items is None:
        return
    work_items.complete_item(work_item_id)


def defer_work_item(
    ctx: ApplicationContext,
    work_item_id: str,
    *,
    error: str,
    retry_delay_seconds: float | None,
) -> None:
    work_items = getattr(ctx, "work_items", None)
    if work_items is None:
        return
    work_items.defer_item(
        work_item_id,
        error=error,
        retry_delay_seconds=0.0 if retry_delay_seconds is None else float(retry_delay_seconds),
    )


def release_work_item(ctx: ApplicationContext, work_item_id: str) -> None:
    work_items = getattr(ctx, "work_items", None)
    if work_items is None:
        return
    work_items.release_item(work_item_id)


async def run_claimed_review_work_item(
    ctx: ApplicationContext,
    *,
    task: TaskRecord,
    family_key: str,
    load_candidates: Callable[[ApplicationContext, dict[str, Any]], list[Any]],
    propose_pairs: Callable[[list[Any]], Awaitable[Any]],
    apply_pairs: Callable[[Any], int],
) -> dict[str, Any] | None:
    claimed_review_items = claim_work_batch(
        ctx,
        task=task,
        family_key=family_key,
        execution_lane="agentic",
        limit=1,
    )
    if not claimed_review_items:
        return None

    review_item = claimed_review_items[0]
    try:
        candidates = load_candidates(ctx, review_item.payload)
        proposed_pairs = await propose_pairs(candidates)
        created = apply_pairs(proposed_pairs)
    except Exception:
        release_work_item(ctx, review_item.id)
        raise

    complete_work_item(ctx, review_item.id)
    result = {
        "created": created,
        "claimed_work_item_count": 1,
        "execution_mode": "agentic_review",
    }
    result.update(
        work_item_result_metadata(
            family_key=family_key,
            execution_lane="agentic",
            seed_source="claimed_review_work_item",
            seed_records=candidates,
            claimed_work_item=review_item,
        )
    )
    return result


async def run_sparse_frontier_review_task(
    ctx: ApplicationContext,
    *,
    task: TaskRecord,
    sampled_batch: SamplingBatch,
    candidates: list[Any],
    family_key: str,
    propose_pairs: Callable[[list[Any]], Awaitable[Any]],
    apply_pairs: Callable[[Any], int],
    should_seed: Callable[[Any, list[Any]], bool],
    enqueue_review_work_item: Callable[[list[Any]], tuple[Any, bool]],
) -> dict[str, Any]:
    created_work_item: Any | None = None
    fallback_pairs = await propose_pairs(candidates)
    created = apply_pairs(fallback_pairs)
    seeded_work_item_count = 0
    if should_seed(fallback_pairs, candidates):
        created_work_item, created_item = enqueue_review_work_item(candidates)
        seeded_work_item_count = 1 if created_item else 0

    return sampling_payload(
        sampled_batch,
        sampled_records=candidates,
        extra=work_item_result_metadata(
            family_key=family_key,
            execution_lane="agentic",
            seed_source="frontier_seed",
            seed_records=candidates,
            created_work_item=created_work_item if seeded_work_item_count else None,
        ),
        created=created,
        seeded_work_item_count=seeded_work_item_count,
    )
