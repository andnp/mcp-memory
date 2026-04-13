from __future__ import annotations

from typing import Any, Awaitable, Callable, cast

from mcp_memory.context import ApplicationContext
from mcp_memory.core.task_handlers.agentic_tool_tracking import (
    finalize_agentic_tool_tracking,
    prefer_deterministic_agentic_counts,
    reset_agentic_tool_tracking,
)
import mcp_memory.core.task_handlers.deduplicator_merge as _deduplicator_merge
import mcp_memory.core.task_handlers.deduplicator_support as _deduplicator_support
from mcp_memory.core.task_handlers.constants import DEFAULT_AGENT_SCAN_LIMIT
from mcp_memory.core.task_handlers.maintenance_framework import requested_sampling_strategy, sampling_payload
from mcp_memory.core.task_handlers.maintenance_housekeeping import _resolve_workspace_id
from mcp_memory.core.task_handlers.maintenance_work_items import (
    claim_work_batch,
    complete_work_item,
    enqueue_review_work_item,
    release_work_item,
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

    if provider is not None:
        claimed_review_items = _claim_dedup_review_work_batch(ctx, task=task, limit=1)
        if claimed_review_items:
            review_item = claimed_review_items[0]
            seed_records = _deduplicator_support.review_seed_records(ctx, review_item.payload)
            facts = [record for record in seed_records if record.type == "fact"]
            if not facts:
                complete_work_item(ctx, review_item.id)
                return {
                    "summary": None,
                    "merged": 0,
                    "archived": 0,
                    "absorbed_observations": 0,
                    "claimed_work_item_count": 1,
                    "execution_mode": "agentic_review",
                }
            run_agent = getattr(provider, "run_agent", None)
            supports_agentic = getattr(provider, "supports_agentic", None)
            if callable(run_agent) and (not callable(supports_agentic) or supports_agentic()):
                reset_agentic_tool_tracking(ctx, task.id)
                try:
                    agentic_result = await cast(Callable[[str], Awaitable[Any]], run_agent)(
                        _deduplicator_support.build_deduplicator_agent_prompt(
                            task,
                            seed_records,
                            strategy_used=_deduplicator_support.review_strategy(review_item.payload),
                        )
                    )
                except Exception:
                    release_work_item(ctx, review_item.id)
                    finalize_agentic_tool_tracking(ctx, task.id)
                    raise
                complete_work_item(ctx, review_item.id)
                normalized = _prefer_deduplicator_tracked_agentic_counts(
                    _deduplicator_support.normalize_deduplicator_agentic_result(agentic_result, seed_records),
                    deterministic_counts=finalize_agentic_tool_tracking(ctx, task.id),
                )
                normalized["claimed_work_item_count"] = 1
                return sampling_payload(
                    _deduplicator_support.review_sampling_batch(review_item.payload, seed_records),
                    sampled_records=seed_records,
                    extra=normalized,
                )

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

    run_agent = getattr(provider, "run_agent", None)
    supports_agentic = getattr(provider, "supports_agentic", None)
    if callable(run_agent) and (not callable(supports_agentic) or supports_agentic()):
        reset_agentic_tool_tracking(ctx, task.id)
        try:
            agentic_result = await cast(Callable[[str], Awaitable[Any]], run_agent)(
                _deduplicator_support.build_deduplicator_agent_prompt(task, seed_records, strategy_used=seed_batch.strategy_used)
            )
        except Exception:
            finalize_agentic_tool_tracking(ctx, task.id)
            raise
        normalized = _prefer_deduplicator_tracked_agentic_counts(
            _deduplicator_support.normalize_deduplicator_agentic_result(agentic_result, seed_records),
            deterministic_counts=finalize_agentic_tool_tracking(ctx, task.id),
        )
        if _deduplicator_result_has_effective_change(normalized):
            return sampling_payload(
                seed_batch,
                sampled_records=seed_records,
                extra=normalized,
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
            extra={**normalized, **seeded_review["metadata"]},
            seeded_work_item_count=seeded_review["seeded_work_item_count"],
        )

    deterministic_result = await _deduplicator_merge.run_deterministic_deduplicator_pass(
        ctx,
        task,
        candidates,
        seed_records,
        provider,
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


def _claim_dedup_review_work_batch(
    ctx: ApplicationContext,
    *,
    task: TaskRecord,
    limit: int,
) -> list[Any]:
    return claim_work_batch(
        ctx,
        task=task,
        family_key=WORK_FAMILY_MEMORY_DEDUP_REVIEW,
        execution_lane=EXECUTION_LANE_AGENTIC,
        limit=limit,
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


def _prefer_deduplicator_tracked_agentic_counts(
    result: dict[str, Any],
    *,
    deterministic_counts: Any,
) -> dict[str, Any]:
    normalized = prefer_deterministic_agentic_counts(result, deterministic_counts=deterministic_counts)
    if int(normalized.get("tool_calls_executed", 0)) > 0:
        return normalized
    if int(normalized.get("mutations", 0)) > 0:
        return normalized
    if normalized.get("tool_names_used"):
        return normalized
    return result


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
