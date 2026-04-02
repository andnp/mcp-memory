from __future__ import annotations

from typing import Any, Awaitable, Callable, cast

from mcp_memory.context import ApplicationContext
from mcp_memory.core.provider_admission import classify_provider_failure
from mcp_memory.core.task_handlers.agentic_tool_tracking import (
    finalize_agentic_tool_tracking,
    prefer_deterministic_agentic_counts,
    reset_agentic_tool_tracking,
)
from mcp_memory.core.task_handlers.constants import DEFAULT_AGENT_SCAN_LIMIT
from mcp_memory.core.task_handlers.maintenance_framework import sample_maintenance_candidates
from mcp_memory.core.task_handlers.maintenance_housekeeping import _resolve_workspace_id
from mcp_memory.core.task_handlers.maintenance_normalization import (
    coerce_text_summary,
    extract_embedded_json_object,
    normalize_tag_values,
)
from mcp_memory.core.task_handlers.maintenance_work_items import (
    complete_work_item,
    defer_work_item,
    release_work_item,
)
import mcp_memory.core.task_handlers.taxonomist_support as _taxonomist_support
from mcp_memory.core.tasks import TaskRecord


async def handle_taxonomist_task(
    ctx: ApplicationContext,
    task: TaskRecord,
    provider: Any = None,
) -> dict[str, Any]:
    if ctx.repository is None:
        return {"updated": 0}

    workspace_id = _resolve_workspace_id(ctx, task)
    all_candidates = ctx.repository.list_memories(
        workspace_id=workspace_id,
        limit=int(task.data.get("limit", DEFAULT_AGENT_SCAN_LIMIT)),
    )
    sampled_batch = sample_maintenance_candidates(
        ctx,
        task,
        all_candidates,
        allowed_strategies=_taxonomist_support.TAXONOMIST_ALLOWED_STRATEGIES,
        strategy_weights=_taxonomist_support.TAXONOMIST_STRATEGY_WEIGHTS,
        limit=min(len(all_candidates), DEFAULT_AGENT_SCAN_LIMIT),
    )
    candidates = sampled_batch.records
    provider_call_budget = _taxonomist_support.provider_call_budget(ctx, provider)
    untagged_candidates = [record for record in candidates if not normalize_tag_values(record.tags)]

    if not untagged_candidates:
        return _taxonomist_support.build_taxonomist_result(
            sampled_batch,
            sampled_records=candidates,
            updated=0,
            provider_call_budget=provider_call_budget,
            provider_calls_used=0,
            provider_deferred_reason_code=None,
            provider_deferred_retry_delay_seconds=None,
            claimed_work_item_count=0,
            execution_mode="precheck_skipped",
            reason="no_untagged_candidates",
        )

    seeded_work_items = _taxonomist_support.seed_taxonomist_work_items(
        ctx,
        task=task,
        workspace_id=workspace_id,
        candidates=untagged_candidates,
    )
    if _taxonomist_support.supports_agentic_execution(provider) and provider_call_budget > 0:
        if not _taxonomist_support.has_ready_tagging_work(ctx, workspace_id=workspace_id):
            return _taxonomist_support.build_taxonomist_result(
                sampled_batch,
                sampled_records=candidates,
                updated=0,
                provider_call_budget=provider_call_budget,
                provider_calls_used=0,
                provider_deferred_reason_code=None,
                provider_deferred_retry_delay_seconds=None,
                claimed_work_item_count=0,
                execution_mode="precheck_skipped",
                reason="no_ready_work_items",
            )
    if _taxonomist_support.supports_agentic_execution(provider) and seeded_work_items and provider_call_budget > 0:
        agentic_result = await _run_taxonomist_agentic_pass(
            ctx,
            task=task,
            provider=provider,
            workspace_id=workspace_id,
            candidate_memory_ids={record.id for record in untagged_candidates},
            seeded_work_items=seeded_work_items,
            provider_call_budget=provider_call_budget,
        )
        return _taxonomist_support.build_taxonomist_result(
            sampled_batch,
            sampled_records=candidates,
            updated=agentic_result["updated"],
            provider_call_budget=provider_call_budget,
            provider_calls_used=agentic_result["provider_calls_used"],
            provider_deferred_reason_code=agentic_result["provider_deferred_reason_code"],
            provider_deferred_retry_delay_seconds=agentic_result["provider_deferred_retry_delay_seconds"],
            claimed_work_item_count=agentic_result["claimed_work_item_count"],
            tool_calls_executed=agentic_result["tool_calls_executed"],
            mutations=agentic_result["mutations"],
            tool_names_used=agentic_result["tool_names_used"],
            compatible_batch_calls=agentic_result["compatible_batch_calls"],
            execution_mode="agentic_mcp",
            summary=agentic_result["summary"],
            compatibility_group=agentic_result["compatibility_group"],
            work_item_batch_limit=agentic_result["work_item_batch_limit"],
            max_batches_per_run=agentic_result["max_batches_per_run"],
        )

    json_result = await _run_taxonomist_json_pass(
        ctx,
        task=task,
        provider=provider,
        candidates=untagged_candidates,
        provider_call_budget=provider_call_budget,
    )
    return _taxonomist_support.build_taxonomist_result(
        sampled_batch,
        sampled_records=candidates,
        updated=json_result["updated"],
        provider_call_budget=provider_call_budget,
        provider_calls_used=json_result["provider_calls_used"],
        provider_deferred_reason_code=json_result["provider_deferred_reason_code"],
        provider_deferred_retry_delay_seconds=json_result["provider_deferred_retry_delay_seconds"],
        claimed_work_item_count=json_result["claimed_work_item_count"],
    )


async def handle_tag_normalizer_task(
    ctx: ApplicationContext,
    task: TaskRecord,
) -> dict[str, Any]:
    if ctx.repository is None:
        return {"updated": 0, "seeded_enrichment_count": 0}

    workspace_id = _resolve_workspace_id(ctx, task)
    all_candidates = ctx.repository.list_memories(
        workspace_id=workspace_id,
        limit=int(task.data.get("limit", DEFAULT_AGENT_SCAN_LIMIT)),
    )
    sampled_batch = sample_maintenance_candidates(
        ctx,
        task,
        all_candidates,
        allowed_strategies=_taxonomist_support.TAXONOMIST_ALLOWED_STRATEGIES,
        strategy_weights=_taxonomist_support.TAXONOMIST_STRATEGY_WEIGHTS,
        limit=min(len(all_candidates), DEFAULT_AGENT_SCAN_LIMIT),
    )
    candidates = sampled_batch.records
    normalization_candidates = [
        record for record in candidates if _taxonomist_support.needs_tag_normalization(record.tags, normalize_tag_values)
    ]
    seeded_work_items = _taxonomist_support.seed_tag_normalizer_work_items(
        ctx,
        task=task,
        workspace_id=workspace_id,
        candidates=normalization_candidates,
        normalize_tag_values=normalize_tag_values,
    )
    claimed_work_items = _taxonomist_support.claim_tag_normalizer_work_batch(
        ctx,
        task=task,
        workspace_id=workspace_id,
        limit=min(len(normalization_candidates), DEFAULT_AGENT_SCAN_LIMIT),
    )

    updated = 0
    seeded_enrichment_count = 0
    finalized_work_item_ids: set[str] = set()
    for work_item in claimed_work_items:
        memory_id = _taxonomist_support.work_memory_id(work_item.payload)
        if memory_id is None:
            complete_work_item(ctx, work_item.id)
            finalized_work_item_ids.add(work_item.id)
            continue

        record = ctx.repository.get_memory(memory_id)
        if record is None:
            complete_work_item(ctx, work_item.id)
            finalized_work_item_ids.add(work_item.id)
            continue

        normalized_tags = normalize_tag_values(record.tags)
        if normalized_tags != record.tags:
            refreshed = ctx.repository.update_memory(record.id, tags=normalized_tags)
            if refreshed is not None:
                updated += 1
                record = refreshed

        if not record.tags:
            _, created = _taxonomist_support.enqueue_taxonomist_enrichment_work_item(
                ctx,
                task=task,
                memory_id=record.id,
                workspace_id=workspace_id,
            )
            if created:
                seeded_enrichment_count += 1

        complete_work_item(ctx, work_item.id)
        finalized_work_item_ids.add(work_item.id)

    for record in candidates:
        if record.tags:
            continue
        _, created = _taxonomist_support.enqueue_taxonomist_enrichment_work_item(
            ctx,
            task=task,
            memory_id=record.id,
            workspace_id=workspace_id,
        )
        if created:
            seeded_enrichment_count += 1

    for work_item in claimed_work_items:
        if work_item.id in finalized_work_item_ids:
            continue
        release_work_item(ctx, work_item.id)

    return _taxonomist_support.build_tag_normalizer_result(
        sampled_batch,
        sampled_records=candidates,
        updated=updated,
        seeded_work_item_count=len(seeded_work_items),
        claimed_work_item_count=len(claimed_work_items),
        seeded_enrichment_count=seeded_enrichment_count,
    )


async def _run_taxonomist_agentic_pass(
    ctx: ApplicationContext,
    *,
    task: TaskRecord,
    provider: Any,
    workspace_id: str | None,
    candidate_memory_ids: set[str],
    seeded_work_items: dict[str, Any],
    provider_call_budget: int,
) -> dict[str, Any]:
    run_agent = getattr(provider, "run_agent", None)
    assert callable(run_agent)
    work_item_batch_limit = _taxonomist_support.work_item_batch_limit(task)
    max_batches_per_run = _taxonomist_support.max_batches_per_run(task)
    reset_agentic_tool_tracking(ctx, task.id)
    try:
        agentic_result = await cast(Callable[[str], Awaitable[Any]], run_agent)(
            _taxonomist_support.build_agent_prompt(
                task,
                workspace_id=workspace_id,
                provider_call_budget=provider_call_budget,
                work_item_batch_limit=work_item_batch_limit,
                max_batches_per_run=max_batches_per_run,
            )
        )
    except Exception as exc:
        deterministic_counts = finalize_agentic_tool_tracking(ctx, task.id)
        normalized_counts = prefer_deterministic_agentic_counts(
            {
                "tool_calls_executed": 0,
                "mutations": 0,
                "tool_names_used": [],
            },
            deterministic_counts=deterministic_counts,
        )
        running_items = _taxonomist_support.list_running_work_items(
            ctx,
            task_id=task.id,
            workspace_id=workspace_id,
            limit=max(_taxonomist_support.TAXONOMIST_DEFAULT_PROVIDER_CALL_BUDGET, DEFAULT_AGENT_SCAN_LIMIT),
        )
        if _taxonomist_support.is_provider_deferred_error(exc):
            reason = classify_provider_failure(exc)
            _taxonomist_support.record_provider_deferred_event(
                ctx,
                task=task,
                provider=provider,
                reason_code=reason.reason_code,
                reason_category=reason.reason_category,
                retry_delay_seconds=reason.retry_delay_seconds,
            )
            for work_item in running_items:
                defer_work_item(
                    ctx,
                    work_item.id,
                    error=reason.reason_code,
                    retry_delay_seconds=reason.retry_delay_seconds,
                )
            return {
                "updated": _taxonomist_support.count_updated_records(ctx, candidate_memory_ids),
                "provider_calls_used": 0,
                "provider_deferred_reason_code": reason.reason_code,
                "provider_deferred_retry_delay_seconds": reason.retry_delay_seconds,
                "claimed_work_item_count": _taxonomist_support.count_claimed_work_items(ctx, seeded_work_items),
                "tool_calls_executed": normalized_counts["tool_calls_executed"],
                "mutations": normalized_counts["mutations"],
                "compatible_batch_calls": 0,
                "compatibility_group": "lightweight_review",
                "work_item_batch_limit": work_item_batch_limit,
                "max_batches_per_run": max_batches_per_run,
                "summary": None,
                "tool_names_used": normalized_counts["tool_names_used"],
            }
        for work_item in running_items:
            release_work_item(ctx, work_item.id)
        raise

    for work_item in _taxonomist_support.list_running_work_items(
        ctx,
        task_id=task.id,
        workspace_id=workspace_id,
        limit=max(_taxonomist_support.TAXONOMIST_DEFAULT_PROVIDER_CALL_BUDGET, DEFAULT_AGENT_SCAN_LIMIT),
    ):
        release_work_item(ctx, work_item.id)
    normalized = prefer_deterministic_agentic_counts(
        _taxonomist_support.normalize_agentic_result(
            agentic_result,
            coerce_text_summary=coerce_text_summary,
            extract_embedded_json_object=extract_embedded_json_object,
        ),
        deterministic_counts=finalize_agentic_tool_tracking(ctx, task.id),
    )
    return {
        "updated": _taxonomist_support.count_updated_records(ctx, candidate_memory_ids),
        "provider_calls_used": 1,
        "provider_deferred_reason_code": None,
        "provider_deferred_retry_delay_seconds": None,
        "claimed_work_item_count": _taxonomist_support.count_claimed_work_items(ctx, seeded_work_items),
        "tool_calls_executed": normalized["tool_calls_executed"],
        "mutations": normalized["mutations"],
        "compatible_batch_calls": normalized["compatible_batch_calls"],
        "compatibility_group": "lightweight_review",
        "work_item_batch_limit": work_item_batch_limit,
        "max_batches_per_run": max_batches_per_run,
        "summary": normalized["summary"],
        "tool_names_used": normalized["tool_names_used"],
    }


async def _run_taxonomist_json_pass(
    ctx: ApplicationContext,
    *,
    task: TaskRecord,
    provider: Any,
    candidates: list[Any],
    provider_call_budget: int,
) -> dict[str, Any]:
    assert ctx.repository is not None
    seeded_work_items: dict[str, Any] = {}
    claimed_work_items = _taxonomist_support.claim_taxonomist_work_batch(
        ctx,
        task=task,
        workspace_id=_resolve_workspace_id(ctx, task) or "global",
        candidates=candidates,
        limit=provider_call_budget,
        seeded_work_items=seeded_work_items,
    )
    claimed_by_memory_id = {
        _taxonomist_support.work_memory_id(record.payload): record
        for record in claimed_work_items
        if _taxonomist_support.work_memory_id(record.payload) is not None
    }
    finalized_work_item_ids: set[str] = set()
    updated = 0
    provider_calls_used = 0
    provider_deferred_reason_code: str | None = None
    provider_deferred_retry_delay_seconds: float | None = None
    for record in candidates:
        normalized_tags = normalize_tag_values(record.tags)
        work_item = claimed_by_memory_id.get(record.id)
        if provider is not None and work_item is not None and provider_calls_used < provider_call_budget:
            try:
                normalized_tags = await _taxonomist_support.provider_normalize_tags(
                    provider,
                    record,
                    normalized_tags,
                    normalize_tag_values,
                )
                provider_calls_used += 1
            except Exception as exc:
                if _taxonomist_support.is_provider_deferred_error(exc):
                    reason = classify_provider_failure(exc)
                    provider_deferred_reason_code = reason.reason_code
                    provider_deferred_retry_delay_seconds = reason.retry_delay_seconds
                    _taxonomist_support.record_provider_deferred_event(
                        ctx,
                        task=task,
                        provider=provider,
                        reason_code=reason.reason_code,
                        reason_category=reason.reason_category,
                        retry_delay_seconds=reason.retry_delay_seconds,
                    )
                    defer_work_item(
                        ctx,
                        work_item.id,
                        error=reason.reason_code,
                        retry_delay_seconds=reason.retry_delay_seconds,
                    )
                    finalized_work_item_ids.add(work_item.id)
                    provider = None
                    continue
                raise
        if normalized_tags == record.tags:
            if work_item is not None:
                complete_work_item(ctx, work_item.id)
                finalized_work_item_ids.add(work_item.id)
            continue
        refreshed = ctx.repository.update_memory(record.id, tags=normalized_tags)
        if refreshed is not None:
            updated += 1
        if work_item is not None:
            complete_work_item(ctx, work_item.id)
            finalized_work_item_ids.add(work_item.id)
    for work_item in claimed_work_items:
        if work_item.id in finalized_work_item_ids:
            continue
        release_work_item(ctx, work_item.id)
    return {
        "updated": updated,
        "provider_calls_used": provider_calls_used,
        "provider_deferred_reason_code": provider_deferred_reason_code,
        "provider_deferred_retry_delay_seconds": provider_deferred_retry_delay_seconds,
        "claimed_work_item_count": len(claimed_work_items),
    }
