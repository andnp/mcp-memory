from __future__ import annotations

from typing import Any, Awaitable, Callable, cast

from mcp_memory.context import ApplicationContext, TaskRuntimeContext
from mcp_memory.core.curation_shadow import (
    curator_create_link_execution_enabled,
    curator_normalize_execution_enabled,
    curator_shadow_mode_enabled,
    run_curator_verified_create_link_execution,
    run_curator_shadow_mode,
    run_curator_verified_normalize_execution,
)
from mcp_memory.core.task_handlers.agentic_tool_tracking import (
    finalize_agentic_tool_tracking,
    prefer_deterministic_agentic_counts,
    reset_agentic_tool_tracking,
)
from mcp_memory.core.task_handlers.agentic_guardrails import build_curator_guardrails
from mcp_memory.core.task_handlers.campaigns import campaign_metadata, count_named_tool_calls
import mcp_memory.core.task_handlers.curator_support as _curator_support
from mcp_memory.core.task_handlers.maintenance_framework import sampling_payload
from mcp_memory.core.task_handlers.maintenance_work_items import (
    complete_work_item,
    release_work_item,
    work_item_result_metadata,
)
from mcp_memory.core.task_handlers.tool_loop import run_internal_tool_loop
from mcp_memory.core.tasks import TaskRecord
from mcp_memory.work_item_store import (
    COMPATIBILITY_GROUP_STRUCTURAL_REVIEW,
    EXECUTION_LANE_AGENTIC,
    WORK_FAMILY_MEMORY_CURATION_REVIEW,
    compatibility_group_families,
)


CURATOR_JSON_TOOL_LOOP_MAX_ROUNDS = 10
CURATOR_JSON_TOOL_LOOP_MAX_TOOL_CALLS_PER_ROUND = 8


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
            seed_batch = _curator_support.select_curator_seed_batch(ctx, task)
            seed_records = seed_batch.records
    else:
        seed_batch = _curator_support.select_curator_seed_batch(ctx, task)
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

    if curator_shadow_mode_enabled(ctx):
        return await run_curator_shadow_mode(
            ctx,
            task,
            provider=provider,
            seed_batch=seed_batch,
            sampled_records=sampled_records,
            seed_records=seed_records,
            claimed_work_item=claimed_review_item,
            work_item_metadata=work_item_metadata,
        )

    if curator_normalize_execution_enabled(ctx):
        return await run_curator_verified_normalize_execution(
            ctx,
            task,
            provider=provider,
            seed_batch=seed_batch,
            sampled_records=sampled_records,
            seed_records=seed_records,
            claimed_work_item=claimed_review_item,
            work_item_metadata=work_item_metadata,
        )

    if curator_create_link_execution_enabled(ctx):
        return await run_curator_verified_create_link_execution(
            ctx,
            task,
            provider=provider,
            seed_batch=seed_batch,
            sampled_records=sampled_records,
            seed_records=seed_records,
            claimed_work_item=claimed_review_item,
            work_item_metadata=work_item_metadata,
        )

    curator_guardrails = build_curator_guardrails()
    prompt = _curator_support.build_json_tool_loop_prompt(
        task,
        strategy_used=seed_batch.strategy_used,
        seed_records=seed_records,
        guardrails=curator_guardrails,
    )
    run_agent = getattr(provider, "run_agent", None)
    supports_agentic = getattr(provider, "supports_agentic", None)
    reset_agentic_tool_tracking(ctx, task.id)
    if callable(run_agent) and (not callable(supports_agentic) or supports_agentic()):
        agentic_prompt = _curator_support.build_agentic_prompt(
            task,
            strategy_used=seed_batch.strategy_used,
            seed_records=seed_records,
            guardrails=curator_guardrails,
        )
        validation_retry_count = 0
        agentic_result: Any = None
        normalized_agentic: dict[str, Any] = {}
        try:
            for attempt in range(2):
                agentic_result = await cast(Callable[[str], Awaitable[Any]], run_agent)(agentic_prompt)
                deterministic_counts = finalize_agentic_tool_tracking(ctx, task.id)
                normalized_agentic = prefer_deterministic_agentic_counts(
                    _curator_support.normalize_curator_agentic_result(agentic_result),
                    deterministic_counts=deterministic_counts,
                )
                raw_summary = getattr(agentic_result, "summary", None)
                if (
                    attempt == 0
                    and int(normalized_agentic["tool_calls_executed"]) <= 0
                    and _curator_support.curator_summary_claims_mutating_actions(raw_summary)
                ):
                    validation_retry_count = 1
                    reset_agentic_tool_tracking(ctx, task.id)
                    agentic_prompt = (
                        agentic_prompt
                        + "\n\nvalidation_error: Your previous response claimed maintenance actions, "
                        "but no internal MCP tool calls were observed. Execute the claimed actions "
                        "through the internal MCP tools before returning the final JSON summary."
                    )
                    continue
                break
        except Exception:
            if claimed_review_item is not None:
                release_work_item(ctx, claimed_review_item.id)
            finalize_agentic_tool_tracking(ctx, task.id)
            raise
        if claimed_review_item is not None:
            complete_work_item(ctx, claimed_review_item.id)
        normalized_agentic["summary"] = _curator_support.normalize_curator_summary(
            {"summary": getattr(agentic_result, "summary", None)},
            tool_calls_executed=normalized_agentic["tool_calls_executed"],
        )
        return sampling_payload(
            seed_batch,
            sampled_records=sampled_records,
            seed_records=seed_records,
            extra=work_item_metadata,
            summary=normalized_agentic["summary"],
            execution_mode="agentic_mcp",
            claimed_work_item_count=1 if claimed_review_item is not None else 0,
            tool_calls_executed=normalized_agentic["tool_calls_executed"],
            mutations=normalized_agentic["mutations"],
            tool_names_used=normalized_agentic["tool_names_used"],
            compatible_batch_calls=count_named_tool_calls(
                normalized_agentic["tool_names_used"],
                tool_name="internal_get_compatible_work_batch",
            ),
            validation_retry_count=validation_retry_count,
        )

    try:
        loop_result = await run_internal_tool_loop(
            ctx,
            provider,
            prompt=prompt,
            allowed_tool_names=[
                "internal_search_memory_records",
                "internal_read_memory_record",
                "internal_list_memory_records",
                "internal_get_next_curator_batch",
                "internal_get_compatible_work_batch",
                "task_complete",
                "internal_task_complete",
                "internal_heartbeat_work_item",
                "internal_complete_work_item",
                "internal_defer_work_item",
                "internal_release_work_item",
                "internal_append_memory_content",
                "internal_archive_memory_record",
                "internal_merge_memory_into_canonical",
                "internal_split_memory_record",
                "internal_create_memory_record",
                "internal_update_memory_record",
                "internal_delete_memory_record",
                "internal_create_memory_link",
                "internal_delete_memory_link",
            ],
            max_rounds=CURATOR_JSON_TOOL_LOOP_MAX_ROUNDS,
            max_tool_calls_per_round=CURATOR_JSON_TOOL_LOOP_MAX_TOOL_CALLS_PER_ROUND,
            unsupported_no_tool_response_error=_curator_support.unsupported_curator_no_tool_response_error,
        )
    except Exception:
        if claimed_review_item is not None:
            release_work_item(ctx, claimed_review_item.id)
        finalize_agentic_tool_tracking(ctx, task.id)
        raise
    if claimed_review_item is not None:
        complete_work_item(ctx, claimed_review_item.id)
    deterministic_counts = finalize_agentic_tool_tracking(ctx, task.id)
    tool_calls_executed = loop_result.tool_calls_executed
    mutating_tool_calls = loop_result.mutating_tool_calls
    tool_names_used = loop_result.tool_names_used
    if deterministic_counts is not None:
        tool_calls_executed = int(getattr(deterministic_counts, "total_calls", 0))
        mutating_tool_calls = int(getattr(deterministic_counts, "mutating_calls", 0))
        tool_names_used = list(getattr(deterministic_counts, "tool_names_used", []))
    summary = _curator_support.normalize_curator_summary(
        loop_result.response,
        tool_calls_executed=tool_calls_executed,
    )
    return sampling_payload(
        seed_batch,
        sampled_records=sampled_records,
        seed_records=seed_records,
        extra=work_item_metadata,
        summary=summary,
        execution_mode="json_tool_loop",
        claimed_work_item_count=1 if claimed_review_item is not None else 0,
        tool_calls_executed=tool_calls_executed,
        mutations=mutating_tool_calls,
        tool_names_used=tool_names_used,
        compatible_batch_calls=count_named_tool_calls(
            tool_names_used,
            tool_name="internal_get_compatible_work_batch",
        ),
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
