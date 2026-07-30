from __future__ import annotations

from inspect import isawaitable
import time
from typing import Any

from mcp_memory.context import ApplicationContext
from mcp_memory.core.providers._json_cli import ProviderBackoffError
from mcp_memory.core.providers.interfaces import ProviderAdmissionDeferred
from mcp_memory.core.providers.interfaces import ProviderAuthenticationRequired
from mcp_memory.core.providers.interfaces import ProviderBudgetExceeded
from mcp_memory.core.providers.interfaces import ProviderRateLimitExceeded
from mcp_memory.core.sampling import (
    BOUNDED_NOISE_STRATEGY,
    COLD_STORAGE_STRATEGY,
    NEVER_SURFACED_STRATEGY,
    SamplingBatch,
)
from mcp_memory.core.task_handlers.agentic_result_support import (
    build_tool_usage_summary,
    count_mutating_agentic_tool_calls,
    extract_agentic_tool_names,
    extract_tool_counts,
)
from mcp_memory.core.task_handlers.campaigns import (
    campaign_family_keys,
    campaign_metadata,
    count_named_tool_calls_from_stats,
)
from mcp_memory.core.task_handlers.maintenance_framework import sample_maintenance_candidates, sampling_payload
from mcp_memory.core.tasks import TaskRecord
from mcp_memory.core.ports.work_items import (
    COMPATIBILITY_GROUP_LIGHTWEIGHT_REVIEW,
    EXECUTION_LANE_AGENTIC,
    WORK_FAMILY_MEMORY_TAGGING,
)

TAXONOMIST_ALLOWED_STRATEGIES = (
    COLD_STORAGE_STRATEGY,
    NEVER_SURFACED_STRATEGY,
    BOUNDED_NOISE_STRATEGY,
)
TAXONOMIST_STRATEGY_WEIGHTS = {
    COLD_STORAGE_STRATEGY: 2,
    NEVER_SURFACED_STRATEGY: 3,
    BOUNDED_NOISE_STRATEGY: 1,
}
TAXONOMIST_DEFAULT_PROVIDER_CALL_BUDGET = 1
TAXONOMIST_DEFAULT_WORK_ITEM_BATCH_LIMIT = 5
TAXONOMIST_DEFAULT_MAX_BATCHES_PER_RUN = 4
TAXONOMIST_WORK_ITEM_PRECHECK_LIMIT = 64

_TAXONOMIST_READ_ONLY_TOOL_NAMES = {
    "mcp_mcp-memory-internal_task_complete",
    "mcp_mcp-memory-internal_internal_get_work_batch",
    "mcp_mcp-memory-internal_internal_get_compatible_work_batch",
    "mcp_mcp-memory-internal_internal_read_memory_record",
    "mcp_mcp-memory-internal_internal_search_memory_records",
    "mcp_mcp-memory-internal_internal_list_memory_records",
    "mcp_mcp-memory-internal_internal_task_complete",
}


def select_taxonomist_sampling_batch(
    ctx: ApplicationContext,
    task: TaskRecord,
    candidates: list[Any],
    *,
    limit: int,
) -> SamplingBatch:
    return sample_maintenance_candidates(
        ctx,
        task,
        candidates,
        allowed_strategies=TAXONOMIST_ALLOWED_STRATEGIES,
        strategy_weights=TAXONOMIST_STRATEGY_WEIGHTS,
        limit=limit,
    )


def provider_call_budget(ctx: ApplicationContext, provider: Any) -> int:
    if provider is None:
        return 0
    config = getattr(ctx, "config", None)
    routing = None if config is None else getattr(config, "provider_routing", None)
    configured_limit = None if routing is None else getattr(routing, "model_burst_call_limit", None)
    if isinstance(configured_limit, int) and configured_limit > 0:
        return configured_limit
    return TAXONOMIST_DEFAULT_PROVIDER_CALL_BUDGET


def work_item_batch_limit(task: TaskRecord) -> int:
    raw_limit = task.data.get("work_item_batch_limit", TAXONOMIST_DEFAULT_WORK_ITEM_BATCH_LIMIT)
    try:
        return max(1, int(raw_limit))
    except (TypeError, ValueError):
        return TAXONOMIST_DEFAULT_WORK_ITEM_BATCH_LIMIT


def max_batches_per_run(task: TaskRecord) -> int:
    raw_limit = task.data.get("max_batches_per_run", TAXONOMIST_DEFAULT_MAX_BATCHES_PER_RUN)
    try:
        return max(1, int(raw_limit))
    except (TypeError, ValueError):
        return TAXONOMIST_DEFAULT_MAX_BATCHES_PER_RUN


def supports_agentic_execution(provider: Any) -> bool:
    run_agent = getattr(provider, "run_agent", None)
    supports_agentic = getattr(provider, "supports_agentic", None)
    return callable(run_agent) and (not callable(supports_agentic) or bool(supports_agentic()))


def is_provider_deferred_error(exc: Exception) -> bool:
    return isinstance(
        exc,
        (
            ProviderAdmissionDeferred,
            ProviderAuthenticationRequired,
            ProviderBackoffError,
            ProviderBudgetExceeded,
            ProviderRateLimitExceeded,
        ),
    )


def record_provider_deferred_event(
    ctx: ApplicationContext,
    *,
    task: TaskRecord,
    provider: Any,
    reason_code: str,
    reason_category: str,
    retry_delay_seconds: float | None,
) -> None:
    repository = getattr(ctx, "provider_policy_events", None)
    if repository is None:
        return
    repository.record_event(
        task_name=task.task_name,
        task_id=task.id,
        event_kind="provider_deferred",
        provider_key=getattr(provider, "_provider_key", None),
        provider_name=getattr(provider, "_provider_name", None),
        model_name=getattr(provider, "_model_name", None),
        reason_category=reason_category,
        reason_code=reason_code,
        retry_delay_seconds=retry_delay_seconds,
    )


def claim_taxonomist_work_batch(
    ctx: ApplicationContext,
    *,
    task: TaskRecord,
    workspace_id: str | None,
    candidates: list[Any],
    limit: int,
    seeded_work_items: dict[str, Any],
) -> list[Any]:
    work_items = getattr(ctx, "work_items", None)
    if work_items is None or limit < 1:
        return []
    if not seeded_work_items:
        seeded_work_items.update(seed_taxonomist_work_items(ctx, task=task, workspace_id=workspace_id, candidates=candidates))
    return work_items.claim_batch(
        family_key=WORK_FAMILY_MEMORY_TAGGING,
        execution_lane=EXECUTION_LANE_AGENTIC,
        lease_owner=task.id,
        limit=limit,
        workspace_id=workspace_id,
    )


def seed_taxonomist_work_items(
    ctx: ApplicationContext,
    *,
    task: TaskRecord,
    workspace_id: str | None,
    candidates: list[Any],
) -> dict[str, Any]:
    if getattr(ctx, "work_items", None) is None:
        return {}
    seeded: dict[str, Any] = {}
    for record in candidates:
        if record.tags:
            continue
        work_item, _ = enqueue_taxonomist_enrichment_work_item(
            ctx,
            task=task,
            memory_id=record.id,
            workspace_id=workspace_id,
        )
        seeded[record.id] = work_item
    return seeded


def enqueue_taxonomist_enrichment_work_item(
    ctx: ApplicationContext,
    *,
    task: TaskRecord,
    memory_id: str,
    workspace_id: str | None,
) -> tuple[Any, bool]:
    work_items = getattr(ctx, "work_items", None)
    if work_items is None:
        raise ValueError("work_items repository is not configured")
    return work_items.enqueue_unique(
        family_key=WORK_FAMILY_MEMORY_TAGGING,
        execution_lane=EXECUTION_LANE_AGENTIC,
        workspace_id=workspace_id,
        priority=task.priority,
        idempotency_key=f"memory_tagging:{memory_id}",
        payload={"memory_id": memory_id, "workspace_id": workspace_id},
    )


def normalize_candidate_tags(
    ctx: ApplicationContext,
    record: Any,
    *,
    normalize_tag_values,
) -> tuple[Any, bool]:
    repository = getattr(ctx, "repository", None)
    if repository is None:
        return record, False
    normalized_tags = normalize_tag_values(record.tags)
    if normalized_tags == record.tags:
        return record, False
    refreshed = repository.update_memory(record.id, tags=normalized_tags)
    if refreshed is None:
        return record, False
    return refreshed, True


def inline_normalize_taxonomist_candidates(
    ctx: ApplicationContext,
    candidates: list[Any],
    *,
    normalize_tag_values,
) -> tuple[list[Any], list[Any], int]:
    normalized_candidates: list[Any] = []
    untagged_candidates: list[Any] = []
    updated = 0
    for record in candidates:
        refreshed_record, was_updated = normalize_candidate_tags(
            ctx,
            record,
            normalize_tag_values=normalize_tag_values,
        )
        if was_updated:
            updated += 1
        normalized_candidates.append(refreshed_record)
        if not refreshed_record.tags:
            untagged_candidates.append(refreshed_record)
    return normalized_candidates, untagged_candidates, updated


def work_memory_id(payload: dict[str, Any]) -> str | None:
    memory_id = payload.get("memory_id")
    if isinstance(memory_id, str) and memory_id.strip():
        return memory_id
    return None


def build_agent_prompt(
    task: TaskRecord,
    *,
    workspace_id: str | None,
    provider_call_budget: int,
    work_item_batch_limit: int,
    max_batches_per_run: int,
) -> str:
    lightweight_families = list(campaign_family_keys(COMPATIBILITY_GROUP_LIGHTWEIGHT_REVIEW))
    return (
        "You are the taxonomist maintenance agent for the global memory store.\n"
        "Use the workspace-local internal MCP maintenance tools directly.\n"
        "Treat this run as a lightweight-review campaign: if the current tagging batch is done, keep the same provider session alive while compatible lightweight work remains productive and safe.\n"
        f"Aim to drain eligible {WORK_FAMILY_MEMORY_TAGGING} work for this task in one run without leaving the family boundary.\n"
        f"Repeatedly call internal_get_work_batch using task_id='{task.id}', family_key='{WORK_FAMILY_MEMORY_TAGGING}', execution_lane='{EXECUTION_LANE_AGENTIC}', and limit={work_item_batch_limit}.\n"
        f"Stop when the batch comes back empty, when you finish {max_batches_per_run} batch pulls, or when lease/time safety suggests you should stop.\n"
        "For each claimed work item, read the target memory, normalize its tags to a concise canonical set, use internal_update_memory_record when the tag set should change, and then finalize the work item.\n"
        f"After the active tagging frontier is exhausted, you may continue with lightweight follow-on work by calling internal_get_compatible_work_batch using task_id='{task.id}', compatibility_group='lightweight_review', execution_lane='{EXECUTION_LANE_AGENTIC}', allowed_families={lightweight_families}, and limit={work_item_batch_limit}.\n"
        "For claimed graph_link_review items, inspect payload.candidate_memory_ids, read the candidate memories, and create safe typed links with internal_create_memory_link when justified. For claimed conflict_review items, inspect payload.candidate_memory_ids, read the candidate memories, and add CONTRADICTS links only when the conflict is explicit and evidence-backed.\n"
        "When you claim follow-on lightweight work, complete, release, defer, or heartbeat that work item yourself through the work-item lifecycle tools before moving on.\n"
        "Do not stop the provider session just because the initial tagging loop ended if one more compatible lightweight item would still be high-yield.\n"
        "Use internal_complete_work_item after a successful tag decision, internal_release_work_item when no safe change is needed, and internal_defer_work_item only when a claimed item truly needs delayed retry semantics.\n"
        "Use internal_heartbeat_work_item if you need to extend a claimed lease before finishing it.\n"
        f"You still have only one external model execution for this run, so maximize useful work inside that single run. Provider call budget for this run: {provider_call_budget}.\n"
        "Do not create journal or memory records for routine status traces.\n"
        'When finished, output final JSON only in the form {"summary": "...", "updated_memory_ids": ["..."]}.\n'
    )


def list_running_work_items(
    ctx: ApplicationContext,
    *,
    task_id: str,
    workspace_id: str | None,
    limit: int,
) -> list[Any]:
    work_items = getattr(ctx, "work_items", None)
    if work_items is None:
        return []
    return work_items.list_items(
        family_key=WORK_FAMILY_MEMORY_TAGGING,
        execution_lane=EXECUTION_LANE_AGENTIC,
        status="running",
        workspace_id=workspace_id,
        lease_owner=task_id,
        limit=limit,
    )


def count_claimed_work_items(ctx: ApplicationContext, seeded_work_items: dict[str, Any]) -> int:
    work_items = getattr(ctx, "work_items", None)
    if work_items is None:
        return 0
    claimed = 0
    for work_item in seeded_work_items.values():
        current = work_items.get_item(work_item.id)
        if current.attempt_count > 0:
            claimed += 1
    return claimed


def count_updated_records(ctx: ApplicationContext, memory_ids: set[str]) -> int:
    if ctx.repository is None or not memory_ids:
        return 0
    updated = 0
    for memory_id in memory_ids:
        record = ctx.repository.get_memory(memory_id)
        if record is None:
            continue
        if record.tags:
            updated += 1
    return updated


def has_ready_tagging_work(
    ctx: ApplicationContext,
    *,
    workspace_id: str | None,
) -> bool:
    work_items = getattr(ctx, "work_items", None)
    if work_items is None:
        return False
    now = time.time()

    for item in work_items.list_items(
        family_key=WORK_FAMILY_MEMORY_TAGGING,
        execution_lane=EXECUTION_LANE_AGENTIC,
        workspace_id=workspace_id,
        limit=TAXONOMIST_WORK_ITEM_PRECHECK_LIMIT,
    ):
        if item.status in {"pending", "deferred"} and item.available_at <= now:
            return True
        if item.status == "running" and item.lease_expires_at is not None and item.lease_expires_at <= now:
            return True
    return False


def normalize_agentic_result(agentic_result: Any, *, coerce_text_summary, extract_embedded_json_object) -> dict[str, Any]:
    summary = coerce_text_summary(getattr(agentic_result, "summary", None))
    parsed = getattr(agentic_result, "parsed", None)
    tool_counts = extract_tool_counts(parsed)
    if isinstance(parsed, dict):
        response = parsed.get("response")
        if isinstance(response, str):
            parsed_response = extract_embedded_json_object(response)
            if isinstance(parsed_response, dict):
                summary = coerce_text_summary(parsed_response.get("summary")) or summary
    return {
        "summary": summary,
        "compatible_batch_calls": count_named_tool_calls_from_stats(
            tool_counts,
            tool_name="internal_get_compatible_work_batch",
        ),
    } | build_tool_usage_summary(parsed, read_only_tool_names=_TAXONOMIST_READ_ONLY_TOOL_NAMES)


def build_taxonomist_result(
    sampled_batch: SamplingBatch,
    *,
    sampled_records: list[Any],
    updated: int,
    inline_normalized_count: int = 0,
    provider_call_budget: int,
    provider_calls_used: int,
    provider_deferred_reason_code: str | None,
    provider_deferred_retry_delay_seconds: float | None,
    claimed_work_item_count: int,
    tool_calls_executed: int | None = None,
    mutations: int | None = None,
    tool_names_used: list[str] | None = None,
    compatible_batch_calls: int | None = None,
    summary: str | None = None,
    execution_mode: str | None = None,
    compatibility_group: str | None = None,
    work_item_batch_limit: int | None = None,
    max_batches_per_run: int | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "updated": updated,
        "inline_normalized_count": inline_normalized_count,
        "provider_calls_used": provider_calls_used,
        "provider_call_budget": provider_call_budget,
        "provider_deferred_reason_code": provider_deferred_reason_code,
        "provider_deferred_retry_delay_seconds": provider_deferred_retry_delay_seconds,
        "claimed_work_item_count": claimed_work_item_count,
    }
    metrics.update(
        campaign_metadata(
            compatibility_group=COMPATIBILITY_GROUP_LIGHTWEIGHT_REVIEW,
            origin_family=WORK_FAMILY_MEMORY_TAGGING,
            execution_lane=EXECUTION_LANE_AGENTIC,
        )
    )
    if summary is not None:
        metrics["summary"] = summary
    if tool_calls_executed is not None:
        metrics["tool_calls_executed"] = tool_calls_executed
    if mutations is not None:
        metrics["mutations"] = mutations
    if tool_names_used is not None:
        metrics["tool_names_used"] = tool_names_used
    if compatible_batch_calls is not None:
        metrics["compatible_batch_calls"] = compatible_batch_calls
    if execution_mode is not None:
        metrics["execution_mode"] = execution_mode
    if compatibility_group is not None:
        metrics["compatibility_group"] = compatibility_group
    if work_item_batch_limit is not None:
        metrics["work_item_batch_limit"] = work_item_batch_limit
    if max_batches_per_run is not None:
        metrics["max_batches_per_run"] = max_batches_per_run
    if reason is not None:
        metrics["reason"] = reason
    return sampling_payload(sampled_batch, sampled_records=sampled_records, **metrics)


def _extract_agentic_tool_names(value: object) -> list[str]:
    return extract_agentic_tool_names(value)


def _count_mutating_agentic_tool_calls(value: object) -> int:
    return count_mutating_agentic_tool_calls(value, read_only_tool_names=_TAXONOMIST_READ_ONLY_TOOL_NAMES)


async def provider_normalize_tags(provider: Any, record: Any, normalized_tags: list[str], normalize_tag_values) -> list[str]:
    prompt = (
        "Normalize these tags to a concise canonical set. Return JSON with {\"tags\": [...]} only.\n"
        f"Title: {record.title}\nTags: {normalized_tags}"
    )
    response = provider.ask(prompt)
    if isawaitable(response):
        response = await response
    provider_tags = response.get("tags", [])
    if not isinstance(provider_tags, list):
        return normalized_tags
    return normalize_tag_values([str(tag) for tag in provider_tags]) or normalized_tags
