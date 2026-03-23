from __future__ import annotations

from inspect import isawaitable
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
)
from mcp_memory.core.task_handlers.maintenance_framework import SamplingBatch, sampling_payload
from mcp_memory.core.tasks import TaskRecord
from mcp_memory.work_item_store import (
    EXECUTION_LANE_AGENTIC,
    EXECUTION_LANE_DETERMINISTIC,
    WORK_FAMILY_MEMORY_TAG_NORMALIZATION,
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


def provider_call_budget(ctx: ApplicationContext, provider: Any) -> int:
    if provider is None:
        return 0
    config = getattr(ctx, "config", None)
    routing = None if config is None else getattr(config, "provider_routing", None)
    configured_limit = None if routing is None else getattr(routing, "model_burst_call_limit", None)
    if isinstance(configured_limit, int) and configured_limit > 0:
        return configured_limit
    return TAXONOMIST_DEFAULT_PROVIDER_CALL_BUDGET


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
    workspace_id: str,
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


def claim_tag_normalizer_work_batch(
    ctx: ApplicationContext,
    *,
    task: TaskRecord,
    workspace_id: str,
    limit: int,
) -> list[Any]:
    work_items = getattr(ctx, "work_items", None)
    if work_items is None or limit < 1:
        return []
    return work_items.claim_batch(
        family_key=WORK_FAMILY_MEMORY_TAG_NORMALIZATION,
        execution_lane=EXECUTION_LANE_DETERMINISTIC,
        lease_owner=task.id,
        limit=limit,
        workspace_id=workspace_id,
    )


def seed_taxonomist_work_items(
    ctx: ApplicationContext,
    *,
    task: TaskRecord,
    workspace_id: str,
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


def seed_tag_normalizer_work_items(
    ctx: ApplicationContext,
    *,
    task: TaskRecord,
    workspace_id: str,
    candidates: list[Any],
    normalize_tag_values,
) -> dict[str, Any]:
    work_items = getattr(ctx, "work_items", None)
    if work_items is None:
        return {}
    seeded: dict[str, Any] = {}
    for record in candidates:
        normalized_tags = normalize_tag_values(record.tags)
        if normalized_tags == record.tags:
            continue
        signature = tag_normalization_signature(record.tags)
        work_item, _ = work_items.enqueue_unique(
            family_key=WORK_FAMILY_MEMORY_TAG_NORMALIZATION,
            execution_lane=EXECUTION_LANE_DETERMINISTIC,
            workspace_id=workspace_id,
            priority=task.priority,
            idempotency_key=f"memory_tag_normalization:{record.id}:{signature}",
            payload={"memory_id": record.id, "workspace_id": workspace_id, "observed_tags": list(record.tags)},
        )
        seeded[record.id] = work_item
    return seeded


def enqueue_taxonomist_enrichment_work_item(
    ctx: ApplicationContext,
    *,
    task: TaskRecord,
    memory_id: str,
    workspace_id: str,
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


def needs_tag_normalization(tags: list[str], normalize_tag_values) -> bool:
    normalized_tags = normalize_tag_values(tags)
    return bool(tags) and normalized_tags != tags


def tag_normalization_signature(tags: list[str]) -> str:
    if not tags:
        return "untagged"
    return "|".join(tag.strip() for tag in tags)


def work_memory_id(payload: dict[str, Any]) -> str | None:
    memory_id = payload.get("memory_id")
    if isinstance(memory_id, str) and memory_id.strip():
        return memory_id
    return None


def build_agent_prompt(
    task: TaskRecord,
    *,
    workspace_id: str,
    provider_call_budget: int,
) -> str:
    return (
        "You are the taxonomist maintenance agent for the global memory store.\n"
        "Use the workspace-local internal MCP maintenance tools directly.\n"
        f"Start with internal_get_work_batch using task_id='{task.id}', family_key='{WORK_FAMILY_MEMORY_TAGGING}', execution_lane='{EXECUTION_LANE_AGENTIC}', workspace_id='{workspace_id}', and limit={provider_call_budget}.\n"
        "For each claimed work item, read the target memory, normalize its tags to a concise canonical set, use internal_update_memory_record when the tag set should change, and then finalize the work item.\n"
        "Use internal_complete_work_item after a successful tag decision, internal_release_work_item when no safe change is needed, and internal_defer_work_item only when a claimed item truly needs delayed retry semantics.\n"
        "Use internal_heartbeat_work_item if you need to extend a claimed lease before finishing it.\n"
        "Do not create journal or memory records for routine status traces.\n"
        'When finished, output final JSON only in the form {"summary": "...", "updated_memory_ids": ["..."]}.\n'
    )


def list_running_work_items(
    ctx: ApplicationContext,
    *,
    task_id: str,
    workspace_id: str,
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


def normalize_agentic_result(agentic_result: Any, *, coerce_text_summary, extract_embedded_json_object) -> dict[str, Any]:
    summary = coerce_text_summary(getattr(agentic_result, "summary", None))
    parsed = getattr(agentic_result, "parsed", None)
    if isinstance(parsed, dict):
        response = parsed.get("response")
        if isinstance(response, str):
            parsed_response = extract_embedded_json_object(response)
            if isinstance(parsed_response, dict):
                summary = coerce_text_summary(parsed_response.get("summary")) or summary
    return {"summary": summary}


def build_taxonomist_result(
    sampled_batch: SamplingBatch,
    *,
    sampled_records: list[Any],
    updated: int,
    provider_call_budget: int,
    provider_calls_used: int,
    provider_deferred_reason_code: str | None,
    provider_deferred_retry_delay_seconds: float | None,
    claimed_work_item_count: int,
    summary: str | None = None,
    execution_mode: str | None = None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "updated": updated,
        "provider_calls_used": provider_calls_used,
        "provider_call_budget": provider_call_budget,
        "provider_deferred_reason_code": provider_deferred_reason_code,
        "provider_deferred_retry_delay_seconds": provider_deferred_retry_delay_seconds,
        "claimed_work_item_count": claimed_work_item_count,
    }
    if summary is not None:
        metrics["summary"] = summary
    if execution_mode is not None:
        metrics["execution_mode"] = execution_mode
    return sampling_payload(sampled_batch, sampled_records=sampled_records, **metrics)


def build_tag_normalizer_result(
    sampled_batch: SamplingBatch,
    *,
    sampled_records: list[Any],
    updated: int,
    seeded_work_item_count: int,
    claimed_work_item_count: int,
    seeded_enrichment_count: int,
) -> dict[str, Any]:
    return sampling_payload(
        sampled_batch,
        sampled_records=sampled_records,
        updated=updated,
        seeded_work_item_count=seeded_work_item_count,
        claimed_work_item_count=claimed_work_item_count,
        seeded_enrichment_count=seeded_enrichment_count,
    )


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
