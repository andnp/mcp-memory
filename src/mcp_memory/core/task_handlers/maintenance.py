from __future__ import annotations

from inspect import isawaitable
import json
from typing import Any, Awaitable, Callable, cast
import re

from mcp_memory.context import ApplicationContext
from mcp_memory.core.provider_admission import classify_provider_failure
from mcp_memory.core.providers._json_cli import ProviderBackoffError
from mcp_memory.core.providers.interfaces import ProviderAdmissionDeferred
from mcp_memory.core.providers.interfaces import ProviderAuthenticationRequired
from mcp_memory.core.providers.interfaces import ProviderBudgetExceeded
from mcp_memory.core.providers.interfaces import ProviderRateLimitExceeded
from mcp_memory.core.sampling import (
    ANOMALY_STRATEGY,
    BOUNDED_NOISE_STRATEGY,
    COLD_STORAGE_STRATEGY,
    COOLDOWN_ESCAPE_STRATEGY,
    CONFLICT_FRONTIER_STRATEGY,
    GRAPH_BRIDGE_STRATEGY,
    NEVER_SURFACED_STRATEGY,
    SEMANTIC_STRATEGY,
)
from mcp_memory.core.task_handlers.maintenance_framework import (
    requested_sampling_strategy,
    sample_maintenance_candidates,
    sampling_payload,
)
from mcp_memory.core.task_handlers.maintenance_work_items import (
    claim_work_batch as _claim_work_batch,
    complete_work_item as _complete_work_item,
    defer_work_item as _defer_work_item,
    enqueue_review_work_item as _enqueue_review_work_item,
    payload_memory_records as _payload_memory_records,
    release_work_item as _release_work_item,
    run_claimed_review_work_item as _run_claimed_review_work_item,
    run_sparse_frontier_review_task as _run_sparse_frontier_review_task,
    work_item_result_metadata as _work_item_result_metadata,
)
import mcp_memory.core.task_handlers.curator_support as _curator_support
import mcp_memory.core.task_handlers.deduplicator_merge as _deduplicator_merge
import mcp_memory.core.task_handlers.deduplicator_support as _deduplicator_support
import mcp_memory.core.task_handlers.defragmenter_support as _defragmenter_support
import mcp_memory.core.task_handlers.maintenance_housekeeping as _maintenance_housekeeping
import mcp_memory.core.task_handlers.relationship_proposals as _relationship_proposals
from mcp_memory.core.task_handlers.constants import (
    DEFAULT_AGENT_SCAN_LIMIT,
    CURATOR_TASK_NAME,
)
from mcp_memory.core.task_handlers.agentic_guardrails import (
    build_curator_guardrails,
)
from mcp_memory.core.task_handlers.tool_loop import run_internal_tool_loop
from mcp_memory.core.tasks import TaskRecord
from mcp_memory.work_item_store import (
    WORK_FAMILY_CONFLICT_REVIEW,
    EXECUTION_LANE_AGENTIC,
    EXECUTION_LANE_DETERMINISTIC,
    WORK_FAMILY_GRAPH_LINK_REVIEW,
    WORK_FAMILY_MEMORY_CURATION_REVIEW,
    WORK_FAMILY_MEMORY_DEDUP_REVIEW,
    WORK_FAMILY_MEMORY_TAG_NORMALIZATION,
    WORK_FAMILY_MEMORY_TAGGING,
)


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_:-]+")
DEDUPLICATOR_MAX_SEED_RECORDS = 8
DEDUPLICATOR_SIZE_ANOMALY_SEED_RECORDS = 2
DEDUPLICATOR_OBSERVATION_SEED_RECORDS = 4
DEDUPLICATOR_ALLOWED_STRATEGIES = (
    SEMANTIC_STRATEGY,
    ANOMALY_STRATEGY,
    COOLDOWN_ESCAPE_STRATEGY,
)
DEDUPLICATOR_STRATEGY_WEIGHTS = {
    SEMANTIC_STRATEGY: 4,
    ANOMALY_STRATEGY: 2,
    COOLDOWN_ESCAPE_STRATEGY: 2,
}
CURATOR_JSON_TOOL_LOOP_MAX_ROUNDS = 10
CURATOR_JSON_TOOL_LOOP_MAX_TOOL_CALLS_PER_ROUND = 8
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


handle_project_manager_task = _maintenance_housekeeping.handle_project_manager_task
handle_fact_checker_task = _maintenance_housekeeping.handle_fact_checker_task
handle_sweeper_task = _maintenance_housekeeping.handle_sweeper_task


async def handle_graph_linker_task(
    ctx: ApplicationContext,
    task: TaskRecord,
    provider: Any = None,
) -> dict[str, Any]:
    if ctx.repository is None:
        return {"created": 0}

    if provider is not None:
        claimed_review_result = await _run_claimed_review_work_item(
            ctx,
            task=task,
            family_key=WORK_FAMILY_GRAPH_LINK_REVIEW,
            load_candidates=_graph_link_review_candidates,
            propose_pairs=lambda review_candidates: _relationship_proposals.propose_graph_links(
                ctx,
                review_candidates,
                provider,
            ),
            apply_pairs=lambda proposed_pairs: _apply_graph_link_proposals(ctx, proposed_pairs),
        )
        if claimed_review_result is not None:
            return claimed_review_result

    all_candidates = ctx.repository.list_memories(
        workspace_id=_resolve_workspace_id(ctx, task),
        status="active",
        limit=int(task.data.get("limit", DEFAULT_AGENT_SCAN_LIMIT)),
    )
    sampled_batch = _sample_maintenance_candidates(
        ctx,
        task,
        all_candidates,
        allowed_strategies=GRAPH_LINKER_ALLOWED_STRATEGIES,
        strategy_weights=GRAPH_LINKER_STRATEGY_WEIGHTS,
        limit=min(len(all_candidates), DEFAULT_AGENT_SCAN_LIMIT),
    )
    candidates = sampled_batch.records
    if len(candidates) < 2:
        return sampling_payload(sampled_batch, sampled_records=candidates, created=0)

    proposed_pairs = await _relationship_proposals.propose_graph_links(ctx, candidates, provider)
    created = _apply_graph_link_proposals(ctx, proposed_pairs)

    return sampling_payload(sampled_batch, sampled_records=candidates, created=created)


async def handle_graph_link_discovery_task(
    ctx: ApplicationContext,
    task: TaskRecord,
) -> dict[str, Any]:
    if ctx.repository is None:
        return {"created": 0, "seeded_work_item_count": 0}

    workspace_id = _resolve_workspace_id(ctx, task)
    all_candidates = ctx.repository.list_memories(
        workspace_id=workspace_id,
        status="active",
        limit=int(task.data.get("limit", DEFAULT_AGENT_SCAN_LIMIT)),
    )
    sampled_batch = _sample_maintenance_candidates(
        ctx,
        task,
        all_candidates,
        allowed_strategies=GRAPH_LINKER_ALLOWED_STRATEGIES,
        strategy_weights=GRAPH_LINKER_STRATEGY_WEIGHTS,
        limit=min(len(all_candidates), DEFAULT_AGENT_SCAN_LIMIT),
    )
    candidates = sampled_batch.records
    if len(candidates) < 2:
        return sampling_payload(sampled_batch, sampled_records=candidates, created=0, seeded_work_item_count=0)
    return await _run_sparse_frontier_review_task(
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
        apply_pairs=lambda proposed_pairs: _apply_graph_link_proposals(ctx, proposed_pairs),
        should_seed=lambda proposed_pairs, review_candidates: (
            len(proposed_pairs) < _relationship_proposals.GRAPH_LINKER_FALLBACK_LINK_TARGET
            and len(review_candidates) > _relationship_proposals.GRAPH_LINKER_AI_MIN_CANDIDATES
        ),
        enqueue_review_work_item=lambda review_candidates: _enqueue_graph_link_review_work_item(
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
        claimed_review_result = await _run_claimed_review_work_item(
            ctx,
            task=task,
            family_key=WORK_FAMILY_CONFLICT_REVIEW,
            load_candidates=_conflict_review_candidates,
            propose_pairs=lambda review_candidates: _relationship_proposals.propose_conflicts(
                ctx,
                review_candidates,
                provider,
            ),
            apply_pairs=lambda proposed_pairs: _apply_conflict_proposals(ctx, proposed_pairs),
        )
        if claimed_review_result is not None:
            return claimed_review_result

    all_candidates = [
        record
        for record in ctx.repository.list_memories(
            workspace_id=_resolve_workspace_id(ctx, task),
            status="active",
            limit=int(task.data.get("limit", DEFAULT_AGENT_SCAN_LIMIT)),
        )
        if record.type in {"fact", "plan"}
    ]
    sampled_batch = _sample_maintenance_candidates(
        ctx,
        task,
        all_candidates,
        allowed_strategies=CONFLICT_DETECTOR_ALLOWED_STRATEGIES,
        strategy_weights=CONFLICT_DETECTOR_STRATEGY_WEIGHTS,
        limit=min(len(all_candidates), DEFAULT_AGENT_SCAN_LIMIT),
    )
    candidates = sampled_batch.records
    if len(candidates) < 2:
        return sampling_payload(sampled_batch, sampled_records=candidates, created=0)

    proposed_pairs = await _relationship_proposals.propose_conflicts(ctx, candidates, provider)
    created = _apply_conflict_proposals(ctx, proposed_pairs)

    return sampling_payload(sampled_batch, sampled_records=candidates, created=created)


async def handle_conflict_screening_task(
    ctx: ApplicationContext,
    task: TaskRecord,
) -> dict[str, Any]:
    if ctx.repository is None:
        return {"created": 0, "seeded_work_item_count": 0}

    workspace_id = _resolve_workspace_id(ctx, task)
    all_candidates = [
        record
        for record in ctx.repository.list_memories(
            workspace_id=workspace_id,
            status="active",
            limit=int(task.data.get("limit", DEFAULT_AGENT_SCAN_LIMIT)),
        )
        if record.type in {"fact", "plan"}
    ]
    sampled_batch = _sample_maintenance_candidates(
        ctx,
        task,
        all_candidates,
        allowed_strategies=CONFLICT_DETECTOR_ALLOWED_STRATEGIES,
        strategy_weights=CONFLICT_DETECTOR_STRATEGY_WEIGHTS,
        limit=min(len(all_candidates), DEFAULT_AGENT_SCAN_LIMIT),
    )
    candidates = sampled_batch.records
    if len(candidates) < 2:
        return sampling_payload(sampled_batch, sampled_records=candidates, created=0, seeded_work_item_count=0)

    return await _run_sparse_frontier_review_task(
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
        apply_pairs=lambda proposed_pairs: _apply_conflict_proposals(ctx, proposed_pairs),
        should_seed=lambda proposed_pairs, review_candidates: (
            not proposed_pairs and len(review_candidates) > _relationship_proposals.CONFLICT_DETECTOR_AI_MIN_CANDIDATES
        ),
        enqueue_review_work_item=lambda review_candidates: _enqueue_conflict_review_work_item(
            ctx,
            task=task,
            workspace_id=workspace_id,
            candidates=review_candidates,
            strategy_used=sampled_batch.strategy_used,
        ),
    )


async def handle_defragmenter_task(
    ctx: ApplicationContext,
    task: TaskRecord,
    provider: Any = None,
) -> dict[str, Any]:
    if ctx.repository is None:
        return {"created": 0, "archived": 0}

    all_candidates = [
        record
        for record in ctx.repository.list_memories(
            workspace_id=_resolve_workspace_id(ctx, task),
            status="active",
            limit=int(task.data.get("limit", DEFAULT_AGENT_SCAN_LIMIT)),
        )
        if record.type in {"journal", "observation"}
        and not ctx.repository.has_incoming_link(record.id, "SUPERSEDES")
    ]
    sampled_batch = _sample_maintenance_candidates(
        ctx,
        task,
        all_candidates,
        allowed_strategies=DEFRAGMENTER_ALLOWED_STRATEGIES,
        strategy_weights=DEFRAGMENTER_STRATEGY_WEIGHTS,
        limit=min(len(all_candidates), DEFAULT_AGENT_SCAN_LIMIT),
    )
    candidates = sampled_batch.records
    groups = _defragmenter_support.collect_defragment_groups(candidates)
    if not groups:
        return sampling_payload(sampled_batch, sampled_records=candidates, created=0, archived=0)

    created = 0
    archived = 0
    lines_compressed = 0
    for group in groups:
        source_lines = sum(_defragmenter_support.count_text_lines(item.content) for item in group)
        title, content = await _defragmenter_support.build_defragmented_memory(
            group,
            provider if _defragmenter_support.should_use_provider_for_defragment_group(group, source_lines) else None,
        )
        reflection_lines = _defragmenter_support.count_text_lines(content)
        record = ctx.repository.create_memory(
            title=title,
            content=content,
            workspace_ids=_defragmenter_support.resolve_group_workspace_ids(group, _resolve_workspace_id(ctx, task)),
            tags=_normalize_tag_values([tag for item in group for tag in item.tags] + ["auto-defragmented"]),
            memory_type="reflection",
            metadata={"source_memory_ids": [item.id for item in group], "defragmenter_task_id": task.id},
        )
        assert record is not None
        created += 1
        lines_compressed += max(source_lines - reflection_lines, 0)
        for item in group:
            ctx.repository.add_link(record.id, item.id, "SUPERSEDES", "Auto-defragmented into a reflection.")
            updated = ctx.repository.update_memory(item.id, status="archived")
            if updated is not None:
                archived += 1

    return sampling_payload(
        sampled_batch,
        sampled_records=candidates,
        created=created,
        archived=archived,
        lines_compressed=lines_compressed,
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
                _complete_work_item(ctx, review_item.id)
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
                try:
                    agentic_result = await cast(Callable[[str], Awaitable[Any]], run_agent)(
                        _build_deduplicator_agent_prompt(
                            task,
                            seed_records,
                            strategy_used=_deduplicator_support.review_strategy(review_item.payload),
                        )
                    )
                except Exception:
                    _release_work_item(ctx, review_item.id)
                    raise
                _complete_work_item(ctx, review_item.id)
                normalized = _normalize_deduplicator_agentic_result(agentic_result, seed_records)
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
    seed_batch = _select_deduplicator_seed_batch(
        ctx,
        candidates,
        task_id=task.id,
        strategy=_requested_sampling_strategy(task),
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
        agentic_result = await cast(Callable[[str], Awaitable[Any]], run_agent)(
            _build_deduplicator_agent_prompt(task, seed_records, strategy_used=seed_batch.strategy_used)
        )
        return sampling_payload(
            seed_batch,
            sampled_records=seed_records,
            extra=_normalize_deduplicator_agentic_result(agentic_result, seed_records),
        )

    deterministic_result = await _deduplicator_merge.run_deterministic_deduplicator_pass(
        ctx,
        task,
        candidates,
        seed_records,
        provider,
    )

    return sampling_payload(
        seed_batch,
        sampled_records=seed_records,
        seed_records=seed_records,
        merged=deterministic_result["merged"],
        archived=deterministic_result["archived"],
        absorbed_observations=deterministic_result["absorbed_observations"],
    )


async def handle_dedup_prep_task(
    ctx: ApplicationContext,
    task: TaskRecord,
) -> dict[str, Any]:
    if ctx.repository is None:
        return {"seeded_work_item_count": 0}

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
    seed_batch = _select_deduplicator_seed_batch(
        ctx,
        candidates,
        task_id=task.id,
        strategy=_requested_sampling_strategy(task),
    )
    seed_records = seed_batch.records
    facts = [record for record in seed_records if record.type == "fact"]
    if not facts:
        return sampling_payload(
            seed_batch,
            sampled_records=seed_records,
            seed_records=seed_records,
            seeded_work_item_count=0,
        )

    created_work_item, created = _enqueue_dedup_review_work_item(
        ctx,
        task=task,
        workspace_id=workspace_id,
        seed_records=seed_records,
        strategy_used=seed_batch.strategy_used,
        candidate_count=seed_batch.candidate_count,
    )
    return sampling_payload(
        seed_batch,
        sampled_records=seed_records,
        seed_records=seed_records,
        extra=_work_item_result_metadata(
            family_key=WORK_FAMILY_MEMORY_DEDUP_REVIEW,
            execution_lane=EXECUTION_LANE_AGENTIC,
            seed_source="frontier_seed",
            seed_records=seed_records,
            created_work_item=created_work_item if created else None,
        ),
        seeded_work_item_count=1 if created else 0,
    )


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
    sampled_batch = _sample_maintenance_candidates(
        ctx,
        task,
        all_candidates,
        allowed_strategies=TAXONOMIST_ALLOWED_STRATEGIES,
        strategy_weights=TAXONOMIST_STRATEGY_WEIGHTS,
        limit=min(len(all_candidates), DEFAULT_AGENT_SCAN_LIMIT),
    )
    candidates = sampled_batch.records
    updated = 0
    provider_call_budget = _taxonomist_provider_call_budget(ctx, provider)
    provider_calls_used = 0
    provider_deferred_reason_code: str | None = None
    provider_deferred_retry_delay_seconds: float | None = None
    untagged_candidates = [record for record in candidates if not _normalize_tag_values(record.tags)]

    seeded_work_items = _seed_taxonomist_work_items(
        ctx,
        task=task,
        candidates=untagged_candidates,
    )
    if _taxonomist_supports_agentic_execution(provider) and seeded_work_items and provider_call_budget > 0:
        agentic_result = await _run_taxonomist_agentic_pass(
            ctx,
            task=task,
            provider=provider,
            workspace_id=workspace_id,
            candidate_memory_ids={record.id for record in untagged_candidates},
            seeded_work_items=seeded_work_items,
            provider_call_budget=provider_call_budget,
        )
        updated += agentic_result["updated"]
        provider_calls_used = agentic_result["provider_calls_used"]
        provider_deferred_reason_code = agentic_result["provider_deferred_reason_code"]
        provider_deferred_retry_delay_seconds = agentic_result["provider_deferred_retry_delay_seconds"]
        return sampling_payload(
            sampled_batch,
            sampled_records=candidates,
            updated=updated,
            provider_calls_used=provider_calls_used,
            provider_call_budget=provider_call_budget,
            provider_deferred_reason_code=provider_deferred_reason_code,
            provider_deferred_retry_delay_seconds=provider_deferred_retry_delay_seconds,
            claimed_work_item_count=agentic_result["claimed_work_item_count"],
            execution_mode="agentic_mcp",
            summary=agentic_result["summary"],
        )

    json_result = await _run_taxonomist_json_pass(
        ctx,
        task=task,
        provider=provider,
        candidates=untagged_candidates,
        provider_call_budget=provider_call_budget,
    )
    updated += json_result["updated"]
    provider_calls_used = json_result["provider_calls_used"]
    provider_deferred_reason_code = json_result["provider_deferred_reason_code"]
    provider_deferred_retry_delay_seconds = json_result["provider_deferred_retry_delay_seconds"]
    return sampling_payload(
        sampled_batch,
        sampled_records=candidates,
        updated=updated,
        provider_calls_used=provider_calls_used,
        provider_call_budget=provider_call_budget,
        provider_deferred_reason_code=provider_deferred_reason_code,
        provider_deferred_retry_delay_seconds=provider_deferred_retry_delay_seconds,
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
    sampled_batch = _sample_maintenance_candidates(
        ctx,
        task,
        all_candidates,
        allowed_strategies=TAXONOMIST_ALLOWED_STRATEGIES,
        strategy_weights=TAXONOMIST_STRATEGY_WEIGHTS,
        limit=min(len(all_candidates), DEFAULT_AGENT_SCAN_LIMIT),
    )
    candidates = sampled_batch.records
    normalization_candidates = [record for record in candidates if _needs_tag_normalization(record.tags)]
    seeded_work_items = _seed_tag_normalizer_work_items(
        ctx,
        task=task,
        candidates=normalization_candidates,
    )
    claimed_work_items = _claim_tag_normalizer_work_batch(
        ctx,
        task=task,
        limit=min(len(normalization_candidates), DEFAULT_AGENT_SCAN_LIMIT),
    )

    updated = 0
    seeded_enrichment_count = 0
    finalized_work_item_ids: set[str] = set()
    for work_item in claimed_work_items:
        memory_id = _taxonomist_work_memory_id(work_item.payload)
        if memory_id is None:
            _complete_work_item(ctx, work_item.id)
            finalized_work_item_ids.add(work_item.id)
            continue

        record = ctx.repository.get_memory(memory_id)
        if record is None:
            _complete_work_item(ctx, work_item.id)
            finalized_work_item_ids.add(work_item.id)
            continue

        normalized_tags = _normalize_tag_values(record.tags)
        if normalized_tags != record.tags:
            refreshed = ctx.repository.update_memory(record.id, tags=normalized_tags)
            if refreshed is not None:
                updated += 1
                record = refreshed

        if not record.tags:
            _, created = _enqueue_taxonomist_enrichment_work_item(
                ctx,
                task=task,
                memory_id=record.id,
                workspace_id=workspace_id,
            )
            if created:
                seeded_enrichment_count += 1

        _complete_work_item(ctx, work_item.id)
        finalized_work_item_ids.add(work_item.id)

    for record in candidates:
        if record.tags:
            continue
        _, created = _enqueue_taxonomist_enrichment_work_item(
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
        _release_work_item(ctx, work_item.id)

    return sampling_payload(
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
    workspace_id: str,
    candidate_memory_ids: set[str],
    seeded_work_items: dict[str, Any],
    provider_call_budget: int,
) -> dict[str, Any]:
    run_agent = getattr(provider, "run_agent", None)
    assert callable(run_agent)
    try:
        agentic_result = await cast(Callable[[str], Awaitable[Any]], run_agent)(
            _build_taxonomist_agent_prompt(
                task,
                workspace_id=workspace_id,
                provider_call_budget=provider_call_budget,
            )
        )
    except Exception as exc:
        running_items = _list_taxonomist_running_work_items(ctx, task_id=task.id, workspace_id=workspace_id)
        if _is_taxonomist_provider_deferred_error(exc):
            reason = classify_provider_failure(exc)
            _record_taxonomist_provider_deferred_event(
                ctx,
                task=task,
                provider=provider,
                reason_code=reason.reason_code,
                reason_category=reason.reason_category,
                retry_delay_seconds=reason.retry_delay_seconds,
            )
            for work_item in running_items:
                _defer_work_item(
                    ctx,
                    work_item.id,
                    error=reason.reason_code,
                    retry_delay_seconds=reason.retry_delay_seconds,
                )
            return {
                "updated": _count_taxonomist_updated_records(ctx, candidate_memory_ids),
                "provider_calls_used": 0,
                "provider_deferred_reason_code": reason.reason_code,
                "provider_deferred_retry_delay_seconds": reason.retry_delay_seconds,
                "claimed_work_item_count": _count_claimed_taxonomist_work_items(ctx, seeded_work_items),
                "summary": None,
            }
        for work_item in running_items:
            _release_work_item(ctx, work_item.id)
        raise

    for work_item in _list_taxonomist_running_work_items(ctx, task_id=task.id, workspace_id=workspace_id):
        _release_work_item(ctx, work_item.id)
    normalized = _normalize_taxonomist_agentic_result(agentic_result)
    return {
        "updated": _count_taxonomist_updated_records(ctx, candidate_memory_ids),
        "provider_calls_used": 1,
        "provider_deferred_reason_code": None,
        "provider_deferred_retry_delay_seconds": None,
        "claimed_work_item_count": _count_claimed_taxonomist_work_items(ctx, seeded_work_items),
        "summary": normalized["summary"],
    }


async def _run_taxonomist_json_pass(
    ctx: ApplicationContext,
    *,
    task: TaskRecord,
    provider: Any,
    candidates: list[Any],
    provider_call_budget: int,
) -> dict[str, Any]:
    claimed_work_items = _claim_taxonomist_work_batch(
        ctx,
        task=task,
        candidates=candidates,
        limit=provider_call_budget,
    )
    claimed_by_memory_id = {
        _taxonomist_work_memory_id(record.payload): record
        for record in claimed_work_items
        if _taxonomist_work_memory_id(record.payload) is not None
    }
    finalized_work_item_ids: set[str] = set()
    updated = 0
    provider_calls_used = 0
    provider_deferred_reason_code: str | None = None
    provider_deferred_retry_delay_seconds: float | None = None
    for record in candidates:
        normalized_tags = _normalize_tag_values(record.tags)
        work_item = claimed_by_memory_id.get(record.id)
        if provider is not None and work_item is not None and provider_calls_used < provider_call_budget:
            try:
                normalized_tags = await _provider_normalize_tags(provider, record, normalized_tags)
                provider_calls_used += 1
            except Exception as exc:
                if _is_taxonomist_provider_deferred_error(exc):
                    reason = classify_provider_failure(exc)
                    provider_deferred_reason_code = reason.reason_code
                    provider_deferred_retry_delay_seconds = reason.retry_delay_seconds
                    _record_taxonomist_provider_deferred_event(
                        ctx,
                        task=task,
                        provider=provider,
                        reason_code=reason.reason_code,
                        reason_category=reason.reason_category,
                        retry_delay_seconds=reason.retry_delay_seconds,
                    )
                    _defer_work_item(
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
                _complete_work_item(ctx, work_item.id)
                finalized_work_item_ids.add(work_item.id)
            continue
        refreshed = ctx.repository.update_memory(record.id, tags=normalized_tags)
        if refreshed is not None:
            updated += 1
        if work_item is not None:
            _complete_work_item(ctx, work_item.id)
            finalized_work_item_ids.add(work_item.id)
    for work_item in claimed_work_items:
        if work_item.id in finalized_work_item_ids:
            continue
        _release_work_item(ctx, work_item.id)
    return {
        "updated": updated,
        "provider_calls_used": provider_calls_used,
        "provider_deferred_reason_code": provider_deferred_reason_code,
        "provider_deferred_retry_delay_seconds": provider_deferred_retry_delay_seconds,
        "claimed_work_item_count": len(claimed_work_items),
    }


async def handle_memory_curator_task(
    ctx: ApplicationContext,
    task: TaskRecord,
    provider: Any = None,
) -> dict[str, Any]:
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
            _complete_work_item(ctx, claimed_review_item.id)
            claimed_review_item = None
            seed_batch = _curator_support.select_curator_seed_batch(ctx, task)
            seed_records = seed_batch.records
    else:
        seed_batch = _curator_support.select_curator_seed_batch(ctx, task)
        seed_records = seed_batch.records

    work_item_metadata = _work_item_result_metadata(
        family_key=WORK_FAMILY_MEMORY_CURATION_REVIEW,
        execution_lane=EXECUTION_LANE_AGENTIC,
        seed_source="claimed_review_work_item" if claimed_review_item is not None else "sampled_frontier",
        seed_records=seed_records,
        claimed_work_item=claimed_review_item,
    )

    if not seed_records:
        return sampling_payload(
            seed_batch,
            sampled_records=seed_records,
            extra=work_item_metadata,
            summary=None,
            tool_calls_executed=0,
            mutations=0,
            claimed_work_item_count=0,
            reason="no_seed_records",
        )

    seed_payload = [
        _curator_support.curator_seed_payload_item(record)
        for record in seed_records
    ]
    prompt = (
        f"You are the {CURATOR_TASK_NAME} maintenance agent for the global memory store.\n"
        "Improve the store by merging, refining, rewriting, retagging, relinking, archiving, or deleting archived garbage when justified.\n"
        "Work in high-impact maintenance mode: prefer several coherent high-value improvements in one run when the store clearly supports them.\n"
        f"Start by calling internal_get_next_curator_batch with task_id='{task.id}', strategy='{seed_batch.strategy_used}', and exclude_memory_ids=[] so you can confirm or widen the current frontier before mutating.\n"
        "Treat the seed memories as a starting frontier, not a hard boundary; widen only when they hint at nearby duplicates, contradictions, or oversized clusters.\n"
        "Prefer safe operations with clear lineage and archive before delete when possible.\n"
        f"{build_curator_guardrails()}\n"
        f"Treat memories above {_curator_support.CURATOR_MAX_MEMORY_CHARS} characters as oversized. Prefer splitting them into smaller focused records with links such as DEPENDS_ON or AMENDS instead of growing one blob.\n"
        f"Avoid creating or growing memories past {_curator_support.CURATOR_MAX_MEMORY_CHARS} characters unless no reasonable split exists.\n"
        "Do not create journal or memory records for routine completion, counters, or status-only traces; use task_complete for operational closeout instead.\n"
        "Before stopping, check once more for any adjacent worthwhile maintenance action; no-op is fine when another step would be low-value or unsafe.\n"
        f"When your pass is complete, call task_complete with task_id='{task.id}', task_name='{CURATOR_TASK_NAME}', and a short summary before your final JSON response.\n"
        "When finished, return JSON like {\"summary\": \"...\", \"actions_taken\": N}.\n\n"
        f"Seed memories (compact view):\n{json.dumps(seed_payload, sort_keys=True, ensure_ascii=False)}"
    )
    run_agent = getattr(provider, "run_agent", None)
    supports_agentic = getattr(provider, "supports_agentic", None)
    if callable(run_agent) and (not callable(supports_agentic) or supports_agentic()):
        try:
            agentic_result = await cast(Callable[[str], Awaitable[Any]], run_agent)(
                (
                    f"You are the {CURATOR_TASK_NAME} maintenance agent for the global memory store.\n"
                    "Use the workspace-local internal MCP maintenance tools directly to inspect and mutate memories.\n"
                    f"Start by calling internal_get_next_curator_batch with task_id='{task.id}', strategy='{seed_batch.strategy_used}', and exclude_memory_ids={json.dumps([record.id for record in seed_records], sort_keys=True)} so you can widen beyond the current frontier only when justified.\n"
                    "Treat the provided seed memories as a starting frontier and the active frontier for this run; widen only when they imply nearby duplicates, contradictions, taxonomy cleanup, or oversized clusters.\n"
                    "Aim for multiple coherent, high-value maintenance actions in one run when justified, with clear lineage and archive-before-delete when possible.\n"
                    f"{build_curator_guardrails()}\n"
                    f"Treat memories above {_curator_support.CURATOR_MAX_MEMORY_CHARS} characters as oversized and prefer splitting them into focused linked records.\n"
                    "When you materially rewrite a memory and already understand it, refresh a concise summary in the same tool call.\n"
                    "Do not create journal or memory records for routine completion, counters, or status-only traces; use task_complete for operational closeout instead.\n"
                    "Before finishing, do one more quick search/list/read pass for any adjacent high-value maintenance opportunity.\n"
                    "Do not claim work you did not actually execute through MCP tools.\n"
                    f"When your pass is complete, call task_complete with task_id='{task.id}', task_name='{CURATOR_TASK_NAME}', and a short summary before your final JSON response.\n"
                    "When finished, output final JSON only in the form {\"summary\": \"...\"}.\n\n"
                    f"Sampling strategy: {seed_batch.strategy_used}\n"
                    f"Seed memories (compact view):\n{json.dumps(seed_payload, sort_keys=True, ensure_ascii=False)}"
                )
            )
        except Exception:
            if claimed_review_item is not None:
                _release_work_item(ctx, claimed_review_item.id)
            raise
        if claimed_review_item is not None:
            _complete_work_item(ctx, claimed_review_item.id)
        return sampling_payload(
            seed_batch,
            sampled_records=seed_records,
            seed_records=seed_records,
            extra=work_item_metadata,
            summary=agentic_result.summary,
            execution_mode="agentic_mcp",
            claimed_work_item_count=1 if claimed_review_item is not None else 0,
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
                "task_complete",
                "internal_task_complete",
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
        )
    except Exception:
        if claimed_review_item is not None:
            _release_work_item(ctx, claimed_review_item.id)
        raise
    if claimed_review_item is not None:
        _complete_work_item(ctx, claimed_review_item.id)
    summary = _curator_support.normalize_curator_summary(loop_result.response, tool_calls_executed=loop_result.tool_calls_executed)
    return sampling_payload(
        seed_batch,
        sampled_records=seed_records,
        seed_records=seed_records,
        extra=work_item_metadata,
        summary=summary,
        execution_mode="json_tool_loop",
        claimed_work_item_count=1 if claimed_review_item is not None else 0,
        tool_calls_executed=loop_result.tool_calls_executed,
        mutations=loop_result.mutating_tool_calls,
        tool_names_used=loop_result.tool_names_used,
    )


async def handle_curator_frontier_task(
    ctx: ApplicationContext,
    task: TaskRecord,
) -> dict[str, Any]:
    if ctx.repository is None:
        return {"seeded_work_item_count": 0}

    workspace_id = _resolve_workspace_id(ctx, task)
    seed_batch = _curator_support.select_curator_seed_batch(ctx, task)
    seed_records = seed_batch.records
    if not seed_records:
        return sampling_payload(
            seed_batch,
            sampled_records=seed_records,
            seed_records=seed_records,
            seeded_work_item_count=0,
            reason="no_seed_records",
        )

    created_work_item, created = _enqueue_curator_review_work_item(
        ctx,
        task=task,
        workspace_id=workspace_id,
        seed_records=seed_records,
        strategy_used=seed_batch.strategy_used,
        candidate_count=seed_batch.candidate_count,
    )
    return sampling_payload(
        seed_batch,
        sampled_records=seed_records,
        seed_records=seed_records,
        extra=_work_item_result_metadata(
            family_key=WORK_FAMILY_MEMORY_CURATION_REVIEW,
            execution_lane=EXECUTION_LANE_AGENTIC,
            seed_source="frontier_seed",
            seed_records=seed_records,
            created_work_item=created_work_item if created else None,
        ),
        seeded_work_item_count=1 if created else 0,
    )


async def _provider_normalize_tags(provider: Any, record, normalized_tags: list[str]) -> list[str]:
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
    return _normalize_tag_values([str(tag) for tag in provider_tags]) or normalized_tags


def _taxonomist_provider_call_budget(ctx: ApplicationContext, provider: Any) -> int:
    if provider is None:
        return 0
    config = getattr(ctx, "config", None)
    routing = None if config is None else getattr(config, "provider_routing", None)
    configured_limit = None if routing is None else getattr(routing, "model_burst_call_limit", None)
    if isinstance(configured_limit, int) and configured_limit > 0:
        return configured_limit
    return TAXONOMIST_DEFAULT_PROVIDER_CALL_BUDGET


def _taxonomist_supports_agentic_execution(provider: Any) -> bool:
    run_agent = getattr(provider, "run_agent", None)
    supports_agentic = getattr(provider, "supports_agentic", None)
    return callable(run_agent) and (not callable(supports_agentic) or bool(supports_agentic()))


def _is_taxonomist_provider_deferred_error(exc: Exception) -> bool:
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


def _record_taxonomist_provider_deferred_event(
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


def _apply_graph_link_proposals(
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


def _apply_conflict_proposals(
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


def _claim_dedup_review_work_batch(
    ctx: ApplicationContext,
    *,
    task: TaskRecord,
    limit: int,
) -> list[Any]:
    return _claim_work_batch(
        ctx,
        task=task,
        family_key=WORK_FAMILY_MEMORY_DEDUP_REVIEW,
        execution_lane=EXECUTION_LANE_AGENTIC,
        limit=limit,
    )


def _claim_graph_link_review_work_batch(
    ctx: ApplicationContext,
    *,
    task: TaskRecord,
    limit: int,
) -> list[Any]:
    return _claim_work_batch(
        ctx,
        task=task,
        family_key=WORK_FAMILY_GRAPH_LINK_REVIEW,
        execution_lane=EXECUTION_LANE_AGENTIC,
        limit=limit,
    )


def _claim_conflict_review_work_batch(
    ctx: ApplicationContext,
    *,
    task: TaskRecord,
    limit: int,
) -> list[Any]:
    return _claim_work_batch(
        ctx,
        task=task,
        family_key=WORK_FAMILY_CONFLICT_REVIEW,
        execution_lane=EXECUTION_LANE_AGENTIC,
        limit=limit,
    )


def _enqueue_graph_link_review_work_item(
    ctx: ApplicationContext,
    *,
    task: TaskRecord,
    workspace_id: str,
    candidates: list[Any],
    strategy_used: str | None,
) -> tuple[Any, bool]:
    return _enqueue_review_work_item(
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


def _enqueue_conflict_review_work_item(
    ctx: ApplicationContext,
    *,
    task: TaskRecord,
    workspace_id: str,
    candidates: list[Any],
    strategy_used: str | None,
) -> tuple[Any, bool]:
    return _enqueue_review_work_item(
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


def _enqueue_dedup_review_work_item(
    ctx: ApplicationContext,
    *,
    task: TaskRecord,
    workspace_id: str,
    seed_records: list[Any],
    strategy_used: str | None,
    candidate_count: int,
) -> tuple[Any, bool]:
    return _enqueue_review_work_item(
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
    )


def _claim_curator_review_work_batch(
    ctx: ApplicationContext,
    *,
    task: TaskRecord,
    limit: int,
) -> list[Any]:
    return _claim_work_batch(
        ctx,
        task=task,
        family_key=WORK_FAMILY_MEMORY_CURATION_REVIEW,
        execution_lane=EXECUTION_LANE_AGENTIC,
        limit=limit,
    )


def _enqueue_curator_review_work_item(
    ctx: ApplicationContext,
    *,
    task: TaskRecord,
    workspace_id: str | None,
    seed_records: list[Any],
    strategy_used: str | None,
    candidate_count: int,
) -> tuple[Any, bool]:
    return _enqueue_review_work_item(
        ctx,
        task=task,
        family_key=WORK_FAMILY_MEMORY_CURATION_REVIEW,
        execution_lane=EXECUTION_LANE_AGENTIC,
        workspace_id=workspace_id,
        idempotency_prefix="memory_curation_review",
        payload_memory_ids_key="seed_memory_ids",
        memory_ids=[record.id for record in seed_records],
        strategy_used=strategy_used,
        candidate_count=candidate_count,
    )


def _graph_link_review_candidates(ctx: ApplicationContext, payload: dict[str, Any]) -> list[Any]:
    return _payload_memory_records(ctx, payload, payload_memory_ids_key="candidate_memory_ids")


def _conflict_review_candidates(ctx: ApplicationContext, payload: dict[str, Any]) -> list[Any]:
    return _payload_memory_records(
        ctx,
        payload,
        payload_memory_ids_key="candidate_memory_ids",
        allowed_types={"fact", "plan"},
    )


def _claim_taxonomist_work_batch(
    ctx: ApplicationContext,
    *,
    task: TaskRecord,
    candidates: list[Any],
    limit: int,
) -> list[Any]:
    work_items = getattr(ctx, "work_items", None)
    if work_items is None or limit < 1:
        return []
    _seed_taxonomist_work_items(ctx, task=task, candidates=candidates)
    return work_items.claim_batch(
        family_key=WORK_FAMILY_MEMORY_TAGGING,
        execution_lane=EXECUTION_LANE_AGENTIC,
        lease_owner=task.id,
        limit=limit,
        workspace_id=_resolve_workspace_id(ctx, task),
    )


def _claim_tag_normalizer_work_batch(
    ctx: ApplicationContext,
    *,
    task: TaskRecord,
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
        workspace_id=_resolve_workspace_id(ctx, task),
    )


def _seed_taxonomist_work_items(
    ctx: ApplicationContext,
    *,
    task: TaskRecord,
    candidates: list[Any],
) -> dict[str, Any]:
    if getattr(ctx, "work_items", None) is None:
        return {}
    workspace_id = _resolve_workspace_id(ctx, task)
    seeded: dict[str, Any] = {}
    for record in candidates:
        if record.tags:
            continue
        work_item, _ = _enqueue_taxonomist_enrichment_work_item(
            ctx,
            task=task,
            memory_id=record.id,
            workspace_id=workspace_id,
        )
        seeded[record.id] = work_item
    return seeded


def _seed_tag_normalizer_work_items(
    ctx: ApplicationContext,
    *,
    task: TaskRecord,
    candidates: list[Any],
) -> dict[str, Any]:
    work_items = getattr(ctx, "work_items", None)
    if work_items is None:
        return {}
    workspace_id = _resolve_workspace_id(ctx, task)
    seeded: dict[str, Any] = {}
    for record in candidates:
        normalized_tags = _normalize_tag_values(record.tags)
        if normalized_tags == record.tags:
            continue
        signature = _tag_normalization_signature(record.tags)
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


def _enqueue_taxonomist_enrichment_work_item(
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


def _needs_tag_normalization(tags: list[str]) -> bool:
    normalized_tags = _normalize_tag_values(tags)
    return bool(tags) and normalized_tags != tags


def _tag_normalization_signature(tags: list[str]) -> str:
    if not tags:
        return "untagged"
    return "|".join(tag.strip() for tag in tags)


def _taxonomist_work_memory_id(payload: dict[str, Any]) -> str | None:
    memory_id = payload.get("memory_id")
    if isinstance(memory_id, str) and memory_id.strip():
        return memory_id
    return None


def _build_taxonomist_agent_prompt(
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
        "When finished, output final JSON only in the form {\"summary\": \"...\", \"updated_memory_ids\": [\"...\"]}.\n"
    )


def _list_taxonomist_running_work_items(
    ctx: ApplicationContext,
    *,
    task_id: str,
    workspace_id: str,
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
        limit=max(TAXONOMIST_DEFAULT_PROVIDER_CALL_BUDGET, DEFAULT_AGENT_SCAN_LIMIT),
    )


def _count_claimed_taxonomist_work_items(ctx: ApplicationContext, seeded_work_items: dict[str, Any]) -> int:
    work_items = getattr(ctx, "work_items", None)
    if work_items is None:
        return 0
    claimed = 0
    for work_item in seeded_work_items.values():
        current = work_items.get_item(work_item.id)
        if current.attempt_count > 0:
            claimed += 1
    return claimed


def _count_taxonomist_updated_records(ctx: ApplicationContext, memory_ids: set[str]) -> int:
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


def _normalize_taxonomist_agentic_result(agentic_result: Any) -> dict[str, Any]:
    summary = _coerce_text_summary(getattr(agentic_result, "summary", None))
    parsed = getattr(agentic_result, "parsed", None)
    if isinstance(parsed, dict):
        response = parsed.get("response")
        if isinstance(response, str):
            parsed_response = _extract_embedded_json_object(response)
            if isinstance(parsed_response, dict):
                summary = _coerce_text_summary(parsed_response.get("summary")) or summary
    return {"summary": summary}


# Temporary compatibility aliases for the extraction slices.
CURATOR_MAX_MEMORY_CHARS = _curator_support.CURATOR_MAX_MEMORY_CHARS
CURATOR_MAX_SEED_RECORDS = _curator_support.CURATOR_MAX_SEED_RECORDS
DEFRAGMENTER_ALLOWED_STRATEGIES = _defragmenter_support.DEFRAGMENTER_ALLOWED_STRATEGIES
DEFRAGMENTER_STRATEGY_WEIGHTS = _defragmenter_support.DEFRAGMENTER_STRATEGY_WEIGHTS
_select_curator_seed_records = _curator_support.select_curator_seed_records
_build_support_counts = _curator_support.build_support_counts
_requested_sampling_strategy = requested_sampling_strategy
_sample_maintenance_candidates = sample_maintenance_candidates
_select_deduplicator_seed_batch = _deduplicator_support.select_deduplicator_seed_batch
_build_deduplicator_agent_prompt = _deduplicator_support.build_deduplicator_agent_prompt
_normalize_deduplicator_agentic_result = _deduplicator_support.normalize_deduplicator_agentic_result
_purge_recoverable_journal_entries = _maintenance_housekeeping._purge_recoverable_journal_entries
_cleanup_deleted_thought_embeddings = _maintenance_housekeeping._cleanup_deleted_thought_embeddings
_resolve_workspace_id = _maintenance_housekeeping._resolve_workspace_id
_resolve_workspace_root = _maintenance_housekeeping._resolve_workspace_root

def _coerce_non_negative_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(value, 0)
    return 0


def _coerce_text_summary(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _extract_embedded_json_object(text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        json_start = text.find("{")
        json_end = text.rfind("}") + 1
        if json_start < 0 or json_end <= json_start:
            return None
        try:
            parsed = json.loads(text[json_start:json_end])
        except json.JSONDecodeError:
            return None
    if isinstance(parsed, dict):
        return parsed
    return None


def _extract_agentic_tool_names(value: object) -> list[str]:
    if not isinstance(value, dict):
        return []
    return sorted(str(name) for name, payload in value.items() if isinstance(name, str) and isinstance(payload, dict))


def _count_mutating_agentic_tool_calls(value: object) -> int:
    if not isinstance(value, dict):
        return 0
    read_only_tool_names = {
        "mcp_mcp-memory-internal_task_complete",
        "mcp_mcp-memory-internal_internal_get_next_curator_batch",
        "mcp_mcp-memory-internal_internal_get_next_dedup_batch",
        "mcp_mcp-memory-internal_internal_read_memory_record",
        "mcp_mcp-memory-internal_internal_search_memory_records",
        "mcp_mcp-memory-internal_internal_list_memory_records",
        "mcp_mcp-memory-internal_internal_task_complete",
    }
    total = 0
    for name, payload in value.items():
        if not isinstance(name, str) or name in read_only_tool_names or not isinstance(payload, dict):
            continue
        total += _coerce_non_negative_int(payload.get("count"))
    return total


def _normalize_tag_values(tags: list[str]) -> list[str]:
    aliases = {
        "unit-test": "testing",
        "unit_tests": "testing",
        "tests": "testing",
        "test": "testing",
        "authn": "auth",
        "authz": "auth",
    }
    normalized: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        normalized_tag = tag.strip().lower().replace("_", "-")
        normalized_tag = aliases.get(normalized_tag, normalized_tag)
        if not normalized_tag or normalized_tag in seen:
            continue
        seen.add(normalized_tag)
        normalized.append(normalized_tag)
    return sorted(normalized)


def _has_link(ctx: ApplicationContext, source_id: str, target_id: str, link_type: str) -> bool:
    assert ctx.repository is not None
    return any(
        link.target_id == target_id
        for link in ctx.repository.get_links(source_id, direction="outgoing", link_type=link_type)
    )
