from __future__ import annotations

import json
from typing import Any, Awaitable, Callable, cast

from mcp_memory.context import ApplicationContext
from mcp_memory.core.provider_admission import classify_provider_failure
from mcp_memory.core.sampling import (
    ANOMALY_STRATEGY,
    COOLDOWN_ESCAPE_STRATEGY,
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
import mcp_memory.core.task_handlers.taxonomist_support as _taxonomist_support
import mcp_memory.core.task_handlers.relationship_proposals as _relationship_proposals
import mcp_memory.core.task_handlers.relationship_review_support as _relationship_review_support
from mcp_memory.core.task_handlers.constants import DEFAULT_AGENT_SCAN_LIMIT
from mcp_memory.core.task_handlers.agentic_guardrails import (
    build_curator_guardrails,
)
from mcp_memory.core.task_handlers.campaigns import campaign_metadata
from mcp_memory.core.task_handlers.campaigns import count_named_tool_calls
from mcp_memory.core.task_handlers.tool_loop import run_internal_tool_loop
from mcp_memory.core.tasks import TaskRecord
from mcp_memory.work_item_store import (
    COMPATIBILITY_GROUP_STRUCTURAL_REVIEW,
    WORK_FAMILY_CONFLICT_REVIEW,
    EXECUTION_LANE_AGENTIC,
    WORK_FAMILY_GRAPH_LINK_REVIEW,
    WORK_FAMILY_MEMORY_CURATION_REVIEW,
    WORK_FAMILY_MEMORY_DEDUP_REVIEW,
)


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
            load_candidates=_relationship_review_support.graph_link_review_candidates,
            propose_pairs=lambda review_candidates: _relationship_proposals.propose_graph_links(
                ctx,
                review_candidates,
                provider,
            ),
            apply_pairs=lambda proposed_pairs: _relationship_review_support.apply_graph_link_proposals(ctx, proposed_pairs),
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
        allowed_strategies=_relationship_review_support.GRAPH_LINKER_ALLOWED_STRATEGIES,
        strategy_weights=_relationship_review_support.GRAPH_LINKER_STRATEGY_WEIGHTS,
        limit=min(len(all_candidates), DEFAULT_AGENT_SCAN_LIMIT),
    )
    candidates = sampled_batch.records
    if len(candidates) < 2:
        return sampling_payload(sampled_batch, sampled_records=candidates, created=0)

    proposed_pairs = await _relationship_proposals.propose_graph_links(ctx, candidates, provider)
    created = _relationship_review_support.apply_graph_link_proposals(ctx, proposed_pairs)

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
        allowed_strategies=_relationship_review_support.GRAPH_LINKER_ALLOWED_STRATEGIES,
        strategy_weights=_relationship_review_support.GRAPH_LINKER_STRATEGY_WEIGHTS,
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
        apply_pairs=lambda proposed_pairs: _relationship_review_support.apply_graph_link_proposals(ctx, proposed_pairs),
        should_seed=lambda proposed_pairs, review_candidates: (
            len(proposed_pairs) < _relationship_proposals.GRAPH_LINKER_FALLBACK_LINK_TARGET
            and len(review_candidates) > _relationship_proposals.GRAPH_LINKER_AI_MIN_CANDIDATES
        ),
        enqueue_review_work_item=lambda review_candidates: _relationship_review_support.enqueue_graph_link_review_work_item(
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
            load_candidates=_relationship_review_support.conflict_review_candidates,
            propose_pairs=lambda review_candidates: _relationship_proposals.propose_conflicts(
                ctx,
                review_candidates,
                provider,
            ),
            apply_pairs=lambda proposed_pairs: _relationship_review_support.apply_conflict_proposals(ctx, proposed_pairs),
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
        allowed_strategies=_relationship_review_support.CONFLICT_DETECTOR_ALLOWED_STRATEGIES,
        strategy_weights=_relationship_review_support.CONFLICT_DETECTOR_STRATEGY_WEIGHTS,
        limit=min(len(all_candidates), DEFAULT_AGENT_SCAN_LIMIT),
    )
    candidates = sampled_batch.records
    if len(candidates) < 2:
        return sampling_payload(sampled_batch, sampled_records=candidates, created=0)

    proposed_pairs = await _relationship_proposals.propose_conflicts(ctx, candidates, provider)
    created = _relationship_review_support.apply_conflict_proposals(ctx, proposed_pairs)

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
        allowed_strategies=_relationship_review_support.CONFLICT_DETECTOR_ALLOWED_STRATEGIES,
        strategy_weights=_relationship_review_support.CONFLICT_DETECTOR_STRATEGY_WEIGHTS,
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
        apply_pairs=lambda proposed_pairs: _relationship_review_support.apply_conflict_proposals(ctx, proposed_pairs),
        should_seed=lambda proposed_pairs, review_candidates: (
            not proposed_pairs and len(review_candidates) > _relationship_proposals.CONFLICT_DETECTOR_AI_MIN_CANDIDATES
        ),
        enqueue_review_work_item=lambda review_candidates: _relationship_review_support.enqueue_conflict_review_work_item(
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
        candidates=candidates,
        strategy_used=seed_batch.strategy_used,
        candidate_count=seed_batch.candidate_count,
    )
    return sampling_payload(
        seed_batch,
        sampled_records=seed_records,
        seed_records=seed_records + _deduplicator_support.select_deduplicator_support_records(seed_records, candidates),
        extra=_work_item_result_metadata(
            family_key=WORK_FAMILY_MEMORY_DEDUP_REVIEW,
            execution_lane=EXECUTION_LANE_AGENTIC,
            seed_source="frontier_seed",
            seed_records=seed_records + _deduplicator_support.select_deduplicator_support_records(seed_records, candidates),
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
        allowed_strategies=_taxonomist_support.TAXONOMIST_ALLOWED_STRATEGIES,
        strategy_weights=_taxonomist_support.TAXONOMIST_STRATEGY_WEIGHTS,
        limit=min(len(all_candidates), DEFAULT_AGENT_SCAN_LIMIT),
    )
    candidates = sampled_batch.records
    provider_call_budget = _taxonomist_support.provider_call_budget(ctx, provider)
    untagged_candidates = [record for record in candidates if not _normalize_tag_values(record.tags)]

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
    sampled_batch = _sample_maintenance_candidates(
        ctx,
        task,
        all_candidates,
        allowed_strategies=_taxonomist_support.TAXONOMIST_ALLOWED_STRATEGIES,
        strategy_weights=_taxonomist_support.TAXONOMIST_STRATEGY_WEIGHTS,
        limit=min(len(all_candidates), DEFAULT_AGENT_SCAN_LIMIT),
    )
    candidates = sampled_batch.records
    normalization_candidates = [
        record for record in candidates if _taxonomist_support.needs_tag_normalization(record.tags, _normalize_tag_values)
    ]
    seeded_work_items = _taxonomist_support.seed_tag_normalizer_work_items(
        ctx,
        task=task,
        workspace_id=workspace_id,
        candidates=normalization_candidates,
        normalize_tag_values=_normalize_tag_values,
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
            _, created = _taxonomist_support.enqueue_taxonomist_enrichment_work_item(
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
        _release_work_item(ctx, work_item.id)

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
                _defer_work_item(
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
                "tool_calls_executed": 0,
                "mutations": 0,
                "compatible_batch_calls": 0,
                "compatibility_group": "lightweight_review",
                "work_item_batch_limit": work_item_batch_limit,
                "max_batches_per_run": max_batches_per_run,
                "summary": None,
            }
        for work_item in running_items:
            _release_work_item(ctx, work_item.id)
        raise

    for work_item in _taxonomist_support.list_running_work_items(
        ctx,
        task_id=task.id,
        workspace_id=workspace_id,
        limit=max(_taxonomist_support.TAXONOMIST_DEFAULT_PROVIDER_CALL_BUDGET, DEFAULT_AGENT_SCAN_LIMIT),
    ):
        _release_work_item(ctx, work_item.id)
    normalized = _taxonomist_support.normalize_agentic_result(
        agentic_result,
        coerce_text_summary=_coerce_text_summary,
        extract_embedded_json_object=_extract_embedded_json_object,
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
        normalized_tags = _normalize_tag_values(record.tags)
        work_item = claimed_by_memory_id.get(record.id)
        if provider is not None and work_item is not None and provider_calls_used < provider_call_budget:
            try:
                normalized_tags = await _taxonomist_support.provider_normalize_tags(
                    provider,
                    record,
                    normalized_tags,
                    _normalize_tag_values,
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
    work_item_metadata.update(
        campaign_metadata(
            compatibility_group=COMPATIBILITY_GROUP_STRUCTURAL_REVIEW,
            origin_family=WORK_FAMILY_MEMORY_CURATION_REVIEW,
            execution_lane=EXECUTION_LANE_AGENTIC,
        )
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

    curator_guardrails = build_curator_guardrails()
    prompt = _curator_support.build_json_tool_loop_prompt(
        task,
        strategy_used=seed_batch.strategy_used,
        seed_records=seed_records,
        guardrails=curator_guardrails,
    )
    run_agent = getattr(provider, "run_agent", None)
    supports_agentic = getattr(provider, "supports_agentic", None)
    if callable(run_agent) and (not callable(supports_agentic) or supports_agentic()):
        try:
            agentic_result = await cast(Callable[[str], Awaitable[Any]], run_agent)(
                _curator_support.build_agentic_prompt(
                    task,
                    strategy_used=seed_batch.strategy_used,
                    seed_records=seed_records,
                    guardrails=curator_guardrails,
                )
            )
        except Exception:
            if claimed_review_item is not None:
                _release_work_item(ctx, claimed_review_item.id)
            raise
        if claimed_review_item is not None:
            _complete_work_item(ctx, claimed_review_item.id)
        normalized_agentic = _curator_support.normalize_curator_agentic_result(agentic_result)
        return sampling_payload(
            seed_batch,
            sampled_records=seed_records,
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
        compatible_batch_calls=count_named_tool_calls(
            loop_result.tool_names_used,
            tool_name="internal_get_compatible_work_batch",
        ),
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
    support_records = _curator_support.select_curator_support_records(ctx, task, seed_records)
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
        support_records=support_records,
        strategy_used=seed_batch.strategy_used,
        candidate_count=seed_batch.candidate_count,
    )
    return sampling_payload(
        seed_batch,
        sampled_records=seed_records,
        seed_records=seed_records + support_records,
        extra=_work_item_result_metadata(
            family_key=WORK_FAMILY_MEMORY_CURATION_REVIEW,
            execution_lane=EXECUTION_LANE_AGENTIC,
            seed_source="frontier_seed",
            seed_records=seed_records + support_records,
            created_work_item=created_work_item if created else None,
        ),
        seeded_work_item_count=1 if created else 0,
    )


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
        extra_payload={
            "support_memory_ids": [record.id for record in support_records],
            "packet_record_count": len(seed_records) + len(support_records),
        },
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
    support_records: list[Any],
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
        extra_payload={
            "support_memory_ids": [record.id for record in support_records],
            "packet_record_count": len(seed_records) + len(support_records),
        },
    )


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


