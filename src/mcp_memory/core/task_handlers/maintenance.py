from __future__ import annotations

from typing import Any

from mcp_memory.context import ApplicationContext
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
import mcp_memory.core.task_handlers.curator_handlers as _curator_handlers
import mcp_memory.core.task_handlers.curator_support as _curator_support
import mcp_memory.core.task_handlers.deduplicator_handlers as _deduplicator_handlers
import mcp_memory.core.task_handlers.deduplicator_support as _deduplicator_support
import mcp_memory.core.task_handlers.defragmenter_support as _defragmenter_support
import mcp_memory.core.task_handlers.maintenance_housekeeping as _maintenance_housekeeping
from mcp_memory.core.task_handlers.maintenance_normalization import normalize_tag_values
import mcp_memory.core.task_handlers.relationship_review_handlers as _relationship_review_handlers
import mcp_memory.core.task_handlers.taxonomist_handlers as _taxonomist_handlers
from mcp_memory.core.task_handlers.constants import DEFAULT_AGENT_SCAN_LIMIT
from mcp_memory.core.tasks import TaskRecord


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
handle_project_manager_task = _maintenance_housekeeping.handle_project_manager_task
handle_fact_checker_task = _maintenance_housekeeping.handle_fact_checker_task
handle_sweeper_task = _maintenance_housekeeping.handle_sweeper_task
handle_graph_linker_task = _relationship_review_handlers.handle_graph_linker_task
handle_graph_link_discovery_task = _relationship_review_handlers.handle_graph_link_discovery_task
handle_conflict_detector_task = _relationship_review_handlers.handle_conflict_detector_task
handle_conflict_screening_task = _relationship_review_handlers.handle_conflict_screening_task
handle_memory_curator_task = _curator_handlers.handle_memory_curator_task
handle_deduplicator_task = _deduplicator_handlers.handle_deduplicator_task
handle_dedup_prep_task = _deduplicator_handlers.handle_dedup_prep_task
handle_taxonomist_task = _taxonomist_handlers.handle_taxonomist_task
handle_tag_normalizer_task = _taxonomist_handlers.handle_tag_normalizer_task


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
        support_counts=_build_support_counts(ctx, all_candidates),
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
            tags=normalize_tag_values([tag for item in group for tag in item.tags] + ["auto-defragmented"]),
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
_normalize_tag_values = normalize_tag_values
