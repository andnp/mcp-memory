from __future__ import annotations

from inspect import isawaitable
import json
from typing import Any, Awaitable, Callable, cast
import re

from mcp_memory.context import ApplicationContext
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
    created = 0
    for source_id, target_id, link_type, context in proposed_pairs:
        if source_id == target_id or _has_link(ctx, source_id, target_id, link_type):
            continue
        ctx.repository.add_link(source_id, target_id, link_type, context)
        created += 1

    return sampling_payload(sampled_batch, sampled_records=candidates, created=created)


async def handle_conflict_detector_task(
    ctx: ApplicationContext,
    task: TaskRecord,
    provider: Any = None,
) -> dict[str, Any]:
    if ctx.repository is None:
        return {"created": 0}

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

    return sampling_payload(sampled_batch, sampled_records=candidates, created=created)


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


async def handle_taxonomist_task(
    ctx: ApplicationContext,
    task: TaskRecord,
    provider: Any = None,
) -> dict[str, Any]:
    if ctx.repository is None:
        return {"updated": 0}

    all_candidates = ctx.repository.list_memories(
        workspace_id=_resolve_workspace_id(ctx, task),
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
    for record in candidates:
        normalized_tags = _normalize_tag_values(record.tags)
        if normalized_tags != record.tags:
            refreshed = ctx.repository.update_memory(record.id, tags=normalized_tags)
            if refreshed is not None:
                updated += 1
            continue
        if provider is not None and not record.tags:
            normalized_tags = await _provider_normalize_tags(provider, record, normalized_tags)
        if normalized_tags == record.tags:
            continue
        refreshed = ctx.repository.update_memory(record.id, tags=normalized_tags)
        if refreshed is not None:
            updated += 1
    return sampling_payload(sampled_batch, sampled_records=candidates, updated=updated)


async def handle_memory_curator_task(
    ctx: ApplicationContext,
    task: TaskRecord,
    provider: Any = None,
) -> dict[str, Any]:
    if ctx.repository is None:
        return {"summary": None, "tool_calls_executed": 0, "mutations": 0}
    if provider is None:
        return {"summary": None, "tool_calls_executed": 0, "mutations": 0, "reason": "provider_not_configured"}

    seed_batch = _curator_support.select_curator_seed_batch(ctx, task)
    seed_records = seed_batch.records
    if not seed_records:
        return sampling_payload(
            seed_batch,
            sampled_records=seed_records,
            summary=None,
            tool_calls_executed=0,
            mutations=0,
            reason="no_seed_records",
        )

    seed_payload = [
        _curator_support.curator_seed_payload_item(record)
        for record in seed_records
    ]
    prompt = (
        f"You are the {CURATOR_TASK_NAME} maintenance agent for the global memory store.\n"
        "Your goal is to improve the memory store by merging, refining, rewriting, retagging, relinking, archiving, or deleting archived garbage when justified.\n"
        "Work in high-impact maintenance mode: prefer several coherent improvements in one run when the store clearly supports them, not just the first safe fix.\n"
        f"Start by calling internal_get_next_curator_batch with task_id='{task.id}', strategy='{seed_batch.strategy_used}', and exclude_memory_ids=[] so you can confirm or widen the current frontier before mutating.\n"
        "Treat the seed memories as a starting frontier, not a hard boundary; use search/list/read tools to widen the working set when the seeds hint at nearby duplicates, contradictions, or oversized clusters.\n"
        "Prefer safe operations with clear lineage. Archive before delete whenever possible.\n"
        f"{build_curator_guardrails()}\n"
        f"Treat memories above {_curator_support.CURATOR_MAX_MEMORY_CHARS} characters as oversized. Prefer splitting oversized memories into smaller focused records with links such as DEPENDS_ON or AMENDS instead of continuing to append or merge them into one blob.\n"
        f"Avoid creating or growing memories past {_curator_support.CURATOR_MAX_MEMORY_CHARS} characters unless no reasonable split exists.\n"
        "Before stopping, check whether at least one additional worthwhile maintenance action is still visible through search/list/read; no-op is fine only when another step would be low-value or unsafe.\n"
        "Use the internal maintenance tools to inspect and mutate the store.\n"
        f"When your pass is complete, call internal_task_complete with task_id='{task.id}', task_name='{CURATOR_TASK_NAME}', and a short summary before your final JSON response.\n"
        "When finished, return JSON like {\"summary\": \"...\", \"actions_taken\": N}.\n\n"
        f"Seed memories (compact view):\n{json.dumps(seed_payload, sort_keys=True, ensure_ascii=False)}"
    )
    run_agent = getattr(provider, "run_agent", None)
    supports_agentic = getattr(provider, "supports_agentic", None)
    if callable(run_agent) and (not callable(supports_agentic) or supports_agentic()):
        agentic_result = await cast(Callable[[str], Awaitable[Any]], run_agent)(
            (
                f"You are the {CURATOR_TASK_NAME} maintenance agent for the global memory store.\n"
                "Use the workspace-local internal MCP maintenance tools directly to inspect and mutate memories.\n"
                "Search, read, list, split, merge, archive, create, update, delete, and link records as needed.\n"
                f"Start by calling internal_get_next_curator_batch with task_id='{task.id}', strategy='{seed_batch.strategy_used}', and exclude_memory_ids=[] so you can confirm or widen the active frontier before mutating.\n"
                "Treat the provided seed memories as a starting frontier; widen your search beyond them when they imply adjacent duplicates, contradictions, taxonomy cleanup, or oversized clusters.\n"
                "Aim for multiple coherent, high-value maintenance actions in one run when justified instead of stopping after the first easy mutation.\n"
                "Prefer safe operations with clear lineage. Archive before delete whenever possible.\n"
                f"{build_curator_guardrails()}\n"
                f"Treat memories above {_curator_support.CURATOR_MAX_MEMORY_CHARS} characters as oversized and prefer splitting them into focused linked records.\n"
                "When you create, merge, or materially rewrite a memory and you already understand it, include or refresh a concise summary in the same tool call instead of relying on a later standalone summarizer.\n"
                "Before finishing, do one more quick search/list/read pass to confirm there is not an adjacent high-value maintenance opportunity still sitting nearby.\n"
                "Do not claim work you did not actually execute through MCP tools.\n"
                f"When your pass is complete, call internal_task_complete with task_id='{task.id}', task_name='{CURATOR_TASK_NAME}', and a short summary before your final JSON response.\n"
                "When finished, output final JSON only in the form {\"summary\": \"...\"}.\n\n"
                f"Sampling strategy: {seed_batch.strategy_used}\n"
                f"Seed memories (compact view):\n{json.dumps(seed_payload, sort_keys=True, ensure_ascii=False)}"
            )
        )
        return sampling_payload(
            seed_batch,
            sampled_records=seed_records,
            seed_records=seed_records,
            summary=agentic_result.summary,
            execution_mode="agentic_mcp",
        )

    loop_result = await run_internal_tool_loop(
        ctx,
        provider,
        prompt=prompt,
        allowed_tool_names=[
            "internal_search_memory_records",
            "internal_read_memory_record",
            "internal_list_memory_records",
            "internal_get_next_curator_batch",
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
    summary = _curator_support.normalize_curator_summary(loop_result.response, tool_calls_executed=loop_result.tool_calls_executed)
    return sampling_payload(
        seed_batch,
        sampled_records=seed_records,
        seed_records=seed_records,
        summary=summary,
        execution_mode="json_tool_loop",
        tool_calls_executed=loop_result.tool_calls_executed,
        mutations=loop_result.mutating_tool_calls,
        tool_names_used=loop_result.tool_names_used,
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
