from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from inspect import isawaitable
import json
from pathlib import Path
import re
from typing import Any, Awaitable, Callable, cast

from mcp_memory.context import ApplicationContext
from mcp_memory.core.sampling import (
    ANOMALY_STRATEGY,
    BOUNDED_NOISE_STRATEGY,
    COLD_STORAGE_STRATEGY,
    COOLDOWN_ESCAPE_STRATEGY,
    CONFLICT_FRONTIER_STRATEGY,
    GRAPH_BRIDGE_STRATEGY,
    NEVER_SURFACED_STRATEGY,
    ORPHAN_LOW_SUPPORT_STRATEGY,
    SEMANTIC_STRATEGY,
    SamplingBatch,
)
from mcp_memory.core.task_handlers.maintenance_framework import (
    requested_sampling_strategy,
    sample_maintenance_candidates,
    sampling_payload,
    support_counts_for_candidates,
)
import mcp_memory.core.task_handlers.deduplicator_merge as _deduplicator_merge
import mcp_memory.core.task_handlers.deduplicator_support as _deduplicator_support
from mcp_memory.core.task_handlers.constants import (
    DEFAULT_AGENT_SCAN_LIMIT,
    CURATOR_TASK_NAME,
    DEFAULT_STALE_PLAN_DAYS,
    DEFAULT_SWEEP_RETENTION_DAYS,
)
from mcp_memory.core.task_handlers.agentic_guardrails import (
    build_curator_guardrails,
    build_reflection_synthesis_guardrails,
)
from mcp_memory.core.task_handlers.tool_loop import run_internal_tool_loop
from mcp_memory.core.tasks import TaskRecord


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_:-]+")
DEFRAGMENTER_GROUP_SIMILARITY_THRESHOLD = 0.45
GRAPH_LINKER_AI_MIN_CANDIDATES = 12
GRAPH_LINKER_FALLBACK_LINK_TARGET = 2
CONFLICT_DETECTOR_AI_MIN_CANDIDATES = 15
DEFRAGMENTER_AI_MIN_GROUP_SIZE = 3
DEFRAGMENTER_AI_MIN_SOURCE_LINES = 200
DEDUPLICATOR_MAX_SEED_RECORDS = 8
DEDUPLICATOR_SIZE_ANOMALY_SEED_RECORDS = 2
DEDUPLICATOR_OBSERVATION_SEED_RECORDS = 4
CURATOR_MAX_SEED_RECORDS = 8
CURATOR_SIZE_ANOMALY_SEED_RECORDS = 2
CURATOR_MAX_MEMORY_CHARS = 4000
CURATOR_LARGEST_MEMORY_PASS_INTERVAL = 3
CURATOR_MAX_TITLE_CHARS = 80
CURATOR_MAX_SUMMARY_CHARS = 220
CURATOR_MAX_TAGS = 6
LOW_SIGNAL_GROUPING_TAGS = {
    "anomaly-sample",
    "architecture",
    "audit",
    "auto-defragmented",
    "auto-ingested",
    "background-agent",
    "deduplicator",
    "maintenance",
    "memory-curator",
    "system1",
    "system1-appended",
    "system-design",
    "task-complete",
}
LOW_SIGNAL_TOPIC_TOKENS = {
    "agent",
    "agents",
    "architecture",
    "background",
    "cli",
    "complete",
    "completed",
    "daemon",
    "design",
    "implementation",
    "implementations",
    "infra",
    "infrastructure",
    "lifecycle",
    "maintenance",
    "migration",
    "migrations",
    "observability",
    "overview",
    "project",
    "projects",
    "repo",
    "repository",
    "roadmap",
    "runtime",
    "service",
    "services",
    "system",
    "systems",
    "task",
    "tasks",
    "testing",
    "tool",
    "tools",
    "transport",
}
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
CURATOR_ALLOWED_STRATEGIES = (
    ANOMALY_STRATEGY,
    COLD_STORAGE_STRATEGY,
    NEVER_SURFACED_STRATEGY,
    ORPHAN_LOW_SUPPORT_STRATEGY,
    BOUNDED_NOISE_STRATEGY,
)
CURATOR_STRATEGY_WEIGHTS = {
    ANOMALY_STRATEGY: 3,
    COLD_STORAGE_STRATEGY: 2,
    NEVER_SURFACED_STRATEGY: 2,
    ORPHAN_LOW_SUPPORT_STRATEGY: 2,
    BOUNDED_NOISE_STRATEGY: 1,
}
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
DEFRAGMENTER_ALLOWED_STRATEGIES = (
    COLD_STORAGE_STRATEGY,
    SEMANTIC_STRATEGY,
    ORPHAN_LOW_SUPPORT_STRATEGY,
)
DEFRAGMENTER_STRATEGY_WEIGHTS = {
    COLD_STORAGE_STRATEGY: 3,
    SEMANTIC_STRATEGY: 2,
    ORPHAN_LOW_SUPPORT_STRATEGY: 2,
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


def handle_project_manager_task(
    ctx: ApplicationContext,
    task: TaskRecord,
) -> dict[str, Any]:
    if ctx.db_manager is None:
        return {"updated": 0}

    cutoff = (datetime.now(UTC) - timedelta(days=DEFAULT_STALE_PLAN_DAYS)).isoformat()
    workspace_id = _resolve_workspace_id(ctx, task)
    conn = ctx.db_manager.get_connection()
    if workspace_id is None:
        cursor = conn.execute(
            """
            UPDATE memories
            SET status = 'stale'
            WHERE type = 'plan' AND status = 'active' AND updated_at < ?
            """,
            (cutoff,),
        )
    else:
        cursor = conn.execute(
            """
            UPDATE memories
            SET status = 'stale'
            WHERE id IN (
                SELECT memories.id
                FROM memories
                JOIN memory_workspaces ON memory_workspaces.memory_id = memories.id
                WHERE memories.type = 'plan'
                  AND memories.status = 'active'
                  AND memories.updated_at < ?
                  AND memory_workspaces.workspace_id = ?
            )
            """,
            (cutoff, workspace_id),
        )
    conn.commit()
    return {"updated": cursor.rowcount}


def handle_fact_checker_task(
    ctx: ApplicationContext,
    task: TaskRecord,
) -> dict[str, Any]:
    if ctx.db_manager is None:
        return {"degraded": 0, "restored": 0}

    conn = ctx.db_manager.get_connection()
    rows = conn.execute(
        """
        SELECT memories.id, memories.status, links.target_id, memory_workspaces.workspace_id
        FROM memories
        JOIN links ON links.source_id = memories.id
        LEFT JOIN memory_workspaces ON memory_workspaces.memory_id = memories.id
        WHERE links.target_id LIKE 'ext:%'
        """
    ).fetchall()

    degraded_ids: set[str] = set()
    healthy_ids: set[str] = set()
    workspace_root_override = task.data.get("workspace_root")
    for row in rows:
        workspace_root = _resolve_workspace_root(
            ctx,
            row["workspace_id"],
            workspace_root_override,
        )
        target_path = row["target_id"][4:]
        file_path = Path(target_path)
        if not file_path.is_absolute() and workspace_root is not None:
            file_path = workspace_root / target_path

        if file_path.exists():
            healthy_ids.add(row["id"])
        else:
            degraded_ids.add(row["id"])

    restored_ids = healthy_ids - degraded_ids
    if degraded_ids:
        placeholders = ",".join("?" for _ in degraded_ids)
        conn.execute(
            f"UPDATE memories SET status = 'degraded' WHERE id IN ({placeholders})",
            list(degraded_ids),
        )
    if restored_ids:
        placeholders = ",".join("?" for _ in restored_ids)
        conn.execute(
            f"UPDATE memories SET status = 'active' WHERE id IN ({placeholders}) AND status = 'degraded'",
            list(restored_ids),
        )
    conn.commit()
    return {"degraded": len(degraded_ids), "restored": len(restored_ids)}


def handle_sweeper_task(
    ctx: ApplicationContext,
    task: TaskRecord,
) -> dict[str, Any]:
    if ctx.db_manager is None:
        return {"deleted_tasks": 0, "deleted_journal_entries": 0}

    cutoff = datetime.now(UTC) - timedelta(days=DEFAULT_SWEEP_RETENTION_DAYS)
    cutoff_timestamp = cutoff.timestamp()
    conn = ctx.db_manager.get_connection()
    deleted_tasks = conn.execute(
        "DELETE FROM tasks WHERE status = 'completed' AND updated_at < ?",
        (cutoff_timestamp,),
    ).rowcount
    deleted_journal_entries = conn.execute(
        "DELETE FROM system1_journal WHERE status IN ('processed', 'archived') AND timestamp < ?",
        (cutoff_timestamp,),
    ).rowcount
    conn.commit()
    return {
        "deleted_tasks": deleted_tasks,
        "deleted_journal_entries": deleted_journal_entries,
    }


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

    proposed_pairs = await _propose_graph_links(ctx, candidates, provider)
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

    proposed_pairs = await _propose_conflicts(ctx, candidates, provider)
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
    groups = _collect_defragment_groups(candidates)
    if not groups:
        return sampling_payload(sampled_batch, sampled_records=candidates, created=0, archived=0)

    created = 0
    archived = 0
    lines_compressed = 0
    for group in groups:
        source_lines = sum(_count_text_lines(item.content) for item in group)
        title, content = await _build_defragmented_memory(
            group,
            provider if _should_use_provider_for_defragment_group(group, source_lines) else None,
        )
        reflection_lines = _count_text_lines(content)
        record = ctx.repository.create_memory(
            title=title,
            content=content,
            workspace_ids=_resolve_group_workspace_ids(group, _resolve_workspace_id(ctx, task)),
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

    seed_batch = _select_curator_seed_batch(ctx, task)
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
        _curator_seed_payload_item(record)
        for record in seed_records
    ]
    prompt = (
        f"You are the {CURATOR_TASK_NAME} maintenance agent for the global memory store.\n"
        "Your goal is to improve the memory store by merging, refining, rewriting, retagging, relinking, archiving, or deleting archived garbage when justified.\n"
        "Prefer safe operations with clear lineage. Archive before delete whenever possible.\n"
        f"{build_curator_guardrails()}\n"
        f"Treat memories above {CURATOR_MAX_MEMORY_CHARS} characters as oversized. Prefer splitting oversized memories into smaller focused records with links such as DEPENDS_ON or AMENDS instead of continuing to append or merge them into one blob.\n"
        f"Avoid creating or growing memories past {CURATOR_MAX_MEMORY_CHARS} characters unless no reasonable split exists.\n"
        "Use the internal maintenance tools to inspect and mutate the store.\n"
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
                "Prefer safe operations with clear lineage. Archive before delete whenever possible.\n"
                f"{build_curator_guardrails()}\n"
                f"Treat memories above {CURATOR_MAX_MEMORY_CHARS} characters as oversized and prefer splitting them into focused linked records.\n"
                "When you create, merge, or materially rewrite a memory and you already understand it, include or refresh a concise summary in the same tool call instead of relying on a later standalone summarizer.\n"
                "Do not claim work you did not actually execute through MCP tools.\n"
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
        max_rounds=6,
    )
    summary = _normalize_curator_summary(loop_result.response, tool_calls_executed=loop_result.tool_calls_executed)
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


def _normalize_curator_summary(response: dict[str, Any], *, tool_calls_executed: int) -> str | None:
    raw_summary = response.get("summary")
    summary = raw_summary.strip() if isinstance(raw_summary, str) and raw_summary.strip() else None
    if summary is None:
        return None

    reported_actions_taken = response.get("actions_taken")
    if tool_calls_executed <= 0 and isinstance(reported_actions_taken, int) and reported_actions_taken > 0:
        return (
            f"Provider reported actions_taken={reported_actions_taken} without using internal tools; "
            "no curator maintenance actions were executed."
        )
    return summary


async def _propose_graph_links(
    ctx: ApplicationContext,
    candidates: list,
    provider: Any = None,
) -> list[tuple[str, str, str, str]]:
    fallback = _fallback_graph_links(candidates)
    if (
        provider is None
        or len(fallback) >= GRAPH_LINKER_FALLBACK_LINK_TARGET
        or len(candidates) <= GRAPH_LINKER_AI_MIN_CANDIDATES
    ):
        return fallback

    if provider is not None:
        response = await run_internal_tool_loop(
            ctx,
            provider=provider,
            prompt=_build_linker_prompt(candidates),
            allowed_tool_names=[
                "internal_search_memory_records",
                "internal_read_memory_record",
                "internal_list_memory_records",
            ],
            max_rounds=3,
        )
        proposals = response.response.get("links", [])
        if isinstance(proposals, list):
            normalized: list[tuple[str, str, str, str]] = []
            for item in proposals:
                if not isinstance(item, dict):
                    continue
                source_id = str(item.get("source_id", "")).strip()
                target_id = str(item.get("target_id", "")).strip()
                link_type = str(item.get("link_type", "")).strip() or "DEPENDS_ON"
                context = str(item.get("context", "")).strip() or "Auto-linked by graph linker."
                if source_id and target_id:
                    normalized.append((source_id, target_id, link_type, context))
            if normalized:
                return normalized

    return fallback


async def _propose_conflicts(
    ctx: ApplicationContext,
    candidates: list,
    provider: Any = None,
) -> list[tuple[str, str, str]]:
    fallback = _fallback_conflicts(candidates)
    if provider is None or fallback or len(candidates) <= CONFLICT_DETECTOR_AI_MIN_CANDIDATES:
        return fallback

    if provider is not None:
        response = await run_internal_tool_loop(
            ctx,
            provider=provider,
            prompt=_build_conflict_prompt(candidates),
            allowed_tool_names=[
                "internal_search_memory_records",
                "internal_read_memory_record",
                "internal_list_memory_records",
            ],
            max_rounds=3,
        )
        proposals = response.response.get("conflicts", [])
        if isinstance(proposals, list):
            normalized: list[tuple[str, str, str]] = []
            for item in proposals:
                if not isinstance(item, dict):
                    continue
                left_id = str(item.get("left_id", "")).strip()
                right_id = str(item.get("right_id", "")).strip()
                context = str(item.get("context", "")).strip() or "Potential contradiction detected."
                if left_id and right_id:
                    normalized.append((left_id, right_id, context))
            if normalized:
                return normalized

    return fallback


def _fallback_graph_links(candidates: list) -> list[tuple[str, str, str, str]]:
    proposals: list[tuple[str, str, str, str]] = []
    for source, target in _iter_candidate_pairs(candidates):
        shared_tags = sorted(set(source.tags) & set(target.tags))
        token_overlap = _token_overlap(source.title, target.title)
        if not shared_tags and token_overlap < 0.34:
            continue
        newer, older = _sort_newer_first(source, target)
        link_type = "AMENDS" if newer.type == older.type else "DEPENDS_ON"
        context = (
            f"Auto-linked from shared tags ({', '.join(shared_tags)})"
            if shared_tags
            else "Auto-linked from title similarity."
        )
        proposals.append((newer.id, older.id, link_type, context))
    return proposals[:10]


def _fallback_conflicts(candidates: list) -> list[tuple[str, str, str]]:
    proposals: list[tuple[str, str, str]] = []
    for left, right in _iter_candidate_pairs(candidates):
        if left.type != right.type:
            continue
        if left.content.strip() == right.content.strip():
            continue
        shared_tags = set(left.tags) & set(right.tags)
        title_overlap = _token_overlap(left.title, right.title)
        if title_overlap < 0.5 and not shared_tags:
            continue
        proposals.append((left.id, right.id, "Potential contradiction detected from overlapping titles/tags."))
    return proposals[:10]


def _collect_defragment_groups(candidates: list) -> list[list]:
    groups: list[list] = []
    used: set[str] = set()
    for record in candidates:
        if record.id in used:
            continue
        related = [record]
        used.add(record.id)
        for other in candidates:
            if other.id in used or other.id == record.id:
                continue
            shared_tags = _shared_meaningful_tags(record.tags, other.tags)
            similarity = _topic_token_overlap(record.title + " " + record.content, other.title + " " + other.content)
            if shared_tags or similarity >= DEFRAGMENTER_GROUP_SIMILARITY_THRESHOLD:
                related.append(other)
                used.add(other.id)
        if len(related) >= 2:
            groups.append(related[:5])
    return groups[:3]


async def _build_defragmented_memory(group: list, provider: Any = None) -> tuple[str, str]:
    if provider is not None:
        prompt = (
            "Summarize these memories into one reflection. Return JSON with title and content.\n"
            f"{build_reflection_synthesis_guardrails()}\n\n"
            + "\n".join(f"- {item.title}: {item.content}" for item in group)
        )
        response = provider.ask(prompt)
        if isawaitable(response):
            response = await response
        title = str(response.get("title", "")).strip()
        content = str(response.get("content", "")).strip()
        if title and content:
            return title, content

    title = f"Reflection: {group[0].title}"
    content_lines = ["Consolidated observations:"]
    for item in group:
        snippet = item.summary or item.content.strip().splitlines()[0]
        content_lines.append(f"- {item.title}: {snippet}")
    return title, "\n".join(content_lines)


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


def _select_curator_seed_records(ctx: ApplicationContext, task: TaskRecord) -> list:
    return _select_curator_seed_batch(ctx, task).records


def _select_curator_seed_batch(ctx: ApplicationContext, task: TaskRecord) -> SamplingBatch:
    assert ctx.repository is not None
    candidates = [
        record
        for record in ctx.repository.list_memories(
            workspace_id=_resolve_workspace_id(ctx, task),
            limit=int(task.data.get("limit", DEFAULT_AGENT_SCAN_LIMIT)),
        )
        if not ctx.repository.has_incoming_link(record.id, "SUPERSEDES")
    ]
    if not candidates:
        return SamplingBatch(
            requested_strategy=_requested_sampling_strategy(task),
            strategy_used=_requested_sampling_strategy(task) or "none",
            strategy_fallback_reason=None,
            candidate_count=0,
            records=[],
        )

    sampled_batch = sample_maintenance_candidates(
        ctx,
        task,
        candidates,
        allowed_strategies=CURATOR_ALLOWED_STRATEGIES,
        strategy_weights=CURATOR_STRATEGY_WEIGHTS,
        limit=min(len(candidates), CURATOR_MAX_SEED_RECORDS * 2),
        support_counts=_build_support_counts(ctx, candidates),
    )
    sampled_candidates = sampled_batch.records

    prioritized_candidates = sorted(
        sampled_candidates,
        key=lambda record: (
            0 if record.type in {"journal", "observation"} else 1,
            record.read_count,
            len(record.content.strip()),
            record.updated_at,
        )
    )
    largest_candidates = sorted(
        sampled_candidates,
        key=lambda record: (
            -len(record.content.strip()),
            record.read_count,
            record.updated_at,
        ),
    )

    seed_records: list[Any] = []
    oversized_candidates = [record for record in largest_candidates if _is_oversized_curator_memory(record)]
    anomaly_candidates = oversized_candidates
    if not anomaly_candidates and len(candidates) > CURATOR_MAX_SEED_RECORDS and _should_run_curator_largest_memory_pass(task):
        anomaly_candidates = largest_candidates

    _extend_unique_seed_records(seed_records, anomaly_candidates, CURATOR_SIZE_ANOMALY_SEED_RECORDS)
    _extend_unique_seed_records(seed_records, prioritized_candidates, CURATOR_MAX_SEED_RECORDS)
    return SamplingBatch(
        requested_strategy=sampled_batch.requested_strategy,
        strategy_used=sampled_batch.strategy_used,
        strategy_fallback_reason=sampled_batch.strategy_fallback_reason,
        candidate_count=sampled_batch.candidate_count,
        records=seed_records[:CURATOR_MAX_SEED_RECORDS],
    )


def _curator_seed_payload_item(record) -> dict[str, Any]:
    summary_source = record.summary or record.content
    return {
        "id": record.id,
        "type": record.type,
        "status": record.status,
        "content_size_chars": len(record.content.strip()),
        "oversized_for_curator": _is_oversized_curator_memory(record),
        "title": _truncate_text(record.title, CURATOR_MAX_TITLE_CHARS),
        "summary": _truncate_text(summary_source, CURATOR_MAX_SUMMARY_CHARS),
        "tags": list(record.tags[:CURATOR_MAX_TAGS]),
    }


def _extend_unique_seed_records(seed_records: list[Any], candidates: list[Any], limit: int) -> None:
    seen_ids = {record.id for record in seed_records}
    for record in candidates:
        if record.id in seen_ids:
            continue
        seed_records.append(record)
        seen_ids.add(record.id)
        if len(seed_records) >= limit:
            return


def _is_oversized_curator_memory(record) -> bool:
    return len(record.content.strip()) > CURATOR_MAX_MEMORY_CHARS


def _should_run_curator_largest_memory_pass(task: TaskRecord) -> bool:
    return sum(task.id.encode("utf-8")) % CURATOR_LARGEST_MEMORY_PASS_INTERVAL == 0


def _build_support_counts(ctx: ApplicationContext, candidates: list) -> dict[str, int]:
    return support_counts_for_candidates(ctx, candidates)


def _sample_maintenance_candidates(
    ctx: ApplicationContext,
    task: TaskRecord,
    candidates: list,
    *,
    allowed_strategies: tuple[str, ...],
    strategy_weights: dict[str, int],
    limit: int,
) -> SamplingBatch:
    return sample_maintenance_candidates(
        ctx,
        task,
        candidates,
        allowed_strategies=allowed_strategies,
        strategy_weights=strategy_weights,
        limit=limit,
    )


def _requested_sampling_strategy(task: TaskRecord) -> str | None:
    return requested_sampling_strategy(task)


# Temporary compatibility aliases for the first deduplicator extraction slice.
_select_deduplicator_seed_batch = _deduplicator_support.select_deduplicator_seed_batch
_build_deduplicator_agent_prompt = _deduplicator_support.build_deduplicator_agent_prompt
_normalize_deduplicator_agentic_result = _deduplicator_support.normalize_deduplicator_agentic_result


def _truncate_text(value: str | None, limit: int) -> str:
    text = "" if value is None else " ".join(value.strip().split())
    if len(text) <= limit:
        return text
    return text[: max(limit - 1, 0)].rstrip() + "…"


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
        "mcp_mcp-memory-internal_internal_get_next_dedup_batch",
        "mcp_mcp-memory-internal_internal_read_memory_record",
        "mcp_mcp-memory-internal_internal_search_memory_records",
        "mcp_mcp-memory-internal_internal_list_memory_records",
    }
    total = 0
    for name, payload in value.items():
        if not isinstance(name, str) or name in read_only_tool_names or not isinstance(payload, dict):
            continue
        total += _coerce_non_negative_int(payload.get("count"))
    return total


def _should_use_provider_for_defragment_group(group: list, source_lines: int) -> bool:
    return len(group) >= DEFRAGMENTER_AI_MIN_GROUP_SIZE and source_lines >= DEFRAGMENTER_AI_MIN_SOURCE_LINES


def _iter_candidate_pairs(candidates: Iterable) -> Iterable[tuple[Any, Any]]:
    candidate_list = list(candidates)
    for index, left in enumerate(candidate_list):
        for right in candidate_list[index + 1 :]:
            yield left, right


def _sort_newer_first(left, right):
    return (left, right) if left.updated_at >= right.updated_at else (right, left)


def _token_overlap(left: str, right: str) -> float:
    left_tokens = {token.lower() for token in TOKEN_PATTERN.findall(left)}
    right_tokens = {token.lower() for token in TOKEN_PATTERN.findall(right)}
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _topic_token_overlap(left: str, right: str) -> float:
    left_tokens = _topic_tokens(left)
    right_tokens = _topic_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _topic_tokens(value: str) -> set[str]:
    return {
        token.lower()
        for token in TOKEN_PATTERN.findall(value)
        if token.lower() not in LOW_SIGNAL_TOPIC_TOKENS
    }


def _shared_meaningful_tags(left_tags: list[str], right_tags: list[str]) -> set[str]:
    return _meaningful_tag_set(left_tags) & _meaningful_tag_set(right_tags)


def _meaningful_tag_set(tags: list[str]) -> set[str]:
    meaningful: set[str] = set()
    for tag in tags:
        normalized = tag.strip().lower()
        if (
            not normalized
            or normalized in LOW_SIGNAL_GROUPING_TAGS
            or normalized.endswith("-task")
            or "-task-" in normalized
        ):
            continue
        meaningful.add(normalized)
    return meaningful


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


def _count_text_lines(value: str) -> int:
    text = value.strip()
    if not text:
        return 0
    return text.count("\n") + 1


def _has_link(ctx: ApplicationContext, source_id: str, target_id: str, link_type: str) -> bool:
    assert ctx.repository is not None
    return any(
        link.target_id == target_id
        for link in ctx.repository.get_links(source_id, direction="outgoing", link_type=link_type)
    )


def _build_linker_prompt(candidates: list) -> str:
    entries = json.dumps(
        [
            {
                "id": record.id,
                "type": record.type,
                "status": record.status,
                "title": record.title,
                "summary": _truncate_text(record.summary or record.content, 180),
                "tags": record.tags,
            }
            for record in candidates[:20]
        ],
        sort_keys=True,
        ensure_ascii=False,
    )
    return (
        "Review these active memories and propose only high-confidence typed links.\n"
        "Use DEPENDS_ON when one memory relies on, implements, or is downstream of another.\n"
        "Use AMENDS when a newer memory updates, refines, or corrects an older memory on the same topic.\n"
        "Do not propose weak title-only links, duplicate existing relationships, or symmetric duplicates.\n"
        "If the local candidate list is insufficient, use the internal tools to search and read for better context before proposing links.\n"
        'Return JSON: {"links": [{"source_id": "...", "target_id": "...", "link_type": "DEPENDS_ON|AMENDS", "context": "..."}]}\n\n'
        f"Memories:\n{entries}"
    )


def _build_conflict_prompt(candidates: list) -> str:
    entries = json.dumps(
        [
            {
                "id": record.id,
                "type": record.type,
                "status": record.status,
                "title": record.title,
                "summary": _truncate_text(record.summary or record.content, 180),
                "tags": record.tags,
            }
            for record in candidates[:20]
        ],
        sort_keys=True,
        ensure_ascii=False,
    )
    return (
        "Review these active memories and propose contradictions only when two records make materially incompatible claims.\n"
        "Do not flag mere topic overlap, phrasing differences, or newer refinements of older memories as contradictions.\n"
        "Prefer concrete evidence in the titles, summaries, and any extra context you retrieve with the internal tools.\n"
        "If you need broader context, use the internal tools to search and read before proposing a contradiction.\n"
        'Return JSON: {"conflicts": [{"left_id": "...", "right_id": "...", "context": "..."}]}\n\n'
        f"Memories:\n{entries}"
    )


def _resolve_workspace_id(ctx: ApplicationContext, task: TaskRecord) -> str | None:
    task_workspace = task.data.get("workspace_id")
    if isinstance(task_workspace, str) and task_workspace.strip():
        return task_workspace.strip()
    return None


def _resolve_group_workspace_ids(group: list, fallback_workspace_id: str | None) -> list[str]:
    workspace_ids = sorted(
        {
            workspace_id.strip()
            for item in group
            for workspace_id in getattr(item, "workspace_ids", [])
            if isinstance(workspace_id, str) and workspace_id.strip()
        }
    )
    if workspace_ids:
        return workspace_ids
    if isinstance(fallback_workspace_id, str) and fallback_workspace_id.strip():
        return [fallback_workspace_id.strip()]
    return ["workspace-unknown"]


def _resolve_workspace_root(
    ctx: ApplicationContext,
    workspace_id: str | None,
    workspace_root_override: object | None = None,
) -> Path | None:
    if isinstance(workspace_root_override, str) and workspace_root_override.strip():
        override_path = Path(workspace_root_override).expanduser()
        if override_path.exists():
            return override_path
    if workspace_id is None:
        return None
    if ctx.workspace_root is not None and ctx.workspace_id == workspace_id:
        return ctx.workspace_root
    path = Path(workspace_id).expanduser()
    if path.exists():
        return path
    return None
