from __future__ import annotations

import json
from typing import Any

from mcp_memory.context import ApplicationContext
from mcp_memory.core.sampling import (
    ANOMALY_STRATEGY,
    BOUNDED_NOISE_STRATEGY,
    COLD_STORAGE_STRATEGY,
    NEVER_SURFACED_STRATEGY,
    ORPHAN_LOW_SUPPORT_STRATEGY,
    SamplingBatch,
)
from mcp_memory.core.task_handlers.constants import DEFAULT_AGENT_SCAN_LIMIT
from mcp_memory.core.task_handlers.maintenance_framework import (
    requested_sampling_strategy,
    sample_maintenance_candidates,
    support_counts_for_candidates,
)
from mcp_memory.core.tasks import TaskRecord

CURATOR_MAX_SEED_RECORDS = 16
CURATOR_SIZE_ANOMALY_SEED_RECORDS = 4
CURATOR_RECENCY_SEED_RECORDS = 4
CURATOR_CANDIDATE_POOL_MULTIPLIER = 3
CURATOR_MAX_BATCH_RECORDS = 24
CURATOR_MAX_MEMORY_CHARS = 4000
CURATOR_LARGEST_MEMORY_PASS_INTERVAL = 3
CURATOR_MAX_TITLE_CHARS = 80
CURATOR_MAX_SUMMARY_CHARS = 220
CURATOR_MAX_TAGS = 6
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


def normalize_curator_summary(response: dict[str, Any], *, tool_calls_executed: int) -> str | None:
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


def select_curator_seed_records(ctx: ApplicationContext, task: TaskRecord) -> list:
    return select_curator_seed_batch(ctx, task).records


def select_curator_seed_batch(
    ctx: ApplicationContext,
    task: TaskRecord,
    *,
    seed_limit: int | None = None,
    exclude_memory_ids: set[str] | None = None,
) -> SamplingBatch:
    assert ctx.repository is not None
    limit = _normalize_curator_seed_limit(seed_limit)
    excluded_ids = exclude_memory_ids or set()
    candidates = [
        record
        for record in ctx.repository.list_memories(
            workspace_id=_resolve_workspace_id(ctx, task),
            limit=int(task.data.get("limit", DEFAULT_AGENT_SCAN_LIMIT)),
        )
        if not ctx.repository.has_incoming_link(record.id, "SUPERSEDES")
        and record.id not in excluded_ids
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
        limit=min(len(candidates), max(limit, CURATOR_MAX_SEED_RECORDS) * CURATOR_CANDIDATE_POOL_MULTIPLIER),
        support_counts=build_support_counts(ctx, candidates),
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
    oversized_candidates = [record for record in largest_candidates if is_oversized_curator_memory(record)]
    anomaly_candidates = oversized_candidates
    if not anomaly_candidates and len(candidates) > CURATOR_MAX_SEED_RECORDS and should_run_curator_largest_memory_pass(task):
        anomaly_candidates = largest_candidates

    extend_unique_seed_records(seed_records, anomaly_candidates, CURATOR_SIZE_ANOMALY_SEED_RECORDS)
    extend_unique_seed_records(
        seed_records,
        sort_recent_curator_candidates(candidates),
        min(limit, len(seed_records) + CURATOR_RECENCY_SEED_RECORDS),
    )
    extend_unique_seed_records(seed_records, prioritized_candidates, limit)
    return SamplingBatch(
        requested_strategy=sampled_batch.requested_strategy,
        strategy_used=sampled_batch.strategy_used,
        strategy_fallback_reason=sampled_batch.strategy_fallback_reason,
        candidate_count=sampled_batch.candidate_count,
        records=seed_records[:limit],
    )


def curator_seed_payload_item(record) -> dict[str, Any]:
    summary_source = record.summary or record.content
    return {
        "id": record.id,
        "type": record.type,
        "status": record.status,
        "content_size_chars": len(record.content.strip()),
        "oversized_for_curator": is_oversized_curator_memory(record),
        "title": truncate_text(record.title, CURATOR_MAX_TITLE_CHARS),
        "summary": truncate_text(summary_source, CURATOR_MAX_SUMMARY_CHARS),
        "tags": list(record.tags[:CURATOR_MAX_TAGS]),
    }


def build_json_tool_loop_prompt(
    task: TaskRecord,
    *,
    strategy_used: str,
    seed_records: list[Any],
    guardrails: str,
) -> str:
    seed_payload = [curator_seed_payload_item(record) for record in seed_records]
    return (
        "You are the curator maintenance agent for the global memory store.\n"
        "Improve the store by merging, refining, rewriting, retagging, relinking, archiving, or deleting archived garbage when justified.\n"
        "Work in high-impact maintenance mode: prefer several coherent high-value improvements in one run when the store clearly supports them.\n"
        f"Start by calling internal_get_next_curator_batch with task_id='{task.id}', strategy='{strategy_used}', and exclude_memory_ids=[] so you can confirm or widen the current frontier before mutating.\n"
        "Treat the seed memories as a starting frontier, not a hard boundary; widen only when they hint at nearby duplicates, contradictions, or oversized clusters.\n"
        "Prefer safe operations with clear lineage and archive before delete when possible.\n"
        f"{guardrails}\n"
        f"Treat memories above {CURATOR_MAX_MEMORY_CHARS} characters as oversized. Prefer splitting them into smaller focused records with links such as DEPENDS_ON or AMENDS instead of growing one blob.\n"
        f"Avoid creating or growing memories past {CURATOR_MAX_MEMORY_CHARS} characters unless no reasonable split exists.\n"
        "Do not create journal or memory records for routine completion, counters, or status-only traces; use task_complete for operational closeout instead.\n"
        "Before stopping, check once more for any adjacent worthwhile maintenance action; no-op is fine when another step would be low-value or unsafe.\n"
        f"When your pass is complete, call task_complete with task_id='{task.id}', task_name='memory-curator', and a short summary before your final JSON response.\n"
        "When finished, return JSON like {\"summary\": \"...\", \"actions_taken\": N}.\n\n"
        f"Seed memories (compact view):\n{json.dumps(seed_payload, sort_keys=True, ensure_ascii=False)}"
    )


def build_agentic_prompt(
    task: TaskRecord,
    *,
    strategy_used: str,
    seed_records: list[Any],
    guardrails: str,
) -> str:
    seed_payload = [curator_seed_payload_item(record) for record in seed_records]
    return (
        "You are the memory-curator maintenance agent for the global memory store.\n"
        "Use the workspace-local internal MCP maintenance tools directly to inspect and mutate memories.\n"
        f"Start by calling internal_get_next_curator_batch with task_id='{task.id}', strategy='{strategy_used}', and exclude_memory_ids={json.dumps([record.id for record in seed_records], sort_keys=True)} so you can widen beyond the current frontier only when justified.\n"
        "Treat the provided seed memories as a starting frontier and the active frontier for this run; widen only when they imply nearby duplicates, contradictions, taxonomy cleanup, or oversized clusters.\n"
        "Aim for multiple coherent, high-value maintenance actions in one run when justified, with clear lineage and archive-before-delete when possible.\n"
        f"{guardrails}\n"
        f"Treat memories above {CURATOR_MAX_MEMORY_CHARS} characters as oversized and prefer splitting them into focused linked records.\n"
        "When you materially rewrite a memory and already understand it, refresh a concise summary in the same tool call.\n"
        "Do not create journal or memory records for routine completion, counters, or status-only traces; use task_complete for operational closeout instead.\n"
        "Before finishing, do one more quick search/list/read pass for any adjacent high-value maintenance opportunity.\n"
        "Do not claim work you did not actually execute through MCP tools.\n"
        f"When your pass is complete, call task_complete with task_id='{task.id}', task_name='memory-curator', and a short summary before your final JSON response.\n"
        "When finished, output final JSON only in the form {\"summary\": \"...\"}.\n\n"
        f"Sampling strategy: {strategy_used}\n"
        f"Seed memories (compact view):\n{json.dumps(seed_payload, sort_keys=True, ensure_ascii=False)}"
    )


def review_seed_records(ctx: ApplicationContext, payload: dict[str, Any]) -> list[Any]:
    if ctx.repository is None:
        return []
    seed_ids = payload.get("seed_memory_ids")
    if not isinstance(seed_ids, list):
        return []
    records: list[Any] = []
    for memory_id in seed_ids:
        if not isinstance(memory_id, str):
            continue
        record = ctx.repository.get_memory(memory_id)
        if record is None or record.status != "active":
            continue
        records.append(record)
    return records


def review_sampling_batch(payload: dict[str, Any], seed_records: list[Any]) -> SamplingBatch:
    requested_strategy = payload.get("strategy_used")
    if not isinstance(requested_strategy, str):
        requested_strategy = None
    candidate_count = payload.get("candidate_count")
    if not isinstance(candidate_count, int):
        candidate_count = len(seed_records)
    return SamplingBatch(
        requested_strategy=requested_strategy,
        strategy_used=requested_strategy or "none",
        strategy_fallback_reason=None,
        candidate_count=candidate_count,
        records=seed_records,
    )


def extend_unique_seed_records(seed_records: list[Any], candidates: list[Any], limit: int) -> None:
    seen_ids = {record.id for record in seed_records}
    for record in candidates:
        if record.id in seen_ids:
            continue
        seed_records.append(record)
        seen_ids.add(record.id)
        if len(seed_records) >= limit:
            return


def sort_recent_curator_candidates(candidates: list[Any]) -> list[Any]:
    return sorted(
        candidates,
        key=lambda record: (
            _sort_curator_timestamp(record.created_at),
            _sort_curator_timestamp(record.updated_at),
            -(record.read_count),
        ),
        reverse=True,
    )


def is_oversized_curator_memory(record) -> bool:
    return len(record.content.strip()) > CURATOR_MAX_MEMORY_CHARS


def should_run_curator_largest_memory_pass(task: TaskRecord) -> bool:
    return sum(task.id.encode("utf-8")) % CURATOR_LARGEST_MEMORY_PASS_INTERVAL == 0


def build_support_counts(ctx: ApplicationContext, candidates: list) -> dict[str, int]:
    return support_counts_for_candidates(ctx, candidates)


def _normalize_curator_seed_limit(value: int | None) -> int:
    if value is None:
        return CURATOR_MAX_SEED_RECORDS
    return max(1, min(int(value), CURATOR_MAX_BATCH_RECORDS))


def _sort_curator_timestamp(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def truncate_text(value: str | None, limit: int) -> str:
    text = "" if value is None else " ".join(value.strip().split())
    if len(text) <= limit:
        return text
    return text[: max(limit - 1, 0)].rstrip() + "…"


def _requested_sampling_strategy(task: TaskRecord) -> str | None:
    return requested_sampling_strategy(task)


def _resolve_workspace_id(ctx: ApplicationContext, task: TaskRecord) -> str | None:
    task_workspace = task.data.get("workspace_id")
    if isinstance(task_workspace, str) and task_workspace.strip():
        return task_workspace.strip()
    return None
