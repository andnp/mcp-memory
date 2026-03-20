from __future__ import annotations

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
