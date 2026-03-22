from __future__ import annotations

import json
from typing import Any

from mcp_memory.context import ApplicationContext
from mcp_memory.core.sampling import (
    ANOMALY_STRATEGY,
    COOLDOWN_ESCAPE_STRATEGY,
    SEMANTIC_STRATEGY,
    SamplingBatch,
)
from mcp_memory.core.task_handlers.agentic_guardrails import build_deduplicator_guardrails
from mcp_memory.core.task_handlers.constants import DEDUPLICATOR_TASK_NAME
from mcp_memory.core.task_handlers.maintenance_framework import (
    sample_maintenance_candidates,
    support_counts_for_candidates,
)
from mcp_memory.core.tasks import TaskRecord

DEDUPLICATOR_AI_MIN_COMBINED_LINES = 20
DEDUPLICATOR_HIGH_OVERLAP_THRESHOLD = 0.75
DEDUPLICATOR_MAX_SEED_RECORDS = 8
DEDUPLICATOR_SIZE_ANOMALY_SEED_RECORDS = 2
DEDUPLICATOR_OBSERVATION_SEED_RECORDS = 4
CURATOR_MAX_TITLE_CHARS = 80
CURATOR_MAX_SUMMARY_CHARS = 220
CURATOR_MAX_TAGS = 6

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


def select_deduplicator_seed_batch(
    ctx: ApplicationContext,
    candidates: list,
    *,
    task_id: str,
    strategy: str | None,
) -> SamplingBatch:
    if not candidates:
        return SamplingBatch(
            requested_strategy=strategy,
            strategy_used=strategy or "none",
            strategy_fallback_reason=None,
            candidate_count=0,
            records=[],
        )

    deduplicator_task = TaskRecord(
        id=task_id,
        task_name=DEDUPLICATOR_TASK_NAME,
        data={} if strategy is None else {"strategy": strategy},
        workspace_id=None,
        status="pending",
        priority=0,
        retries_count=0,
        max_retries=0,
        created_at=0.0,
        updated_at=0.0,
        available_at=0.0,
        claimed_at=None,
        started_at=None,
        completed_at=None,
        last_error=None,
    )
    sampled_batch = sample_maintenance_candidates(
        ctx,
        deduplicator_task,
        candidates,
        allowed_strategies=DEDUPLICATOR_ALLOWED_STRATEGIES,
        strategy_weights=DEDUPLICATOR_STRATEGY_WEIGHTS,
        limit=min(len(candidates), DEDUPLICATOR_MAX_SEED_RECORDS * 2),
        support_counts=support_counts_for_candidates(ctx, candidates),
    )
    seed_records = _select_deduplicator_seed_records(sampled_batch.records)
    return SamplingBatch(
        requested_strategy=sampled_batch.requested_strategy,
        strategy_used=sampled_batch.strategy_used,
        strategy_fallback_reason=sampled_batch.strategy_fallback_reason,
        candidate_count=sampled_batch.candidate_count,
        records=seed_records,
    )


def build_deduplicator_agent_prompt(task: TaskRecord, seed_records: list, *, strategy_used: str) -> str:
    seed_payload = [_deduplicator_seed_payload_item(record) for record in seed_records]
    return (
        "You are the deduplicator maintenance agent for the global memory store.\n"
        "Use the workspace-local internal MCP maintenance tools directly to inspect and mutate memories.\n"
        f"Start with internal_get_next_dedup_batch using task_id='{task.id}' to confirm the current seed batch before making changes.\n"
        f"{build_deduplicator_guardrails()}\n"
        "Merge highly similar fact memories into canonical records, preserve lineage with SUPERSEDES links, and absorb matching observations into the most appropriate fact when justified.\n"
        "Prefer internal_merge_memory_into_canonical for every merge or observation absorption so canonical metadata, archived sources, and lineage stay consistent.\n"
        f"When using internal_merge_memory_into_canonical, include metadata with deduplicator_task_id='{task.id}' and preserve merged_source_ids.\n"
        "When the merged canonical fact is clear, include a concise summary in the same merge call so no separate summarizer pass is needed.\n"
        "Only fall back to separate update/archive/link calls when you are creating a brand new canonical fact first and then merging other records into it.\n"
        "Prefer safe, minimal merges. Do not merge records unless the content overlap is strong and the resulting canonical memory stays coherent.\n"
        "Do not claim work you did not actually execute through MCP tools.\n"
        f"When your pass is complete, call task_complete with task_id='{task.id}', task_name='deduplicator', and a short summary before your final JSON response.\n"
        'When finished, output final JSON only in the form {"summary": "...", "merged": N, "archived": N, "absorbed_observations": N}.\n\n'
        f"Sampling strategy: {strategy_used}\n"
        f"Seed memories (compact view):\n{json.dumps(seed_payload, sort_keys=True, ensure_ascii=False)}"
    )


def normalize_deduplicator_agentic_result(agentic_result: Any, seed_records: list) -> dict[str, Any]:
    parsed = agentic_result.parsed if isinstance(getattr(agentic_result, "parsed", None), dict) else {}
    response_payload = parsed
    response_text = parsed.get("response")
    if not {"summary", "merged", "archived", "absorbed_observations"} <= set(response_payload) and isinstance(response_text, str):
        nested = _extract_embedded_json_object(response_text)
        if isinstance(nested, dict):
            response_payload = nested
    raw_tool_stats = parsed.get("stats")
    tool_stats = raw_tool_stats if isinstance(raw_tool_stats, dict) else {}
    raw_tool_payload = tool_stats.get("tools")
    tool_payload = raw_tool_payload if isinstance(raw_tool_payload, dict) else {}
    tool_names_used = _extract_agentic_tool_names(tool_payload.get("byName"))
    return {
        "summary": _coerce_text_summary(getattr(agentic_result, "summary", None)) or _coerce_text_summary(response_payload.get("summary")),
        "merged": _coerce_non_negative_int(response_payload.get("merged")),
        "archived": _coerce_non_negative_int(response_payload.get("archived")),
        "absorbed_observations": _coerce_non_negative_int(response_payload.get("absorbed_observations")),
        "execution_mode": "agentic_mcp",
        "tool_calls_executed": _coerce_non_negative_int(tool_payload.get("totalCalls")),
        "mutations": _count_mutating_agentic_tool_calls(tool_payload.get("byName")),
        "tool_names_used": tool_names_used,
        "seed_memory_ids": [record.id for record in seed_records],
    }


def _select_deduplicator_seed_records(candidates: list) -> list:
    if not candidates:
        return []

    largest_facts = sorted(
        [record for record in candidates if record.type == "fact"],
        key=lambda record: (
            -len(record.content.strip()),
            -record.read_count,
            record.updated_at,
        ),
    )
    prioritized_observations = sorted(
        [record for record in candidates if record.type == "observation"],
        key=lambda record: (
            -record.read_count,
            -len(record.content.strip()),
            record.updated_at,
        ),
    )
    prioritized_facts = sorted(
        [record for record in candidates if record.type == "fact"],
        key=lambda record: (
            -record.read_count,
            -len(record.content.strip()),
            record.updated_at,
        ),
    )

    seed_records: list[Any] = []
    _extend_unique_seed_records(seed_records, largest_facts, DEDUPLICATOR_SIZE_ANOMALY_SEED_RECORDS)
    _extend_unique_seed_records(seed_records, prioritized_observations, DEDUPLICATOR_OBSERVATION_SEED_RECORDS)
    _extend_unique_seed_records(seed_records, prioritized_facts, DEDUPLICATOR_MAX_SEED_RECORDS)
    return seed_records[:DEDUPLICATOR_MAX_SEED_RECORDS]


def _deduplicator_seed_payload_item(record) -> dict[str, Any]:
    summary_source = record.summary or record.content
    return {
        "id": record.id,
        "type": record.type,
        "status": record.status,
        "content_size_chars": len(record.content.strip()),
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
        "mcp_mcp-memory-internal_task_complete",
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
