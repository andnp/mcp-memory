from __future__ import annotations

import json
import re
from typing import Any

from mcp_memory.context import ApplicationContext
from mcp_memory.core.sampling import (
    ANOMALY_STRATEGY,
    COOLDOWN_ESCAPE_STRATEGY,
    SEMANTIC_STRATEGY,
    SamplingBatch,
)
from mcp_memory.core.task_handlers.agentic_guardrails import build_deduplicator_guardrails
from mcp_memory.core.task_handlers.agentic_result_support import (
    build_tool_usage_summary,
    coerce_non_negative_int,
    coerce_text_summary,
    count_mutating_agentic_tool_calls,
    extract_agentic_tool_names,
    extract_embedded_json_object,
)
from mcp_memory.core.task_handlers.constants import DEFAULT_AGENT_SCAN_LIMIT, DEDUPLICATOR_TASK_NAME
from mcp_memory.core.task_handlers.maintenance_framework import (
    requested_sampling_strategy,
    sample_maintenance_candidates,
    sampling_payload,
    support_counts_for_candidates,
)
from mcp_memory.core.task_handlers.maintenance_housekeeping import _resolve_workspace_id
from mcp_memory.core.task_handlers.maintenance_work_items import (
    enqueue_review_work_item,
    work_item_result_metadata,
)
from mcp_memory.core.ports.tasks import TaskRecord
from mcp_memory.core.ports.work_items import (
    EXECUTION_LANE_AGENTIC,
    WORK_FAMILY_MEMORY_DEDUP_REVIEW,
)
from searchkernel.ingestion import embed_in_batches
from searchkernel.utils.similarity import cosine_similarity_lists

# ---------------------------------------------------------------------------
# deduplicator_merge: deterministic merging logic
# ---------------------------------------------------------------------------

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_:-]+")
FACT_DEDUPLICATION_THRESHOLD = 0.72
OBSERVATION_ABSORPTION_THRESHOLD = 0.62
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


async def run_deterministic_deduplicator_pass(
    ctx: ApplicationContext,
    task: TaskRecord,
    candidates: list,
    seed_records: list,
) -> dict[str, int]:
    facts = [record for record in seed_records if record.type == "fact"]
    observations = [record for record in seed_records if record.type == "observation"]
    if not facts:
        return {"merged": 0, "archived": 0, "absorbed_observations": 0}

    active_facts = [record for record in candidates if record.type == "fact"]
    embedding_by_id = _embed_records(ctx, [*active_facts, *observations])
    merged = 0
    archived = 0
    absorbed_observations = 0
    claimed_sources: set[str] = set()

    for group in _collect_similar_fact_groups(facts, embedding_by_id):
        canonical = _choose_canonical_fact(ctx, group)
        for source in group:
            if source.id == canonical.id or source.id in claimed_sources:
                continue
            canonical = await _merge_into_canonical_fact(ctx, canonical, source, task)
            claimed_sources.add(source.id)
            merged += 1
            archived += 1

    active_facts_by_id = {record.id: record for record in active_facts}
    for observation in observations:
        if observation.id in claimed_sources:
            continue
        target = _find_best_fact_target(observation, active_facts, embedding_by_id)
        if target is None:
            continue
        refreshed = active_facts_by_id.get(target.id, target)
        merged_target = await _merge_into_canonical_fact(ctx, refreshed, observation, task)
        active_facts_by_id[merged_target.id] = merged_target
        active_facts = list(active_facts_by_id.values())
        claimed_sources.add(observation.id)
        absorbed_observations += 1
        archived += 1

    return {
        "merged": merged,
        "archived": archived,
        "absorbed_observations": absorbed_observations,
    }


def _embed_records(ctx: ApplicationContext, records: list) -> dict[str, list[float]]:
    embedder = getattr(ctx, "embedder", None)
    if embedder is None:
        return {}
    payloads = [_record_embedding_text(record) for record in records]
    embeddings = embed_in_batches(
        payloads,
        provider=embedder,
        batch_size=max(len(records), 1),
    )
    return {
        record.id: embedding
        for record, embedding in zip(records, embeddings, strict=True)
    }


def _collect_similar_fact_groups(facts: list, embedding_by_id: dict[str, list[float]]) -> list[list]:
    groups: list[list] = []
    used: set[str] = set()
    for fact in facts:
        if fact.id in used:
            continue
        group = [fact]
        used.add(fact.id)
        for other in facts:
            if other.id in used or other.id == fact.id:
                continue
            if _memory_similarity(fact, other, embedding_by_id) >= FACT_DEDUPLICATION_THRESHOLD:
                group.append(other)
                used.add(other.id)
        if len(group) >= 2:
            groups.append(group)
    return groups


def _choose_canonical_fact(ctx: ApplicationContext, facts: list):
    assert ctx.repository is not None
    repository = ctx.repository
    return max(
        facts,
        key=lambda record: (
            repository.count_incoming_links(record.id),
            record.access_score,
            record.updated_at,
            record.created_at,
        ),
    )


def _find_best_fact_target(observation, facts: list, embedding_by_id: dict[str, list[float]]):
    best_score = 0.0
    best_target = None
    for fact in facts:
        score = _memory_similarity(observation, fact, embedding_by_id)
        if score >= OBSERVATION_ABSORPTION_THRESHOLD and score > best_score:
            best_score = score
            best_target = fact
    return best_target


async def _merge_into_canonical_fact(
    ctx: ApplicationContext,
    canonical,
    source,
    task: TaskRecord,
):
    assert ctx.repository is not None
    merged_title, merged_content = await _build_merged_fact_content(canonical, source)
    merged_tags = _normalize_tag_values([*canonical.tags, *source.tags])
    merged_metadata = dict(canonical.metadata)
    merged_source_ids = merged_metadata.get("merged_source_ids", [])
    if not isinstance(merged_source_ids, list):
        merged_source_ids = []
    merged_metadata["merged_source_ids"] = sorted({*map(str, merged_source_ids), source.id})
    merged_metadata["deduplicator_task_id"] = task.id
    updated = ctx.repository.update_memory(
        canonical.id,
        title=merged_title,
        content=merged_content,
        tags=merged_tags,
        metadata=merged_metadata,
        workspace_ids=sorted({*canonical.workspace_ids, *source.workspace_ids}),
    )
    if updated is None:
        return canonical
    ctx.repository.add_link(updated.id, source.id, "SUPERSEDES", "Auto-merged into canonical fact memory.")
    ctx.repository.update_memory(source.id, status="archived")
    return updated


async def _build_merged_fact_content(canonical, source) -> tuple[str, str]:
    if source.content.strip() in canonical.content:
        return canonical.title, canonical.content

    merged_lines = [canonical.content.strip()]
    addition = source.summary or source.content.strip()
    if addition and addition not in canonical.content:
        merged_lines.append(f"Merged from {source.title}:\n{addition}")
    return canonical.title, "\n\n".join(part for part in merged_lines if part)


def _memory_similarity(left, right, embedding_by_id: dict[str, list[float]]) -> float:
    shared_tags = _shared_meaningful_tags(left.tags, right.tags)
    lexical_similarity = _topic_token_overlap(left.title + " " + left.content, right.title + " " + right.content)
    semantic_similarity = 0.0
    if left.id in embedding_by_id and right.id in embedding_by_id:
        semantic_similarity = cosine_similarity_lists(embedding_by_id[left.id], embedding_by_id[right.id])
    if not shared_tags and lexical_similarity < 0.12:
        semantic_similarity = 0.0
    tag_bonus = 0.15 if shared_tags else 0.0
    return max(lexical_similarity, semantic_similarity + tag_bonus)


def _record_embedding_text(record) -> str:
    parts = [record.title, record.summary or "", record.content]
    if record.tags:
        parts.append("tags: " + ", ".join(record.tags))
    parts.append(f"type: {record.type}")
    return "\n".join(part for part in parts if part)


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

# ---------------------------------------------------------------------------
# deduplicator_support: seed selection, prompt building, result normalization
# ---------------------------------------------------------------------------

DEDUPLICATOR_AI_MIN_COMBINED_LINES = 20
DEDUPLICATOR_HIGH_OVERLAP_THRESHOLD = 0.75
DEDUPLICATOR_MAX_SEED_RECORDS = 8
DEDUPLICATOR_MAX_SUPPORT_RECORDS = 4
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

_DEDUPLICATOR_READ_ONLY_TOOL_NAMES = {
    "mcp_mcp-memory-internal_task_complete",
    "mcp_mcp-memory-internal_internal_get_next_dedup_batch",
    "mcp_mcp-memory-internal_internal_read_memory_record",
    "mcp_mcp-memory-internal_internal_search_memory_records",
    "mcp_mcp-memory-internal_internal_list_memory_records",
    "mcp_mcp-memory-internal_internal_task_complete",
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
        strategy_selection_mode=sampled_batch.strategy_selection_mode,
        strategy_selection_reason=sampled_batch.strategy_selection_reason,
        strategy_selection_scores=sampled_batch.strategy_selection_scores,
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
        "Do not call record_thought or create journal/observation memories for routine completion, counters, or status traces; use task_complete for operational closeout only.\n"
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
    return {
        "summary": _coerce_text_summary(getattr(agentic_result, "summary", None)) or _coerce_text_summary(response_payload.get("summary")),
        "merged": _coerce_non_negative_int(response_payload.get("merged")),
        "archived": _coerce_non_negative_int(response_payload.get("archived")),
        "absorbed_observations": _coerce_non_negative_int(response_payload.get("absorbed_observations")),
        "execution_mode": "agentic_mcp",
        "seed_memory_ids": [record.id for record in seed_records],
    } | build_tool_usage_summary(parsed, read_only_tool_names=_DEDUPLICATOR_READ_ONLY_TOOL_NAMES)


def review_seed_records(ctx: ApplicationContext, payload: dict[str, Any]) -> list[Any]:
    if ctx.repository is None:
        return []
    ordered_ids = _ordered_packet_memory_ids(payload)
    if not ordered_ids:
        return []
    records: list[Any] = []
    for memory_id in ordered_ids:
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


def review_strategy(payload: dict[str, Any]) -> str:
    strategy = payload.get("strategy_used")
    if isinstance(strategy, str) and strategy.strip():
        return strategy
    return "none"


def select_deduplicator_support_records(seed_records: list[Any], candidates: list[Any]) -> list[Any]:
    if not seed_records:
        return []
    seed_ids = {record.id for record in seed_records}
    seed_tags = {tag for record in seed_records for tag in record.tags}
    ranked = sorted(
        [record for record in candidates if record.id not in seed_ids],
        key=lambda record: (
            0 if record.type == "observation" else 1,
            -len(seed_tags.intersection(record.tags)),
            -record.read_count,
            -len(record.content.strip()),
            str(record.updated_at),
        ),
    )
    support_records: list[Any] = []
    for record in ranked:
        if not seed_tags.intersection(record.tags):
            continue
        support_records.append(record)
        if len(support_records) >= DEDUPLICATOR_MAX_SUPPORT_RECORDS:
            break
    return support_records


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
    return coerce_non_negative_int(value)


def _coerce_text_summary(value: object) -> str | None:
    return coerce_text_summary(value)


def _extract_embedded_json_object(text: str) -> dict[str, Any] | None:
    return extract_embedded_json_object(text)


def _extract_agentic_tool_names(value: object) -> list[str]:
    return extract_agentic_tool_names(value)


def _count_mutating_agentic_tool_calls(value: object) -> int:
    return count_mutating_agentic_tool_calls(value, read_only_tool_names=_DEDUPLICATOR_READ_ONLY_TOOL_NAMES)


def _ordered_packet_memory_ids(payload: dict[str, Any]) -> list[str]:
    ordered_ids: list[str] = []
    for key in ("seed_memory_ids", "support_memory_ids"):
        memory_ids = payload.get(key)
        if not isinstance(memory_ids, list):
            continue
        for memory_id in memory_ids:
            if not isinstance(memory_id, str) or memory_id in ordered_ids:
                continue
            ordered_ids.append(memory_id)
    return ordered_ids

# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

async def handle_deduplicator_task(
    ctx: ApplicationContext,
    task: TaskRecord,
    provider: Any = None,
) -> dict[str, Any]:
    if ctx.repository is None:
        return {"merged": 0, "archived": 0, "absorbed_observations": 0}
    del provider

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
    seed_batch = select_deduplicator_seed_batch(
        ctx,
        candidates,
        task_id=task.id,
        strategy=requested_sampling_strategy(task),
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

    deterministic_result = await run_deterministic_deduplicator_pass(
        ctx,
        task,
        candidates,
        seed_records,
    )

    if _deduplicator_result_has_effective_change(deterministic_result):
        return sampling_payload(
            seed_batch,
            sampled_records=seed_records,
            seed_records=seed_records,
            merged=deterministic_result["merged"],
            archived=deterministic_result["archived"],
            absorbed_observations=deterministic_result["absorbed_observations"],
        )

    seeded_review = _seed_dedup_review_from_seed_batch(
        ctx,
        task=task,
        workspace_id=workspace_id,
        seed_batch=seed_batch,
        seed_records=seed_records,
        candidates=candidates,
    )

    return sampling_payload(
        seed_batch,
        sampled_records=seed_records,
        seed_records=seeded_review["packet_records"],
        extra=seeded_review["metadata"],
        merged=deterministic_result["merged"],
        archived=deterministic_result["archived"],
        absorbed_observations=deterministic_result["absorbed_observations"],
        seeded_work_item_count=seeded_review["seeded_work_item_count"],
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
    support_records = select_deduplicator_support_records(seed_records, candidates)
    return enqueue_review_work_item(
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


def _deduplicator_result_has_effective_change(result: dict[str, Any]) -> bool:
    return any(
        isinstance(result.get(metric), int) and result.get(metric, 0) > 0
        for metric in ("merged", "archived", "absorbed_observations")
    )


def _seed_dedup_review_from_seed_batch(
    ctx: ApplicationContext,
    *,
    task: TaskRecord,
    workspace_id: str | None,
    seed_batch: Any,
    seed_records: list[Any],
    candidates: list[Any],
) -> dict[str, Any]:
    created_work_item, created = _enqueue_dedup_review_work_item(
        ctx,
        task=task,
        workspace_id=workspace_id,
        seed_records=seed_records,
        candidates=candidates,
        strategy_used=seed_batch.strategy_used,
        candidate_count=seed_batch.candidate_count,
    )
    packet_records = seed_records + select_deduplicator_support_records(seed_records, candidates)
    seeded_work_item_count = 1 if created else 0
    return {
        "packet_records": packet_records,
        "seeded_work_item_count": seeded_work_item_count,
        "metadata": work_item_result_metadata(
            family_key=WORK_FAMILY_MEMORY_DEDUP_REVIEW,
            execution_lane=EXECUTION_LANE_AGENTIC,
            seed_source="frontier_seed",
            seed_records=packet_records,
            created_work_item=created_work_item if created else None,
        ),
    }