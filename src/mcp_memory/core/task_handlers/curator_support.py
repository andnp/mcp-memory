from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, cast

from mcp_memory.context import ApplicationContext
from mcp_memory.core.sampling import (
    ANOMALY_STRATEGY,
    BOUNDED_NOISE_STRATEGY,
    COLD_STORAGE_STRATEGY,
    NEVER_SURFACED_STRATEGY,
    ORPHAN_LOW_SUPPORT_STRATEGY,
    SEMANTIC_STRATEGY,
    SamplingBatch,
)
from mcp_memory.core.task_handlers.campaigns import campaign_family_keys
from mcp_memory.core.task_handlers.constants import CURATOR_TASK_NAME, DEFAULT_AGENT_SCAN_LIMIT
from mcp_memory.core.task_handlers.maintenance_framework import (
    requested_sampling_strategy,
    sample_maintenance_candidates,
    support_counts_for_candidates,
)
from mcp_memory.core.tasks import TaskRecord
from mcp_memory.work_item_store import COMPATIBILITY_GROUP_STRUCTURAL_REVIEW

CURATOR_MAX_SEED_RECORDS = 16
CURATOR_SIZE_ANOMALY_SEED_RECORDS = 6
CURATOR_RECENCY_SEED_RECORDS = 4
CURATOR_CANDIDATE_POOL_MULTIPLIER = 3
CURATOR_MAX_BATCH_RECORDS = 24
CURATOR_MAX_MEMORY_CHARS = 3000
CURATOR_MAX_SUPPORT_RECORDS = 8
CURATOR_LARGEST_MEMORY_PASS_INTERVAL = 3
CURATOR_RETRIEVAL_FRICTION_SEED_RECORDS = 4
CURATOR_MAX_TITLE_CHARS = 80
CURATOR_MAX_SUMMARY_CHARS = 220
CURATOR_MAX_TAGS = 6
CURATOR_LOW_READ_REVIEW_THRESHOLD = 3
CURATOR_THIN_SPLIT_CHILD_MAX_CHARS = 800
CURATOR_ALLOWED_STRATEGIES = (
    SEMANTIC_STRATEGY,
    ANOMALY_STRATEGY,
    COLD_STORAGE_STRATEGY,
    NEVER_SURFACED_STRATEGY,
    ORPHAN_LOW_SUPPORT_STRATEGY,
    BOUNDED_NOISE_STRATEGY,
)
CURATOR_STRATEGY_WEIGHTS = {
    SEMANTIC_STRATEGY: 2,
    ANOMALY_STRATEGY: 3,
    COLD_STORAGE_STRATEGY: 2,
    NEVER_SURFACED_STRATEGY: 2,
    ORPHAN_LOW_SUPPORT_STRATEGY: 2,
    BOUNDED_NOISE_STRATEGY: 1,
}

CURATOR_SIZE_BAND_TARGET = "target"
CURATOR_SIZE_BAND_ACCEPTABLE = "acceptable"
CURATOR_SIZE_BAND_OVERSIZED = "oversized"
CURATOR_UNSUPPORTED_NO_TOOL_MUTATION_SUMMARY = (
    "Provider claimed maintenance actions without MCP tool execution; "
    "no curator maintenance actions were executed."
)
_CURATOR_MUTATION_SUMMARY_PATTERNS = (
    re.compile(r"\bsplit(?:ting)?\b"),
    re.compile(r"\bmerg(?:e|ed|ing)\b"),
    re.compile(r"\barchiv(?:e|ed|ing)\b"),
    re.compile(r"\bdelet(?:e|ed|ing)\b"),
    re.compile(r"\brewrot(?:e|ten)\b|\brewrit(?:e|ing)\b"),
    re.compile(r"\bretitl(?:e|ed|ing)\b|\bresummar(?:ize|ized|izing)\b|\bretagg(?:ed|ing)\b"),
    re.compile(r"\bclean(?:ed|ing)?(?: up)? tags?\b"),
    re.compile(r"\bimprov(?:e|ed|ing)\s+(?:the\s+)?(?:summary|summaries|title|titles|tag|tags)\b"),
    re.compile(r"\bupdat(?:e|ed|ing)\s+(?:the\s+)?(?:summary|summaries|title|titles|tag|tags)\b"),
    re.compile(r"\bnormaliz(?:e|ed|ing)\s+(?:the\s+)?tags?\b"),
)
_CURATOR_NON_MUTATING_SUMMARY_FRAGMENTS = (
    "no-op",
    "no op",
    "no change",
    "no changes",
    "no mutation",
    "no mutations",
    "no maintenance actions",
    "did not ",
    "didn't ",
    "without changes",
    "without mutation",
    "without mutations",
    "declined",
)


@dataclass(frozen=True)
class CuratorSizePolicy:
    target_max_chars: int
    acceptable_max_chars: int
    split_threshold_chars: int


CURATOR_SIZE_POLICY = CuratorSizePolicy(
    target_max_chars=1600,
    acceptable_max_chars=CURATOR_MAX_MEMORY_CHARS,
    split_threshold_chars=CURATOR_MAX_MEMORY_CHARS,
)


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
    if tool_calls_executed <= 0 and curator_summary_claims_mutating_actions(summary):
        return CURATOR_UNSUPPORTED_NO_TOOL_MUTATION_SUMMARY
    return summary


def normalize_curator_agentic_result(response: Any) -> dict[str, Any]:
    parsed = response.parsed if isinstance(getattr(response, "parsed", None), dict) else {}
    tool_stats: dict[str, Any] = {}
    tool_payload: dict[str, Any] = {}
    tool_counts: dict[str, Any] = {}
    raw_tool_stats = parsed.get("stats") if isinstance(parsed, dict) else None
    if isinstance(raw_tool_stats, dict):
        tool_stats = raw_tool_stats
        raw_tool_payload = tool_stats.get("tools")
        if isinstance(raw_tool_payload, dict):
            tool_payload = raw_tool_payload
            raw_tool_counts = tool_payload.get("byName")
            if isinstance(raw_tool_counts, dict):
                tool_counts = raw_tool_counts
    tool_names_used = _extract_agentic_tool_names(tool_counts)
    summary = getattr(response, "summary", None)
    tool_calls_executed = _coerce_non_negative_int(tool_payload.get("totalCalls"))
    if tool_calls_executed <= 0 and curator_summary_claims_mutating_actions(summary):
        summary = CURATOR_UNSUPPORTED_NO_TOOL_MUTATION_SUMMARY
    return {
        "summary": summary,
        "tool_calls_executed": tool_calls_executed,
        "mutations": _count_mutating_agentic_tool_calls(tool_counts),
        "tool_names_used": tool_names_used,
    }


def curator_summary_claims_mutating_actions(summary: str | None) -> bool:
    normalized = _normalize_curator_text(summary)
    if not normalized:
        return False
    if any(fragment in normalized for fragment in _CURATOR_NON_MUTATING_SUMMARY_FRAGMENTS):
        return False
    return any(pattern.search(normalized) for pattern in _CURATOR_MUTATION_SUMMARY_PATTERNS)


def unsupported_curator_no_tool_response_error(response: dict[str, Any], tool_calls_executed: int) -> str | None:
    if tool_calls_executed > 0:
        return None
    raw_summary = response.get("summary")
    summary = raw_summary.strip() if isinstance(raw_summary, str) and raw_summary.strip() else None
    if not curator_summary_claims_mutating_actions(summary):
        return None
    return (
        "Do not claim curator maintenance actions in the summary without first issuing tool_calls. "
        "If no tools were used, return a no-op summary that does not describe mutations."
    )


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

    sampling_task = TaskRecord(
        id=task.id,
        task_name=CURATOR_TASK_NAME,
        data=dict(task.data),
        workspace_id=task.workspace_id,
        status=task.status,
        priority=task.priority,
        retries_count=task.retries_count,
        max_retries=task.max_retries,
        created_at=task.created_at,
        updated_at=task.updated_at,
        available_at=task.available_at,
        claimed_at=task.claimed_at,
        started_at=task.started_at,
        completed_at=task.completed_at,
        last_error=task.last_error,
        execution_epoch=task.execution_epoch,
        subprocess_pid=task.subprocess_pid,
        active_request_id=task.active_request_id,
        cancellation_requested_at=task.cancellation_requested_at,
        cancelled_at=task.cancelled_at,
        cancellation_reason=task.cancellation_reason,
        cancelled_by=task.cancelled_by,
    )
    sampled_batch = sample_maintenance_candidates(
        ctx,
        sampling_task,
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
    retrieval_friction_candidates = sorted(
        [record for record in sampled_candidates if retrieval_friction_flags(record)],
        key=lambda record: (
            len(retrieval_friction_flags(record)),
            1 if getattr(record, "last_surfaced_at", None) else 0,
            _sort_curator_timestamp(getattr(record, "last_surfaced_at", None)),
            -record.read_count,
            -len(record.content.strip()),
            _sort_curator_timestamp(record.updated_at),
        ),
        reverse=True,
    )

    seed_records: list[Any] = []
    oversized_candidates = [record for record in largest_candidates if is_oversized_curator_memory(record)]
    anomaly_candidates = oversized_candidates
    if not anomaly_candidates and len(candidates) > CURATOR_MAX_SEED_RECORDS and should_run_curator_largest_memory_pass(task):
        anomaly_candidates = largest_candidates

    extend_unique_seed_records(seed_records, anomaly_candidates, CURATOR_SIZE_ANOMALY_SEED_RECORDS)
    extend_unique_seed_records(
        seed_records,
        retrieval_friction_candidates,
        min(limit, len(seed_records) + CURATOR_RETRIEVAL_FRICTION_SEED_RECORDS),
    )
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
        strategy_selection_mode=sampled_batch.strategy_selection_mode,
        strategy_selection_reason=sampled_batch.strategy_selection_reason,
        strategy_selection_scores=sampled_batch.strategy_selection_scores,
    )


def curator_seed_payload_item(record) -> dict[str, Any]:
    summary_source = record.summary or record.content
    retrieval_flags = retrieval_friction_flags(record)
    content_size_chars = len(record.content.strip())
    return {
        "id": record.id,
        "type": record.type,
        "status": record.status,
        "read_count": record.read_count,
        "last_surfaced_at": getattr(record, "last_surfaced_at", None),
        "content_size_chars": content_size_chars,
        "size_band": curator_size_band_for_char_count(content_size_chars),
        "oversized_for_curator": is_oversized_curator_memory(record),
        "retrieval_friction_flags": retrieval_flags,
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
    structural_families = list(campaign_family_keys(COMPATIBILITY_GROUP_STRUCTURAL_REVIEW))
    return (
        "You are the curator maintenance agent for the global memory store.\n"
        "Improve retrieval quality with justified maintenance: merge, rewrite, retag, relink, split, archive, or delete only when clearly justified.\n"
        "Prefer focused durable memories with specific titles/summaries. Treat frequently surfaced but rarely read records as retrieval-friction candidates.\n"
        f"{_curator_size_policy_prompt()}\n"
        "Your workflow is a loop, not a one-shot response.\n"
        f"1. Immediately call internal_get_next_curator_batch with task_id='{task.id}', strategy='{strategy_used}', and exclude_memory_ids=[] to fetch the next active curator batch.\n"
        "2. For each returned memory, use read/search/list plus mutation tools as needed.\n"
        "3. For the highest-risk records, you must spend some budget on adjacency discovery before concluding no-op. Use search/list/read to inspect nearby duplicates, canonicals, contradictions, taxonomy cleanup, or split candidates when records look broad, noisy, heavily linked, or frequently surfaced.\n"
        "4. Build a shortlist of concrete mutations, score each roughly for confidence and impact, and execute every safe candidate that clears the high-confidence/medium-impact bar.\n"
        "5. A local read alone is not enough for suspicious records; either mutate the nearby cluster or make an explicit no-op decision after adjacency review.\n"
        "6. After finishing that batch, call internal_get_next_curator_batch again with exclude_memory_ids covering the memories you already reviewed in this run.\n"
        "7. Keep looping until internal_get_next_curator_batch returns no records worth processing. Do not stop after a single batch.\n"
        "8. Summarize only after the loop is done. Do not report incremental results between batches.\n"
        "Treat seed memories as a starting frontier, not a hard boundary; widen only for nearby duplicates, contradictions, or oversized clusters.\n"
        f"When the current frontier is exhausted, you may continue by calling internal_get_compatible_work_batch with task_id='{task.id}', compatibility_group='structural_review', execution_lane='agentic', allowed_families={json.dumps(structural_families, separators=(",", ":"))}, and limit=1.\n"
        "For claimed memory_curation_review items, continue curator work from payload.seed_memory_ids. For claimed memory_dedup_review items, prefer internal_merge_memory_into_canonical and preserve lineage.\n"
        "For claimed conflict_review items, inspect payload.candidate_memory_ids and add symmetric CONTRADICTS links only when the conflict is clear and well-supported.\n"
        f"{guardrails}\n"
        f"Treat memories above {CURATOR_SIZE_POLICY.split_threshold_chars} characters as oversized and prefer splitting them into focused linked records. Avoid growing a memory past that size unless no reasonable split exists.\n"
        "Do not create journal or memory records for routine completion, counters, or status-only traces; use task_complete for closeout instead.\n"
        "Before stopping, check once more for adjacent worthwhile maintenance. No-op is fine only when no concrete safe cleanup remains above the high-confidence/medium-impact bar, and your final summary must make clear whether adjacency review was actually performed.\n"
        f"When your full looping pass is complete, call task_complete with task_id='{task.id}', task_name='memory-curator', and a short summary before your final JSON response.\n"
        "Return final JSON like {\"summary\":\"...\",\"actions_taken\":N}.\n\n"
        f"Seed memories (compact view):\n{json.dumps(seed_payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))}"
    )


def build_agentic_prompt(
    task: TaskRecord,
    *,
    strategy_used: str,
    seed_records: list[Any],
    guardrails: str,
) -> str:
    structural_families = list(campaign_family_keys(COMPATIBILITY_GROUP_STRUCTURAL_REVIEW))
    return (
        "You are the memory-curator maintenance agent for the global memory store.\n"
        "Use the workspace-local internal MCP maintenance tools directly to inspect and mutate memories.\n"
        "Treat this run as a structural-review campaign: the session should keep going while compatible structural work remains high-value and safe.\n"
        "Good memory anatomy: one focused durable takeaway plus enough evidence to stand alone, a title/summary that names the conclusion, and tags that make the record discoverable later.\n"
        f"{_curator_size_policy_prompt()}\n"
        "Bad memory smells from live retrieval telemetry include generic summaries, mixed-topic blobs, thin split-child fragments, repeated overlap across neighboring memories, and memories that keep getting surfaced but almost never opened.\n"
        "Your standing curator jobs are: retitle vague memories; resummarize generic memories; rewrite memories into more durable language; trim noise, filler, and unnecessary detail; retag or normalize taxonomy; split oversized or mixed-topic records; merge near-duplicates; reorganize overlapping clusters into a smaller clearer set; archive/delete low-value leftovers after preserving lineage; and relink or remove misleading edges when clearly justified.\n"
        "Tool mapping: use internal_update_memory_record for retitling, resummarizing, rewriting for durability, trimming noise, and retagging; use internal_split_memory_record for decompositions; use internal_merge_memory_into_canonical for canonicalization; use internal_archive_memory_record or internal_delete_memory_record for safe cleanup of leftovers; and use internal_create_memory_link or internal_delete_memory_link for structural edge cleanup.\n"
        "Actively look for multi-memory cleanups, not just single-record edits. If two similar memories should become one canonical memory, or three-to-five overlapping memories should become a smaller set of cleaner focused records, do that reorganization instead of merely describing it. Think in rotations in memory space: 5 noisy memories can become 3 durable ones, 3 overlapping memories can become 2 organized ones, and 1 giant blob can become several focused memories.\n"
        "Prefer mutations that improve future retrieval decisions: fewer clearer durable memories, stronger canonicals, better summaries/titles, less overlap, less filler, and cleaner neighborhood structure. Judge success at the neighborhood level, not just the single-record level. Prefer reversible low-risk cleanup when ambiguity remains.\n"
        "Your workflow is a loop, not a single batch.\n"
        f"1. Immediately call internal_get_next_curator_batch with task_id='{task.id}', strategy='{strategy_used}', and exclude_memory_ids=[] to fetch the next active curator batch.\n"
        "2. For each memory in the returned batch, use search/read/list and any other available tools to gain enough context to make high-value edits.\n"
        "3. For the highest-risk records or clusters in that batch, you must spend some budget on adjacency discovery before concluding no-op. Use search/list/read to inspect nearby duplicates, canonicals, contradictions, taxonomy cleanup opportunities, split candidates, merge candidates, or reorganization opportunities when records look broad, append-heavy, heavily linked, frequently surfaced, or otherwise noisy.\n"
        "4. Build a shortlist of concrete possible mutations from that review, including multi-memory reorganizations when appropriate, and score each one roughly for confidence and impact. Execute every safe candidate that is high-confidence and at least medium-impact instead of merely reporting it.\n"
        "5. Split, archive, merge, rewrite, relabel, resummarize, relink, canonicalize, or otherwise improve the memory base when justified. If a record is structurally acceptable but its title, summary, or wording is still weak, prefer a lightweight internal_update_memory_record rather than defaulting to no-op. A local read of the current batch alone is not enough to declare the frontier healthy when suspicious records or suspicious clusters exist.\n"
        "6. After finishing that batch, call internal_get_next_curator_batch again with exclude_memory_ids covering the memories you already reviewed in this run.\n"
        "7. Keep looping until internal_get_next_curator_batch returns no more records worth processing. You must not stop after a single batch.\n"
        "8. Only report final results after all looping work is complete. Do not emit incremental progress summaries between batches.\n"
        "Treat the fetched frontier as the active working set for this run; widen only when it implies nearby duplicates, contradictions, taxonomy cleanup, merge opportunities, or oversized clusters.\n"
        "When you successfully split or merge a cluster, do one extra neighborhood cleanup pass before moving on: remove obsolete links, tighten summaries/titles/tags on the new children or canonicals if needed, and archive/delete stale leftovers when the resulting structure is clearly better and lineage is preserved.\n"
        f"After finishing the active frontier, you may claim more compatible structural work by calling internal_get_compatible_work_batch with task_id='{task.id}', compatibility_group='structural_review', execution_lane='agentic', allowed_families={json.dumps(structural_families, separators=(",", ":"))}, and limit=1.\n"
        "For claimed memory_curation_review items, continue curator work from payload.seed_memory_ids. For claimed memory_dedup_review items, use payload.seed_memory_ids, prefer internal_merge_memory_into_canonical, and preserve SUPERSEDES lineage plus archive-before-delete cleanup.\n"
        "For claimed conflict_review items, inspect payload.candidate_memory_ids and add symmetric CONTRADICTS links only when the conflict is clear and well-supported.\n"
        "When you claim a follow-on structural work item, you must finish it yourself with internal_complete_work_item, internal_release_work_item, or internal_defer_work_item; use internal_heartbeat_work_item if you need more lease time.\n"
        "Aim for multiple coherent, high-value maintenance actions in one run when justified, with clear lineage and archive-before-delete when possible.\n"
        "Do not end the provider session merely because the initial handler-local frontier is complete if a compatible structural follow-on item is still worth doing.\n"
        f"{guardrails}\n"
        f"Treat memories above {CURATOR_SIZE_POLICY.split_threshold_chars} characters as oversized and prefer splitting them into focused linked records.\n"
        "If you split or rewrite a record, make each resulting memory self-contained enough to stand alone in search results.\n"
        "When you materially rewrite a memory and already understand it, refresh a concise summary in the same tool call.\n"
        "Do not create journal or memory records for routine completion, counters, or status-only traces; use task_complete for operational closeout instead.\n"
        "Before finishing, do one more quick search/list/read pass for any adjacent high-value maintenance opportunity, and make your final summary explicit about whether risky records received adjacency review before a no-op decision, whether any shortlisted mutations cleared the high-confidence/medium-impact bar, and which 1-3 mutation classes were considered but declined with a brief reason when relevant.\n"
        "Do not claim work you did not actually execute through MCP tools.\n"
        f"When your full looping pass is complete, call task_complete with task_id='{task.id}', task_name='memory-curator', and a short summary before your final JSON response.\n"
        "When finished, output final JSON only in the form {\"summary\": \"...\"}.\n\n"
        f"Sampling strategy: {strategy_used}\n"
        "Do not expect inline seed-memory payloads in this prompt; discover the real working set through the MCP tools."
    )


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


def select_curator_support_records(
    ctx: ApplicationContext,
    task: TaskRecord,
    seed_records: list[Any],
    *,
    support_limit: int = CURATOR_MAX_SUPPORT_RECORDS,
) -> list[Any]:
    if ctx.repository is None or not seed_records or support_limit < 1:
        return []

    seed_ids = {record.id for record in seed_records}
    seed_tags = {tag for record in seed_records for tag in record.tags}
    adjacent_ids = _seed_adjacent_memory_ids(ctx, seed_records)
    candidates = [
        record
        for record in ctx.repository.list_memories(
            workspace_id=_resolve_workspace_id(ctx, task),
            limit=int(task.data.get("limit", DEFAULT_AGENT_SCAN_LIMIT)),
        )
        if record.id not in seed_ids and not ctx.repository.has_incoming_link(record.id, "SUPERSEDES")
    ]
    ranked = sorted(
        candidates,
        key=lambda record: (
            -int(record.id in adjacent_ids),
            -len(seed_tags.intersection(record.tags)),
            0 if record.type in {"observation", "journal"} else 1,
            -len(record.content.strip()),
            -record.read_count,
            str(record.updated_at),
        ),
    )

    support_records: list[Any] = []
    for record in ranked:
        if record.id in seed_ids:
            continue
        if record.id not in adjacent_ids and not seed_tags.intersection(record.tags):
            continue
        support_records.append(record)
        if len(support_records) >= support_limit:
            break
    return support_records


def review_sampling_batch(payload: dict[str, Any], seed_records: list[Any]) -> SamplingBatch:
    requested_strategy = payload.get("strategy_used")
    if not isinstance(requested_strategy, str):
        requested_strategy = None
    candidate_count = payload.get("candidate_count")
    if not isinstance(candidate_count, int):
        candidate_count = len(seed_records)
    strategy_selection_mode = payload.get("strategy_selection_mode")
    if not isinstance(strategy_selection_mode, str):
        strategy_selection_mode = None
    strategy_selection_reason = payload.get("strategy_selection_reason")
    if not isinstance(strategy_selection_reason, str):
        strategy_selection_reason = None
    strategy_selection_scores = payload.get("strategy_selection_scores")
    if not isinstance(strategy_selection_scores, dict):
        strategy_selection_scores = None
    return SamplingBatch(
        requested_strategy=requested_strategy,
        strategy_used=requested_strategy or "none",
        strategy_fallback_reason=None,
        candidate_count=candidate_count,
        records=seed_records,
        strategy_selection_mode=strategy_selection_mode,
        strategy_selection_reason=strategy_selection_reason,
        strategy_selection_scores=strategy_selection_scores,
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
    return len(record.content.strip()) > CURATOR_SIZE_POLICY.split_threshold_chars


def curator_size_band_for_char_count(content_size_chars: int) -> str:
    if content_size_chars <= CURATOR_SIZE_POLICY.target_max_chars:
        return CURATOR_SIZE_BAND_TARGET
    if content_size_chars <= CURATOR_SIZE_POLICY.acceptable_max_chars:
        return CURATOR_SIZE_BAND_ACCEPTABLE
    return CURATOR_SIZE_BAND_OVERSIZED


def retrieval_friction_flags(record) -> list[str]:
    flags: list[str] = []
    normalized_summary = _normalize_curator_text(record.summary)
    if normalized_summary.startswith("covers ") or normalized_summary.startswith("added "):
        flags.append("generic_summary")
    if record.type == "observation" and not record.tags:
        flags.append("untagged_observation")
    if getattr(record, "last_surfaced_at", None) and record.read_count <= CURATOR_LOW_READ_REVIEW_THRESHOLD:
        flags.append("surfaced_low_read")
    if record.metadata.get("split_from_memory_id") and len(record.content.strip()) <= CURATOR_THIN_SPLIT_CHILD_MAX_CHARS:
        flags.append("thin_split_child")
    if is_oversized_curator_memory(record):
        flags.append("oversized_blob")
    return flags


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


def _normalize_curator_text(value: str | None) -> str:
    return " ".join((value or "").strip().lower().split())


def _curator_size_policy_prompt() -> str:
    target_max = CURATOR_SIZE_POLICY.target_max_chars
    split_threshold = CURATOR_SIZE_POLICY.split_threshold_chars
    return (
        f"Size policy: target band is {target_max} chars or less; acceptable band is {target_max + 1}-{split_threshold} chars; oversized is anything above {split_threshold} chars. "
        "Leave alone when a memory already has one focused durable takeaway plus enough evidence to stand alone. "
        "Rewrite or trim when it is still one takeaway but the acceptable band carries filler, drift, or avoidable detail. "
        "Split when it crosses the oversized threshold or carries multiple takeaways. "
        "Merge when a memory is too thin to stand alone or mostly duplicates a nearby canonical."
    )


def _requested_sampling_strategy(task: TaskRecord) -> str | None:
    return requested_sampling_strategy(task)


def _resolve_workspace_id(ctx: ApplicationContext, task: TaskRecord) -> str | None:
    task_workspace = task.data.get("workspace_id")
    if isinstance(task_workspace, str) and task_workspace.strip():
        return task_workspace.strip()
    return None


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


def _seed_adjacent_memory_ids(ctx: ApplicationContext, seed_records: list[Any]) -> set[str]:
    if ctx.repository is None:
        return set()
    adjacent_ids: set[str] = set()
    for record in seed_records:
        outgoing = ctx.repository.get_links(record.id, direction="outgoing")
        incoming = ctx.repository.get_links(record.id, direction="incoming")
        adjacent_ids.update(link.target_id for link in outgoing if isinstance(link.target_id, str))
        adjacent_ids.update(link.source_id for link in incoming if isinstance(link.source_id, str))
    return adjacent_ids


def _coerce_non_negative_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(value, 0)
    return 0


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
        "mcp_mcp-memory-internal_internal_get_compatible_work_batch",
        "mcp_mcp-memory-internal_internal_read_memory_record",
        "mcp_mcp-memory-internal_internal_search_memory_records",
        "mcp_mcp-memory-internal_internal_list_memory_records",
        "mcp_mcp-memory-internal_internal_task_complete",
    }
    total = 0
    for name, payload in value.items():
        if not isinstance(name, str) or name in read_only_tool_names or not isinstance(payload, dict):
            continue
        payload_dict = cast(dict[str, Any], payload)
        total += _coerce_non_negative_int(payload_dict.get("count"))
    return total
