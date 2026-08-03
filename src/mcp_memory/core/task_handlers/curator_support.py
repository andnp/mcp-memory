from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import re
from typing import Any, cast
from uuid import UUID

from mcp_memory.context import ApplicationContext
from mcp_memory.core.curation_candidates import CuratorCandidateRequest, CuratorSamplingContext
from mcp_memory.core.curation_identity import candidate_revision_token, graph_token, record_token
from mcp_memory.core.curation_models import (
    CampaignHypothesis,
    CampaignRetrievalProblem,
    campaign_hypothesis_from_payload,
)
from mcp_memory.core.ports.curation import CandidateDisposition, CurationCandidateState
from mcp_memory.core.sampling import (
    ANOMALY_STRATEGY,
    BOUNDED_NOISE_STRATEGY,
    COLD_STORAGE_STRATEGY,
    NEVER_SURFACED_STRATEGY,
    ORPHAN_LOW_SUPPORT_STRATEGY,
    QUALITY_SIGNAL_STRATEGY,
    SEMANTIC_STRATEGY,
    SamplingBatch,
)

from mcp_memory.core.task_handlers.constants import DEFAULT_AGENT_SCAN_LIMIT
from mcp_memory.core.task_handlers.maintenance_framework import (
    requested_sampling_strategy,
    sample_maintenance_candidates,
    support_counts_for_candidates,
)
from mcp_memory.core.ports.tasks import TaskRecord
from mcp_memory.integrations.memory_retrieval import build_memory_retrieval_facade


def _sampling_task_id(task: Any) -> str:
    return task.task_id if hasattr(task, "task_id") else task.id

CURATOR_MAX_SEED_RECORDS = 16
CURATOR_SIZE_ANOMALY_SEED_RECORDS = 6
CURATOR_RECENCY_SEED_RECORDS = 4
CURATOR_CANDIDATE_POOL_MULTIPLIER = 3
CURATOR_MAX_BATCH_RECORDS = 24
CURATOR_MAX_MEMORY_CHARS = 3000
CURATOR_MAX_SUPPORT_RECORDS = 8
CURATOR_MAX_QUALITY_FEEDBACK_SEEDS = 1
CURATOR_LARGEST_MEMORY_PASS_INTERVAL = 3
CURATOR_STABILIZATION_WINDOW_SECONDS = 3600.0
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
    QUALITY_SIGNAL_STRATEGY,
)
_CURATOR_BACKEND_STRATEGIES = (
    "retrieval-quality",
    "oversized/thin",
    COLD_STORAGE_STRATEGY,
    NEVER_SURFACED_STRATEGY,
    "orphan/low-support",
    QUALITY_SIGNAL_STRATEGY,
    "seeded-random",
)

CURATOR_SIZE_BAND_TARGET = "target"
CURATOR_SIZE_BAND_ACCEPTABLE = "acceptable"
CURATOR_SIZE_BAND_OVERSIZED = "oversized"
_CURATOR_MUTATION_SUMMARY_PATTERNS = (
    re.compile(r"\bsplit(?:ting)?\b"),
    re.compile(r"\bmerg(?:e|ed|ing)\b"),
    re.compile(r"\barchiv(?:e|ed|ing)\b"),
    re.compile(r"\bdelet(?:e|ed|ing)\b"),
    re.compile(r"\b(?:add(?:ed|ing)?|creat(?:e|ed|ing)|remov(?:e|ed|ing)|link(?:ed|ing)|restor(?:e|ed|ing))\b"),
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
_CURATOR_QUALITY_FEEDBACK_REASONS = frozenset(
    {"retrieval_regression", "zero_result_regression", "acceptance_not_met"}
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



def curator_summary_claims_mutating_actions(summary: str | None) -> bool:
    normalized = _normalize_curator_text(summary)
    if not normalized:
        return False
    for clause in re.split(r"[.!?;]+|\b(?:but|however)\b", normalized):
        clause = clause.strip()
        if not clause or clause.startswith(("no ", "none ", "without ")):
            continue
        if any(fragment in clause for fragment in _CURATOR_NON_MUTATING_SUMMARY_FRAGMENTS):
            continue
        if any(pattern.search(clause) for pattern in _CURATOR_MUTATION_SUMMARY_PATTERNS):
            return True
    return False



def select_curator_seed_records(ctx: ApplicationContext, task: TaskRecord) -> list:
    return select_curator_seed_batch(ctx, task).records


def _acquire_curator_candidates(
    ctx: ApplicationContext,
    request: CuratorCandidateRequest,
) -> SamplingBatch:
    return _select_curator_seed_batch(
        ctx,
        CuratorSamplingContext(request.task_id, request.requested_strategy, request.workspace_id),
        seed_limit=request.limit,
        exclude_memory_ids=set(request.exclude_memory_ids),
        backend_query_limit=request.limit,
        campaign_hypothesis=request.campaign_hypothesis,
    )


def acquire_curator_candidates(
    ctx: ApplicationContext,
    request: CuratorCandidateRequest,
) -> SamplingBatch:
    """Compatibility import for the application candidate boundary."""
    return _acquire_curator_candidates(ctx, request)


def select_curator_seed_batch(
    ctx: ApplicationContext,
    task: Any,
    *,
    seed_limit: int | None = None,
    exclude_memory_ids: set[str] | None = None,
) -> SamplingBatch:
    return _select_curator_seed_batch(
        ctx,
        CuratorSamplingContext(task.id, _requested_sampling_strategy(task), task.workspace_id),
        seed_limit=seed_limit,
        exclude_memory_ids=exclude_memory_ids,
        backend_query_limit=int(task.data["limit"]) if "limit" in task.data else None,
        campaign_hypothesis=campaign_hypothesis_from_payload(task.data),
    )


def _select_curator_seed_batch(
    ctx: ApplicationContext,
    task: Any,
    *,
    seed_limit: int | None = None,
    exclude_memory_ids: set[str] | None = None,
    backend_query_limit: int | None = None,
    campaign_hypothesis: CampaignHypothesis | None = None,
) -> SamplingBatch:
    assert ctx.repository is not None
    limit = _normalize_curator_seed_limit(seed_limit)
    excluded_ids = exclude_memory_ids or set()
    excluded_ids = exclude_memory_ids or set()
    backend_candidates = _query_curator_backend_candidates(
        ctx,
        task,
        frontier_limit=limit + len(excluded_ids),
        configured_limit=backend_query_limit,
        expand_for_exclusions=backend_query_limit is not None and not hasattr(task, "data"),
    )
    explicit_hypothesis = (
        campaign_hypothesis if _is_explicit_campaign_hypothesis(campaign_hypothesis) else None
    )
    goal_candidates: list[Any] = []
    if explicit_hypothesis is not None:
        goal_candidates = _query_campaign_goal_candidates(
            ctx,
            explicit_hypothesis,
            backend_candidates,
            limit=min(
                CURATOR_MAX_BATCH_RECORDS,
                max(limit, CURATOR_MAX_SEED_RECORDS) * CURATOR_CANDIDATE_POOL_MULTIPLIER,
            ),
        )
        candidates = [
            record
            for record in (goal_candidates + backend_candidates)
            if record.id not in excluded_ids
        ]
        candidates = list({record.id: record for record in candidates}.values())
    else:
        candidates = [record for record in backend_candidates if record.id not in excluded_ids]
    if not candidates:
        return SamplingBatch(
            requested_strategy=requested_sampling_strategy(task),
            strategy_used=requested_sampling_strategy(task) or "none",
            strategy_fallback_reason=None,
            candidate_count=0,
            records=[],
        )

    goal_ids = {record.id for record in goal_candidates}
    sampled_batch = sample_maintenance_candidates(
        ctx,
        task,
        [record for record in candidates if record.id not in goal_ids],
        allowed_strategies=CURATOR_ALLOWED_STRATEGIES,
        limit=min(len(candidates), max(limit, CURATOR_MAX_SEED_RECORDS) * CURATOR_CANDIDATE_POOL_MULTIPLIER),
        support_counts=build_support_counts(ctx, candidates),
    )
    sampled_candidates = sampled_batch.records
    quality_feedback_candidates = _quality_feedback_candidates(ctx, candidates)

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
    if explicit_hypothesis is not None:
        extend_unique_seed_records(seed_records, goal_candidates, limit)
    extend_unique_seed_records(
        seed_records,
        quality_feedback_candidates,
        min(limit, CURATOR_MAX_QUALITY_FEEDBACK_SEEDS),
    )
    oversized_candidates = [record for record in largest_candidates if is_oversized_curator_memory(record)]
    anomaly_candidates = oversized_candidates
    if (
        not anomaly_candidates
        and len(candidates) > CURATOR_MAX_SEED_RECORDS
        and sum(_sampling_task_id(task).encode("utf-8")) % CURATOR_LARGEST_MEMORY_PASS_INTERVAL == 0
    ):
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
        candidate_count=len(candidates) if explicit_hypothesis is not None else sampled_batch.candidate_count,
        records=seed_records[:limit],
        strategy_selection_mode=sampled_batch.strategy_selection_mode,
        strategy_selection_reason=sampled_batch.strategy_selection_reason,
        strategy_selection_scores=sampled_batch.strategy_selection_scores,
    )


def curator_seed_payload_item(
    record,
    *,
    selection_reason: str | None = None,
    selection_signals: dict[str, float] | None = None,
    selection_scores: dict[str, float] | None = None,
    quality_feedback: dict[str, Any] | None = None,
) -> dict[str, Any]:
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
        "selection_reason": selection_reason,
        "selection_signals": dict(selection_signals or {}),
        "selection_scores": dict(selection_scores or {}),
        "quality_feedback": dict(quality_feedback or {}),
        "title": truncate_text(record.title, CURATOR_MAX_TITLE_CHARS),
        "summary": truncate_text(summary_source, CURATOR_MAX_SUMMARY_CHARS),
        "tags": list(record.tags[:CURATOR_MAX_TAGS]),
    }



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
    candidate_by_id = {
        record.id: record
        for record in _query_curator_backend_candidates(
            ctx,
            task,
            frontier_limit=max(support_limit, CURATOR_MAX_SUPPORT_RECORDS),
        )
    }
    for memory_id in adjacent_ids:
        record = ctx.repository.get_memory(memory_id)
        if record is not None:
            candidate_by_id[record.id] = record
    candidates = [
        record
        for record in candidate_by_id.values()
        if record.id not in seed_ids and not ctx.repository.has_incoming_link(record.id, "SUPERSEDES")
    ]
    candidates = filter_curator_candidates(ctx, candidates)
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
    normalized_summary = _normalize_curator_text(getattr(record, "summary", None))
    if normalized_summary.startswith(("covers ", "added ")):
        flags.append("generic_summary")
    if record.type == "observation" and not record.tags:
        flags.append("untagged_observation")
    if getattr(record, "last_surfaced_at", None) and record.read_count <= CURATOR_LOW_READ_REVIEW_THRESHOLD:
        flags.append("surfaced_low_read")
    metadata = getattr(record, "metadata", {})
    if isinstance(metadata, dict) and metadata.get("split_from_memory_id") and len(record.content.strip()) <= CURATOR_THIN_SPLIT_CHILD_MAX_CHARS:
        flags.append("thin_split_child")
    if is_oversized_curator_memory(record):
        flags.append("oversized_blob")
    return flags


def should_run_curator_largest_memory_pass(task: TaskRecord) -> bool:
    return sum(task.id.encode("utf-8")) % CURATOR_LARGEST_MEMORY_PASS_INTERVAL == 0


def build_support_counts(ctx: ApplicationContext, candidates: list) -> dict[str, int]:
    return support_counts_for_candidates(ctx, candidates)


def filter_curator_candidates(
    ctx: ApplicationContext,
    candidates: list[Any],
    *,
    now: datetime | None = None,
    suppress_destructive_actions: bool = True,
    preserve_memory_ids: set[str] | frozenset[str] = frozenset(),
) -> list[Any]:
    """Apply persisted no-op cooldown and edit stabilization to a candidate pool."""
    curation = getattr(ctx, "curation", None)
    get_state = getattr(curation, "get_candidate_state", None)
    put_state = getattr(curation, "put_candidate_state", None)
    if not callable(get_state):
        return candidates

    current_time = now or datetime.now(UTC)
    eligible: list[Any] = []
    for record in candidates:
        memory_id = _candidate_uuid(record.id)
        if memory_id is None:
            eligible.append(record)
            continue
        identity = curator_candidate_revision_token(ctx, record)
        state = cast(CurationCandidateState | None, get_state(memory_id))
        if state is not None and state.last_observed_revision_token != identity:
            if (
                callable(put_state)
                and state.disposition is CandidateDisposition.ESCALATED
                and state.last_disposition_reason in _CURATOR_QUALITY_FEEDBACK_REASONS
            ):
                put_state(
                    state.model_copy(
                        update={
                            "last_observed_revision_token": identity,
                            "cooldown_until": None,
                            "consecutive_no_op_count": 0,
                        }
                    )
                )
            elif callable(put_state):
                put_state(
                    state.model_copy(
                        update={
                            "last_observed_revision_token": identity,
                            "disposition": CandidateDisposition.PENDING,
                            "consecutive_no_op_count": 0,
                            "cooldown_until": None,
                            "last_disposition_reason": "candidate_revision_or_adjacency_changed",
                        }
                    )
                )
            state = None
        if record.id in preserve_memory_ids:
            eligible.append(record)
            continue
        if state is not None and state.cooldown_until is not None and state.cooldown_until > current_time:
            continue
        if suppress_destructive_actions and _has_recent_stabilizing_edit(ctx, memory_id, current_time):
            continue
        eligible.append(record)
    return eligible


def curator_candidate_revision_token(ctx: ApplicationContext, record: Any) -> str:
    """Return the stable semantic-plus-adjacency identity used by candidate state."""
    edges = _candidate_edges(ctx, record.id)
    return candidate_revision_token(record_token(record), graph_token(record.id, edges))


def _candidate_edges(ctx: ApplicationContext, memory_id: str) -> list[dict[str, Any]]:
    repository = getattr(ctx, "repository", None)
    get_links = getattr(repository, "get_links", None)
    if not callable(get_links):
        return []
    links = [
        *cast(list[Any], get_links(memory_id, direction="outgoing")),
        *cast(list[Any], get_links(memory_id, direction="incoming")),
    ]
    return [
        {
            "source_id": link.source_id,
            "target_id": link.target_id,
            "type": getattr(link, "link_type", getattr(link, "type", "")),
            "context": getattr(link, "context", None),
        }
        for link in links
    ]


def _candidate_uuid(memory_id: Any) -> UUID | None:
    try:
        return UUID(str(memory_id))
    except (TypeError, ValueError):
        return None


def _has_recent_stabilizing_edit(ctx: ApplicationContext, memory_id: UUID, now: datetime) -> bool:
    history = getattr(ctx, "mutation_history", None)
    list_events = getattr(history, "list_events", None)
    if not callable(list_events):
        return False
    try:
        events = cast(list[Any], list_events(memory_id=memory_id, limit=50))
    except Exception:
        return False
    for event in events:
        created_at = getattr(event, "created_at", None)
        if created_at is None:
            continue
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=UTC)
        age = (now - created_at).total_seconds()
        family = getattr(event, "family", None)
        family_name = "" if family is None else str(family)
        if 0 <= age <= CURATOR_STABILIZATION_WINDOW_SECONDS and (
            str(getattr(event, "actor_kind", "")) == "user" or family_name not in {"", "curator"}
        ):
            return True
    return False


def _query_curator_backend_candidates(
    ctx: ApplicationContext,
    task: Any,
    *,
    frontier_limit: int,
    configured_limit: int | None = None,
    expand_for_exclusions: bool = False,
) -> list[Any]:
    """Build a bounded global candidate pool from the accepted backend seams."""
    if ctx.repository is None:
        return []

    query_candidates = getattr(ctx.repository, "query_maintenance_candidates", None)
    if not callable(query_candidates):
        return []

    query_limit = _curator_backend_query_limit(
        task,
        frontier_limit,
        configured_limit=configured_limit,
        expand_for_exclusions=expand_for_exclusions,
    )
    candidates_by_id: dict[str, Any] = {}
    for strategy in _CURATOR_BACKEND_STRATEGIES:
        records = cast(
            list[Any],
            query_candidates(
                strategy,
                limit=query_limit,
                status="active",
                include_superseded=False,
                seed=_sampling_task_id(task),
            ),
        )
        for record in records:
            candidates_by_id[record.id] = record
    list_candidate_states = getattr(getattr(ctx, "curation", None), "list_candidate_states", None)
    if callable(list_candidate_states):
        escalated_states = cast(
            list[CurationCandidateState],
            list_candidate_states(
                disposition=CandidateDisposition.ESCALATED,
                limit=CURATOR_MAX_BATCH_RECORDS,
            ),
        )
        for state in escalated_states:
            record = ctx.repository.get_memory(str(state.memory_id))
            if record is not None and record.status == "active":
                candidates_by_id[record.id] = record

    return filter_curator_candidates(ctx, list(candidates_by_id.values()))


def _is_explicit_campaign_hypothesis(hypothesis: CampaignHypothesis | None) -> bool:
    if hypothesis is None:
        return False
    return bool(
        hypothesis.expected_memory_ids
        or (hypothesis.query and hypothesis.query.strip())
        or hypothesis.retrieval_problem is not CampaignRetrievalProblem.HEURISTIC
    )


def _query_campaign_goal_candidates(
    ctx: ApplicationContext,
    hypothesis: CampaignHypothesis,
    backend_candidates: list[Any],
    *,
    limit: int,
) -> list[Any]:
    if ctx.repository is None:
        return []
    candidate_by_id: dict[str, Any] = {}
    anchor_ids = {str(memory_id) for memory_id in hypothesis.expected_memory_ids}
    expected_ids = [str(memory_id) for memory_id in hypothesis.expected_memory_ids]
    for memory_id in expected_ids:
        record = ctx.repository.get_memory(memory_id)
        if record is not None and record.status == "active":
            candidate_by_id[record.id] = record

    retrieval = getattr(ctx, "memory_retrieval", None)
    if retrieval is None and ctx.relational_search is not None:
        retrieval = build_memory_retrieval_facade(
            ctx.repository,
            config=getattr(ctx, "config", None),
            vector_store=getattr(ctx, "vector_store", None),
            embedder=getattr(ctx, "embedder", None),
            embedding_maintenance=getattr(ctx, "embedding_maintenance", None),
            native_search=ctx.relational_search,
        )
    if retrieval is not None and hypothesis.query and hypothesis.query.strip():
        outcome = retrieval.search_sync(
            hypothesis.query.strip(),
            limit=limit,
            workspace_id=None,
            status="active",
            include_superseded=False,
        )
        for result in outcome.results:
            memory_id = getattr(result.record, "source_id", None)
            if isinstance(memory_id, str):
                record = ctx.repository.get_memory(memory_id)
                if record is not None and record.status == "active":
                    candidate_by_id[record.id] = record
                    anchor_ids.add(record.id)

    anchors = list(candidate_by_id.values())
    adjacent_ids = _seed_adjacent_memory_ids(ctx, anchors)
    for memory_id in sorted(adjacent_ids):
        record = ctx.repository.get_memory(memory_id)
        if record is not None and record.status == "active":
            candidate_by_id[record.id] = record

    anchor_tags = {tag for record in anchors for tag in record.tags}
    support_candidates = [
        record
        for record in backend_candidates
        if record.id not in candidate_by_id
        and (record.id in adjacent_ids or anchor_tags.intersection(record.tags))
    ]
    support_candidates.sort(
        key=lambda record: (
            -int(record.id in adjacent_ids),
            -len(anchor_tags.intersection(record.tags)),
            -record.read_count,
            str(record.updated_at),
        )
    )
    for record in support_candidates:
        candidate_by_id[record.id] = record

    filtered = filter_curator_candidates(
        ctx,
        list(candidate_by_id.values()),
        preserve_memory_ids=anchor_ids,
    )
    by_id = {record.id: record for record in filtered}
    ordered: list[Any] = []
    for memory_id in expected_ids:
        record = by_id.get(memory_id)
        if record is not None:
            ordered.append(record)
    ordered_ids = {record.id for record in ordered}
    for record in filtered:
        if record.id not in ordered_ids:
            ordered.append(record)
    return ordered[:limit]


def _quality_feedback_candidates(ctx: ApplicationContext, candidates: list[Any]) -> list[Any]:
    get_state = getattr(getattr(ctx, "curation", None), "get_candidate_state", None)
    if not callable(get_state):
        return []
    feedback_candidates: list[tuple[int, str, Any]] = []
    for record in candidates:
        memory_id = _candidate_uuid(record.id)
        if memory_id is None:
            continue
        state = cast(CurationCandidateState | None, get_state(memory_id))
        if (
            state is not None
            and state.disposition is CandidateDisposition.ESCALATED
            and state.last_disposition_reason in _CURATOR_QUALITY_FEEDBACK_REASONS
        ):
            feedback_candidates.append(
                (
                    -state.escalation_count,
                    str(state.last_run_id or ""),
                    record,
                )
            )
    feedback_candidates.sort(key=lambda item: (item[0], item[1], item[2].id))
    return [record for _, _, record in feedback_candidates]


def curator_quality_feedback(ctx: ApplicationContext, record: Any) -> dict[str, Any] | None:
    get_state = getattr(getattr(ctx, "curation", None), "get_candidate_state", None)
    if not callable(get_state):
        return None
    memory_id = _candidate_uuid(record.id)
    if memory_id is None:
        return None
    state = cast(CurationCandidateState | None, get_state(memory_id))
    if (
        state is None
        or state.disposition is not CandidateDisposition.ESCALATED
        or state.last_disposition_reason not in _CURATOR_QUALITY_FEEDBACK_REASONS
    ):
        return None
    quality_evidence = state.coverage_evidence_json.get("quality_regression")
    if (
        state.last_disposition_reason == "acceptance_not_met"
        and isinstance(quality_evidence, dict)
        and quality_evidence.get("neutral_reason") is not None
    ):
        return None
    feedback = {
        "reason": state.last_disposition_reason,
        "escalation_count": state.escalation_count,
        "last_run_id": str(state.last_run_id) if state.last_run_id is not None else None,
        "last_strategy": state.last_escalated_strategy,
    }
    if isinstance(quality_evidence, dict):
        for field in (
            "retrieval_regression_count",
            "zero_result_change",
            "acceptance_met",
            "neutral_reason",
        ):
            if field in quality_evidence:
                feedback[field] = quality_evidence[field]
    return feedback


def _curator_backend_query_limit(
    task: Any,
    frontier_limit: int,
    *,
    configured_limit: int | None = None,
    expand_for_exclusions: bool = False,
) -> int:
    if configured_limit is None:
        configured_limit = DEFAULT_AGENT_SCAN_LIMIT
    requested_limit = max(configured_limit, frontier_limit) if expand_for_exclusions else configured_limit
    return max(
        1,
        min(
            requested_limit,
            CURATOR_MAX_BATCH_RECORDS * CURATOR_CANDIDATE_POOL_MULTIPLIER,
        ),
    )


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
    for key in ("seed_memory_ids", "candidate_memory_ids", "support_memory_ids"):
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
