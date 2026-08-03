from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from mcp_memory.management.analytics_common import _bucket_starts
from mcp_memory.management.models import (
    NerdRetrievalCallerKindRowPayload,
    NerdRetrievalConversionMemoryRowPayload,
    NerdRetrievalFunnelPayload,
    NerdRetrievalEngagementEvidencePayload,
    NerdRetrievalMemoryRowPayload,
    NerdRetrievalPayload,
    NerdRetrievalQueryFamilyRowPayload,
    NerdRetrievalSummaryPayload,
    NerdRetrievalTagRowPayload,
    NerdRetrievalTagTimelinePayload,
    NerdTimeCountBucketPayload,
)
from mcp_memory.management.reporting_rows import MemoryToolEventRow, ScopedMemoryRow


_RETRIEVAL_MEMORY_LIMIT = 100
_RETRIEVAL_QUERY_FAMILY_LIMIT = 25
_RETRIEVAL_TAG_LIMIT = 25
_RETRIEVAL_TAG_TIMELINE_LIMIT = 10


@dataclass
class _RetrievalMemoryAccumulator:
    title: str
    memory_type: str
    status: str
    tags: list[str]
    read_count: int = 0
    search_count: int = 0
    converted_search_count: int = 0
    last_read_at: float | None = None
    last_search_at: float | None = None


@dataclass
class _RetrievalTagAccumulator:
    read_count: int = 0
    search_count: int = 0


@dataclass
class _RetrievalCallerKindAccumulator:
    search_invocations: int = 0
    search_hits: int = 0
    zero_result_searches: int = 0
    read_events: int = 0


@dataclass
class _RetrievalQueryFamilyAccumulator:
    label: str
    search_invocations: int = 0
    search_hits: int = 0
    zero_result_searches: int = 0
    converted_search_hits: int = 0
    unique_search_memory_ids: set[str] = field(default_factory=set)


@dataclass
class _RetrievalSearchInvocationAccumulator:
    caller_kind: str
    query_family_key: str
    query_family_label: str
    surfaced_memory_ids: set[str] = field(default_factory=set)
    zero_result: bool = False


@dataclass(frozen=True)
class _RetrievalSearchHitRecord:
    query_family_key: str
    memory_id: str
    created_at: float


def build_retrieval_analytics(
    memory_rows: list[ScopedMemoryRow],
    *,
    retrieval_rows: list[MemoryToolEventRow],
    cutoff: float,
    generated_at: float,
    bucket_seconds: int,
) -> NerdRetrievalPayload:
    bucket_starts = _bucket_starts(cutoff=cutoff, generated_at=generated_at, bucket_seconds=bucket_seconds)
    if not bucket_starts:
        return NerdRetrievalPayload()

    memory_info = {row.id: row for row in memory_rows}
    memory_accumulators: dict[str, _RetrievalMemoryAccumulator] = {}
    tag_accumulators: dict[str, _RetrievalTagAccumulator] = {}
    caller_kind_accumulators: dict[str, _RetrievalCallerKindAccumulator] = {}
    query_family_accumulators: dict[str, _RetrievalQueryFamilyAccumulator] = {}
    search_invocation_accumulators: dict[str, _RetrievalSearchInvocationAccumulator] = {}
    tag_read_buckets: dict[str, Counter[int]] = {}
    tag_search_buckets: dict[str, Counter[int]] = {}
    search_invocation_ids: set[str] = set()
    zero_result_search_invocation_ids: set[str] = set()
    unique_search_memories: set[str] = set()
    unique_read_memories: set[str] = set()
    read_event_times_by_memory: dict[str, list[float]] = {}
    search_hit_records: list[_RetrievalSearchHitRecord] = []
    converted_search_hits = 0
    read_events = 0
    search_hits = 0
    external_search_hits: list[tuple[MemoryToolEventRow, str, str]] = []
    external_reads_by_memory: dict[str, list[float]] = {}

    for row in retrieval_rows:
        event_kind = row.event_kind
        invocation_id = row.invocation_id
        created_at = row.created_at
        caller_kind = _normalize_caller_kind(row.caller_kind)
        is_internal = caller_kind == "internal"
        if event_kind == "search":
            if not is_internal and row.memory_id is not None and row.memory_id in memory_info:
                external_search_hits.append(
                    (
                        row,
                        _normalize_query_family_key(row.query_text),
                        row.memory_id,
                    )
                )
            search_invocation_ids.add(invocation_id)
            search_invocation = search_invocation_accumulators.setdefault(
                invocation_id,
                _RetrievalSearchInvocationAccumulator(
                    caller_kind=caller_kind,
                    query_family_key=_normalize_query_family_key(row.query_text),
                    query_family_label=_format_query_family_label(row.query_text),
                ),
            )
            if row.result_count == 0:
                zero_result_search_invocation_ids.add(invocation_id)
                search_invocation.zero_result = True

        memory_id = row.memory_id
        if memory_id is None or memory_id not in memory_info:
            if event_kind == "read":
                caller_kind_accumulators.setdefault(caller_kind, _RetrievalCallerKindAccumulator()).read_events += 1
            continue

        if event_kind == "read" and not is_internal:
            external_reads_by_memory.setdefault(memory_id, []).append(created_at)
        info = memory_info[memory_id]
        memory_accumulator = memory_accumulators.setdefault(
            memory_id,
            _RetrievalMemoryAccumulator(
                title=info.title,
                memory_type=info.memory_type,
                status=info.status,
                tags=list(info.tags),
            ),
        )
        bucket_start = int(created_at // bucket_seconds) * bucket_seconds
        for tag in info.tags:
            tag_accumulator = tag_accumulators.setdefault(tag, _RetrievalTagAccumulator())
            if event_kind == "read":
                tag_accumulator.read_count += 1
                tag_read_buckets.setdefault(tag, Counter())[bucket_start] += 1
            elif event_kind == "search":
                tag_accumulator.search_count += 1
                tag_search_buckets.setdefault(tag, Counter())[bucket_start] += 1

        if event_kind == "read":
            memory_accumulator.read_count += 1
            memory_accumulator.last_read_at = created_at if memory_accumulator.last_read_at is None else max(memory_accumulator.last_read_at, created_at)
            unique_read_memories.add(memory_id)
            read_events += 1
            caller_kind_accumulators.setdefault(caller_kind, _RetrievalCallerKindAccumulator()).read_events += 1
            read_event_times_by_memory.setdefault(memory_id, []).append(created_at)
        elif event_kind == "search":
            memory_accumulator.search_count += 1
            memory_accumulator.last_search_at = created_at if memory_accumulator.last_search_at is None else max(memory_accumulator.last_search_at, created_at)
            unique_search_memories.add(memory_id)
            search_hits += 1
            search_invocation_accumulators[invocation_id].surfaced_memory_ids.add(memory_id)
            search_hit_records.append(
                _RetrievalSearchHitRecord(
                    query_family_key=search_invocation_accumulators[invocation_id].query_family_key,
                    memory_id=memory_id,
                    created_at=created_at,
                )
            )

    for search_hit_record in search_hit_records:
        read_times = read_event_times_by_memory.get(search_hit_record.memory_id, [])
        if not any(read_time >= search_hit_record.created_at for read_time in read_times):
            continue
        converted_search_hits += 1
        memory_accumulators[search_hit_record.memory_id].converted_search_count += 1
        query_family_accumulator = query_family_accumulators.setdefault(
            search_hit_record.query_family_key,
            _RetrievalQueryFamilyAccumulator(label=search_hit_record.query_family_key),
        )
        query_family_accumulator.converted_search_hits += 1

    for search_invocation in search_invocation_accumulators.values():
        caller_kind_accumulator = caller_kind_accumulators.setdefault(
            search_invocation.caller_kind,
            _RetrievalCallerKindAccumulator(),
        )
        caller_kind_accumulator.search_invocations += 1
        caller_kind_accumulator.search_hits += len(search_invocation.surfaced_memory_ids)
        if search_invocation.zero_result:
            caller_kind_accumulator.zero_result_searches += 1

        query_family_accumulator = query_family_accumulators.setdefault(
            search_invocation.query_family_key,
            _RetrievalQueryFamilyAccumulator(label=search_invocation.query_family_label),
        )
        query_family_accumulator.search_invocations += 1
        query_family_accumulator.search_hits += len(search_invocation.surfaced_memory_ids)
        if search_invocation.zero_result:
            query_family_accumulator.zero_result_searches += 1
        query_family_accumulator.unique_search_memory_ids.update(search_invocation.surfaced_memory_ids)

    by_caller_kind = [
        NerdRetrievalCallerKindRowPayload(
            key=caller_kind,
            label=_format_caller_kind_label(caller_kind),
            search_invocations=accumulator.search_invocations,
            search_hits=accumulator.search_hits,
            zero_result_searches=accumulator.zero_result_searches,
            read_events=accumulator.read_events,
            total_events=accumulator.search_invocations + accumulator.read_events,
        )
        for caller_kind, accumulator in sorted(
            caller_kind_accumulators.items(),
            key=lambda item: (
                -(item[1].search_invocations + item[1].read_events),
                -item[1].search_hits,
                -item[1].read_events,
                item[0],
            ),
        )
    ]

    top_query_families = [
        NerdRetrievalQueryFamilyRowPayload(
            key=family_key,
            label=accumulator.label,
            search_invocations=accumulator.search_invocations,
            search_hits=accumulator.search_hits,
            zero_result_searches=accumulator.zero_result_searches,
            unique_search_memories=len(accumulator.unique_search_memory_ids),
            converted_search_hits=accumulator.converted_search_hits,
            conversion_rate=0.0 if accumulator.search_hits == 0 else round(accumulator.converted_search_hits / accumulator.search_hits, 4),
        )
        for family_key, accumulator in sorted(
            query_family_accumulators.items(),
            key=lambda item: (
                -item[1].search_invocations,
                -item[1].search_hits,
                -len(item[1].unique_search_memory_ids),
                item[0],
            ),
        )[:_RETRIEVAL_QUERY_FAMILY_LIMIT]
    ]

    top_zero_result_query_families = [row for row in top_query_families if row.zero_result_searches > 0]

    low_conversion_memories = [
        NerdRetrievalConversionMemoryRowPayload(
            memory_id=memory_id,
            title=accumulator.title,
            memory_type=accumulator.memory_type,
            status=accumulator.status,
            tags=accumulator.tags,
            read_count=accumulator.read_count,
            search_count=accumulator.search_count,
            converted_search_count=accumulator.converted_search_count,
            conversion_rate=0.0 if accumulator.search_count == 0 else round(accumulator.converted_search_count / accumulator.search_count, 4),
            last_read_at=accumulator.last_read_at,
            last_search_at=accumulator.last_search_at,
        )
        for memory_id, accumulator in sorted(
            memory_accumulators.items(),
            key=lambda item: (
                1.0 if item[1].search_count == 0 else round(item[1].converted_search_count / item[1].search_count, 4),
                -item[1].search_count,
                item[1].title.lower(),
                item[0],
            ),
        )
        if accumulator.search_count > 0
    ][:_RETRIEVAL_QUERY_FAMILY_LIMIT]

    top_read_memories = [
        NerdRetrievalMemoryRowPayload(
            memory_id=memory_id,
            title=accumulator.title,
            memory_type=accumulator.memory_type,
            status=accumulator.status,
            tags=accumulator.tags,
            read_count=accumulator.read_count,
            search_count=accumulator.search_count,
            total_count=accumulator.read_count + accumulator.search_count,
            last_read_at=accumulator.last_read_at,
            last_search_at=accumulator.last_search_at,
        )
        for memory_id, accumulator in sorted(
            memory_accumulators.items(),
            key=lambda item: (-item[1].read_count, -(item[1].last_read_at or 0.0), item[1].title.lower(), item[0]),
        )
        if accumulator.read_count > 0
    ][:_RETRIEVAL_MEMORY_LIMIT]

    top_search_memories = [
        NerdRetrievalMemoryRowPayload(
            memory_id=memory_id,
            title=accumulator.title,
            memory_type=accumulator.memory_type,
            status=accumulator.status,
            tags=accumulator.tags,
            read_count=accumulator.read_count,
            search_count=accumulator.search_count,
            total_count=accumulator.read_count + accumulator.search_count,
            last_read_at=accumulator.last_read_at,
            last_search_at=accumulator.last_search_at,
        )
        for memory_id, accumulator in sorted(
            memory_accumulators.items(),
            key=lambda item: (-item[1].search_count, -(item[1].last_search_at or 0.0), item[1].title.lower(), item[0]),
        )
        if accumulator.search_count > 0
    ][:_RETRIEVAL_MEMORY_LIMIT]

    ordered_tag_items = sorted(
        tag_accumulators.items(),
        key=lambda item: (-(item[1].read_count + item[1].search_count), -item[1].search_count, -item[1].read_count, item[0]),
    )
    top_tags = [
        NerdRetrievalTagRowPayload(
            key=tag,
            label=tag,
            read_count=accumulator.read_count,
            search_count=accumulator.search_count,
            total_count=accumulator.read_count + accumulator.search_count,
        )
        for tag, accumulator in ordered_tag_items[:_RETRIEVAL_TAG_LIMIT]
    ]

    tag_timelines = [
        NerdRetrievalTagTimelinePayload(
            key=tag,
            label=tag,
            read_buckets=[
                NerdTimeCountBucketPayload(bucket_start=float(bucket_start), count=tag_read_buckets.get(tag, Counter()).get(bucket_start, 0))
                for bucket_start in bucket_starts
            ],
            search_buckets=[
                NerdTimeCountBucketPayload(bucket_start=float(bucket_start), count=tag_search_buckets.get(tag, Counter()).get(bucket_start, 0))
                for bucket_start in bucket_starts
            ],
        )
        for tag, _accumulator in ordered_tag_items[:_RETRIEVAL_TAG_TIMELINE_LIMIT]
    ]

    engagement_evidence = _build_engagement_evidence(
        external_search_hits,
        external_reads_by_memory,
    )
    return NerdRetrievalPayload(
        summary=NerdRetrievalSummaryPayload(
            search_invocations=len(search_invocation_ids),
            search_hits=search_hits,
            zero_result_searches=len(zero_result_search_invocation_ids),
            read_events=read_events,
            unique_search_memories=len(unique_search_memories),
            unique_read_memories=len(unique_read_memories),
        ),
        funnel=NerdRetrievalFunnelPayload(
            search_hits=search_hits,
            converted_search_hits=converted_search_hits,
            conversion_rate=0.0 if search_hits == 0 else round(converted_search_hits / search_hits, 4),
        ),
        by_caller_kind=by_caller_kind,
        top_query_families=top_query_families,
        top_zero_result_query_families=top_zero_result_query_families,
        top_read_memories=top_read_memories,
        top_search_memories=top_search_memories,
        low_conversion_memories=low_conversion_memories,
        engagement_evidence=engagement_evidence,
        top_tags=top_tags,
        tag_timelines=tag_timelines,
    )


def _build_engagement_evidence(
    search_hits: list[tuple[MemoryToolEventRow, str, str]],
    reads_by_memory: dict[str, list[float]],
) -> list[NerdRetrievalEngagementEvidencePayload]:
    exposures_by_memory: Counter[str] = Counter(memory_id for _row, _family, memory_id in search_hits)
    read_by_invocation: dict[str, set[str]] = {}
    for row, _family, memory_id in search_hits:
        if any(read_at >= row.created_at for read_at in reads_by_memory.get(memory_id, ())):
            read_by_invocation.setdefault(row.invocation_id, set()).add(memory_id)

    evidence: list[NerdRetrievalEngagementEvidencePayload] = []
    for row, family, memory_id in search_hits:
        reads = reads_by_memory.get(memory_id, ())
        converted = any(read_at >= row.created_at for read_at in reads)
        co_result_read = bool(read_by_invocation.get(row.invocation_id, set()) - {memory_id})
        if converted:
            kind, strength = "search_to_read", "strong"
        elif co_result_read:
            kind, strength = "skipped_co_result", "conditional"
        elif exposures_by_memory[memory_id] >= 2:
            kind, strength = "repeated_exposure", "weak"
        else:
            kind, strength = "unknown", "neutral"
        evidence.append(
            NerdRetrievalEngagementEvidencePayload(
                memory_id=memory_id,
                query_family_key=family,
                evidence_kind=kind,
                strength=strength,
                exposure_count=exposures_by_memory[memory_id],
                co_result_read=co_result_read,
                graph_provenance=dict(row.graph_provenance),
            )
        )
    return evidence


def _normalize_caller_kind(value: object) -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized:
            return normalized
    return "unknown"


def _format_caller_kind_label(value: str) -> str:
    return value.replace("_", " ").title()


def _normalize_query_family_key(value: object) -> str:
    if not isinstance(value, str):
        return "empty-query"
    normalized = " ".join(value.lower().split())
    return normalized or "empty-query"


def _format_query_family_label(value: object) -> str:
    if not isinstance(value, str):
        return "(empty query)"
    normalized = " ".join(value.split())
    return normalized or "(empty query)"
