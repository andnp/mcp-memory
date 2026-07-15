from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass

from mcp_memory.management.analytics_common import _bucket_starts, _datetime_to_timestamp
from mcp_memory.management.models import (
    NerdCountSeriesPayload,
    NerdQualityDrilldownPayload,
    NerdQualityMemoryRowPayload,
    NerdQualityProducerAttributionPayload,
    NerdQualityProducerPayload,
    NerdQualityRemediationPayload,
    NerdQualitySignalDrilldownPayload,
    NerdStatPayload,
    NerdTimeCountBucketPayload,
)
from mcp_memory.management.reporting_rows import ScopedMemoryRow


_QUALITY_SIGNAL_DEFS: tuple[tuple[str, str], ...] = (
    ("trace_like_memory_count", "Trace-like memories"),
    ("generic_summary_count", "Generic summaries"),
    ("untagged_observation_count", "Untagged observations"),
    ("oversized_memory_count", "Oversized memories"),
)

_QUALITY_REMEDIATION_DEFS: tuple[tuple[str, str], ...] = (
    ("tagged_observation_updates", "Tagged observation updates"),
    ("concrete_summary_updates", "Concrete summary updates"),
    ("split_lineage_updates", "Split-lineage updates"),
)


@dataclass(frozen=True)
class _MemoryQualitySignals:
    trace_like_memory_count: int = 0
    generic_summary_count: int = 0
    untagged_observation_count: int = 0
    oversized_memory_count: int = 0
    observation_count: int = 0

    @property
    def untagged_observation_rate(self) -> float:
        if self.observation_count <= 0:
            return 0.0
        return self.untagged_observation_count / self.observation_count


def build_memory_quality_signals(memory_rows: list[ScopedMemoryRow]) -> _MemoryQualitySignals:
    trace_like_memory_count = 0
    generic_summary_count = 0
    untagged_observation_count = 0
    oversized_memory_count = 0
    observation_count = 0

    for row in memory_rows:
        memory_type = row.memory_type
        tags = row.tags
        content_bytes = row.content_bytes
        quality_keys = set(_quality_signal_keys_for_row(row))

        if "trace_like_memory_count" in quality_keys:
            trace_like_memory_count += 1
        if "generic_summary_count" in quality_keys:
            generic_summary_count += 1
        if "oversized_memory_count" in quality_keys or content_bytes >= 4_000:
            oversized_memory_count += 1
        if memory_type == "observation":
            observation_count += 1
            if "untagged_observation_count" in quality_keys or not tags:
                untagged_observation_count += 1

    return _MemoryQualitySignals(
        trace_like_memory_count=trace_like_memory_count,
        generic_summary_count=generic_summary_count,
        untagged_observation_count=untagged_observation_count,
        oversized_memory_count=oversized_memory_count,
        observation_count=observation_count,
    )


def build_quality_signal_series(
    memory_rows: list[ScopedMemoryRow],
    *,
    cutoff: float,
    generated_at: float,
    bucket_seconds: int,
) -> list[NerdCountSeriesPayload]:
    series: list[NerdCountSeriesPayload] = []
    for key, label in _QUALITY_SIGNAL_DEFS:
        def predicate(row, signal_key=key) -> bool:
            return signal_key in _quality_signal_keys_for_row(row)

        if not any(predicate(row) for row in memory_rows):
            continue
        series.append(
            NerdCountSeriesPayload(
                key=key,
                label=label,
                buckets=_build_backlog_series(
                    memory_rows,
                    cutoff=cutoff,
                    generated_at=generated_at,
                    bucket_seconds=bucket_seconds,
                    predicate=predicate,
                ),
            )
        )
    return series


def build_quality_drilldown(memory_rows: list[ScopedMemoryRow], *, limit_per_signal: int = 12) -> NerdQualityDrilldownPayload:
    if not memory_rows:
        return NerdQualityDrilldownPayload()

    signals: list[NerdQualitySignalDrilldownPayload] = []
    for key, label in _QUALITY_SIGNAL_DEFS:
        rows = [row for row in memory_rows if key in _quality_signal_keys_for_row(row)]
        rows.sort(
            key=lambda row: (
                -(_datetime_to_timestamp(row.updated_at) or 0.0),
                row.title.lower(),
                row.id,
            )
        )
        signals.append(
            NerdQualitySignalDrilldownPayload(
                key=key,
                label=label,
                count=len(rows),
                records=[_build_quality_memory_row(row) for row in rows[:limit_per_signal]],
            )
        )
    return NerdQualityDrilldownPayload(
        signals=signals,
        producer_attributions=build_quality_producer_attributions(memory_rows),
    )


def build_quality_producer_attributions(
    memory_rows: list[ScopedMemoryRow],
) -> list[NerdQualityProducerAttributionPayload]:
    grouped: Counter[tuple[str, str, str, str, str, str, str]] = Counter()
    labels = dict(_QUALITY_SIGNAL_DEFS)
    for row in memory_rows:
        producer = _producer_for_row(row)
        producer_key = (
            producer.task_id,
            producer.task_name,
            producer.tool_name,
            producer.provider_key,
            producer.provider_name,
            producer.model_name,
        )
        for signal_key in _quality_signal_keys_for_row(row):
            grouped[(signal_key, *producer_key)] += 1

    attributions: list[NerdQualityProducerAttributionPayload] = []
    for (signal_key, task_id, task_name, tool_name, provider_key, provider_name, model_name), count in sorted(
        grouped.items(), key=lambda item: (-item[1], item[0])
    ):
        attributions.append(
            NerdQualityProducerAttributionPayload(
                signal_key=signal_key,
                signal_label=labels[signal_key],
                count=count,
                repeated=count > 1,
                producer=NerdQualityProducerPayload(
                    task_id=task_id,
                    task_name=task_name,
                    tool_name=tool_name,
                    provider_key=provider_key,
                    provider_name=provider_name,
                    model_name=model_name,
                ),
            )
        )
    return attributions


def build_quality_remediation(
    memory_rows: list[ScopedMemoryRow],
    *,
    cutoff: float,
    generated_at: float,
    bucket_seconds: int,
) -> NerdQualityRemediationPayload:
    if not memory_rows:
        return NerdQualityRemediationPayload()

    tagged_observation_count = sum(1 for row in memory_rows if row.memory_type == "observation" and row.tags)
    concrete_summary_count = sum(1 for row in memory_rows if _is_concrete_summary(row.summary))
    split_lineage_count = sum(1 for row in memory_rows if row.has_split_lineage)

    stats = [
        NerdStatPayload(
            key="tagged_observation_count",
            label="Tagged observations",
            value=float(tagged_observation_count),
            unit="count",
        ),
        NerdStatPayload(
            key="concrete_summary_count",
            label="Concrete summaries",
            value=float(concrete_summary_count),
            unit="count",
        ),
        NerdStatPayload(
            key="split_lineage_count",
            label="Split-lineage memories",
            value=float(split_lineage_count),
            unit="count",
        ),
    ]

    remediation_predicates = {
        "tagged_observation_updates": lambda row: row.memory_type == "observation" and bool(row.tags),
        "concrete_summary_updates": lambda row: _is_concrete_summary(row.summary),
        "split_lineage_updates": lambda row: row.has_split_lineage,
    }

    activity: list[NerdCountSeriesPayload] = []
    bucket_starts = _bucket_starts(cutoff=cutoff, generated_at=generated_at, bucket_seconds=bucket_seconds)
    for key, label in _QUALITY_REMEDIATION_DEFS:
        predicate = remediation_predicates[key]
        counts: Counter[int] = Counter()
        for row in memory_rows:
            if not predicate(row):
                continue
            updated_at = _datetime_to_timestamp(row.updated_at)
            if updated_at is None or updated_at < cutoff or updated_at > generated_at:
                continue
            counts[int(updated_at // bucket_seconds) * bucket_seconds] += 1
        if not counts:
            continue
        activity.append(
            NerdCountSeriesPayload(
                key=key,
                label=label,
                buckets=[
                    NerdTimeCountBucketPayload(bucket_start=float(bucket_start), count=counts.get(bucket_start, 0))
                    for bucket_start in bucket_starts
                ],
            )
        )
    return NerdQualityRemediationPayload(stats=stats, activity=activity)


def _build_backlog_series(
    memory_rows: list[ScopedMemoryRow],
    *,
    cutoff: float,
    generated_at: float,
    bucket_seconds: int,
    predicate,
) -> list[NerdTimeCountBucketPayload]:
    bucket_starts = _bucket_starts(cutoff=cutoff, generated_at=generated_at, bucket_seconds=bucket_seconds)
    if not bucket_starts:
        return []

    baseline_count = 0
    created_counts: Counter[int] = Counter()
    for row in memory_rows:
        if not predicate(row):
            continue
        created_at = _datetime_to_timestamp(row.created_at)
        if created_at is None:
            continue
        if created_at < cutoff:
            baseline_count += 1
            continue
        if created_at <= generated_at:
            created_counts[int(created_at // bucket_seconds) * bucket_seconds] += 1

    running_count = baseline_count
    series: list[NerdTimeCountBucketPayload] = []
    for bucket_start in bucket_starts:
        running_count += created_counts.get(bucket_start, 0)
        series.append(NerdTimeCountBucketPayload(bucket_start=float(bucket_start), count=running_count))
    return series


def _normalize_quality_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.strip().lower().split())


def _looks_like_trace_title(title: str) -> bool:
    return title.startswith("task_complete") or title.startswith("task complete") or title.startswith("task_complete_record")


def _quality_signal_keys_for_row(row: ScopedMemoryRow) -> list[str]:
    keys: list[str] = []
    if _looks_like_trace_title(_normalize_quality_text(row.title)):
        keys.append("trace_like_memory_count")
    if _normalize_quality_text(row.summary).startswith("covers "):
        keys.append("generic_summary_count")
    if row.memory_type == "observation" and not row.tags:
        keys.append("untagged_observation_count")
    if row.content_bytes >= 4_000:
        keys.append("oversized_memory_count")
    return keys


def _build_quality_memory_row(row: ScopedMemoryRow) -> NerdQualityMemoryRowPayload:
    return NerdQualityMemoryRowPayload(
        memory_id=row.id,
        title=row.title,
        summary=row.summary,
        memory_type=row.memory_type,
        status=row.status,
        updated_at="" if row.updated_at is None else row.updated_at.isoformat(),
        tags=list(row.tags),
        producer=_producer_for_row(row),
    )


def _producer_for_row(row: ScopedMemoryRow) -> NerdQualityProducerPayload:
    metadata = row.metadata
    nested = metadata.get("producer")
    producer = nested if isinstance(nested, Mapping) else {}

    def first_string(*keys: str) -> str:
        for source in (producer, metadata):
            for key in keys:
                value = source.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return "unknown"

    tool_name = first_string("tool_name", "tool", "producer_tool_name", "producer_tool")
    if tool_name == "unknown":
        tool_names = metadata.get("tool_names_used")
        if isinstance(tool_names, list):
            names = sorted({item.strip() for item in tool_names if isinstance(item, str) and item.strip()})
            if len(names) == 1:
                tool_name = names[0]
            elif names:
                tool_name = ",".join(names)
    if tool_name == "unknown":
        if metadata.get("created_via_ingest") is True:
            tool_name = (
                "internal_ingest_append_memory"
                if metadata.get("appended_via_ingest") is True
                else "internal_ingest_create_memory"
            )

    provider = metadata.get("provider")
    provider_mapping = provider if isinstance(provider, Mapping) else {}

    def provider_string(*keys: str) -> str:
        for source in (producer, provider_mapping, metadata):
            for key in keys:
                value = source.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return "unknown"

    return NerdQualityProducerPayload(
        task_id=first_string("task_id", "producer_task_id", "ingest_task_id"),
        task_name=first_string("task_name", "producer_task", "producer_task_name", "ingest_task_name"),
        tool_name=tool_name,
        provider_key=provider_string("provider_key", "producer_provider_key", "producer_provider", "key"),
        provider_name=provider_string("provider_name", "producer_provider_name", "name"),
        model_name=provider_string("model_name", "producer_model_name", "model"),
    )


def _is_concrete_summary(value: object) -> bool:
    normalized = _normalize_quality_text(value)
    return bool(normalized) and not normalized.startswith("covers ")
