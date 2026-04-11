from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
import json

from mcp_memory.management.models import (
    NerdCountSeriesPayload,
    NerdQualityDrilldownPayload,
    NerdQualityMemoryRowPayload,
    NerdQualityRemediationPayload,
    NerdQualitySignalDrilldownPayload,
    NerdStatPayload,
    NerdTimeCountBucketPayload,
)
from mcp_memory.management.reporting_queries import split_csv_values


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


def build_memory_quality_signals(memory_rows) -> _MemoryQualitySignals:
    trace_like_memory_count = 0
    generic_summary_count = 0
    untagged_observation_count = 0
    oversized_memory_count = 0
    observation_count = 0

    for row in memory_rows:
        memory_type = str(row["type"])
        tags = split_csv_values(row["tags_csv"])
        content_bytes = int(row["content_bytes"] or 0)
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
    memory_rows,
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


def build_quality_drilldown(memory_rows, *, limit_per_signal: int = 12) -> NerdQualityDrilldownPayload:
    if not memory_rows:
        return NerdQualityDrilldownPayload()

    signals: list[NerdQualitySignalDrilldownPayload] = []
    for key, label in _QUALITY_SIGNAL_DEFS:
        rows = [row for row in memory_rows if key in _quality_signal_keys_for_row(row)]
        rows.sort(
            key=lambda row: (
                -(_iso_to_timestamp(row["updated_at"]) or 0.0),
                str(row["title"]).lower(),
                str(row["id"]),
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
    return NerdQualityDrilldownPayload(signals=signals)


def build_quality_remediation(
    memory_rows,
    *,
    cutoff: float,
    generated_at: float,
    bucket_seconds: int,
) -> NerdQualityRemediationPayload:
    if not memory_rows:
        return NerdQualityRemediationPayload()

    tagged_observation_count = sum(
        1
        for row in memory_rows
        if str(row["type"]) == "observation" and split_csv_values(row["tags_csv"])
    )
    concrete_summary_count = sum(
        1
        for row in memory_rows
        if _is_concrete_summary(row["summary"])
    )
    split_lineage_count = sum(
        1
        for row in memory_rows
        if _has_split_lineage(_memory_row_metadata(row))
    )

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
        "tagged_observation_updates": lambda row: str(row["type"]) == "observation" and bool(split_csv_values(row["tags_csv"])),
        "concrete_summary_updates": lambda row: _is_concrete_summary(row["summary"]),
        "split_lineage_updates": lambda row: _has_split_lineage(_memory_row_metadata(row)),
    }

    activity: list[NerdCountSeriesPayload] = []
    bucket_starts = _bucket_starts(cutoff=cutoff, generated_at=generated_at, bucket_seconds=bucket_seconds)
    for key, label in _QUALITY_REMEDIATION_DEFS:
        predicate = remediation_predicates[key]
        counts: Counter[int] = Counter()
        for row in memory_rows:
            if not predicate(row):
                continue
            updated_at = _iso_to_timestamp(row["updated_at"])
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
    memory_rows,
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
        created_at = _iso_to_timestamp(row["created_at"])
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


def _bucket_starts(*, cutoff: float, generated_at: float, bucket_seconds: int) -> list[int]:
    if generated_at < cutoff:
        return []
    start_bucket = int(cutoff // bucket_seconds) * bucket_seconds
    end_bucket = int(generated_at // bucket_seconds) * bucket_seconds
    if end_bucket < start_bucket:
        return []
    return list(range(start_bucket, end_bucket + bucket_seconds, bucket_seconds))


def _normalize_quality_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.strip().lower().split())


def _looks_like_trace_title(title: str) -> bool:
    return title.startswith("task_complete") or title.startswith("task complete") or title.startswith("task_complete_record")


def _quality_signal_keys_for_row(row) -> list[str]:
    keys: list[str] = []
    if _looks_like_trace_title(_normalize_quality_text(row["title"])):
        keys.append("trace_like_memory_count")
    if _normalize_quality_text(row["summary"]).startswith("covers "):
        keys.append("generic_summary_count")
    if str(row["type"]) == "observation" and not split_csv_values(row["tags_csv"]):
        keys.append("untagged_observation_count")
    if int(row["content_bytes"] or 0) >= 4_000:
        keys.append("oversized_memory_count")
    return keys


def _build_quality_memory_row(row) -> NerdQualityMemoryRowPayload:
    return NerdQualityMemoryRowPayload(
        memory_id=str(row["id"]),
        title=str(row["title"]),
        summary=row["summary"] if isinstance(row["summary"], str) else None,
        memory_type=str(row["type"]),
        status=str(row["status"]),
        updated_at=str(row["updated_at"]),
        tags=split_csv_values(row["tags_csv"]),
    )


def _memory_row_metadata(row) -> dict[str, object]:
    raw = row["metadata"]
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _has_split_lineage(metadata: dict[str, object]) -> bool:
    return any(
        key in metadata
        for key in ("split_from_memory_id", "split_child_count", "split_child_memory_ids", "split_group_id")
    )


def _is_concrete_summary(value: object) -> bool:
    normalized = _normalize_quality_text(value)
    return bool(normalized) and not normalized.startswith("covers ")


def _iso_to_timestamp(value: str | None) -> float | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.timestamp()
