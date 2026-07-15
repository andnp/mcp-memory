from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from mcp_memory.management.analytics_reporting import (
    build_memory_quality_signals,
    build_quality_drilldown,
    build_quality_remediation,
    build_quality_signal_series,
)
from mcp_memory.management.analytics_quality import build_quality_producer_attributions
from mcp_memory.management.reporting_rows import ScopedMemoryRow


pytestmark = pytest.mark.small


def test_quality_analytics_builders_preserve_existing_quality_contracts() -> None:
    now = datetime(2026, 4, 11, 12, 0, tzinfo=UTC)
    memory_rows = [
        ScopedMemoryRow(
            id="memory-trace",
            title="task_complete: deduplicator-1",
            summary="Concrete summary.",
            memory_type="fact",
            status="active",
            content_bytes=128,
            created_at=now - timedelta(hours=4),
            updated_at=now - timedelta(hours=3),
        ),
        ScopedMemoryRow(
            id="memory-observation",
            title="Broad observation",
            summary="Covers several related findings.",
            memory_type="observation",
            status="active",
            content_bytes=96,
            created_at=now - timedelta(hours=3),
            updated_at=now - timedelta(hours=2),
        ),
        ScopedMemoryRow(
            id="memory-oversized",
            title="Oversized durable memory",
            summary="Detailed but concrete summary.",
            memory_type="fact",
            status="active",
            content_bytes=4_200,
            tags=["durable"],
            has_split_lineage=True,
            created_at=now - timedelta(hours=2),
            updated_at=now - timedelta(hours=1),
        ),
        ScopedMemoryRow(
            id="memory-tagged-observation",
            title="Tagged observation",
            summary="Concrete summary.",
            memory_type="observation",
            status="active",
            content_bytes=128,
            tags=["tagged", "quality"],
            created_at=now - timedelta(minutes=45),
            updated_at=now - timedelta(minutes=30),
        ),
    ]

    quality_signals = build_memory_quality_signals(memory_rows)
    quality_signal_series = build_quality_signal_series(
        memory_rows,
        cutoff=(now - timedelta(hours=24)).timestamp(),
        generated_at=now.timestamp(),
        bucket_seconds=3600,
    )
    quality_drilldown = build_quality_drilldown(memory_rows)
    quality_remediation = build_quality_remediation(
        memory_rows,
        cutoff=(now - timedelta(hours=24)).timestamp(),
        generated_at=now.timestamp(),
        bucket_seconds=3600,
    )

    assert quality_signals.trace_like_memory_count == 1
    assert quality_signals.generic_summary_count == 1
    assert quality_signals.untagged_observation_count == 1
    assert quality_signals.untagged_observation_rate == 0.5
    assert quality_signals.oversized_memory_count == 1

    assert [series.key for series in quality_signal_series] == [
        "trace_like_memory_count",
        "generic_summary_count",
        "untagged_observation_count",
        "oversized_memory_count",
    ]
    assert [series.buckets[-1].count for series in quality_signal_series] == [1, 1, 1, 1]

    drilldown_counts = {signal.key: signal.count for signal in quality_drilldown.signals}
    assert drilldown_counts == {
        "trace_like_memory_count": 1,
        "generic_summary_count": 1,
        "untagged_observation_count": 1,
        "oversized_memory_count": 1,
    }
    trace_like_records = next(signal.records for signal in quality_drilldown.signals if signal.key == "trace_like_memory_count")
    oversized_records = next(signal.records for signal in quality_drilldown.signals if signal.key == "oversized_memory_count")
    assert trace_like_records[0].title == "task_complete: deduplicator-1"
    assert oversized_records[0].tags == ["durable"]

    remediation_stats = {stat.key: stat.value for stat in quality_remediation.stats}
    assert remediation_stats == {
        "tagged_observation_count": 1.0,
        "concrete_summary_count": 3.0,
        "split_lineage_count": 1.0,
    }
    remediation_activity = {
        series.key: sum(bucket.count for bucket in series.buckets)
        for series in quality_remediation.activity
    }
    assert remediation_activity == {
        "tagged_observation_updates": 1,
        "concrete_summary_updates": 3,
        "split_lineage_updates": 1,
    }


def test_quality_analytics_attributes_repeated_defects_and_unknown_provenance() -> None:
    rows = [
        ScopedMemoryRow(
            id="generic-1",
            title="First generic write",
            summary="Covers several findings.",
            memory_type="fact",
            status="active",
            metadata={
                "producer": {
                    "task_id": "task-1",
                    "task_name": "summary-writer",
                    "tool_name": "internal_create_memory_record",
                    "provider_key": "provider-a",
                    "provider_name": "Provider A",
                    "model_name": "model-a",
                }
            },
        ),
        ScopedMemoryRow(
            id="generic-2",
            title="Second generic write",
            summary="Covers another finding.",
            memory_type="fact",
            status="active",
            metadata={
                "producer": {
                    "task_id": "task-1",
                    "task_name": "summary-writer",
                    "tool_name": "internal_create_memory_record",
                    "provider_key": "provider-a",
                    "provider_name": "Provider A",
                    "model_name": "model-a",
                }
            },
        ),
        ScopedMemoryRow(
            id="unknown-trace",
            title="task_complete: unknown",
            summary="Concrete.",
            memory_type="journal",
            status="active",
        ),
    ]

    attributions = build_quality_producer_attributions(rows)
    generic = next(item for item in attributions if item.signal_key == "generic_summary_count")
    unknown_trace = next(item for item in attributions if item.signal_key == "trace_like_memory_count")

    assert generic.count == 2
    assert generic.repeated is True
    assert generic.producer.task_id == "task-1"
    assert generic.producer.task_name == "summary-writer"
    assert generic.producer.tool_name == "internal_create_memory_record"
    assert generic.producer.provider_key == "provider-a"
    assert unknown_trace.count == 1
    assert unknown_trace.repeated is False
    assert unknown_trace.producer.task_id == "unknown"
    assert unknown_trace.producer.tool_name == "unknown"
    assert unknown_trace.producer.provider_key == "unknown"

    drilldown = build_quality_drilldown(rows)
    generic_record = next(signal for signal in drilldown.signals if signal.key == "generic_summary_count").records[0]
    assert generic_record.producer.task_id == "task-1"
