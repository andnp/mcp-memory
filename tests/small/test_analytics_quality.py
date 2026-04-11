from __future__ import annotations

from datetime import UTC, datetime, timedelta
import json

import pytest

from mcp_memory.management.analytics_quality import (
    build_memory_quality_signals,
    build_quality_drilldown,
    build_quality_remediation,
    build_quality_signal_series,
)


pytestmark = pytest.mark.small


def test_quality_analytics_builders_preserve_existing_quality_contracts() -> None:
    now = datetime(2026, 4, 11, 12, 0, tzinfo=UTC)
    memory_rows = [
        {
            "id": "memory-trace",
            "title": "task_complete: deduplicator-1",
            "summary": "Concrete summary.",
            "type": "fact",
            "status": "active",
            "tags_csv": "",
            "content_bytes": 128,
            "metadata": "",
            "created_at": (now - timedelta(hours=4)).isoformat(),
            "updated_at": (now - timedelta(hours=3)).isoformat(),
        },
        {
            "id": "memory-observation",
            "title": "Broad observation",
            "summary": "Covers several related findings.",
            "type": "observation",
            "status": "active",
            "tags_csv": "",
            "content_bytes": 96,
            "metadata": "",
            "created_at": (now - timedelta(hours=3)).isoformat(),
            "updated_at": (now - timedelta(hours=2)).isoformat(),
        },
        {
            "id": "memory-oversized",
            "title": "Oversized durable memory",
            "summary": "Detailed but concrete summary.",
            "type": "fact",
            "status": "active",
            "tags_csv": "durable",
            "content_bytes": 4_200,
            "metadata": json.dumps({"split_from_memory_id": "source-1", "split_group_id": "split-1"}),
            "created_at": (now - timedelta(hours=2)).isoformat(),
            "updated_at": (now - timedelta(hours=1)).isoformat(),
        },
        {
            "id": "memory-tagged-observation",
            "title": "Tagged observation",
            "summary": "Concrete summary.",
            "type": "observation",
            "status": "active",
            "tags_csv": "tagged,quality",
            "content_bytes": 128,
            "metadata": "",
            "created_at": (now - timedelta(minutes=45)).isoformat(),
            "updated_at": (now - timedelta(minutes=30)).isoformat(),
        },
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
