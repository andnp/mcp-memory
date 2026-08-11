from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from mcp_memory.core.task_handlers.maintenance_work_items import (
    enqueue_producer_remediation_signals,
)
from mcp_memory.core.ports.work_items import EXECUTION_LANE_AGENTIC, WORK_FAMILY_OPERATOR_REVIEW
from mcp_memory.management.analytics_quality import build_quality_remediation_signals
from mcp_memory.management.reporting_rows import ScopedMemoryRow
from mcp_memory.storage.sqlite_work_item_store import SQLiteWorkItemRepository


pytestmark = pytest.mark.small


def _row(
    memory_id: str,
    *,
    created_at: datetime,
    title: str = "A generic write",
    producer: dict[str, str] | None = None,
) -> ScopedMemoryRow:
    return ScopedMemoryRow(
        id=memory_id,
        title=title,
        summary="Covers several findings." if title == "A generic write" else "Concrete summary.",
        memory_type="fact",
        status="active",
        created_at=created_at,
        metadata={} if producer is None else {"producer": producer},
    )


def test_repeated_producer_defects_cross_threshold_per_family_and_window() -> None:
    now = datetime(2026, 4, 11, 12, tzinfo=UTC)
    producer = {
        "task_id": "task-1",
        "task_name": "summary-writer",
        "tool_name": "internal_create_memory_record",
        "provider_key": "provider-a",
        "provider_name": "Provider A",
        "model_name": "model-a",
    }
    rows = [
        _row("generic-1", created_at=now - timedelta(hours=2), producer=producer),
        _row("generic-2", created_at=now - timedelta(hours=1), producer=producer),
        _row("trace-1", created_at=now - timedelta(hours=2), title="task_complete: one", producer=producer),
        _row("trace-2", created_at=now - timedelta(hours=1), title="task_complete: two", producer=producer),
        _row("isolated", created_at=now - timedelta(minutes=30), producer={**producer, "model_name": "model-b"}),
    ]

    signals = build_quality_remediation_signals(
        rows,
        cutoff=(now - timedelta(days=1)).timestamp(),
        generated_at=now.timestamp(),
    )

    assert [(signal.defect_family, signal.count) for signal in signals] == [
        ("generic_summary_count", 2),
        ("trace_like_memory_count", 2),
    ]
    assert signals[0].threshold == 2
    assert signals[0].producer.model_name == "model-a"
    assert signals[0].window_end - signals[0].window_start == 86_400
    assert "isolated" not in signals[0].memory_ids

    split_window_signals = build_quality_remediation_signals(
        [
            _row("old-generic", created_at=now - timedelta(days=1, hours=1), producer=producer),
            _row("new-generic", created_at=now - timedelta(hours=1), producer=producer),
        ],
        cutoff=(now - timedelta(days=2)).timestamp(),
        generated_at=now.timestamp(),
    )
    assert split_window_signals == []

    changed_policy_signals = build_quality_remediation_signals(
        rows,
        cutoff=(now - timedelta(days=1)).timestamp(),
        generated_at=now.timestamp(),
        policy_version="cur-065-v2",
    )
    assert [signal.idempotency_key for signal in changed_policy_signals] != [
        signal.idempotency_key for signal in signals
    ]


def test_producer_remediation_signals_enqueue_idempotently(db_manager) -> None:
    now = datetime(2026, 4, 11, 12, tzinfo=UTC)
    producer = {"task_id": "task-1", "task_name": "summary-writer", "model_name": "model-a"}
    rows = [
        _row("generic-1", created_at=now - timedelta(hours=2), producer=producer),
        _row("generic-2", created_at=now - timedelta(hours=1), producer=producer),
    ]
    signals = build_quality_remediation_signals(
        rows,
        cutoff=(now - timedelta(days=1)).timestamp(),
        generated_at=now.timestamp(),
    )
    repository = SQLiteWorkItemRepository(db_manager)

    first = enqueue_producer_remediation_signals(repository, signals)
    second = enqueue_producer_remediation_signals(repository, signals)

    assert len(first) == len(second) == 1
    assert first[0].id == second[0].id
    assert first[0].family_key == WORK_FAMILY_OPERATOR_REVIEW
    assert first[0].execution_lane == EXECUTION_LANE_AGENTIC
    assert first[0].payload["work_item_kind"] == "producer_remediation"
    assert first[0].payload["operator_review_required"] is True
    assert len(repository.list_items(family_key=WORK_FAMILY_OPERATOR_REVIEW)) == 1
