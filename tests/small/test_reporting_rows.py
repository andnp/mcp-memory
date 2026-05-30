from __future__ import annotations

from datetime import UTC, datetime

import pytest

from mcp_memory.management.reporting_rows import (
    adapt_memory_count_row,
    adapt_scoped_memory_row,
    adapt_task_run_row,
)
from mcp_memory.management.reporting_rows import TaskResultView


pytestmark = pytest.mark.small


def test_adapt_scoped_memory_row_parses_datetimes_and_split_lineage() -> None:
    row = adapt_scoped_memory_row(
        {
            "id": "memory-1",
            "title": "Typed row",
            "summary": "Summary",
            "type": "fact",
            "status": "active",
            "created_at": "2026-04-12T12:34:56",
            "updated_at": "2026-04-12T12:35:56+00:00",
            "last_accessed_at": None,
            "last_surfaced_at": "",
            "content_bytes": 42,
            "workspace_ids_csv": "workspace-a,workspace-b",
            "tags_csv": "tag-a,tag-b",
            "metadata": '{"split_from_memory_id": "parent-1"}',
        }
    )

    assert row.created_at == datetime(2026, 4, 12, 12, 34, 56, tzinfo=UTC)
    assert row.updated_at == datetime(2026, 4, 12, 12, 35, 56, tzinfo=UTC)
    assert row.last_accessed_at is None
    assert row.last_surfaced_at is None
    assert row.workspace_ids == ["workspace-a", "workspace-b"]
    assert row.tags == ["tag-a", "tag-b"]
    assert row.has_split_lineage is True


def test_adapters_reject_boolean_values_for_required_numeric_fields() -> None:
    with pytest.raises(TypeError):
        adapt_memory_count_row({"type": "fact", "status": "active", "count": True})

    with pytest.raises(TypeError):
        adapt_task_run_row(
            {
                "status": "completed",
                "completed_at": True,
                "duration_seconds": 1.0,
                "result_json": "{}",
            }
        )

    with pytest.raises(TypeError):
        adapt_task_run_row(
            {
                "status": "completed",
                "completed_at": 1.0,
                "duration_seconds": False,
                "result_json": "{}",
            }
        )


def test_adapt_task_run_row_coerces_invalid_result_json_to_empty_mapping() -> None:
    row = adapt_task_run_row(
        {
            "status": "completed",
            "completed_at": "123.0",
            "duration_seconds": 2,
            "result_json": "not-json",
        }
    )

    assert isinstance(row.result, TaskResultView)
    assert row.result == {}
    assert row.result.raw_payload == {}