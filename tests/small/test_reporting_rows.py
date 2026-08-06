from __future__ import annotations

from datetime import UTC, datetime

import pytest

from mcp_memory.management.reporting_rows import (
    adapt_memory_count_row,
    adapt_scoped_memory_row,
    adapt_task_run_row,
    coerce_task_result_view,
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
    assert row.metadata == {"split_from_memory_id": "parent-1"}
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


def test_coerce_task_result_view_exposes_generic_mutation_and_tool_call_counts() -> None:
    result = coerce_task_result_view(
        {
            "mutations": 4,
            "tool_calls_executed": 7,
            "meaningful_actions": 2,
        }
    )

    assert result.mutation_count == 4
    assert result.tool_calls_executed == 7
    assert result.meaningful_actions == 2
    assert result.metadata.mutations == 4


def test_coerce_task_result_view_exposes_curator_yield_telemetry() -> None:
    result = coerce_task_result_view(
        {
            "curation_outcome": "applied",
            "curation_campaign_result": {
                "budget_usage": {"accepted_mutations": 2, "planner_attempts": 2, "provider_calls": 3},
                "verification_failure_count": 1,
                "receipts": [
                    {"operation": "normalize_memory"},
                    {"operation": "create_link"},
                ],
            },
        }
    )

    assert result.curation_outcome == "applied"
    assert result.curation_accepted_mutation_count == 2
    assert result.curation_verification_failure_count == 1
    assert result.curation_provider_failure_count == 0
    assert result.curation_retry_count == 1
    assert result.metadata.provider_calls_used == 3
    assert result.curation_mutation_categories == {
        "content_tag": 1,
        "structural_link": 1,
    }
