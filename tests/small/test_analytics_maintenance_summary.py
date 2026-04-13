from __future__ import annotations

import pytest

from mcp_memory.management.analytics_maintenance_summary import build_maintenance_summary
from mcp_memory.management.reporting_rows import MaintenanceTaskRunRow
from mcp_memory.management.result_views import coerce_task_result_view


pytestmark = pytest.mark.small


def test_build_maintenance_summary_rolls_up_family_agent_and_delta_series() -> None:
    payload = build_maintenance_summary(
        maintenance_rows=[
            MaintenanceTaskRunRow(
                task_id="task-organization",
                task_name="project-manager",
                status="completed",
                completed_at=120.0,
                duration_seconds=0.0,
                result=coerce_task_result_view({"updated": 2, "meaningful_actions": 2}),
            ),
            MaintenanceTaskRunRow(
                task_id="task-verification",
                task_name="graph-linker",
                status="failed",
                completed_at=180.0,
                duration_seconds=0.0,
                result=coerce_task_result_view({"updated": 3, "lines_compressed": 5}),
            ),
            MaintenanceTaskRunRow(
                task_id="task-compaction",
                task_name="deduplicator",
                status="completed",
                completed_at=240.0,
                duration_seconds=0.0,
                result=coerce_task_result_view({"merged": 4, "archived": 1, "meaningful_actions": 3}),
            ),
        ],
        cutoff=100.0,
        generated_at=300.0,
        bucket_seconds=60,
    )

    organization = next(row for row in payload.by_family if row.key == "organization")
    verification = next(row for row in payload.by_family if row.key == "verification")
    compaction = next(row for row in payload.by_family if row.key == "compaction")
    deduplicator = next(row for row in payload.by_agent if row.key == "deduplicator")
    compaction_series = next(row for row in payload.family_delta_series if row.key == "compaction")

    assert organization.task_names == ["project-manager"]
    assert organization.updated_count == 2
    assert organization.meaningful_actions == 2
    assert organization.delta_total == 2

    assert verification.task_names == ["graph-linker"]
    assert verification.failed_runs == 1
    assert verification.updated_count == 3
    assert verification.lines_compressed == 5
    assert verification.delta_total == 3

    assert compaction.task_names == ["deduplicator"]
    assert compaction.completed_runs == 1
    assert compaction.merged_count == 4
    assert compaction.archived_count == 1
    assert compaction.meaningful_actions == 3
    assert compaction.delta_total == 5

    assert deduplicator.family_key == "compaction"
    assert deduplicator.delta_per_completed_run == 5.0
    assert deduplicator.actions_per_completed_run == 3.0

    assert [bucket.bucket_start for bucket in compaction_series.buckets] == [60.0, 120.0, 180.0, 240.0, 300.0]
    assert sum(bucket.merged_count for bucket in compaction_series.buckets) == 4
    assert sum(bucket.archived_count for bucket in compaction_series.buckets) == 1