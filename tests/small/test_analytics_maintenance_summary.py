from __future__ import annotations

import pytest

from mcp_memory.core.task_handlers import CURATOR_TASK_NAME
from mcp_memory.management.analytics_reporting import build_maintenance_summary
from mcp_memory.management.reporting_rows import MaintenanceTaskRunRow, coerce_task_result_view

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


def test_build_maintenance_summary_uses_generic_mutation_counts_for_curator_runs() -> None:
    payload = build_maintenance_summary(
        maintenance_rows=[
            MaintenanceTaskRunRow(
                task_id="task-curator",
                task_name=CURATOR_TASK_NAME,
                status="completed",
                completed_at=180.0,
                duration_seconds=0.0,
                result=coerce_task_result_view({"mutations": 4, "tool_calls_executed": 7}),
            ),
        ],
        cutoff=100.0,
        generated_at=240.0,
        bucket_seconds=60,
    )

    compaction = next(row for row in payload.by_family if row.key == "compaction")
    curator = next(row for row in payload.by_agent if row.key == CURATOR_TASK_NAME)
    compaction_series = next(row for row in payload.family_delta_series if row.key == "compaction")

    assert compaction.task_names == [CURATOR_TASK_NAME]
    assert compaction.meaningful_actions == 0
    assert compaction.mutation_count == 4
    assert compaction.delta_total == 4

    assert curator.mutation_count == 4
    assert curator.delta_total == 4
    assert curator.actions_per_completed_run == 4.0
    assert curator.delta_per_completed_run == 4.0

    assert sum(bucket.mutation_count for bucket in compaction_series.buckets) == 4
    assert sum(bucket.created_count for bucket in compaction_series.buckets) == 0


def test_build_maintenance_summary_reports_curator_no_ops_and_failures() -> None:
    def campaign(
        *,
        outcome: str,
        accepted_mutations: int = 0,
        planner_attempts: int = 1,
        verification_failure_count: int = 0,
        receipts: list[dict[str, str]] | None = None,
    ) -> dict[str, object]:
        return {
            "curation_outcome": outcome,
            "curation_campaign_result": {
                "budget_usage": {
                    "accepted_mutations": accepted_mutations,
                    "planner_attempts": planner_attempts,
                },
                "verification_failure_count": verification_failure_count,
                "receipts": receipts or [],
            },
        }

    payload = build_maintenance_summary(
        maintenance_rows=[
            MaintenanceTaskRunRow(
                task_id="task-no-op",
                task_name=CURATOR_TASK_NAME,
                status="completed",
                completed_at=180.0,
                duration_seconds=0.0,
                result=coerce_task_result_view(campaign(outcome="no_op")),
            ),
            MaintenanceTaskRunRow(
                task_id="task-failure",
                task_name=CURATOR_TASK_NAME,
                status="failed",
                completed_at=240.0,
                duration_seconds=0.0,
                result=coerce_task_result_view(campaign(outcome="provider_failed")),
            ),
            MaintenanceTaskRunRow(
                task_id="task-applied",
                task_name=CURATOR_TASK_NAME,
                status="completed",
                completed_at=300.0,
                duration_seconds=0.0,
                result=coerce_task_result_view(
                    campaign(
                        outcome="applied",
                        accepted_mutations=2,
                        planner_attempts=2,
                        verification_failure_count=1,
                        receipts=[
                            {"operation": "normalize_memory"},
                            {"operation": "create_link"},
                        ],
                    )
                ),
            ),
        ],
        cutoff=100.0,
        generated_at=360.0,
        bucket_seconds=60,
    )

    curator = next(row for row in payload.by_agent if row.key == CURATOR_TASK_NAME)

    assert curator.valid_plan_count == 2
    assert curator.valid_plan_rate == pytest.approx(2 / 3, abs=0.0001)
    assert curator.no_op_count == 1
    assert curator.no_op_rate == pytest.approx(1 / 3, abs=0.0001)
    assert curator.accepted_mutation_count == 2
    assert curator.verification_failure_count == 1
    assert curator.provider_failure_count == 1
    assert curator.retry_count == 1
    assert curator.mutation_categories == {"content_tag": 1, "structural_link": 1}