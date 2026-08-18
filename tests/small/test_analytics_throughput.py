from __future__ import annotations

import pytest

from mcp_memory.management.analytics_reporting import build_nerd_metrics_throughput_rollups
from mcp_memory.management.reporting_rows import ProviderUsageRow, TaskRunRow, coerce_task_result_view

pytestmark = pytest.mark.small


def test_build_nerd_metrics_throughput_rollups_aggregates_buckets_and_premium_counters() -> None:
    rollups = build_nerd_metrics_throughput_rollups(
        task_rows=[
            TaskRunRow(
                completed_at=120.0,
                status="completed",
                duration_seconds=2.0,
                result=coerce_task_result_view({
                    "provider_calls_used": 2,
                    "claimed_work_item_count": 6,
                    "mutations": 4,
                    "tool_calls_executed": 3,
                    "compatible_batch_calls": 1,
                }),
            ),
            TaskRunRow(
                completed_at=150.0,
                status="failed",
                duration_seconds=4.0,
                result=coerce_task_result_view({
                    "provider_calls_used": 1,
                    "claimed_work_item_count": 2,
                    "mutations": 1,
                    "tool_calls_executed": 5,
                    "compatible_batch_calls": 2,
                }),
            ),
            TaskRunRow(
                completed_at=170.0,
                status="retry",
                duration_seconds=0.0,
                result=coerce_task_result_view({}),
            ),
        ],
        provider_rows=[
            ProviderUsageRow(
                created_at=130.0,
                provider_key="gemini-cli",
                provider_name="Gemini CLI",
                model_name="gemini-3-flash-preview",
                status="success",
                duration_seconds=0.25,
            ),
            ProviderUsageRow(
                created_at=140.0,
                provider_key="gemini-cli",
                provider_name="Gemini CLI",
                model_name="gemini-3-flash-preview",
                status="error",
                duration_seconds=0.75,
            ),
            ProviderUsageRow(
                created_at=150.0,
                provider_key="copilot-mini",
                provider_name="Copilot CLI",
                model_name="gpt-5-mini",
                status="skipped",
                duration_seconds=0.0,
            ),
        ],
        bucket_seconds=60,
    )

    assert len(rollups.agent_throughput) == 1
    assert rollups.agent_throughput[0].bucket_start == 120.0
    assert rollups.agent_throughput[0].total_runs == 3
    assert rollups.agent_throughput[0].completed_runs == 1
    assert rollups.agent_throughput[0].failed_runs == 1
    assert rollups.agent_throughput[0].retry_runs == 1
    assert rollups.agent_throughput[0].avg_duration_seconds == 2.0

    assert len(rollups.provider_latency) == 1
    assert rollups.provider_latency[0].provider_key == "gemini-cli"
    assert rollups.provider_latency[0].call_count == 2
    assert rollups.provider_latency[0].failure_count == 1
    assert rollups.provider_latency[0].avg_duration_seconds == 0.5
    assert rollups.provider_latency[0].p95_duration_seconds == 0.75

    assert rollups.provider_failures == 1
    assert rollups.provider_skips == 1
    assert rollups.provider_p95_latency == 0.75
    assert rollups.provider_failure_rate == 0.5
    assert rollups.provider_skip_rate == pytest.approx(1 / 3)
    assert rollups.provider_call_count == 3
    assert rollups.provider_claimed_work_item_count == 8
    assert rollups.provider_mutations == 5
    assert rollups.provider_tool_calls == 8
    assert rollups.compatible_batch_calls == 3


def test_build_nerd_metrics_attributes_provider_rows_by_task_id() -> None:
    rollups = build_nerd_metrics_throughput_rollups(
        task_rows=[
            TaskRunRow(
                task_id="curator-task",
                completed_at=120.0,
                status="completed",
                duration_seconds=1.0,
                result=coerce_task_result_view({"mutations": 2}),
            )
        ],
        provider_rows=[
            ProviderUsageRow(
                task_id="curator-task",
                provider_key="copilot",
                provider_name="Copilot",
                model_name="model",
                status="success",
                duration_seconds=1.0,
                created_at=120.0,
            ),
            ProviderUsageRow(
                task_id="other-task",
                provider_key="copilot",
                provider_name="Copilot",
                model_name="model",
                status="success",
                duration_seconds=1.0,
                created_at=120.0,
            ),
        ],
        bucket_seconds=60,
    )

    assert rollups.provider_call_count == 1
