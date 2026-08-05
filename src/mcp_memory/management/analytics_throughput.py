from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean

from mcp_memory.management.analytics_common import _percentile
from mcp_memory.management.models import AgentThroughputBucketPayload, ProviderLatencyBucketPayload
from mcp_memory.management.reporting_rows import ProviderUsageRow, TaskRunRow


@dataclass
class _TaskBucketAccumulator:
    total_runs: int = 0
    completed_runs: int = 0
    failed_runs: int = 0
    retry_runs: int = 0
    durations: list[float] = field(default_factory=list)


@dataclass
class _ProviderBucketAccumulator:
    call_count: int = 0
    failure_count: int = 0
    durations: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class NerdMetricsThroughputRollups:
    agent_throughput: list[AgentThroughputBucketPayload] = field(default_factory=list)
    provider_latency: list[ProviderLatencyBucketPayload] = field(default_factory=list)
    provider_failures: int = 0
    provider_skips: int = 0
    provider_p95_latency: float = 0.0
    provider_failure_rate: float = 0.0
    provider_skip_rate: float = 0.0
    provider_call_count: int = 0
    provider_claimed_work_item_count: int = 0
    provider_mutations: int = 0
    provider_tool_calls: int = 0
    compatible_batch_calls: int = 0


def build_nerd_metrics_throughput_rollups(
    *,
    task_rows: list[TaskRunRow],
    provider_rows: list[ProviderUsageRow],
    bucket_seconds: int,
) -> NerdMetricsThroughputRollups:
    task_buckets: dict[float, _TaskBucketAccumulator] = {}
    for row in task_rows:
        bucket_start = float(int(row.completed_at // bucket_seconds) * bucket_seconds)
        bucket = task_buckets.setdefault(bucket_start, _TaskBucketAccumulator())
        bucket.total_runs += 1
        status = row.status
        if status == "completed":
            bucket.completed_runs += 1
        elif status == "failed":
            bucket.failed_runs += 1
        elif status == "retry":
            bucket.retry_runs += 1
        bucket.durations.append(row.duration_seconds)

    agent_throughput = [
        AgentThroughputBucketPayload(
            bucket_start=bucket_start,
            total_runs=bucket.total_runs,
            completed_runs=bucket.completed_runs,
            failed_runs=bucket.failed_runs,
            retry_runs=bucket.retry_runs,
            avg_duration_seconds=round(mean(bucket.durations), 4) if bucket.durations else 0.0,
        )
        for bucket_start, bucket in sorted(task_buckets.items())
    ]

    provider_buckets: dict[tuple[float, str, str, str], _ProviderBucketAccumulator] = {}
    all_provider_durations: list[float] = []
    provider_failures = 0
    provider_skips = 0
    provider_call_count = 0
    provider_claimed_work_item_count = 0
    provider_mutations = 0
    provider_tool_calls = 0
    compatible_batch_calls = 0

    for row in task_rows:
        result_metadata = row.result.metadata
        provider_calls_used = result_metadata.provider_calls_used or 0
        provider_call_count += provider_calls_used
        if provider_calls_used > 0:
            provider_claimed_work_item_count += result_metadata.claimed_work_item_count or 0
            provider_mutations += result_metadata.mutations or 0
            provider_tool_calls += result_metadata.tool_calls_executed or 0
        compatible_batch_calls += result_metadata.compatible_batch_calls or 0

    for row in provider_rows:
        status = row.status
        if status == "skipped":
            provider_skips += 1
            continue

        bucket_start = float(int(row.created_at // bucket_seconds) * bucket_seconds)
        key = (
            bucket_start,
            row.provider_key,
            row.provider_name,
            row.model_name,
        )
        bucket = provider_buckets.setdefault(key, _ProviderBucketAccumulator())
        duration = row.duration_seconds
        bucket.call_count += 1
        bucket.durations.append(duration)
        all_provider_durations.append(duration)
        if status != "success":
            bucket.failure_count += 1
            provider_failures += 1

    provider_latency = [
        ProviderLatencyBucketPayload(
            bucket_start=bucket_start,
            provider_key=provider_key,
            provider_name=provider_name,
            model_name=model_name,
            call_count=bucket.call_count,
            failure_count=bucket.failure_count,
            avg_duration_seconds=round(mean(bucket.durations), 4) if bucket.durations else 0.0,
            p95_duration_seconds=round(_percentile(bucket.durations, 0.95), 4),
        )
        for (bucket_start, provider_key, provider_name, model_name), bucket in sorted(provider_buckets.items())
    ]

    executed_provider_rows = len(provider_rows) - provider_skips
    provider_failure_rate = 0.0 if executed_provider_rows <= 0 else provider_failures / executed_provider_rows
    provider_skip_rate = 0.0 if not provider_rows else provider_skips / len(provider_rows)

    return NerdMetricsThroughputRollups(
        agent_throughput=agent_throughput,
        provider_latency=provider_latency,
        provider_failures=provider_failures,
        provider_skips=provider_skips,
        provider_p95_latency=_percentile(all_provider_durations, 0.95),
        provider_failure_rate=provider_failure_rate,
        provider_skip_rate=provider_skip_rate,
        provider_call_count=provider_call_count,
        provider_claimed_work_item_count=provider_claimed_work_item_count,
        provider_mutations=provider_mutations,
        provider_tool_calls=provider_tool_calls,
        compatible_batch_calls=compatible_batch_calls,
    )
