from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
import math
from statistics import mean, median
import time

from mcp_memory.core.task_handlers import (
    CONFLICT_DETECTOR_TASK_NAME,
    CURATOR_TASK_NAME,
    DEFRAGMENTER_TASK_NAME,
    DEDUPLICATOR_TASK_NAME,
    FACT_CHECKER_TASK_NAME,
    GRAPH_LINKER_TASK_NAME,
    PROJECT_MANAGER_TASK_NAME,
    SWEEPER_TASK_NAME,
    TAXONOMIST_TASK_NAME,
)
from mcp_memory.management.agent_run_reporting import (
    decode_run_result,
    extract_ingest_audit,
    extract_run_result_metadata,
    format_result_summary,
)
from mcp_memory.management.health_reporting import build_search_health
from mcp_memory.management.models import (
    AgentThroughputBucketPayload,
    GraphTopologyPayload,
    MaintenanceEventPayload,
    MemoryTimelineBucketPayload,
    NerdRetrievalCallerKindRowPayload,
    MemoryLifecyclePayload,
    NerdAlertPayload,
    NerdCompositionPayload,
    NerdCountBucketPayload,
    NerdCountSeriesPayload,
    NerdMaintenanceAgentYieldPayload,
    NerdDistributionsPayload,
    NerdGrowthDynamicsPayload,
    NerdLifecycleTrendsPayload,
    NerdMaintenanceDeltaBucketPayload,
    NerdMaintenanceDeltaSeriesPayload,
    NerdMaintenancePayload,
    NerdMaintenanceSummaryPayload,
    NerdMaintenanceSummaryRowPayload,
    NerdMetricsPayload,
    NerdProviderPolicyPayload,
    NerdProviderPolicyProviderPayload,
    NerdProviderPolicyTaskPayload,
    NerdQualityDrilldownPayload,
    NerdQualityMemoryRowPayload,
    NerdQualityRemediationPayload,
    NerdQualitySignalDrilldownPayload,
    NerdRetrievalMemoryRowPayload,
    NerdRetrievalConversionMemoryRowPayload,
    NerdRetrievalFunnelPayload,
    NerdRetrievalPayload,
    NerdRetrievalQueryFamilyRowPayload,
    NerdRetrievalSummaryPayload,
    NerdRetrievalTagRowPayload,
    NerdRetrievalTagTimelinePayload,
    NerdStatPayload,
    NerdShareSeriesPayload,
    NerdTimeCountBucketPayload,
    NerdTimeShareBucketPayload,
    NerdTimelinesPayload,
    ProviderLatencyBucketPayload,
    QueueSnapshotPayload,
    SearchQualityPayload,
)
from mcp_memory.management.reporting_queries import (
    build_queue_diagnostics,
    list_memory_tool_event_rows_since,
    list_maintenance_task_run_rows_since,
    list_provider_usage_rows_since,
    list_runtime_log_rows_since,
    list_scoped_link_rows,
    list_scoped_memory_rows,
    list_task_run_rows_since,
    split_csv_values,
)
from mcp_memory.management.route_audit import build_task_route_audit


_TAG_LIMIT = 10
_DYNAMICS_LIMIT = 5
_RETRIEVAL_MEMORY_LIMIT = 100
_RETRIEVAL_QUERY_FAMILY_LIMIT = 25
_RETRIEVAL_TAG_LIMIT = 25
_RETRIEVAL_TAG_TIMELINE_LIMIT = 10
_AGE_BUCKETS: tuple[tuple[str, str, int, int | None], ...] = (
    ("lt_1d", "< 1 day", 0, 86_400),
    ("1d_to_7d", "1-7 days", 86_400, 7 * 86_400),
    ("7d_to_30d", "7-30 days", 7 * 86_400, 30 * 86_400),
    ("30d_to_90d", "30-90 days", 30 * 86_400, 90 * 86_400),
    ("gte_90d", ">= 90 days", 90 * 86_400, None),
)
_SIZE_BUCKETS: tuple[tuple[str, str, int, int | None], ...] = (
    ("0b_to_255b", "0-255 B", 0, 256),
    ("256b_to_1kb", "256 B-1 KiB", 256, 1_024),
    ("1kb_to_4kb", "1-4 KiB", 1_024, 4_096),
    ("4kb_to_16kb", "4-16 KiB", 4_096, 16_384),
    ("gte_16kb", ">= 16 KiB", 16_384, None),
)
_STATUS_EVENT_DEFS: tuple[tuple[str, str], ...] = (
    ("stale", "Stale promotions"),
    ("degraded", "Degraded markings"),
    ("archived", "Archived markings"),
    ("restored", "Restored markings"),
)
_RUN_REPORTED_DELTA_KEYS: tuple[tuple[str, str], ...] = (
    ("created", "created_count"),
    ("merged", "merged_count"),
    ("updated", "updated_count"),
    ("archived", "archived_count"),
    ("degraded", "degraded_count"),
    ("restored", "restored_count"),
    ("meaningful_actions", "meaningful_actions"),
    ("lines_compressed", "lines_compressed"),
)
_MAINTENANCE_FAMILY_DEFS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "organization",
        "Organization & taxonomy",
        (PROJECT_MANAGER_TASK_NAME, TAXONOMIST_TASK_NAME),
    ),
    (
        "verification",
        "Verification & linkage",
        (FACT_CHECKER_TASK_NAME, GRAPH_LINKER_TASK_NAME, CONFLICT_DETECTOR_TASK_NAME),
    ),
    (
        "compaction",
        "Compaction & curation",
        (DEFRAGMENTER_TASK_NAME, DEDUPLICATOR_TASK_NAME, CURATOR_TASK_NAME),
    ),
    (
        "retention",
        "Retention",
        (SWEEPER_TASK_NAME,),
    ),
)
_MAINTENANCE_FAMILY_BY_TASK = {
    task_name: (family_key, family_label)
    for family_key, family_label, task_names in _MAINTENANCE_FAMILY_DEFS
    for task_name in task_names
}
_PROVENANCE_TAG_PREFIXES: tuple[str, ...] = (
    "system1",
    "auto-",
    "merged-by-",
    "deduplicator-task-",
)
_PROVENANCE_TAG_SUFFIXES: tuple[str, ...] = (
    "-session",
    "-process",
)
_PROVENANCE_PROCESS_TAGS: frozenset[str] = frozenset(
    {
        "workflow",
        "testing-strategy",
        "user-preferences",
        "coding-standards",
        "documentation-style",
        "memory-operating-model",
        "startup",
    }
)
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


@dataclass
class _MaintenanceSummaryAccumulator:
    task_names: set[str] = field(default_factory=set)
    total_runs: int = 0
    completed_runs: int = 0
    failed_runs: int = 0
    retry_runs: int = 0
    created_count: int = 0
    merged_count: int = 0
    updated_count: int = 0
    archived_count: int = 0
    degraded_count: int = 0
    restored_count: int = 0
    meaningful_actions: int = 0
    lines_compressed: int = 0

    @property
    def delta_total(self) -> int:
        return (
            self.created_count
            + self.merged_count
            + self.updated_count
            + self.archived_count
            + self.degraded_count
            + self.restored_count
        )


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


@dataclass
class _RetrievalMemoryAccumulator:
    title: str
    memory_type: str
    status: str
    tags: list[str]
    read_count: int = 0
    search_count: int = 0
    converted_search_count: int = 0
    last_read_at: float | None = None
    last_search_at: float | None = None


@dataclass
class _RetrievalTagAccumulator:
    read_count: int = 0
    search_count: int = 0


@dataclass
class _RetrievalCallerKindAccumulator:
    search_invocations: int = 0
    search_hits: int = 0
    zero_result_searches: int = 0
    read_events: int = 0


@dataclass
class _RetrievalQueryFamilyAccumulator:
    label: str
    search_invocations: int = 0
    search_hits: int = 0
    zero_result_searches: int = 0
    converted_search_hits: int = 0
    unique_search_memory_ids: set[str] = field(default_factory=set)


@dataclass
class _RetrievalSearchInvocationAccumulator:
    caller_kind: str
    query_family_key: str
    query_family_label: str
    surfaced_memory_ids: set[str] = field(default_factory=set)
    zero_result: bool = False


@dataclass(frozen=True)
class _RetrievalSearchHitRecord:
    query_family_key: str
    memory_id: str
    created_at: float


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


@dataclass
class _ProviderPolicyTaskAccumulator:
    route_exhaustion_count: int = 0
    legacy_fallback_denied_count: int = 0
    admission_skip_count: int = 0
    skip_provider_counts: Counter[tuple[str, str, str | None]] = field(default_factory=Counter)
    active_admission_providers: set[tuple[str, str]] = field(default_factory=set)


@dataclass
class _ProviderPolicyProviderAccumulator:
    provider_name: str
    admission_skip_count: int = 0
    task_counts: Counter[str] = field(default_factory=Counter)
    reason_counts: Counter[str] = field(default_factory=Counter)
    active_admission_reason: str | None = None
    active_admission_category: str | None = None
    active_retry_delay_seconds: float | None = None


def build_nerd_metrics(
    *,
    db_manager,
    workspace_id: str | None,
    task_queue,
    provider_usage_repo,
    config,
    ai_json_provider,
    ai_agent_provider,
    ai_provider_registry,
    relational_search,
    window_hours: int = 24,
    bucket_minutes: int = 60,
    now: float | None = None,
) -> NerdMetricsPayload:
    if db_manager is None:
        return NerdMetricsPayload(
            generated_at=time.time() if now is None else now,
            window_hours=window_hours,
            bucket_minutes=bucket_minutes,
        )

    generated_at = time.time() if now is None else now
    bucket_seconds = max(bucket_minutes * 60, 60)
    cutoff = generated_at - (window_hours * 3600)
    task_rows = list_task_run_rows_since(db_manager, cutoff=cutoff, workspace_id=workspace_id)
    maintenance_rows = list_maintenance_task_run_rows_since(db_manager, cutoff=cutoff, workspace_id=workspace_id)
    provider_rows = list_provider_usage_rows_since(db_manager, cutoff=cutoff, workspace_id=workspace_id)
    provider_policy_log_rows = list_runtime_log_rows_since(
        db_manager,
        cutoff=cutoff,
        workspace_id=workspace_id,
        logger_name="mcp_memory.core.provider_policy",
        level="WARNING",
    )
    memory_rows = list_scoped_memory_rows(db_manager, workspace_id)
    retrieval_rows = list_memory_tool_event_rows_since(db_manager, cutoff=cutoff, workspace_id=workspace_id)

    task_buckets: dict[float, _TaskBucketAccumulator] = {}
    for row in task_rows:
        bucket_start = float(int(float(row["completed_at"]) // bucket_seconds) * bucket_seconds)
        bucket = task_buckets.setdefault(bucket_start, _TaskBucketAccumulator())
        bucket.total_runs += 1
        status = str(row["status"])
        if status == "completed":
            bucket.completed_runs += 1
        elif status == "failed":
            bucket.failed_runs += 1
        elif status == "retry":
            bucket.retry_runs += 1
        bucket.durations.append(float(row["duration_seconds"] or 0.0))

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
    for row in provider_rows:
        status = str(row["status"])
        if status == "skipped":
            provider_skips += 1
            continue
        bucket_start = float(int(float(row["created_at"]) // bucket_seconds) * bucket_seconds)
        key = (
            bucket_start,
            str(row["provider_key"]),
            str(row["provider_name"]),
            str(row["model_name"]),
        )
        bucket = provider_buckets.setdefault(key, _ProviderBucketAccumulator())
        duration = float(row["duration_seconds"] or 0.0)
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

    queue_rows = build_queue_diagnostics(task_queue, workspace_id, limit=200, now=generated_at)
    runnable_queue_rows = [row for row in queue_rows if row.pending_state == "runnable"]
    queue_snapshot = QueueSnapshotPayload(
        runnable_count=len(runnable_queue_rows),
        scheduled_count=sum(1 for row in queue_rows if row.pending_state == "scheduled"),
        oldest_age_seconds=round(max((row.age_seconds for row in runnable_queue_rows), default=0.0), 4),
    )

    graph_topology = build_graph_topology(db_manager, workspace_id, memory_rows=memory_rows)
    memory_lifecycle = build_memory_lifecycle(db_manager, workspace_id, memory_rows=memory_rows)
    composition = build_composition(memory_rows)
    distributions = build_distributions(memory_rows, generated_at=generated_at)
    quality_signals = build_memory_quality_signals(memory_rows)
    timelines = build_timelines(
        memory_rows,
        cutoff=cutoff,
        generated_at=generated_at,
        bucket_seconds=bucket_seconds,
    )
    maintenance = build_maintenance_events(
        maintenance_rows,
        cutoff=cutoff,
        generated_at=generated_at,
        bucket_seconds=bucket_seconds,
    )
    lifecycle_trends = build_lifecycle_trends(
        memory_rows,
        maintenance_rows=maintenance_rows,
        cutoff=cutoff,
        generated_at=generated_at,
        bucket_seconds=bucket_seconds,
    )
    growth_dynamics = build_growth_dynamics(
        memory_rows,
        cutoff=cutoff,
        generated_at=generated_at,
        bucket_seconds=bucket_seconds,
    )
    quality_drilldown = build_quality_drilldown(memory_rows)
    quality_remediation = build_quality_remediation(
        memory_rows,
        cutoff=cutoff,
        generated_at=generated_at,
        bucket_seconds=bucket_seconds,
    )
    retrieval = build_retrieval_analytics(
        memory_rows,
        retrieval_rows=retrieval_rows,
        cutoff=cutoff,
        generated_at=generated_at,
        bucket_seconds=bucket_seconds,
    )
    maintenance_summary = build_maintenance_summary(
        maintenance_rows,
        cutoff=cutoff,
        generated_at=generated_at,
        bucket_seconds=bucket_seconds,
    )
    search_quality = build_search_quality(
        search_health=build_search_health(relational_search),
        graph_topology=graph_topology,
        memory_lifecycle=memory_lifecycle,
    )
    route_audit = build_task_route_audit(
        config=config,
        workspace_id=workspace_id,
        provider_usage_repo=provider_usage_repo,
        ai_json_provider=ai_json_provider,
        ai_agent_provider=ai_agent_provider,
        ai_provider_registry=ai_provider_registry,
    )
    provider_policy = build_provider_policy_rollups(
        provider_rows=provider_rows,
        provider_policy_log_rows=provider_policy_log_rows,
        provider_usage_repo=provider_usage_repo,
        workspace_id=workspace_id,
    )

    executed_provider_rows = len(provider_rows) - provider_skips
    provider_failure_rate = 0.0 if executed_provider_rows <= 0 else provider_failures / executed_provider_rows
    provider_skip_rate = 0.0 if not provider_rows else provider_skips / len(provider_rows)

    stats = [
        NerdStatPayload(key="queue_oldest_age", label="Oldest runnable age", value=queue_snapshot.oldest_age_seconds, unit="s"),
        NerdStatPayload(key="runs_last_window", label="Runs in window", value=float(len(task_rows)), unit="runs"),
        NerdStatPayload(key="failed_runs_last_window", label="Failed runs in window", value=float(sum(1 for row in task_rows if str(row["status"]) == "failed")), unit="runs"),
        NerdStatPayload(key="provider_calls_last_window", label="Provider calls in window", value=float(len(provider_rows)), unit="calls"),
        NerdStatPayload(key="provider_failures_last_window", label="Provider failures in window", value=float(provider_failures), unit="calls"),
        NerdStatPayload(key="provider_skips_last_window", label="Provider skips in window", value=float(provider_skips), unit="calls"),
        NerdStatPayload(key="provider_p95_latency", label="Provider p95 latency", value=round(_percentile(all_provider_durations, 0.95), 4), unit="s"),
        NerdStatPayload(key="provider_failure_rate", label="Provider failure rate", value=round(provider_failure_rate, 4), unit="pct"),
        NerdStatPayload(key="provider_skip_rate", label="Provider skip rate", value=round(provider_skip_rate, 4), unit="pct"),
        NerdStatPayload(key="orphan_rate", label="Orphan rate", value=round(graph_topology.orphan_rate, 4), unit="pct"),
        NerdStatPayload(key="cold_memory_rate", label="Cold memory rate", value=round(memory_lifecycle.cold_memory_rate, 4), unit="pct"),
        NerdStatPayload(key="search_fallback_count", label="Search fallback count", value=float(search_quality.fallback_count), unit="count"),
        NerdStatPayload(key="trace_like_memory_count", label="Trace-like memories", value=float(quality_signals.trace_like_memory_count), unit="count"),
        NerdStatPayload(key="generic_summary_count", label="Generic summaries", value=float(quality_signals.generic_summary_count), unit="count"),
        NerdStatPayload(key="untagged_observation_count", label="Untagged observations", value=float(quality_signals.untagged_observation_count), unit="count"),
        NerdStatPayload(key="untagged_observation_rate", label="Untagged observation rate", value=round(quality_signals.untagged_observation_rate, 4), unit="pct"),
        NerdStatPayload(key="oversized_memory_count", label="Oversized memories", value=float(quality_signals.oversized_memory_count), unit="count"),
    ]

    return NerdMetricsPayload(
        generated_at=generated_at,
        window_hours=window_hours,
        bucket_minutes=bucket_minutes,
        stats=stats,
        composition=composition,
        distributions=distributions,
        timelines=timelines,
        maintenance=maintenance,
        lifecycle_trends=lifecycle_trends,
        growth_dynamics=growth_dynamics,
        quality_drilldown=quality_drilldown,
        quality_remediation=quality_remediation,
        retrieval=retrieval,
        maintenance_summary=maintenance_summary,
        queue_snapshot=queue_snapshot,
        graph_topology=graph_topology,
        memory_lifecycle=memory_lifecycle,
        search_quality=search_quality,
        route_audit=route_audit,
        provider_policy=provider_policy,
        alerts=build_nerd_alerts(
            queue_snapshot=queue_snapshot,
            graph_topology=graph_topology,
            memory_lifecycle=memory_lifecycle,
            search_quality=search_quality,
            route_audit=route_audit,
            provider_failure_rate=provider_failure_rate,
            quality_signals=quality_signals,
        ),
        agent_throughput=agent_throughput,
        provider_latency=provider_latency,
    )


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


def build_provider_policy_rollups(
    *,
    provider_rows,
    provider_policy_log_rows,
    provider_usage_repo,
    workspace_id: str | None,
) -> NerdProviderPolicyPayload:
    by_task: dict[str, _ProviderPolicyTaskAccumulator] = {}
    by_provider: dict[tuple[str, str], _ProviderPolicyProviderAccumulator] = {}
    total_route_exhaustion_count = 0
    total_legacy_fallback_denied_count = 0
    total_admission_skip_count = 0

    for row in provider_policy_log_rows:
        message = str(row["message"])
        task_name = _runtime_log_task_name(_runtime_log_data(row))
        accumulator = by_task.setdefault(task_name, _ProviderPolicyTaskAccumulator())
        if message == "Provider routing exhausted all configured routes":
            accumulator.route_exhaustion_count += 1
            total_route_exhaustion_count += 1
        elif message == "Legacy fallback provider is unavailable due to admission control":
            accumulator.legacy_fallback_denied_count += 1
            total_legacy_fallback_denied_count += 1

    for row in provider_rows:
        if str(row["status"]) != "skipped":
            continue
        task_name = _coerce_row_str(row["task_name"]) or "unknown"
        provider_key = str(row["provider_key"])
        provider_name = str(row["provider_name"])
        model_name = str(row["model_name"])
        reason_code = _coerce_row_str(row["reason_code"])

        task_accumulator = by_task.setdefault(task_name, _ProviderPolicyTaskAccumulator())
        task_accumulator.admission_skip_count += 1
        task_accumulator.skip_provider_counts[(provider_key, model_name, reason_code)] += 1

        provider_accumulator = by_provider.setdefault(
            (provider_key, model_name),
            _ProviderPolicyProviderAccumulator(provider_name=provider_name),
        )
        provider_accumulator.admission_skip_count += 1
        provider_accumulator.task_counts[task_name] += 1
        if reason_code is not None:
            provider_accumulator.reason_counts[reason_code] += 1
        total_admission_skip_count += 1

    for summary in provider_usage_repo.summarize_usage(workspace_id=workspace_id):
        if summary.active_admission_reason is None:
            continue
        provider_accumulator = by_provider.setdefault(
            (summary.provider_key, summary.model_name),
            _ProviderPolicyProviderAccumulator(provider_name=summary.provider_name),
        )
        provider_accumulator.active_admission_reason = summary.active_admission_reason
        provider_accumulator.active_admission_category = summary.active_admission_category
        provider_accumulator.active_retry_delay_seconds = summary.active_retry_delay_seconds
        if summary.task_name is not None:
            by_task.setdefault(summary.task_name, _ProviderPolicyTaskAccumulator()).active_admission_providers.add(
                (summary.provider_key, summary.model_name)
            )

    task_rows = [
        NerdProviderPolicyTaskPayload(
            task_name=task_name,
            route_exhaustion_count=accumulator.route_exhaustion_count,
            legacy_fallback_denied_count=accumulator.legacy_fallback_denied_count,
            admission_skip_count=accumulator.admission_skip_count,
            top_skip_provider_key=_counter_top_value(accumulator.skip_provider_counts, index=0),
            top_skip_model_name=_counter_top_value(accumulator.skip_provider_counts, index=1),
            top_skip_reason_code=_counter_top_value(accumulator.skip_provider_counts, index=2),
            active_admission_provider_count=len(accumulator.active_admission_providers),
        )
        for task_name, accumulator in by_task.items()
        if (
            accumulator.route_exhaustion_count > 0
            or accumulator.legacy_fallback_denied_count > 0
            or accumulator.admission_skip_count > 0
            or accumulator.active_admission_providers
        )
    ]
    task_rows.sort(
        key=lambda row: (
            -(row.route_exhaustion_count + row.legacy_fallback_denied_count + row.admission_skip_count),
            row.task_name,
        )
    )

    provider_rows_payload = [
        NerdProviderPolicyProviderPayload(
            provider_key=provider_key,
            provider_name=accumulator.provider_name,
            model_name=model_name,
            admission_skip_count=accumulator.admission_skip_count,
            distinct_task_count=len(accumulator.task_counts),
            top_task_name=_string_counter_top_key(accumulator.task_counts),
            top_reason_code=_string_counter_top_key(accumulator.reason_counts),
            active_admission_reason=accumulator.active_admission_reason,
            active_admission_category=accumulator.active_admission_category,
            active_retry_delay_seconds=accumulator.active_retry_delay_seconds,
        )
        for (provider_key, model_name), accumulator in by_provider.items()
        if accumulator.admission_skip_count > 0 or accumulator.active_admission_reason is not None
    ]
    provider_rows_payload.sort(key=lambda row: (-row.admission_skip_count, row.provider_key, row.model_name))

    return NerdProviderPolicyPayload(
        stats=[
            NerdStatPayload(
                key="provider_policy_route_exhaustion_count",
                label="Route exhaustion warnings",
                value=float(total_route_exhaustion_count),
                unit="count",
            ),
            NerdStatPayload(
                key="provider_policy_legacy_fallback_denied_count",
                label="Legacy fallback denials",
                value=float(total_legacy_fallback_denied_count),
                unit="count",
            ),
            NerdStatPayload(
                key="provider_policy_admission_skip_count",
                label="Admission-control skips",
                value=float(total_admission_skip_count),
                unit="count",
            ),
        ],
        by_task=task_rows,
        by_provider=provider_rows_payload,
    )


def build_graph_topology(db_manager, workspace_id: str | None, *, memory_rows=None) -> GraphTopologyPayload:
    if memory_rows is None:
        memory_rows = list_scoped_memory_rows(db_manager, workspace_id)
    memory_ids = {str(row["id"]) for row in memory_rows}
    if not memory_ids:
        return GraphTopologyPayload()

    link_rows = list_scoped_link_rows(db_manager, workspace_id, memory_ids)
    degree_by_memory = {memory_id: 0 for memory_id in memory_ids}
    support_by_memory: set[str] = set()
    link_type_counts: dict[str, int] = {}

    for row in link_rows:
        link_type = str(row["type"])
        link_type_counts[link_type] = link_type_counts.get(link_type, 0) + 1
        source_id = str(row["source_id"])
        target_id = str(row["target_id"])
        if source_id in degree_by_memory:
            degree_by_memory[source_id] += 1
        if target_id in degree_by_memory:
            degree_by_memory[target_id] += 1
            if link_type in {"DEPENDS_ON", "AMENDS"}:
                support_by_memory.add(target_id)

    total_memories = len(memory_ids)
    orphan_count = sum(1 for degree in degree_by_memory.values() if degree == 0)
    total_links = len(link_rows)
    average_degree = sum(degree_by_memory.values()) / total_memories
    return GraphTopologyPayload(
        total_memories=total_memories,
        total_links=total_links,
        average_degree=round(average_degree, 4),
        orphan_count=orphan_count,
        orphan_rate=round(orphan_count / total_memories, 4),
        graph_supported_count=len(support_by_memory),
        graph_supported_rate=round(len(support_by_memory) / total_memories, 4),
        link_type_counts=link_type_counts,
    )


def build_memory_lifecycle(db_manager, workspace_id: str | None, *, memory_rows=None) -> MemoryLifecyclePayload:
    if memory_rows is None:
        memory_rows = list_scoped_memory_rows(db_manager, workspace_id)
    if not memory_rows:
        return MemoryLifecyclePayload()

    by_status: dict[str, int] = {}
    by_type: dict[str, int] = {}
    content_sizes: list[int] = []
    cold_memory_count = 0
    never_surfaced_count = 0
    stale_count = 0
    degraded_count = 0

    for row in memory_rows:
        status = str(row["status"])
        memory_type = str(row["type"])
        by_status[status] = by_status.get(status, 0) + 1
        by_type[memory_type] = by_type.get(memory_type, 0) + 1

        content_size = int(row["content_bytes"] or 0)
        content_sizes.append(content_size)
        if row["last_accessed_at"] is None:
            cold_memory_count += 1
        if row["last_surfaced_at"] is None:
            never_surfaced_count += 1
        if status == "stale":
            stale_count += 1
        if status == "degraded":
            degraded_count += 1

    total_memories = len(memory_rows)
    return MemoryLifecyclePayload(
        by_status=by_status,
        by_type=by_type,
        total_content_bytes=sum(content_sizes),
        median_content_bytes=round(float(median(content_sizes)), 4) if content_sizes else 0.0,
        cold_memory_count=cold_memory_count,
        cold_memory_rate=round(cold_memory_count / total_memories, 4),
        never_surfaced_count=never_surfaced_count,
        stale_count=stale_count,
        degraded_count=degraded_count,
    )


def build_composition(memory_rows) -> NerdCompositionPayload:
    if not memory_rows:
        return NerdCompositionPayload()

    workspace_counts: Counter[str] = Counter()
    tag_counts: Counter[str] = Counter()
    content_tag_counts: Counter[str] = Counter()
    provenance_tag_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()

    for row in memory_rows:
        type_counts[str(row["type"])] += 1
        status_counts[str(row["status"])] += 1
        workspace_counts.update(split_csv_values(row["workspace_ids_csv"]))
        tags = split_csv_values(row["tags_csv"])
        tag_counts.update(tags)
        for tag in tags:
            if is_provenance_process_tag(tag):
                provenance_tag_counts[tag] += 1
            else:
                content_tag_counts[tag] += 1

    return NerdCompositionPayload(
        by_workspace=_count_buckets(workspace_counts),
        by_tag=_count_buckets(tag_counts, limit=_TAG_LIMIT, other_key="other", other_label="Other"),
        by_content_tag=_count_buckets(content_tag_counts, limit=_TAG_LIMIT, other_key="other", other_label="Other"),
        by_provenance_tag=_count_buckets(
            provenance_tag_counts,
            limit=_TAG_LIMIT,
            other_key="other",
            other_label="Other",
        ),
        by_type=_count_buckets(type_counts),
        by_status=_count_buckets(status_counts),
    )


def build_distributions(memory_rows, *, generated_at: float) -> NerdDistributionsPayload:
    created_counts = {key: 0 for key, _, _, _ in _AGE_BUCKETS}
    updated_counts = {key: 0 for key, _, _, _ in _AGE_BUCKETS}
    size_counts = {key: 0 for key, _, _, _ in _SIZE_BUCKETS}

    for row in memory_rows:
        created_at = _iso_to_timestamp(row["created_at"])
        updated_at = _iso_to_timestamp(row["updated_at"])
        content_bytes = int(row["content_bytes"] or 0)
        if created_at is not None:
            created_counts[_bucket_key_for_value(max(generated_at - created_at, 0.0), _AGE_BUCKETS)] += 1
        if updated_at is not None:
            updated_counts[_bucket_key_for_value(max(generated_at - updated_at, 0.0), _AGE_BUCKETS)] += 1
        size_counts[_bucket_key_for_value(float(content_bytes), _SIZE_BUCKETS)] += 1

    return NerdDistributionsPayload(
        created_age_buckets=_materialize_bucket_counts(_AGE_BUCKETS, created_counts),
        updated_age_buckets=_materialize_bucket_counts(_AGE_BUCKETS, updated_counts),
        content_size_buckets=_materialize_bucket_counts(_SIZE_BUCKETS, size_counts),
    )


def build_timelines(
    memory_rows,
    *,
    cutoff: float,
    generated_at: float,
    bucket_seconds: int,
) -> NerdTimelinesPayload:
    if not memory_rows or generated_at < cutoff:
        return NerdTimelinesPayload()

    start_bucket = int(cutoff // bucket_seconds) * bucket_seconds
    end_bucket = int(generated_at // bucket_seconds) * bucket_seconds
    if end_bucket < start_bucket:
        return NerdTimelinesPayload()

    created_counts: Counter[int] = Counter()
    updated_counts: Counter[int] = Counter()
    created_bytes: Counter[int] = Counter()
    baseline_bytes = 0

    for row in memory_rows:
        content_bytes = int(row["content_bytes"] or 0)
        created_at = _iso_to_timestamp(row["created_at"])
        updated_at = _iso_to_timestamp(row["updated_at"])
        if created_at is not None:
            if created_at < cutoff:
                baseline_bytes += content_bytes
            elif created_at <= generated_at:
                created_bucket = int(created_at // bucket_seconds) * bucket_seconds
                created_counts[created_bucket] += 1
                created_bytes[created_bucket] += content_bytes
        if updated_at is not None and cutoff <= updated_at <= generated_at:
            updated_bucket = int(updated_at // bucket_seconds) * bucket_seconds
            updated_counts[updated_bucket] += 1

    running_bytes = baseline_bytes
    memory_activity: list[MemoryTimelineBucketPayload] = []
    for bucket_start in range(start_bucket, end_bucket + bucket_seconds, bucket_seconds):
        running_bytes += created_bytes.get(bucket_start, 0)
        memory_activity.append(
            MemoryTimelineBucketPayload(
                bucket_start=float(bucket_start),
                created_count=created_counts.get(bucket_start, 0),
                updated_count=updated_counts.get(bucket_start, 0),
                total_content_bytes=running_bytes,
            )
        )
    return NerdTimelinesPayload(memory_activity=memory_activity)


def build_maintenance_events(
    maintenance_rows,
    *,
    cutoff: float,
    generated_at: float,
    bucket_seconds: int,
) -> NerdMaintenancePayload:
    if not maintenance_rows or generated_at < cutoff:
        return NerdMaintenancePayload()

    events: list[MaintenanceEventPayload] = []
    for row in maintenance_rows:
        completed_at = float(row["completed_at"] or 0.0)
        if completed_at < cutoff or completed_at > generated_at:
            continue

        result = decode_run_result(row["result_json"])
        result_metadata = extract_run_result_metadata(result)
        ingest_audit = extract_ingest_audit(result)
        events.append(
            MaintenanceEventPayload(
                task_id=str(row["task_id"]),
                task_name=str(row["task_name"]),
                status=str(row["status"]),
                completed_at=completed_at,
                bucket_start=float(int(completed_at // bucket_seconds) * bucket_seconds),
                duration_seconds=float(row["duration_seconds"] or 0.0),
                result_summary=format_result_summary(result) or _coerce_row_str(row["error_text"]),
                strategy_used=result_metadata.strategy_used,
                impact_summary=_build_maintenance_impact_summary(result, ingest_audit=ingest_audit),
                candidate_count=result_metadata.candidate_count,
                group_count=result_metadata.group_count,
                created_count=_prefer_nonzero_int(_result_int(result, "created"), ingest_audit.created_count),
                merged_count=_result_int(result, "merged"),
                updated_count=_result_int(result, "updated"),
                archived_count=_result_int(result, "archived"),
                lines_compressed=_result_int(result, "lines_compressed"),
                meaningful_actions=_prefer_nonzero_int(
                    _result_int(result, "meaningful_actions"),
                    ingest_audit.meaningful_actions,
                ),
            )
        )
    return NerdMaintenancePayload(events=events)


def build_search_quality(*, search_health, graph_topology: GraphTopologyPayload, memory_lifecycle: MemoryLifecyclePayload) -> SearchQualityPayload:
    return SearchQualityPayload(
        semantic_enabled=search_health.semantic_enabled,
        degraded=search_health.degraded,
        fallback_count=search_health.fallback_count,
        rebuild_count=search_health.rebuild_count,
        graph_supported_count=graph_topology.graph_supported_count,
        graph_supported_rate=graph_topology.graph_supported_rate,
        never_surfaced_count=memory_lifecycle.never_surfaced_count,
        last_error=search_health.last_error,
    )


def build_lifecycle_trends(
    memory_rows,
    *,
    maintenance_rows,
    cutoff: float,
    generated_at: float,
    bucket_seconds: int,
) -> NerdLifecycleTrendsPayload:
    bucket_starts = _bucket_starts(cutoff=cutoff, generated_at=generated_at, bucket_seconds=bucket_seconds)
    if not bucket_starts:
        return NerdLifecycleTrendsPayload()

    never_surfaced_backlog = _build_backlog_series(
        memory_rows,
        cutoff=cutoff,
        generated_at=generated_at,
        bucket_seconds=bucket_seconds,
        predicate=lambda row: row["last_surfaced_at"] is None,
    )
    cold_tail = _build_backlog_series(
        memory_rows,
        cutoff=cutoff,
        generated_at=generated_at,
        bucket_seconds=bucket_seconds,
        predicate=lambda row: row["last_accessed_at"] is None,
    )

    event_buckets: dict[str, Counter[int]] = {key: Counter() for key, _ in _STATUS_EVENT_DEFS}
    observed_event_keys: set[str] = set()
    for row in maintenance_rows:
        completed_at = float(row["completed_at"] or 0.0)
        if completed_at < cutoff or completed_at > generated_at:
            continue
        result = decode_run_result(row["result_json"])
        bucket_start = int(completed_at // bucket_seconds) * bucket_seconds
        for key, _label in _STATUS_EVENT_DEFS:
            value = _result_int(result, key)
            if value is None:
                continue
            observed_event_keys.add(key)
            if value > 0:
                event_buckets[key][bucket_start] += value

    status_events = [
        NerdCountSeriesPayload(
            key=key,
            label=label,
            buckets=[
                NerdTimeCountBucketPayload(bucket_start=float(bucket_start), count=event_buckets[key].get(bucket_start, 0))
                for bucket_start in bucket_starts
            ],
        )
        for key, label in _STATUS_EVENT_DEFS
        if key in observed_event_keys
    ]
    return NerdLifecycleTrendsPayload(
        status_events=status_events,
        never_surfaced_backlog=never_surfaced_backlog,
        cold_tail=cold_tail,
        quality_signals=_build_quality_signal_series(
            memory_rows,
            cutoff=cutoff,
            generated_at=generated_at,
            bucket_seconds=bucket_seconds,
        ),
    )


def build_growth_dynamics(
    memory_rows,
    *,
    cutoff: float,
    generated_at: float,
    bucket_seconds: int,
) -> NerdGrowthDynamicsPayload:
    bucket_starts = _bucket_starts(cutoff=cutoff, generated_at=generated_at, bucket_seconds=bucket_seconds)
    if not bucket_starts or not memory_rows:
        return NerdGrowthDynamicsPayload()

    tag_counts: Counter[str] = Counter()
    workspace_counts: Counter[str] = Counter()
    for row in memory_rows:
        tag_counts.update(split_csv_values(row["tags_csv"]))
        workspace_counts.update(split_csv_values(row["workspace_ids_csv"]))

    top_tag_keys, include_other_tags = _select_top_keys(tag_counts, limit=_DYNAMICS_LIMIT)
    top_workspace_keys, include_other_workspaces = _select_top_keys(workspace_counts, limit=_DYNAMICS_LIMIT)

    return NerdGrowthDynamicsPayload(
        top_tag_trends=_build_cumulative_dimension_count_series(
            memory_rows,
            cutoff=cutoff,
            generated_at=generated_at,
            bucket_seconds=bucket_seconds,
            bucket_starts=bucket_starts,
            top_keys=top_tag_keys,
            include_other=include_other_tags,
            extractor=lambda row: split_csv_values(row["tags_csv"]),
        ),
        workspace_contribution_share=_build_cumulative_dimension_share_series(
            memory_rows,
            cutoff=cutoff,
            generated_at=generated_at,
            bucket_seconds=bucket_seconds,
            bucket_starts=bucket_starts,
            top_keys=top_workspace_keys,
            include_other=include_other_workspaces,
            extractor=lambda row: split_csv_values(row["workspace_ids_csv"]),
        ),
    )


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


def build_maintenance_summary(
    maintenance_rows,
    *,
    cutoff: float,
    generated_at: float,
    bucket_seconds: int,
) -> NerdMaintenanceSummaryPayload:
    bucket_starts = _bucket_starts(cutoff=cutoff, generated_at=generated_at, bucket_seconds=bucket_seconds)
    if not bucket_starts or not maintenance_rows:
        return NerdMaintenanceSummaryPayload()

    family_accumulators: dict[str, _MaintenanceSummaryAccumulator] = {}
    agent_accumulators: dict[str, _MaintenanceSummaryAccumulator] = {}
    agent_families: dict[str, tuple[str, str]] = {}
    family_bucket_totals: dict[str, dict[int, dict[str, int]]] = {}

    for row in maintenance_rows:
        completed_at = float(row["completed_at"] or 0.0)
        if completed_at < cutoff or completed_at > generated_at:
            continue
        task_name = str(row["task_name"])
        family_key, family_label = _maintenance_family_for_task(task_name)
        family_acc = family_accumulators.setdefault(family_key, _MaintenanceSummaryAccumulator())
        agent_acc = agent_accumulators.setdefault(task_name, _MaintenanceSummaryAccumulator())
        agent_families[task_name] = (family_key, family_label)

        result = decode_run_result(row["result_json"])
        deltas = _run_reported_delta_counts(result)
        _update_maintenance_summary_accumulator(family_acc, task_name=task_name, status=str(row["status"]), deltas=deltas)
        _update_maintenance_summary_accumulator(agent_acc, task_name=task_name, status=str(row["status"]), deltas=deltas)

        bucket_start = int(completed_at // bucket_seconds) * bucket_seconds
        family_bucket = family_bucket_totals.setdefault(family_key, {}).setdefault(
            bucket_start,
            {attribute_name: 0 for _, attribute_name in _RUN_REPORTED_DELTA_KEYS},
        )
        for _key, attribute_name in _RUN_REPORTED_DELTA_KEYS:
            family_bucket[attribute_name] += deltas[attribute_name]

    by_family = [
        NerdMaintenanceSummaryRowPayload(
            key=family_key,
            label=family_label,
            task_names=sorted(accumulator.task_names),
            total_runs=accumulator.total_runs,
            completed_runs=accumulator.completed_runs,
            failed_runs=accumulator.failed_runs,
            retry_runs=accumulator.retry_runs,
            created_count=accumulator.created_count,
            merged_count=accumulator.merged_count,
            updated_count=accumulator.updated_count,
            archived_count=accumulator.archived_count,
            degraded_count=accumulator.degraded_count,
            restored_count=accumulator.restored_count,
            meaningful_actions=accumulator.meaningful_actions,
            lines_compressed=accumulator.lines_compressed,
            delta_total=accumulator.delta_total,
        )
        for family_key, family_label, _task_names in _MAINTENANCE_FAMILY_DEFS
        if (accumulator := family_accumulators.get(family_key)) is not None
    ]

    by_agent = [
        _build_agent_yield_payload(
            task_name=task_name,
            family_key=agent_families[task_name][0],
            family_label=agent_families[task_name][1],
            accumulator=accumulator,
        )
        for task_name, accumulator in sorted(
            agent_accumulators.items(),
            key=lambda item: (-item[1].delta_total, -item[1].meaningful_actions, item[0]),
        )
    ]

    family_delta_series = [
        NerdMaintenanceDeltaSeriesPayload(
            key=family_key,
            label=family_label,
            task_names=sorted(family_accumulators[family_key].task_names),
            buckets=[
                NerdMaintenanceDeltaBucketPayload(
                    bucket_start=float(bucket_start),
                    **family_bucket_totals.get(family_key, {}).get(
                        bucket_start,
                        {attribute_name: 0 for _, attribute_name in _RUN_REPORTED_DELTA_KEYS},
                    ),
                )
                for bucket_start in bucket_starts
            ],
        )
        for family_key, family_label, _task_names in _MAINTENANCE_FAMILY_DEFS
        if family_key in family_accumulators
    ]

    return NerdMaintenanceSummaryPayload(
        by_family=by_family,
        by_agent=by_agent,
        family_delta_series=family_delta_series,
    )


def build_retrieval_analytics(
    memory_rows,
    *,
    retrieval_rows,
    cutoff: float,
    generated_at: float,
    bucket_seconds: int,
) -> NerdRetrievalPayload:
    bucket_starts = _bucket_starts(cutoff=cutoff, generated_at=generated_at, bucket_seconds=bucket_seconds)
    if not bucket_starts:
        return NerdRetrievalPayload()

    memory_info = {
        str(row["id"]): {
            "title": str(row["title"]),
            "memory_type": str(row["type"]),
            "status": str(row["status"]),
            "tags": split_csv_values(row["tags_csv"]),
        }
        for row in memory_rows
    }
    memory_accumulators: dict[str, _RetrievalMemoryAccumulator] = {}
    tag_accumulators: dict[str, _RetrievalTagAccumulator] = {}
    caller_kind_accumulators: dict[str, _RetrievalCallerKindAccumulator] = {}
    query_family_accumulators: dict[str, _RetrievalQueryFamilyAccumulator] = {}
    search_invocation_accumulators: dict[str, _RetrievalSearchInvocationAccumulator] = {}
    tag_read_buckets: dict[str, Counter[int]] = {}
    tag_search_buckets: dict[str, Counter[int]] = {}
    search_invocation_ids: set[str] = set()
    zero_result_search_invocation_ids: set[str] = set()
    unique_search_memories: set[str] = set()
    unique_read_memories: set[str] = set()
    read_event_times_by_memory: dict[str, list[float]] = {}
    search_hit_records: list[_RetrievalSearchHitRecord] = []
    converted_search_hits = 0
    read_events = 0
    search_hits = 0

    for row in retrieval_rows:
        event_kind = str(row["event_kind"])
        invocation_id = str(row["invocation_id"])
        created_at = float(row["created_at"] or 0.0)
        caller_kind = _normalize_caller_kind(row["caller_kind"])
        if event_kind == "search":
            search_invocation_ids.add(invocation_id)
            search_invocation = search_invocation_accumulators.setdefault(
                invocation_id,
                _RetrievalSearchInvocationAccumulator(
                    caller_kind=caller_kind,
                    query_family_key=_normalize_query_family_key(row["query_text"]),
                    query_family_label=_format_query_family_label(row["query_text"]),
                ),
            )
            if int(row["result_count"] or 0) == 0:
                zero_result_search_invocation_ids.add(invocation_id)
                search_invocation.zero_result = True

        memory_id = None if row["memory_id"] is None else str(row["memory_id"])
        if memory_id is None or memory_id not in memory_info:
            if event_kind == "read":
                caller_kind_accumulators.setdefault(caller_kind, _RetrievalCallerKindAccumulator()).read_events += 1
            continue

        info = memory_info[memory_id]
        memory_accumulator = memory_accumulators.setdefault(
            memory_id,
            _RetrievalMemoryAccumulator(
                title=info["title"],
                memory_type=info["memory_type"],
                status=info["status"],
                tags=list(info["tags"]),
            ),
        )
        bucket_start = int(created_at // bucket_seconds) * bucket_seconds
        for tag in info["tags"]:
            tag_accumulator = tag_accumulators.setdefault(tag, _RetrievalTagAccumulator())
            if event_kind == "read":
                tag_accumulator.read_count += 1
                tag_read_buckets.setdefault(tag, Counter())[bucket_start] += 1
            elif event_kind == "search":
                tag_accumulator.search_count += 1
                tag_search_buckets.setdefault(tag, Counter())[bucket_start] += 1

        if event_kind == "read":
            memory_accumulator.read_count += 1
            memory_accumulator.last_read_at = created_at if memory_accumulator.last_read_at is None else max(memory_accumulator.last_read_at, created_at)
            unique_read_memories.add(memory_id)
            read_events += 1
            caller_kind_accumulators.setdefault(caller_kind, _RetrievalCallerKindAccumulator()).read_events += 1
            read_event_times_by_memory.setdefault(memory_id, []).append(created_at)
        elif event_kind == "search":
            memory_accumulator.search_count += 1
            memory_accumulator.last_search_at = created_at if memory_accumulator.last_search_at is None else max(memory_accumulator.last_search_at, created_at)
            unique_search_memories.add(memory_id)
            search_hits += 1
            search_invocation_accumulators[invocation_id].surfaced_memory_ids.add(memory_id)
            search_hit_records.append(
                _RetrievalSearchHitRecord(
                    query_family_key=search_invocation_accumulators[invocation_id].query_family_key,
                    memory_id=memory_id,
                    created_at=created_at,
                )
            )

    for search_hit_record in search_hit_records:
        read_times = read_event_times_by_memory.get(search_hit_record.memory_id, [])
        if not any(read_time >= search_hit_record.created_at for read_time in read_times):
            continue
        converted_search_hits += 1
        memory_accumulators[search_hit_record.memory_id].converted_search_count += 1
        query_family_accumulator = query_family_accumulators.setdefault(
            search_hit_record.query_family_key,
            _RetrievalQueryFamilyAccumulator(label=search_hit_record.query_family_key),
        )
        query_family_accumulator.converted_search_hits += 1

    for search_invocation in search_invocation_accumulators.values():
        caller_kind_accumulator = caller_kind_accumulators.setdefault(
            search_invocation.caller_kind,
            _RetrievalCallerKindAccumulator(),
        )
        caller_kind_accumulator.search_invocations += 1
        caller_kind_accumulator.search_hits += len(search_invocation.surfaced_memory_ids)
        if search_invocation.zero_result:
            caller_kind_accumulator.zero_result_searches += 1

        query_family_accumulator = query_family_accumulators.setdefault(
            search_invocation.query_family_key,
            _RetrievalQueryFamilyAccumulator(label=search_invocation.query_family_label),
        )
        query_family_accumulator.search_invocations += 1
        query_family_accumulator.search_hits += len(search_invocation.surfaced_memory_ids)
        if search_invocation.zero_result:
            query_family_accumulator.zero_result_searches += 1
        query_family_accumulator.unique_search_memory_ids.update(search_invocation.surfaced_memory_ids)

    by_caller_kind = [
        NerdRetrievalCallerKindRowPayload(
            key=caller_kind,
            label=_format_caller_kind_label(caller_kind),
            search_invocations=accumulator.search_invocations,
            search_hits=accumulator.search_hits,
            zero_result_searches=accumulator.zero_result_searches,
            read_events=accumulator.read_events,
            total_events=accumulator.search_invocations + accumulator.read_events,
        )
        for caller_kind, accumulator in sorted(
            caller_kind_accumulators.items(),
            key=lambda item: (
                -(item[1].search_invocations + item[1].read_events),
                -item[1].search_hits,
                -item[1].read_events,
                item[0],
            ),
        )
    ]

    top_query_families = [
        NerdRetrievalQueryFamilyRowPayload(
            key=family_key,
            label=accumulator.label,
            search_invocations=accumulator.search_invocations,
            search_hits=accumulator.search_hits,
            zero_result_searches=accumulator.zero_result_searches,
            unique_search_memories=len(accumulator.unique_search_memory_ids),
            converted_search_hits=accumulator.converted_search_hits,
            conversion_rate=0.0 if accumulator.search_hits == 0 else round(accumulator.converted_search_hits / accumulator.search_hits, 4),
        )
        for family_key, accumulator in sorted(
            query_family_accumulators.items(),
            key=lambda item: (
                -item[1].search_invocations,
                -item[1].search_hits,
                -len(item[1].unique_search_memory_ids),
                item[0],
            ),
        )[:_RETRIEVAL_QUERY_FAMILY_LIMIT]
    ]

    top_zero_result_query_families = [
        row
        for row in top_query_families
        if row.zero_result_searches > 0
    ]

    low_conversion_memories = [
        NerdRetrievalConversionMemoryRowPayload(
            memory_id=memory_id,
            title=accumulator.title,
            memory_type=accumulator.memory_type,
            status=accumulator.status,
            tags=accumulator.tags,
            read_count=accumulator.read_count,
            search_count=accumulator.search_count,
            converted_search_count=accumulator.converted_search_count,
            conversion_rate=0.0 if accumulator.search_count == 0 else round(accumulator.converted_search_count / accumulator.search_count, 4),
            last_read_at=accumulator.last_read_at,
            last_search_at=accumulator.last_search_at,
        )
        for memory_id, accumulator in sorted(
            memory_accumulators.items(),
            key=lambda item: (
                1.0 if item[1].search_count == 0 else round(item[1].converted_search_count / item[1].search_count, 4),
                -item[1].search_count,
                item[1].title.lower(),
                item[0],
            ),
        )
        if accumulator.search_count > 0
    ][:_RETRIEVAL_QUERY_FAMILY_LIMIT]

    top_read_memories = [
        NerdRetrievalMemoryRowPayload(
            memory_id=memory_id,
            title=accumulator.title,
            memory_type=accumulator.memory_type,
            status=accumulator.status,
            tags=accumulator.tags,
            read_count=accumulator.read_count,
            search_count=accumulator.search_count,
            total_count=accumulator.read_count + accumulator.search_count,
            last_read_at=accumulator.last_read_at,
            last_search_at=accumulator.last_search_at,
        )
        for memory_id, accumulator in sorted(
            memory_accumulators.items(),
            key=lambda item: (-item[1].read_count, -(item[1].last_read_at or 0.0), item[1].title.lower(), item[0]),
        )
        if accumulator.read_count > 0
    ][:_RETRIEVAL_MEMORY_LIMIT]

    top_search_memories = [
        NerdRetrievalMemoryRowPayload(
            memory_id=memory_id,
            title=accumulator.title,
            memory_type=accumulator.memory_type,
            status=accumulator.status,
            tags=accumulator.tags,
            read_count=accumulator.read_count,
            search_count=accumulator.search_count,
            total_count=accumulator.read_count + accumulator.search_count,
            last_read_at=accumulator.last_read_at,
            last_search_at=accumulator.last_search_at,
        )
        for memory_id, accumulator in sorted(
            memory_accumulators.items(),
            key=lambda item: (-item[1].search_count, -(item[1].last_search_at or 0.0), item[1].title.lower(), item[0]),
        )
        if accumulator.search_count > 0
    ][:_RETRIEVAL_MEMORY_LIMIT]

    ordered_tag_items = sorted(
        tag_accumulators.items(),
        key=lambda item: (-(item[1].read_count + item[1].search_count), -item[1].search_count, -item[1].read_count, item[0]),
    )
    top_tags = [
        NerdRetrievalTagRowPayload(
            key=tag,
            label=tag,
            read_count=accumulator.read_count,
            search_count=accumulator.search_count,
            total_count=accumulator.read_count + accumulator.search_count,
        )
        for tag, accumulator in ordered_tag_items[:_RETRIEVAL_TAG_LIMIT]
    ]

    tag_timelines = [
        NerdRetrievalTagTimelinePayload(
            key=tag,
            label=tag,
            read_buckets=[
                NerdTimeCountBucketPayload(bucket_start=float(bucket_start), count=tag_read_buckets.get(tag, Counter()).get(bucket_start, 0))
                for bucket_start in bucket_starts
            ],
            search_buckets=[
                NerdTimeCountBucketPayload(bucket_start=float(bucket_start), count=tag_search_buckets.get(tag, Counter()).get(bucket_start, 0))
                for bucket_start in bucket_starts
            ],
        )
        for tag, _accumulator in ordered_tag_items[:_RETRIEVAL_TAG_TIMELINE_LIMIT]
    ]

    return NerdRetrievalPayload(
        summary=NerdRetrievalSummaryPayload(
            search_invocations=len(search_invocation_ids),
            search_hits=search_hits,
            zero_result_searches=len(zero_result_search_invocation_ids),
            read_events=read_events,
            unique_search_memories=len(unique_search_memories),
            unique_read_memories=len(unique_read_memories),
        ),
        funnel=NerdRetrievalFunnelPayload(
            search_hits=search_hits,
            converted_search_hits=converted_search_hits,
            conversion_rate=0.0 if search_hits == 0 else round(converted_search_hits / search_hits, 4),
        ),
        by_caller_kind=by_caller_kind,
        top_query_families=top_query_families,
        top_zero_result_query_families=top_zero_result_query_families,
        top_read_memories=top_read_memories,
        top_search_memories=top_search_memories,
        low_conversion_memories=low_conversion_memories,
        top_tags=top_tags,
        tag_timelines=tag_timelines,
    )


def _normalize_caller_kind(value: object) -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized:
            return normalized
    return "unknown"


def _format_caller_kind_label(value: str) -> str:
    return value.replace("_", " ").title()


def _normalize_query_family_key(value: object) -> str:
    if not isinstance(value, str):
        return "empty-query"
    normalized = " ".join(value.lower().split())
    return normalized or "empty-query"


def _format_query_family_label(value: object) -> str:
    if not isinstance(value, str):
        return "(empty query)"
    normalized = " ".join(value.split())
    return normalized or "(empty query)"


def _runtime_log_data(row) -> dict[str, object]:
    raw_data = row["data_json"]
    if not isinstance(raw_data, str) or not raw_data.strip():
        return {}
    try:
        decoded = json.loads(raw_data)
    except ValueError:
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _runtime_log_task_name(log_data: dict[str, object]) -> str:
    extra = log_data.get("extra")
    if isinstance(extra, dict):
        task_name = extra.get("task_name")
        if isinstance(task_name, str) and task_name.strip():
            return task_name.strip()
    return "unknown"


def _counter_top_value(counter: Counter[tuple[str, str, str | None]], *, index: int) -> str | None:
    if not counter:
        return None
    top_key = max(counter.items(), key=lambda item: (item[1], item[0]))[0]
    value = top_key[index]
    return value if isinstance(value, str) and value else None


def _string_counter_top_key(counter: Counter[str]) -> str | None:
    if not counter:
        return None
    return max(counter.items(), key=lambda item: (item[1], item[0]))[0]


def build_nerd_alerts(
    *,
    queue_snapshot: QueueSnapshotPayload,
    graph_topology: GraphTopologyPayload,
    memory_lifecycle: MemoryLifecyclePayload,
    search_quality: SearchQualityPayload,
    route_audit,
    provider_failure_rate: float,
    quality_signals: _MemoryQualitySignals,
) -> list[NerdAlertPayload]:
    alerts: list[NerdAlertPayload] = []
    if queue_snapshot.oldest_age_seconds >= 300.0:
        alerts.append(
            NerdAlertPayload(
                key="queue_oldest_age",
                severity="warning",
                label="Queue backlog aging",
                message="The oldest runnable task is older than 5 minutes.",
                value=round(queue_snapshot.oldest_age_seconds, 4),
                threshold=300.0,
                unit="s",
            )
        )
    if provider_failure_rate >= 0.2:
        alerts.append(
            NerdAlertPayload(
                key="provider_failure_rate",
                severity="error",
                label="Provider failures elevated",
                message="Provider failures exceeded 20% in the selected window.",
                value=round(provider_failure_rate, 4),
                threshold=0.2,
                unit="pct",
            )
        )
    if graph_topology.orphan_rate >= 0.25:
        alerts.append(
            NerdAlertPayload(
                key="orphan_rate",
                severity="warning",
                label="Orphan rate rising",
                message="More than 25% of scoped memories have no links.",
                value=round(graph_topology.orphan_rate, 4),
                threshold=0.25,
                unit="pct",
            )
        )
    if search_quality.degraded or search_quality.fallback_count > 0:
        alerts.append(
            NerdAlertPayload(
                key="search_health",
                severity="warning" if search_quality.degraded else "info",
                label="Search health degraded",
                message="Search has entered a degraded or fallback mode; ranking quality may be reduced.",
                value=float(search_quality.fallback_count),
                threshold=0.0,
                unit="count",
            )
        )
    if memory_lifecycle.cold_memory_rate >= 0.5:
        alerts.append(
            NerdAlertPayload(
                key="cold_memory_rate",
                severity="info",
                label="Cold memory tail growing",
                message="At least half of scoped memories have never been accessed.",
                value=round(memory_lifecycle.cold_memory_rate, 4),
                threshold=0.5,
                unit="pct",
            )
        )
    if quality_signals.trace_like_memory_count > 0:
        alerts.append(
            NerdAlertPayload(
                key="trace_like_memory_count",
                severity="warning",
                label="Trace-like memories detected",
                message="At least one stored memory looks like a routine completion or status trace; route these through task_complete instead.",
                value=float(quality_signals.trace_like_memory_count),
                threshold=0.0,
                unit="count",
            )
        )
    if quality_signals.untagged_observation_rate >= 0.25:
        alerts.append(
            NerdAlertPayload(
                key="untagged_observation_rate",
                severity="warning",
                label="Observation tagging coverage low",
                message="At least 25% of scoped observations are missing tags, which makes retrieval and curation noisier.",
                value=round(quality_signals.untagged_observation_rate, 4),
                threshold=0.25,
                unit="pct",
            )
        )
    if quality_signals.generic_summary_count > 0:
        alerts.append(
            NerdAlertPayload(
                key="generic_summary_count",
                severity="info",
                label="Generic summaries present",
                message="Some stored summaries still use generic language like 'Covers ...'; tighter summaries should improve scanability.",
                value=float(quality_signals.generic_summary_count),
                threshold=0.0,
                unit="count",
            )
        )
    if quality_signals.oversized_memory_count > 0:
        alerts.append(
            NerdAlertPayload(
                key="oversized_memory_count",
                severity="info",
                label="Oversized memories detected",
                message="Some scoped memories exceed the split-warning threshold and may benefit from defragmentation.",
                value=float(quality_signals.oversized_memory_count),
                threshold=0.0,
                unit="count",
            )
        )
    curator_route = next((item for item in route_audit if item.task_name == "memory-curator"), None)
    if (
        curator_route is not None
        and curator_route.configured_primary_route is not None
        and curator_route.recent_provider_key is not None
        and not curator_route.recent_provider_key.startswith(curator_route.configured_primary_route)
    ):
        alerts.append(
            NerdAlertPayload(
                key="curator_route_fallback",
                severity="warning",
                label="Curator off premium lane",
                message="Recent curator runs used a non-primary provider route; review premium-lane fallback behavior.",
                value=float(curator_route.recent_failure_count),
                threshold=0.0,
                unit="count",
            )
        )
    return alerts


def _percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = max(math.ceil(len(sorted_values) * ratio) - 1, 0)
    return float(sorted_values[index])


def _normalize_quality_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.strip().lower().split())


def _looks_like_trace_title(title: str) -> bool:
    return title.startswith("task_complete") or title.startswith("task complete") or title.startswith("task_complete_record")


def is_provenance_process_tag(tag: str) -> bool:
    normalized = tag.strip().lower()
    if not normalized:
        return False
    return (
        normalized in _PROVENANCE_PROCESS_TAGS
        or normalized.startswith(_PROVENANCE_TAG_PREFIXES)
        or normalized.endswith(_PROVENANCE_TAG_SUFFIXES)
    )


def _count_buckets(
    counts: Counter[str],
    *,
    limit: int | None = None,
    other_key: str | None = None,
    other_label: str | None = None,
) -> list[NerdCountBucketPayload]:
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    if limit is not None and len(ordered) > limit:
        kept = ordered[:limit]
        other_count = sum(count for _, count in ordered[limit:])
        ordered = kept
        if other_count > 0 and other_key is not None and other_label is not None:
            return [
                *[NerdCountBucketPayload(key=key, label=key, count=count) for key, count in ordered],
                NerdCountBucketPayload(key=other_key, label=other_label, count=other_count),
            ]
    return [NerdCountBucketPayload(key=key, label=key, count=count) for key, count in ordered]


def _materialize_bucket_counts(bucket_defs, counts: dict[str, int]) -> list[NerdCountBucketPayload]:
    return [
        NerdCountBucketPayload(key=key, label=label, count=counts.get(key, 0))
        for key, label, _, _ in bucket_defs
    ]


def _build_maintenance_impact_summary(result: dict[str, object], *, ingest_audit) -> str | None:
    parts: list[str] = []
    for key, label in (
        ("created", "created"),
        ("merged", "merged"),
        ("updated", "updated"),
        ("archived", "archived"),
        ("restored", "restored"),
        ("degraded", "degraded"),
        ("lines_compressed", "lines"),
        ("meaningful_actions", "actions"),
    ):
        value = _result_int(result, key)
        if value is not None and value > 0:
            parts.append(f"{label}={value}")

    return ", ".join(parts[:6]) if parts else None


def _bucket_starts(*, cutoff: float, generated_at: float, bucket_seconds: int) -> list[int]:
    if generated_at < cutoff:
        return []
    start_bucket = int(cutoff // bucket_seconds) * bucket_seconds
    end_bucket = int(generated_at // bucket_seconds) * bucket_seconds
    if end_bucket < start_bucket:
        return []
    return list(range(start_bucket, end_bucket + bucket_seconds, bucket_seconds))


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


def _build_quality_signal_series(
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


def _select_top_keys(counts: Counter[str], *, limit: int) -> tuple[list[str], bool]:
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [key for key, _count in ordered[:limit]], len(ordered) > limit


def _dimension_labels(top_keys: list[str], *, include_other: bool) -> list[tuple[str, str]]:
    labels = [(key, key) for key in top_keys]
    if include_other:
        labels.append(("other", "Other"))
    return labels


def _build_cumulative_dimension_count_series(
    memory_rows,
    *,
    cutoff: float,
    generated_at: float,
    bucket_seconds: int,
    bucket_starts: list[int],
    top_keys: list[str],
    include_other: bool,
    extractor,
) -> list[NerdCountSeriesPayload]:
    if not top_keys and not include_other:
        return []

    tracked_keys = _dimension_labels(top_keys, include_other=include_other)
    running_counts = {key: 0 for key, _label in tracked_keys}
    created_counts = {key: Counter() for key, _label in tracked_keys}

    for row in memory_rows:
        contributions = _map_dimension_contributions(extractor(row), top_keys=top_keys, include_other=include_other)
        if not contributions:
            continue
        created_at = _iso_to_timestamp(row["created_at"])
        if created_at is None or created_at > generated_at:
            continue
        if created_at < cutoff:
            for key, count in contributions.items():
                running_counts[key] += count
            continue
        bucket_start = int(created_at // bucket_seconds) * bucket_seconds
        for key, count in contributions.items():
            created_counts[key][bucket_start] += count

    buckets_by_key: dict[str, list[NerdTimeCountBucketPayload]] = {key: [] for key, _label in tracked_keys}
    for bucket_start in bucket_starts:
        for key, _label in tracked_keys:
            running_counts[key] += created_counts[key].get(bucket_start, 0)
            buckets_by_key[key].append(
                NerdTimeCountBucketPayload(bucket_start=float(bucket_start), count=running_counts[key])
            )

    return [
        NerdCountSeriesPayload(key=key, label=label, buckets=buckets_by_key[key])
        for key, label in tracked_keys
    ]


def _build_cumulative_dimension_share_series(
    memory_rows,
    *,
    cutoff: float,
    generated_at: float,
    bucket_seconds: int,
    bucket_starts: list[int],
    top_keys: list[str],
    include_other: bool,
    extractor,
) -> list[NerdShareSeriesPayload]:
    if not top_keys and not include_other:
        return []

    tracked_keys = _dimension_labels(top_keys, include_other=include_other)
    running_counts = {key: 0 for key, _label in tracked_keys}
    created_counts = {key: Counter() for key, _label in tracked_keys}

    for row in memory_rows:
        contributions = _map_dimension_contributions(extractor(row), top_keys=top_keys, include_other=include_other)
        if not contributions:
            continue
        created_at = _iso_to_timestamp(row["created_at"])
        if created_at is None or created_at > generated_at:
            continue
        if created_at < cutoff:
            for key, count in contributions.items():
                running_counts[key] += count
            continue
        bucket_start = int(created_at // bucket_seconds) * bucket_seconds
        for key, count in contributions.items():
            created_counts[key][bucket_start] += count

    buckets_by_key: dict[str, list[NerdTimeShareBucketPayload]] = {key: [] for key, _label in tracked_keys}
    for bucket_start in bucket_starts:
        for key, _label in tracked_keys:
            running_counts[key] += created_counts[key].get(bucket_start, 0)
        total_count = sum(running_counts.values())
        for key, _label in tracked_keys:
            share = 0.0 if total_count == 0 else round(running_counts[key] / total_count, 4)
            buckets_by_key[key].append(
                NerdTimeShareBucketPayload(
                    bucket_start=float(bucket_start),
                    count=running_counts[key],
                    share=share,
                )
            )

    return [
        NerdShareSeriesPayload(key=key, label=label, buckets=buckets_by_key[key])
        for key, label in tracked_keys
    ]


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


def _map_dimension_contributions(values: list[str], *, top_keys: list[str], include_other: bool) -> Counter[str]:
    contributions: Counter[str] = Counter()
    tracked_key_set = set(top_keys)
    for value in values:
        if value in tracked_key_set:
            contributions[value] += 1
        elif include_other:
            contributions["other"] += 1
    return contributions


def _maintenance_family_for_task(task_name: str) -> tuple[str, str]:
    family = _MAINTENANCE_FAMILY_BY_TASK.get(task_name)
    if family is not None:
        return family
    return ("other", "Other")


def _run_reported_delta_counts(result: dict[str, object]) -> dict[str, int]:
    return {
        attribute_name: max(_result_int(result, result_key) or 0, 0)
        for result_key, attribute_name in _RUN_REPORTED_DELTA_KEYS
    }


def _update_maintenance_summary_accumulator(
    accumulator: _MaintenanceSummaryAccumulator,
    *,
    task_name: str,
    status: str,
    deltas: dict[str, int],
) -> None:
    accumulator.task_names.add(task_name)
    accumulator.total_runs += 1
    if status == "completed":
        accumulator.completed_runs += 1
    elif status == "failed":
        accumulator.failed_runs += 1
    elif status == "retry":
        accumulator.retry_runs += 1
    for attribute_name, value in deltas.items():
        setattr(accumulator, attribute_name, getattr(accumulator, attribute_name) + value)


def _build_agent_yield_payload(
    *,
    task_name: str,
    family_key: str,
    family_label: str,
    accumulator: _MaintenanceSummaryAccumulator,
) -> NerdMaintenanceAgentYieldPayload:
    completed_runs = max(accumulator.completed_runs, 0)
    denominator = completed_runs if completed_runs > 0 else accumulator.total_runs
    return NerdMaintenanceAgentYieldPayload(
        key=task_name,
        label=task_name,
        family_key=family_key,
        family_label=family_label,
        task_names=sorted(accumulator.task_names),
        total_runs=accumulator.total_runs,
        completed_runs=accumulator.completed_runs,
        failed_runs=accumulator.failed_runs,
        retry_runs=accumulator.retry_runs,
        created_count=accumulator.created_count,
        merged_count=accumulator.merged_count,
        updated_count=accumulator.updated_count,
        archived_count=accumulator.archived_count,
        degraded_count=accumulator.degraded_count,
        restored_count=accumulator.restored_count,
        meaningful_actions=accumulator.meaningful_actions,
        lines_compressed=accumulator.lines_compressed,
        delta_total=accumulator.delta_total,
        actions_per_completed_run=0.0 if denominator == 0 else round(accumulator.meaningful_actions / denominator, 4),
        lines_per_completed_run=0.0 if denominator == 0 else round(accumulator.lines_compressed / denominator, 4),
        delta_per_completed_run=0.0 if denominator == 0 else round(accumulator.delta_total / denominator, 4),
    )


def _bucket_key_for_value(value: float, bucket_defs) -> str:
    for key, _label, minimum, maximum in bucket_defs:
        if value < minimum:
            continue
        if maximum is None or value < maximum:
            return key
    return str(bucket_defs[-1][0])


def _coerce_row_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _result_int(result: dict[str, object], key: str) -> int | None:
    value = result.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _prefer_nonzero_int(primary: int | None, fallback: int | None) -> int | None:
    if primary is not None and primary > 0:
        return primary
    if fallback is not None and fallback > 0:
        return fallback
    return primary if primary is not None else fallback


def _iso_to_timestamp(value: str | None) -> float | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.timestamp()
