from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
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
    NerdCountSeriesPayload,
    MemoryTimelineBucketPayload,
    NerdGrowthDynamicsPayload,
    NerdLifecycleTrendsPayload,
    MemoryLifecyclePayload,
    NerdAlertPayload,
    NerdCompositionPayload,
    NerdCountBucketPayload,
    NerdMaintenanceAgentYieldPayload,
    NerdDistributionsPayload,
    NerdMaintenanceDeltaBucketPayload,
    NerdMaintenanceDeltaSeriesPayload,
    NerdMaintenancePayload,
    NerdMaintenanceSummaryPayload,
    NerdMaintenanceSummaryRowPayload,
    NerdMetricsPayload,
    NerdStatPayload,
    NerdTimeCountBucketPayload,
    NerdTimeShareBucketPayload,
    NerdTimelinesPayload,
    ProviderLatencyBucketPayload,
    QueueSnapshotPayload,
    SearchQualityPayload,
    NerdShareSeriesPayload,
)
from mcp_memory.management.reporting_queries import (
    build_queue_diagnostics,
    list_maintenance_task_run_rows_since,
    list_provider_usage_rows_since,
    list_scoped_link_rows,
    list_scoped_memory_rows,
    list_task_run_rows_since,
    split_csv_values,
)
from mcp_memory.management.route_audit import build_task_route_audit


_TAG_LIMIT = 10
_DYNAMICS_LIMIT = 5
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
    memory_rows = list_scoped_memory_rows(db_manager, workspace_id)

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
    for row in provider_rows:
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
        if str(row["status"]) != "success":
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

    provider_failure_rate = 0.0 if not provider_rows else provider_failures / len(provider_rows)

    stats = [
        NerdStatPayload(key="queue_oldest_age", label="Oldest runnable age", value=queue_snapshot.oldest_age_seconds, unit="s"),
        NerdStatPayload(key="runs_last_window", label="Runs in window", value=float(len(task_rows)), unit="runs"),
        NerdStatPayload(key="failed_runs_last_window", label="Failed runs in window", value=float(sum(1 for row in task_rows if str(row["status"]) == "failed")), unit="runs"),
        NerdStatPayload(key="provider_calls_last_window", label="Provider calls in window", value=float(len(provider_rows)), unit="calls"),
        NerdStatPayload(key="provider_failures_last_window", label="Provider failures in window", value=float(provider_failures), unit="calls"),
        NerdStatPayload(key="provider_p95_latency", label="Provider p95 latency", value=round(_percentile(all_provider_durations, 0.95), 4), unit="s"),
        NerdStatPayload(key="provider_failure_rate", label="Provider failure rate", value=round(provider_failure_rate, 4), unit="pct"),
        NerdStatPayload(key="orphan_rate", label="Orphan rate", value=round(graph_topology.orphan_rate, 4), unit="pct"),
        NerdStatPayload(key="cold_memory_rate", label="Cold memory rate", value=round(memory_lifecycle.cold_memory_rate, 4), unit="pct"),
        NerdStatPayload(key="search_fallback_count", label="Search fallback count", value=float(search_quality.fallback_count), unit="count"),
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
        maintenance_summary=maintenance_summary,
        queue_snapshot=queue_snapshot,
        graph_topology=graph_topology,
        memory_lifecycle=memory_lifecycle,
        search_quality=search_quality,
        route_audit=route_audit,
        alerts=build_nerd_alerts(
            queue_snapshot=queue_snapshot,
            graph_topology=graph_topology,
            memory_lifecycle=memory_lifecycle,
            search_quality=search_quality,
            route_audit=route_audit,
            provider_failure_rate=provider_failure_rate,
        ),
        agent_throughput=agent_throughput,
        provider_latency=provider_latency,
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


def build_nerd_alerts(
    *,
    queue_snapshot: QueueSnapshotPayload,
    graph_topology: GraphTopologyPayload,
    memory_lifecycle: MemoryLifecyclePayload,
    search_quality: SearchQualityPayload,
    route_audit,
    provider_failure_rate: float,
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
