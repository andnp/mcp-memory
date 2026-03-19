from __future__ import annotations

from dataclasses import dataclass, field
import math
from statistics import mean, median
import time

from mcp_memory.management.health_reporting import build_search_health
from mcp_memory.management.models import (
    AgentThroughputBucketPayload,
    GraphTopologyPayload,
    MemoryLifecyclePayload,
    NerdAlertPayload,
    NerdMetricsPayload,
    NerdStatPayload,
    ProviderLatencyBucketPayload,
    QueueSnapshotPayload,
    SearchQualityPayload,
)
from mcp_memory.management.overview_reporting import build_queue_diagnostics
from mcp_memory.management.route_audit import build_task_route_audit


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
    conn = db_manager.get_connection()

    task_run_query = "SELECT status, completed_at, duration_seconds FROM task_runs WHERE completed_at >= ?"
    provider_query = (
        "SELECT provider_key, provider_name, model_name, status, duration_seconds, created_at "
        "FROM provider_usage WHERE created_at >= ?"
    )
    task_params: list[object] = [cutoff]
    provider_params: list[object] = [cutoff]
    if workspace_id is not None:
        task_run_query += " AND workspace_id = ?"
        provider_query += " AND workspace_id = ?"
        task_params.append(workspace_id)
        provider_params.append(workspace_id)

    task_rows = conn.execute(task_run_query, task_params).fetchall()
    provider_rows = conn.execute(provider_query, provider_params).fetchall()

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

    graph_topology = build_graph_topology(db_manager, workspace_id)
    memory_lifecycle = build_memory_lifecycle(db_manager, workspace_id)
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


def build_graph_topology(db_manager, workspace_id: str | None) -> GraphTopologyPayload:
    memory_rows = _list_scoped_memories(db_manager, workspace_id)
    memory_ids = {str(row["id"]) for row in memory_rows}
    if not memory_ids:
        return GraphTopologyPayload()

    link_rows = _list_scoped_links(db_manager, workspace_id, memory_ids)
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


def build_memory_lifecycle(db_manager, workspace_id: str | None) -> MemoryLifecyclePayload:
    memory_rows = _list_scoped_memories(db_manager, workspace_id)
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


def _list_scoped_memories(db_manager, workspace_id: str | None):
    if db_manager is None:
        return []
    conn = db_manager.get_connection()
    query = (
        "SELECT memories.id, memories.type, memories.status, memories.last_accessed_at, memories.last_surfaced_at, "
        "LENGTH(COALESCE(memories.content, '')) AS content_bytes FROM memories"
    )
    params: list[object] = []
    if workspace_id is not None:
        query += (
            " WHERE EXISTS (SELECT 1 FROM memory_workspaces WHERE memory_workspaces.memory_id = memories.id "
            "AND memory_workspaces.workspace_id = ?)"
        )
        params.append(workspace_id)
    return conn.execute(query, params).fetchall()


def _list_scoped_links(db_manager, workspace_id: str | None, memory_ids: set[str]):
    if db_manager is None or not memory_ids:
        return []
    conn = db_manager.get_connection()
    if workspace_id is None:
        return conn.execute("SELECT source_id, target_id, type FROM links").fetchall()

    placeholders = ",".join("?" for _ in memory_ids)
    params = [*memory_ids, *memory_ids]
    query = (
        f"SELECT source_id, target_id, type FROM links WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})"
    )
    return conn.execute(query, params).fetchall()


def _percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = max(math.ceil(len(sorted_values) * ratio) - 1, 0)
    return float(sorted_values[index])
