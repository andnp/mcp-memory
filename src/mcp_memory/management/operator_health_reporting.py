from __future__ import annotations

from statistics import mean
import time

from mcp_memory.management.analytics_provider_policy import build_provider_policy_rollups
from mcp_memory.management.models import (
    MemoryToolLatencyMetricPayload,
    MemoryToolLatencyPayload,
    OperatorConversationDigestPayload,
    OperatorHealthSnapshotPayload,
    OperatorLogDigestPayload,
    OperatorMemoryActivityPayload,
    OperatorTaskDigestPayload,
    OperatorWarningDigestPayload,
)
from mcp_memory.management.reporting_rows import list_memory_tool_event_rows_since
from mcp_memory.management.reporting_rows import list_provider_policy_event_rows_since, list_provider_usage_rows_since, list_runtime_log_rows_since


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    bounded_percentile = min(max(percentile, 0.0), 1.0)
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * bounded_percentile))))
    return float(ordered[index])


def summarize_memory_tool_latency(
    db_manager,
    *,
    workspace_id: str | None,
    window_minutes: int,
    slow_threshold_ms: float,
) -> MemoryToolLatencyPayload:
    effective_window_minutes = max(window_minutes, 1)
    cutoff = time.time() - (effective_window_minutes * 60)
    rows = list_memory_tool_event_rows_since(
        db_manager,
        cutoff=cutoff,
        workspace_id=workspace_id,
    )
    durations_by_kind: dict[str, dict[str, float]] = {}
    for row in rows:
        duration_ms = row.duration_ms
        if duration_ms is None:
            continue
        event_kind = row.event_kind or "unknown"
        invocation_id = row.invocation_id or f"{event_kind}:{len(durations_by_kind.get(event_kind, {}))}"
        durations_by_kind.setdefault(event_kind, {})[invocation_id] = max(
            duration_ms,
            durations_by_kind.setdefault(event_kind, {}).get(invocation_id, duration_ms),
        )

    return MemoryToolLatencyPayload(
        window_minutes=effective_window_minutes,
        slow_threshold_ms=slow_threshold_ms,
        by_event_kind=[
            MemoryToolLatencyMetricPayload(
                event_kind=event_kind,
                count=len(duration_map),
                avg_duration_ms=round(mean(duration_map.values()), 3),
                p95_duration_ms=round(_percentile(list(duration_map.values()), 0.95), 3),
                max_duration_ms=round(max(duration_map.values()), 3),
                slow_count=sum(1 for duration in duration_map.values() if duration >= slow_threshold_ms),
            )
            for event_kind, duration_map in sorted(durations_by_kind.items())
        ],
    )


def summarize_provider_policy(
    db_manager,
    *,
    provider_usage_repo,
    workspace_id: str | None,
    window_minutes: int,
):
    effective_window_minutes = max(window_minutes, 1)
    cutoff = time.time() - (effective_window_minutes * 60)
    return build_provider_policy_rollups(
        provider_rows=list_provider_usage_rows_since(
            db_manager,
            cutoff=cutoff,
            workspace_id=workspace_id,
        ),
        provider_policy_event_rows=list_provider_policy_event_rows_since(
            db_manager,
            cutoff=cutoff,
            workspace_id=workspace_id,
        ),
        provider_policy_log_rows=list_runtime_log_rows_since(
            db_manager,
            cutoff=cutoff,
            workspace_id=workspace_id,
            logger_name="mcp_memory.core.provider_policy",
            level="WARNING",
        ),
        provider_usage_repo=provider_usage_repo,
        workspace_id=workspace_id,
    )


def build_operator_health_snapshot_payload(
    *,
    generated_at: float,
    log_window_minutes: int,
    conversation_window_hours: int,
    health,
    log_summary,
    recent_errors,
    recent_warnings,
    recent_run_status_counts: dict[str, int],
    recent_runs,
    recent_failures,
    recent_retries,
    conversation_counts: dict[str, int],
    recent_conversations,
    updated_last_15_minutes: int,
    updated_last_hour: int,
    updated_last_day: int,
    recent_memories,
    tool_latency,
    provider_policy,
) -> OperatorHealthSnapshotPayload:
    alerts = build_operator_health_alerts(
        health=health,
        log_summary=log_summary,
        recent_failures=recent_failures,
        recent_retries=recent_retries,
        conversation_counts=conversation_counts,
        provider_policy=provider_policy,
    )
    return OperatorHealthSnapshotPayload(
        generated_at=generated_at,
        status="warn" if alerts else "ok",
        alerts=alerts,
        health=health,
        logs=OperatorLogDigestPayload(
            window_minutes=max(log_window_minutes, 0),
            total=log_summary.total,
            by_level=log_summary.by_level,
            recent_errors=recent_errors,
        ),
        warnings=OperatorWarningDigestPayload(
            window_minutes=max(log_window_minutes, 0),
            total=log_summary.by_level.get("WARNING", 0),
            recent=recent_warnings,
        ),
        tasks=OperatorTaskDigestPayload(
            recent_status_counts=recent_run_status_counts,
            recent=recent_runs,
            recent_failure_count=len(recent_failures),
            recent_retry_count=len(recent_retries),
            recent_failures=recent_failures,
            recent_retries=recent_retries,
        ),
        conversations=OperatorConversationDigestPayload(
            window_hours=max(conversation_window_hours, 0),
            total=sum(conversation_counts.values()),
            by_status=conversation_counts,
            recent=recent_conversations,
        ),
        memory_activity=OperatorMemoryActivityPayload(
            updated_last_15_minutes=updated_last_15_minutes,
            updated_last_hour=updated_last_hour,
            updated_last_day=updated_last_day,
            recent=recent_memories,
        ),
        tool_latency=tool_latency,
        provider_policy=provider_policy,
    )


def build_operator_health_alerts(
    *,
    health,
    log_summary,
    recent_failures,
    recent_retries,
    conversation_counts: dict[str, int],
    provider_policy,
) -> list[str]:
    alerts: list[str] = []
    if not health.runtime_active:
        alerts.append("runtime_inactive")
    if health.search.degraded:
        alerts.append("search_degraded")
    if health.execution_attempts.stale_attempt_count > 0:
        alerts.append("stale_execution_attempts")
    if health.execution_attempts.dead_subprocess_count > 0:
        alerts.append("dead_execution_subprocesses")
    if log_summary.by_level.get("ERROR", 0) > 0:
        alerts.append("recent_error_logs")
    if recent_failures:
        alerts.append("recent_task_failures")
    if recent_retries:
        alerts.append("recent_task_retries")
    if conversation_counts.get("error", 0) > 0:
        alerts.append("recent_ai_errors")
    if any(stat.value > 0 for stat in provider_policy.stats[:3]):
        alerts.append("provider_policy_churn")
    return alerts