from __future__ import annotations

from dataclasses import dataclass, field
import time

from mcp_memory.management.models import QueueDiagnosticPayload
from mcp_memory.management.reporting_queries import (
    build_queue_diagnostics,
    list_memory_tool_event_rows_since,
    list_maintenance_task_run_rows_since,
    list_provider_policy_event_rows_since,
    list_provider_usage_rows_since,
    list_runtime_log_rows_since,
    list_scoped_memory_rows,
    list_task_run_rows_since,
)
from mcp_memory.management.reporting_rows import (
    MaintenanceTaskRunRow,
    MemoryToolEventRow,
    ProviderPolicyEventRow,
    ProviderUsageRow,
    RuntimeLogRow,
    ScopedMemoryRow,
    TaskRunRow,
)


@dataclass(frozen=True)
class NerdMetricsReadModel:
    generated_at: float
    cutoff: float
    bucket_seconds: int
    task_rows: list[TaskRunRow] = field(default_factory=list)
    maintenance_rows: list[MaintenanceTaskRunRow] = field(default_factory=list)
    provider_rows: list[ProviderUsageRow] = field(default_factory=list)
    provider_policy_event_rows: list[ProviderPolicyEventRow] = field(default_factory=list)
    provider_policy_log_rows: list[RuntimeLogRow] = field(default_factory=list)
    memory_rows: list[ScopedMemoryRow] = field(default_factory=list)
    retrieval_rows: list[MemoryToolEventRow] = field(default_factory=list)
    queue_rows: list[QueueDiagnosticPayload] = field(default_factory=list)


def load_nerd_metrics_read_model(
    *,
    db_manager,
    workspace_id: str | None,
    task_queue,
    window_hours: int = 24,
    bucket_minutes: int = 60,
    now: float | None = None,
) -> NerdMetricsReadModel:
    generated_at = time.time() if now is None else now
    bucket_seconds = max(bucket_minutes * 60, 60)
    cutoff = generated_at - (window_hours * 3600)

    if db_manager is None:
        return NerdMetricsReadModel(
            generated_at=generated_at,
            cutoff=cutoff,
            bucket_seconds=bucket_seconds,
        )

    return NerdMetricsReadModel(
        generated_at=generated_at,
        cutoff=cutoff,
        bucket_seconds=bucket_seconds,
        task_rows=list_task_run_rows_since(db_manager, cutoff=cutoff, workspace_id=workspace_id),
        maintenance_rows=list_maintenance_task_run_rows_since(db_manager, cutoff=cutoff, workspace_id=workspace_id),
        provider_rows=list_provider_usage_rows_since(db_manager, cutoff=cutoff, workspace_id=workspace_id),
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
        memory_rows=list_scoped_memory_rows(db_manager, workspace_id),
        retrieval_rows=list_memory_tool_event_rows_since(db_manager, cutoff=cutoff, workspace_id=workspace_id),
        queue_rows=[]
        if task_queue is None
        else build_queue_diagnostics(task_queue, workspace_id, limit=200, now=generated_at),
    )