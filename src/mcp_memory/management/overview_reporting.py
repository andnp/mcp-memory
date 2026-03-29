from __future__ import annotations

from mcp_memory.core.task_handlers import TRIGGERABLE_BACKGROUND_TASK_NAMES
from mcp_memory.management.agent_run_reporting import build_agent_runs, build_recent_agent_runs
from mcp_memory.management.health_reporting import build_embedding_status, build_execution_attempt_health, build_search_health
from mcp_memory.management.models import (
    CacheHealthPayload,
    JournalSummary,
    MemoryMetricsPayload,
    OverviewCounts,
    OverviewPayload,
    PremiumUsageSummaryPayload,
    ProviderUsagePayload,
    RuntimeLogPayload,
    StorageSummary,
    TaskStatusSummary,
)
from mcp_memory.management.reporting_queries import (
    build_queue_diagnostics,
    fetch_memory_count_rows,
    fetch_memory_metrics_row,
    fetch_pending_journal_metrics_row,
    fetch_task_count_rows,
    row_int,
    summarize_copilot_premium_requests,
)
from mcp_memory.serialization import compact_memory_record_payload, task_payload


def build_memory_counts(db_manager, workspace_id: str | None) -> tuple[dict[str, int], dict[str, int], int]:
    rows = fetch_memory_count_rows(db_manager, workspace_id)
    by_type: dict[str, int] = {}
    by_status: dict[str, int] = {}
    total = 0
    for row in rows:
        count = row_int(row, "count")
        total += count
        by_type[str(row["type"])] = by_type.get(str(row["type"]), 0) + count
        by_status[str(row["status"])] = by_status.get(str(row["status"]), 0) + count
    return by_type, by_status, total


def build_task_counts(db_manager, workspace_id: str | None) -> dict[str, int]:
    rows = fetch_task_count_rows(db_manager, workspace_id)
    return {str(row["status"]): row_int(row, "count") for row in rows}


def build_memory_metrics(db_manager, workspace_id: str | None, task_queue) -> MemoryMetricsPayload:
    if db_manager is None:
        return MemoryMetricsPayload(
            total_memories=0,
            total_memory_lines=0,
            total_summary_lines=0,
            total_lines_compressed=0,
            thought_buffer_entries=0,
            thought_buffer_lines=0,
        )

    memory_row = fetch_memory_metrics_row(db_manager, workspace_id)
    journal_row = fetch_pending_journal_metrics_row(db_manager, workspace_id)

    total_lines_compressed = sum(
        max(agent_run.total_lines_compressed, 0)
        for agent_run in task_queue.summarize_task_runs(
            list(TRIGGERABLE_BACKGROUND_TASK_NAMES),
            workspace_id=workspace_id,
        )
    )
    return MemoryMetricsPayload(
        total_memories=row_int(memory_row, "total_memories"),
        total_memory_lines=row_int(memory_row, "total_memory_lines"),
        total_summary_lines=row_int(memory_row, "total_summary_lines"),
        total_lines_compressed=total_lines_compressed,
        thought_buffer_entries=row_int(journal_row, "thought_buffer_entries"),
        thought_buffer_lines=row_int(journal_row, "thought_buffer_lines"),
    )


def build_provider_usage(provider_usage_repo, workspace_id: str | None) -> list[ProviderUsagePayload]:
    return [
        ProviderUsagePayload(
            task_name=summary.task_name,
            provider_key=summary.provider_key,
            provider_name=summary.provider_name,
            model_name=summary.model_name,
            calls_last_hour=summary.calls_last_hour,
            calls_last_day=summary.calls_last_day,
            failures_last_hour=summary.failures_last_hour,
            failures_last_day=summary.failures_last_day,
            skips_last_hour=summary.skips_last_hour,
            skips_last_day=summary.skips_last_day,
            avg_duration_last_hour=summary.avg_duration_last_hour,
            avg_duration_last_day=summary.avg_duration_last_day,
            top_failure_reason_last_day=summary.top_failure_reason_last_day,
            top_skip_reason_last_day=summary.top_skip_reason_last_day,
            active_admission_reason=summary.active_admission_reason,
            active_admission_category=summary.active_admission_category,
            active_retry_delay_seconds=summary.active_retry_delay_seconds,
        )
        for summary in provider_usage_repo.summarize_usage(workspace_id=workspace_id)
    ]


def build_overview(
    *,
    memory_queries,
    repository,
    task_queue,
    runtime_info,
    db_manager,
    workspace_id: str | None,
    provider_usage_repo,
    runtime_logs_repo,
    embedder,
    relational_search,
    cache: CacheHealthPayload,
    recent_limit: int = 10,
    failed_limit: int = 10,
) -> OverviewPayload:
    records = [] if memory_queries is None else memory_queries.list_memories(
        workspace_id=workspace_id,
        limit=recent_limit,
    )
    top_read_records = [] if repository is None else repository.list_most_read_memories(
        workspace_id=workspace_id,
        limit=10,
    )
    by_type, by_status, total_memories = build_memory_counts(db_manager, workspace_id)
    recent_records = [compact_memory_record_payload(record) for record in records]
    top_read_payloads = [compact_memory_record_payload(record) for record in top_read_records]
    top_read_active_payloads = [payload for payload in top_read_payloads if payload.status == "active"]
    task_counts = build_task_counts(db_manager, workspace_id)
    failed_tasks = [
        task_payload(task)
        for task in task_queue.list_tasks(
            status="failed",
            workspace_id=workspace_id,
            limit=failed_limit,
        )
    ]
    sqlite_bytes = 0
    sqlite_path = runtime_info.db_path
    if sqlite_path is not None and sqlite_path.exists():
        sqlite_bytes = sqlite_path.stat().st_size

    memory_metrics = build_memory_metrics(db_manager, workspace_id, task_queue)
    premium_usage = PremiumUsageSummaryPayload(
        **summarize_copilot_premium_requests(db_manager, workspace_id=workspace_id)
    )
    agent_runs = build_agent_runs(task_queue, workspace_id)
    provider_usage = build_provider_usage(provider_usage_repo, workspace_id)
    recent_agent_runs = build_recent_agent_runs(db_manager, workspace_id, limit=20)
    recent_logs = [
        RuntimeLogPayload(
            id=record.id,
            created_at=record.created_at,
            level=record.level,
            logger_name=record.logger_name,
            source=record.source,
            message=record.message,
            data=record.data,
        )
        for record in runtime_logs_repo.list_logs(limit=10)
    ]

    return OverviewPayload(
        memories=OverviewCounts(total=total_memories, by_type=by_type, by_status=by_status),
        embeddings=build_embedding_status(embedder),
        search=build_search_health(relational_search),
        cache=cache,
        execution_attempts=build_execution_attempt_health(db_manager, workspace_id),
        memory_metrics=memory_metrics,
        premium_usage=premium_usage,
        queue_diagnostics=build_queue_diagnostics(task_queue, workspace_id),
        agent_runs=agent_runs,
        provider_usage=provider_usage,
        recent_agent_runs=recent_agent_runs,
        recent_logs=recent_logs,
        recent_memories=recent_records,
        top_read_memories=top_read_payloads,
        top_read_memories_active=top_read_active_payloads,
        tasks=TaskStatusSummary(
            by_status=task_counts,
            failed_count=task_counts.get("failed", 0),
        ),
        failed_tasks=failed_tasks,
        journal=JournalSummary(pending_count=memory_metrics.thought_buffer_entries),
        storage=StorageSummary(
            sqlite_bytes=sqlite_bytes,
            sqlite_path=str(sqlite_path) if sqlite_path is not None else None,
        ),
    )
