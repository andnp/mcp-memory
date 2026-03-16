from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import time

from mcp_memory.context import ApplicationContext
from mcp_memory.core import MemoryPipeline
from mcp_memory.core.task_handlers import TRIGGERABLE_BACKGROUND_TASK_NAMES
from mcp_memory.embeddings import describe_embedder
from mcp_memory.management.models import (
    AgentRunHistoryPayload,
    AgentRunPayload,
    AIConversationListPayload,
    AIConversationPayload,
    EmbeddingStatusPayload,
    HealthPayload,
    JournalSummary,
    MemoryListPayload,
    MemoryDetailPayload,
    MemoryMetricsPayload,
    OverviewCounts,
    OverviewPayload,
    ProviderUsagePayload,
    RuntimeLogListPayload,
    RuntimeLogPrunePayload,
    RuntimeLogPayload,
    RuntimeLogSummaryPayload,
    SearchHealthPayload,
    StorageSummary,
    TaskListPayload,
    TaskStatusSummary,
)
from mcp_memory.provider_usage_store import ProviderUsageRepository
from mcp_memory.runtime_log_store import RuntimeLogRepository
from mcp_memory.serialization import (
    compact_memory_record_payload,
    link_payload,
    memory_record_payload,
    task_payload,
)


class ManagementService:
    def __init__(self, ctx: ApplicationContext, controller) -> None:
        pipeline = MemoryPipeline.from_context(ctx, controller)
        self._controller = controller
        self._db_manager = ctx.db_manager
        self._workspace_id = None
        self._runtime_info = pipeline.runtime_info
        self._journal = pipeline.journal
        self._task_queue = pipeline.task_queue
        self._memory_queries = pipeline.memory_queries
        self._repository = ctx.repository
        self._provider_usage = ProviderUsageRepository(ctx.db_manager, workspace_id=None)
        self._runtime_logs = RuntimeLogRepository(
            ctx.db_manager,
            workspace_id=None,
            config=None if ctx.config is None else ctx.config.logging,
        )
        self._embedder = ctx.embedder
        self._relational_search = ctx.relational_search
        self._dashboard_static_path = Path(__file__).with_name("static") / "index.html"

    def get_health(self):
        embedder_status = self._build_embedding_status()
        return HealthPayload(
            status="ok",
            workspace_id=self._runtime_info.workspace_id,
            workspace_root=str(self._runtime_info.workspace_root) if self._runtime_info.workspace_root is not None else None,
            memory_path=str(self._runtime_info.memory_path) if self._runtime_info.memory_path is not None else None,
            db_path=str(self._runtime_info.db_path) if self._runtime_info.db_path is not None else None,
            runtime_active=bool(getattr(self._controller, "has_runtime", self._runtime_info.runtime_active)),
            client_count=int(getattr(self._controller, "client_count", self._runtime_info.client_count)),
            task_queue_enabled=self._runtime_info.task_queue_enabled,
            embeddings=embedder_status,
            search=self._build_search_health(),
        )

    def get_overview(self, recent_limit: int = 10, failed_limit: int = 10):
        records = [] if self._memory_queries is None else self._memory_queries.list_memories(
            workspace_id=self._workspace_id,
            limit=recent_limit,
        )
        top_read_records = [] if self._repository is None else self._repository.list_most_read_memories(
            workspace_id=self._workspace_id,
            limit=10,
        )
        by_type, by_status, total_memories = self._build_memory_counts()
        recent_records = [compact_memory_record_payload(record) for record in records]
        top_read_payloads = [compact_memory_record_payload(record) for record in top_read_records]
        task_counts = self._build_task_counts()
        failed_tasks = [
            task_payload(task)
            for task in self._task_queue.list_tasks(
                status="failed",
                workspace_id=self._workspace_id,
                limit=failed_limit,
            )
        ]
        sqlite_bytes = 0
        sqlite_path = self._runtime_info.db_path
        if sqlite_path is not None and sqlite_path.exists():
            sqlite_bytes = sqlite_path.stat().st_size

        memory_metrics = self._build_memory_metrics()
        journal_counts = {"pending": memory_metrics.thought_buffer_entries}
        agent_runs = self._build_agent_runs()
        provider_usage = self._build_provider_usage()
        recent_agent_runs = self._build_recent_agent_runs()
        recent_logs = self.list_logs(limit=10).logs

        return OverviewPayload(
            memories=OverviewCounts(total=total_memories, by_type=by_type, by_status=by_status),
            embeddings=self._build_embedding_status(),
            search=self._build_search_health(),
            memory_metrics=memory_metrics,
            agent_runs=agent_runs,
            provider_usage=provider_usage,
            recent_agent_runs=recent_agent_runs,
            recent_logs=recent_logs,
            recent_memories=recent_records,
            top_read_memories=top_read_payloads,
            tasks=TaskStatusSummary(
                by_status=task_counts,
                failed_count=task_counts.get("failed", 0),
            ),
            failed_tasks=failed_tasks,
            journal=JournalSummary(pending_count=journal_counts.get("pending", 0)),
            storage=StorageSummary(
                sqlite_bytes=sqlite_bytes,
                sqlite_path=str(sqlite_path) if sqlite_path is not None else None,
            ),
        )

    def _build_embedding_status(self) -> EmbeddingStatusPayload:
        status = describe_embedder(self._embedder)
        if status is None:
            return EmbeddingStatusPayload()
        return EmbeddingStatusPayload(
            model_name=status.model_name,
            backend=status.backend,
            model_cached=status.model_cached,
        )

    def _build_search_health(self) -> SearchHealthPayload:
        if self._relational_search is None:
            return SearchHealthPayload()
        health = self._relational_search.get_health()
        return SearchHealthPayload(
            semantic_enabled=health.semantic_enabled,
            available=health.available,
            degraded=health.degraded,
            fallback_count=health.fallback_count,
            rebuild_count=health.rebuild_count,
            last_error=health.last_error,
            last_failure_at=health.last_failure_at,
            last_recovery_at=health.last_recovery_at,
            last_integrity_check_at=health.last_integrity_check_at,
            integrity_check_error=health.integrity_check_error,
        )

    def repair_search_index(self) -> dict[str, int | bool | str | None]:
        if self._relational_search is None:
            raise ValueError("search_not_initialized")
        return self._relational_search.rebuild_semantic_index()

    def enqueue_background_task(
        self,
        task_name: str,
        *,
        force: bool = False,
    ) -> dict:
        if task_name not in TRIGGERABLE_BACKGROUND_TASK_NAMES:
            raise ValueError(f"unknown_background_task:{task_name}")

        payload = {"workspace_id": None}
        if force:
            task = self._task_queue.enqueue(task_name=task_name, workspace_id=None, data=payload)
            return {"status": "enqueued", "created": True, "task": task_payload(task)}

        task, created = self._task_queue.enqueue_unique(
            task_name=task_name,
            workspace_id=None,
            data=payload,
        )
        return {
            "status": "enqueued" if created else "already_pending",
            "created": created,
            "task": task_payload(task),
        }

    def enqueue_all_background_tasks(self, *, force: bool = False) -> list[dict]:
        return [
            self.enqueue_background_task(task_name, force=force)
            for task_name in TRIGGERABLE_BACKGROUND_TASK_NAMES
        ]

    def cancel_task(
        self,
        task_id: str,
        *,
        cancelled_by: str = "cli",
        reason: str = "cancelled_by_user",
    ) -> dict:
        task = self._task_queue.request_cancel(
            task_id,
            cancelled_by=cancelled_by,
            reason=reason,
        )
        signal_sent = False
        if task.status == "running" and task.subprocess_pid is not None:
            signal_sent = _terminate_process(task.subprocess_pid)
        return {
            "status": "cancelled" if task.status == "cancelled" else "cancellation_requested",
            "signal_sent": signal_sent,
            "task": task_payload(self._task_queue.get_task(task_id)),
        }

    def list_ai_conversations(
        self,
        *,
        request_id: str | None = None,
        task_name: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> AIConversationListPayload:
        return AIConversationListPayload(
            conversations=[
                AIConversationPayload(
                    id=record.id,
                    request_id=record.request_id,
                    attempt=record.attempt,
                    workspace_id=record.workspace_id,
                    task_name=record.task_name,
                    task_id=record.task_id,
                    provider_key=record.provider_key,
                    provider_name=record.provider_name,
                    model_name=record.model_name,
                    subprocess_pid=record.subprocess_pid,
                    prompt_text=record.prompt_text,
                    response_text=record.response_text,
                    parsed=record.parsed,
                    status=record.status,
                    error_text=record.error_text,
                    started_at=record.started_at,
                    completed_at=record.completed_at,
                    duration_seconds=record.duration_seconds,
                )
                for record in self._provider_usage.list_conversations(
                    request_id=request_id,
                    task_name=task_name,
                    status=status,
                    limit=limit,
                )
            ]
        )

    def get_memory_detail(self, memory_id: str):
        if self._memory_queries is None:
            raise ValueError("repository_not_initialized")

        record = self._memory_queries.get_memory(memory_id)
        if record is None:
            raise ValueError("memory_not_found")

        outgoing = self._memory_queries.get_links(memory_id, direction="outgoing")
        incoming = self._memory_queries.get_links(memory_id, direction="incoming")
        superseded = [
            memory_record_payload(target)
            for target in self._memory_queries.get_superseded_records(memory_id)
        ]

        return MemoryDetailPayload(
            record=memory_record_payload(record),
            relationships={
                "incoming": [link_payload(link) for link in incoming],
                "outgoing": [link_payload(link) for link in outgoing],
            },
            superseded=superseded,
        )

    def create_memory_link(
        self,
        *,
        source_id: str,
        target_id: str,
        link_type: str,
        context: str = "",
    ) -> dict:
        if self._memory_queries is None:
            raise ValueError("repository_not_initialized")

        source = self._memory_queries.get_memory(source_id)
        if source is None:
            raise ValueError("source_memory_not_found")
        if not target_id.startswith("ext:") and self._memory_queries.get_memory(target_id) is None:
            raise ValueError("target_memory_not_found")

        assert self._repository is not None
        link = self._repository.add_link(
            source_id=source_id,
            target_id=target_id,
            link_type=link_type,
            context=context,
        )
        return {"status": "created", "link": link_payload(link)}

    def delete_memory_link(
        self,
        *,
        source_id: str,
        target_id: str,
        link_type: str,
    ) -> dict:
        if self._memory_queries is None:
            raise ValueError("repository_not_initialized")

        assert self._repository is not None
        deleted = self._repository.remove_link(
            source_id=source_id,
            target_id=target_id,
            link_type=link_type,
        )
        if not deleted:
            raise ValueError("link_not_found")
        return {"status": "deleted"}

    def list_tasks(
        self,
        status: str | None = None,
        workspace_id: str | None = None,
        limit: int = 20,
    ) -> TaskListPayload:
        tasks = self._task_queue.list_tasks(
            status=status,
            workspace_id=workspace_id,
            limit=limit,
        )
        return TaskListPayload(tasks=[task_payload(task) for task in tasks])

    def list_memories(
        self,
        workspace_id: str | None = None,
        memory_type: str | None = None,
        status: str | None = None,
        limit: int = 20,
    ) -> MemoryListPayload:
        if self._memory_queries is None:
            return MemoryListPayload()
        records = self._memory_queries.list_memories(
            workspace_id=workspace_id,
            memory_type=memory_type,
            status=status,
            limit=limit,
        )
        return MemoryListPayload(
            records=[compact_memory_record_payload(record) for record in records]
        )

    def list_logs(
        self,
        *,
        level: str | None = None,
        logger_name: str | None = None,
        source: str | None = None,
        query: str | None = None,
        after: float | None = None,
        before: float | None = None,
        limit: int = 50,
    ) -> RuntimeLogListPayload:
        if self._db_manager is None:
            return RuntimeLogListPayload()
        return RuntimeLogListPayload(
            logs=[
                RuntimeLogPayload(
                    id=record.id,
                    created_at=record.created_at,
                    level=record.level,
                    logger_name=record.logger_name,
                    source=record.source,
                    message=record.message,
                    data=record.data,
                )
                for record in self._runtime_logs.list_logs(
                    level=level,
                    logger_name=logger_name,
                    source=source,
                    query=query,
                    after=after,
                    before=before,
                    limit=limit,
                )
            ]
        )

    def summarize_logs(
        self,
        *,
        level: str | None = None,
        logger_name: str | None = None,
        source: str | None = None,
        query: str | None = None,
        after: float | None = None,
        before: float | None = None,
    ) -> RuntimeLogSummaryPayload:
        if self._db_manager is None:
            return RuntimeLogSummaryPayload()
        summary = self._runtime_logs.summarize_logs(
            level=level,
            logger_name=logger_name,
            source=source,
            query=query,
            after=after,
            before=before,
        )
        return RuntimeLogSummaryPayload(
            total=summary.total,
            by_level=summary.by_level,
            by_source=summary.by_source,
        )

    def prune_logs(
        self,
        *,
        max_runtime_logs: int | None = None,
        max_log_age_days: int | None = None,
    ) -> RuntimeLogPrunePayload:
        config = self._runtime_logs.retention_policy
        deleted = self._runtime_logs.prune_logs(
            max_runtime_logs=max_runtime_logs,
            max_age_days=max_log_age_days,
        )
        resolved_max_runtime_logs = config.max_runtime_logs if max_runtime_logs is None else max_runtime_logs
        resolved_max_log_age_days = config.max_log_age_days if max_log_age_days is None else max_log_age_days
        return RuntimeLogPrunePayload(
            deleted=deleted,
            max_runtime_logs=resolved_max_runtime_logs,
            max_log_age_days=resolved_max_log_age_days,
        )

    def load_dashboard_html(self):
        return self._dashboard_static_path.read_text(encoding="utf-8")

    def _build_memory_counts(self) -> tuple[dict[str, int], dict[str, int], int]:
        if self._db_manager is None:
            return {}, {}, 0
        conn = self._db_manager.get_connection()
        query = (
            "SELECT memories.type, memories.status, COUNT(*) AS count "
            "FROM memories JOIN memory_workspaces ON memory_workspaces.memory_id = memories.id"
        )
        params: list[object] = []
        if self._workspace_id is not None:
            query += " WHERE memory_workspaces.workspace_id = ?"
            params.append(self._workspace_id)
        query += " GROUP BY memories.type, memories.status"
        rows = conn.execute(query, params).fetchall()
        by_type: dict[str, int] = {}
        by_status: dict[str, int] = {}
        total = 0
        for row in rows:
            count = int(row["count"])
            total += count
            by_type[str(row["type"])] = by_type.get(str(row["type"]), 0) + count
            by_status[str(row["status"])] = by_status.get(str(row["status"]), 0) + count
        return by_type, by_status, total

    def _build_task_counts(self) -> dict[str, int]:
        if self._db_manager is None:
            return {}
        conn = self._db_manager.get_connection()
        query = "SELECT status, COUNT(*) AS count FROM tasks"
        params: list[object] = []
        if self._workspace_id is not None:
            query += " WHERE workspace_id = ?"
            params.append(self._workspace_id)
        query += " GROUP BY status"
        rows = conn.execute(query, params).fetchall()
        return {str(row["status"]): int(row["count"]) for row in rows}

    def _build_memory_metrics(self) -> MemoryMetricsPayload:
        if self._db_manager is None:
            return MemoryMetricsPayload(
                total_memories=0,
                total_memory_lines=0,
                total_summary_lines=0,
                total_lines_compressed=0,
                thought_buffer_entries=0,
                thought_buffer_lines=0,
            )

        conn = self._db_manager.get_connection()
        memory_query = (
            "SELECT "
            "COUNT(*) AS total_memories, "
            "COALESCE(SUM(CASE WHEN memories.content = '' THEN 0 ELSE 1 + LENGTH(memories.content) - LENGTH(REPLACE(memories.content, CHAR(10), '')) END), 0) AS total_memory_lines, "
            "COALESCE(SUM(CASE WHEN memories.summary IS NULL OR memories.summary = '' THEN 0 ELSE 1 + LENGTH(memories.summary) - LENGTH(REPLACE(memories.summary, CHAR(10), '')) END), 0) AS total_summary_lines "
            "FROM memories JOIN memory_workspaces ON memory_workspaces.memory_id = memories.id"
        )
        memory_params: list[object] = []
        if self._workspace_id is not None:
            memory_query += " WHERE memory_workspaces.workspace_id = ?"
            memory_params.append(self._workspace_id)
        memory_row = conn.execute(memory_query, memory_params).fetchone()

        journal_query = (
            "SELECT "
            "COUNT(*) AS thought_buffer_entries, "
            "COALESCE(SUM(CASE WHEN content = '' THEN 0 ELSE 1 + LENGTH(content) - LENGTH(REPLACE(content, CHAR(10), '')) END), 0) AS thought_buffer_lines "
            "FROM system1_journal WHERE status = 'pending'"
        )
        journal_params: list[object] = []
        if self._workspace_id is not None:
            journal_query += " AND workspace_id = ?"
            journal_params.append(self._workspace_id)
        journal_row = conn.execute(journal_query, journal_params).fetchone()

        total_lines_compressed = sum(
            max(agent_run.total_lines_compressed, 0)
            for agent_run in self._task_queue.summarize_task_runs(
                list(TRIGGERABLE_BACKGROUND_TASK_NAMES),
                workspace_id=self._workspace_id,
            )
        )
        return MemoryMetricsPayload(
            total_memories=0 if memory_row is None else int(memory_row["total_memories"]),
            total_memory_lines=0 if memory_row is None else int(memory_row["total_memory_lines"]),
            total_summary_lines=0 if memory_row is None else int(memory_row["total_summary_lines"]),
            total_lines_compressed=total_lines_compressed,
            thought_buffer_entries=0 if journal_row is None else int(journal_row["thought_buffer_entries"]),
            thought_buffer_lines=0 if journal_row is None else int(journal_row["thought_buffer_lines"]),
        )

    def _build_agent_runs(self) -> list[AgentRunPayload]:
        now = time.time()
        running_tasks = self._task_queue.list_tasks(
            status="running",
            workspace_id=self._workspace_id,
            limit=200,
        )
        pending_tasks = self._task_queue.list_tasks(
            status="pending",
            workspace_id=self._workspace_id,
            limit=200,
        )
        running_by_name: dict[str, int] = {}
        for task in running_tasks:
            running_by_name[task.task_name] = running_by_name.get(task.task_name, 0) + 1

        next_pending_by_name: dict[str, float] = {}
        for task in pending_tasks:
            current = next_pending_by_name.get(task.task_name)
            if current is None or task.available_at < current:
                next_pending_by_name[task.task_name] = task.available_at

        summaries = self._task_queue.summarize_task_runs(
            list(TRIGGERABLE_BACKGROUND_TASK_NAMES),
            workspace_id=self._workspace_id,
        )
        payloads: list[AgentRunPayload] = []
        for summary in summaries:
            seconds_since_last_completion = None
            if summary.last_completed_at is not None:
                seconds_since_last_completion = max(now - summary.last_completed_at, 0.0)
            next_available_at = next_pending_by_name.get(summary.task_name)
            seconds_until_next_run = None
            if next_available_at is not None:
                seconds_until_next_run = max(next_available_at - now, 0.0)
            payloads.append(
                AgentRunPayload(
                    task_name=summary.task_name,
                    running_count=running_by_name.get(summary.task_name, 0),
                    total_runs=summary.total_runs,
                    completed_runs=summary.completed_runs,
                    failed_runs=summary.failed_runs,
                    cancelled_runs=summary.cancelled_runs,
                    retry_runs=summary.retry_runs,
                    avg_duration_seconds=summary.avg_duration_seconds,
                    total_lines_compressed=summary.total_lines_compressed,
                    last_status=summary.last_status,
                    last_completed_at=summary.last_completed_at,
                    seconds_since_last_completion=seconds_since_last_completion,
                    last_error=summary.last_error,
                    last_result_summary=_format_result_summary(summary.last_result),
                    next_available_at=next_available_at,
                    seconds_until_next_run=seconds_until_next_run,
                )
            )
        return payloads

    def _build_provider_usage(self) -> list[ProviderUsagePayload]:
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
                avg_duration_last_hour=summary.avg_duration_last_hour,
                avg_duration_last_day=summary.avg_duration_last_day,
            )
            for summary in self._provider_usage.summarize_usage()
        ]

    def _build_recent_agent_runs(self, limit: int = 20) -> list[AgentRunHistoryPayload]:
        rows = [] if self._db_manager is None else self._db_manager.get_connection().execute(
            f"SELECT * FROM task_runs WHERE task_name IN ({','.join('?' for _ in TRIGGERABLE_BACKGROUND_TASK_NAMES)})"
            + (" AND workspace_id = ?" if self._workspace_id is not None else "")
            + " ORDER BY completed_at DESC, started_at DESC LIMIT ?",
            [
                *TRIGGERABLE_BACKGROUND_TASK_NAMES,
                *([self._workspace_id] if self._workspace_id is not None else []),
                limit,
            ],
        ).fetchall()
        return [
            AgentRunHistoryPayload(
                task_name=str(row["task_name"]),
                status=str(row["status"]),
                started_at=float(row["started_at"]),
                completed_at=float(row["completed_at"]),
                duration_seconds=float(row["duration_seconds"] or 0.0),
                error_text=row["error_text"],
                result_summary=_format_result_summary(_decode_run_result(row["result_json"])),
            )
            for row in rows
        ]


def _format_result_summary(result: dict[str, object]) -> str | None:
    if not result:
        return None
    preferred_keys = (
        "created",
        "merged",
        "updated",
        "archived",
        "absorbed_observations",
        "degraded",
        "restored",
        "deleted_tasks",
        "deleted_journal_entries",
        "claimed_entry_ids",
        "deleted_entry_ids",
        "released_entry_ids",
        "meaningful_actions",
        "processed_entry_ids",
        "created_memory_ids",
        "lines_compressed",
    )
    formatted_parts: list[str] = []
    for key in preferred_keys:
        if key not in result:
            continue
        value = result[key]
        if isinstance(value, list):
            formatted_parts.append(f"{key}={len(value)}")
        else:
            formatted_parts.append(f"{key}={value}")

    if formatted_parts:
        return ", ".join(formatted_parts)

    for key in sorted(result):
        value = result[key]
        if isinstance(value, (str, int, float, bool)):
            formatted_parts.append(f"{key}={value}")
    return ", ".join(formatted_parts) if formatted_parts else None


def _decode_run_result(raw_result: object) -> dict[str, object]:
    if not isinstance(raw_result, str) or not raw_result.strip():
        return {}
    try:
        decoded = json.loads(raw_result)
    except ValueError:
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _terminate_process(pid: int) -> bool:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return False
    except PermissionError:
        return False
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        if not _is_process_alive(pid):
            return True
        time.sleep(0.05)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return not _is_process_alive(pid)


def _is_process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
