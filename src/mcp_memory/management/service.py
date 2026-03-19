from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import signal
from statistics import mean, median
import time

from mcp_memory.context import ApplicationContext
from mcp_memory.core import MemoryPipeline
from mcp_memory.core.journal_operations import RecordThoughtOperation
from mcp_memory.core.provider_policy import select_provider_for_task
from mcp_memory.core.task_handlers import TRIGGERABLE_BACKGROUND_TASK_NAMES
from mcp_memory.core.task_handlers import task_priority
from mcp_memory.core.task_handlers import SUMMARIZE_MEMORY_TASK_NAME
from mcp_memory.core.task_policy import DEFAULT_AGENTIC_TASK_NAMES, DEFAULT_LOW_PRIORITY_TASK_NAMES, task_class_for_task
from mcp_memory.core.tasks import TaskRecord
from mcp_memory.embeddings import describe_embedder
from mcp_memory.management.models import (
    AgentRunHistoryPayload,
    AgentRunHistoryListPayload,
    AgentRunPayload,
    AIConversationListPayload,
    AIConversationPayload,
    AgentThroughputBucketPayload,
    EmbeddingStatusPayload,
    GraphTopologyPayload,
    HealthPayload,
    JournalSummary,
    MemoryLifecyclePayload,
    MemoryListPayload,
    MemoryDetailPayload,
    MemoryMetricsPayload,
    MemorySearchResultPayload,
    MemorySearchPayload,
    NerdAlertPayload,
    NerdMetricsPayload,
    NerdStatPayload,
    OverviewCounts,
    OverviewPayload,
    ProviderUsagePayload,
    ProviderLatencyBucketPayload,
    QueueSnapshotPayload,
    QueueDiagnosticPayload,
    RunResultMetadataPayload,
    RuntimeLogListPayload,
    RuntimeLogPrunePayload,
    RuntimeLogPayload,
    RuntimeLogSummaryPayload,
    SearchQualityPayload,
    SearchHealthPayload,
    StorageSummary,
    TaskRouteAuditPayload,
    TaskListPayload,
    TaskStatusSummary,
)
from mcp_memory.provider_usage_store import ProviderUsageRepository
from mcp_memory.runtime_log_store import RuntimeLogRepository
from mcp_memory.serialization import (
    compact_memory_record_payload,
    link_payload,
    memory_record_payload,
    search_result_payload,
    task_payload,
)


class ManagementService:
    def __init__(self, ctx: ApplicationContext, controller) -> None:
        pipeline = MemoryPipeline.from_context(ctx, controller)
        self._controller = controller
        self._db_manager = ctx.db_manager
        self._workspace_id = ctx.workspace_id
        self._runtime_info = pipeline.runtime_info
        self._journal = pipeline.journal
        self._task_queue = pipeline.task_queue
        self._memory_queries = pipeline.memory_queries
        self._repository = ctx.repository
        self._provider_usage = ProviderUsageRepository(ctx.db_manager, workspace_id=self._workspace_id)
        self._runtime_logs = RuntimeLogRepository(
            ctx.db_manager,
            workspace_id=self._workspace_id,
            config=None if ctx.config is None else ctx.config.logging,
        )
        self._embedder = ctx.embedder
        self._relational_search = ctx.relational_search
        self._config = ctx.config
        self._ai_json_provider = getattr(ctx, "ai_json_provider", None) or getattr(ctx, "ai_provider", None)
        self._ai_agent_provider = getattr(ctx, "ai_agent_provider", None)
        self._ai_provider_registry = getattr(ctx, "ai_provider_registry", None) or {}
        self._dashboard_static_root = Path(__file__).with_name("static")
        self._dashboard_static_path = self._dashboard_static_root / "index.html"
        self._dashboard_dist_path = self._dashboard_static_root / "dist" / "index.html"
        self._dashboard_asset_root = self._dashboard_static_root / "dist" / "assets"

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
        top_read_active_records = [] if self._repository is None else self._repository.list_most_read_memories(
            workspace_id=self._workspace_id,
            limit=10,
            status="active",
        )
        by_type, by_status, total_memories = self._build_memory_counts()
        recent_records = [compact_memory_record_payload(record) for record in records]
        top_read_payloads = [compact_memory_record_payload(record) for record in top_read_records]
        top_read_active_payloads = [compact_memory_record_payload(record) for record in top_read_active_records]
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
        recent_agent_runs = self.list_recent_agent_runs().runs
        recent_logs = self.list_logs(limit=10).logs

        return OverviewPayload(
            memories=OverviewCounts(total=total_memories, by_type=by_type, by_status=by_status),
            embeddings=self._build_embedding_status(),
            search=self._build_search_health(),
            memory_metrics=memory_metrics,
            queue_diagnostics=self._build_queue_diagnostics(),
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
            task = self._task_queue.enqueue(
                task_name=task_name,
                workspace_id=None,
                data=payload,
                priority=task_priority(task_name),
            )
            return {"status": "enqueued", "created": True, "task": task_payload(task)}

        task, created = self._task_queue.enqueue_unique(
            task_name=task_name,
            workspace_id=None,
            data=payload,
            priority=task_priority(task_name),
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

    def list_recent_agent_runs(self, limit: int = 20) -> AgentRunHistoryListPayload:
        return AgentRunHistoryListPayload(runs=self._build_recent_agent_runs(limit=limit))

    def get_nerd_metrics(
        self,
        *,
        window_hours: int = 24,
        bucket_minutes: int = 60,
        now: float | None = None,
    ) -> NerdMetricsPayload:
        if self._db_manager is None:
            return NerdMetricsPayload(
                generated_at=time.time() if now is None else now,
                window_hours=window_hours,
                bucket_minutes=bucket_minutes,
            )

        generated_at = time.time() if now is None else now
        bucket_seconds = max(bucket_minutes * 60, 60)
        cutoff = generated_at - (window_hours * 3600)
        conn = self._db_manager.get_connection()

        task_run_query = "SELECT status, completed_at, duration_seconds FROM task_runs WHERE completed_at >= ?"
        provider_query = (
            "SELECT provider_key, provider_name, model_name, status, duration_seconds, created_at "
            "FROM provider_usage WHERE created_at >= ?"
        )
        task_params: list[object] = [cutoff]
        provider_params: list[object] = [cutoff]
        if self._workspace_id is not None:
            task_run_query += " AND workspace_id = ?"
            provider_query += " AND workspace_id = ?"
            task_params.append(self._workspace_id)
            provider_params.append(self._workspace_id)

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

        queue_rows = self._build_queue_diagnostics(limit=200)
        runnable_queue_rows = [row for row in queue_rows if row.pending_state == "runnable"]
        queue_snapshot = QueueSnapshotPayload(
            runnable_count=len(runnable_queue_rows),
            scheduled_count=sum(1 for row in queue_rows if row.pending_state == "scheduled"),
            oldest_age_seconds=round(max((row.age_seconds for row in runnable_queue_rows), default=0.0), 4),
        )

        graph_topology = self._build_graph_topology()
        memory_lifecycle = self._build_memory_lifecycle()
        search_quality = self._build_search_quality(graph_topology=graph_topology, memory_lifecycle=memory_lifecycle)
        route_audit = self._build_route_audit()

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
            alerts=self._build_nerd_alerts(
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

    def record_thought(self, content: str) -> dict[str, object]:
        if self._journal.journal is None:
            raise ValueError("journal_not_initialized")
        return RecordThoughtOperation(
            self._journal.journal,
            self._task_queue.task_queue,
            self._runtime_info.workspace_id,
        ).execute(content)

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

    def search_memories(
        self,
        *,
        query: str,
        workspace_id: str | None = None,
        memory_type: str | None = None,
        status: str | None = None,
        include_superseded: bool = False,
        limit: int = 10,
        debug: bool = False,
    ) -> MemorySearchPayload:
        if self._relational_search is None:
            return MemorySearchPayload()
        return MemorySearchPayload(
            results=[
                MemorySearchResultPayload(**search_result_payload(result))
                for result in self._relational_search.search_memories(
                    query,
                    workspace_id=workspace_id,
                    limit=limit,
                    memory_type=memory_type,
                    status=status,
                    include_superseded=include_superseded,
                    debug=debug,
                )
            ]
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
        html_path = self._dashboard_dist_path if self._dashboard_dist_path.exists() else self._dashboard_static_path
        return html_path.read_text(encoding="utf-8")

    def resolve_dashboard_asset_path(self, asset_path: str) -> Path | None:
        if not asset_path.strip() or not self._dashboard_asset_root.exists():
            return None
        candidate = (self._dashboard_asset_root / asset_path).resolve()
        asset_root = self._dashboard_asset_root.resolve()
        if asset_root not in candidate.parents and candidate != asset_root:
            return None
        if not candidate.is_file():
            return None
        return candidate

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

    def _build_queue_diagnostics(self, limit: int = 8) -> list[QueueDiagnosticPayload]:
        now = time.time()
        pending_tasks = self._task_queue.list_tasks(
            status="pending",
            workspace_id=self._workspace_id,
            limit=200,
        )
        ordered = sorted(
            pending_tasks,
            key=lambda task: (
                0 if task.available_at <= now else 1,
                task.priority,
                task.available_at,
                task.created_at,
            ),
        )
        diagnostics: list[QueueDiagnosticPayload] = []
        for task in ordered[:limit]:
            runnable = task.available_at <= now
            diagnostics.append(
                QueueDiagnosticPayload(
                    task_id=task.id,
                    task_name=task.task_name,
                    workspace_id=task.workspace_id,
                    priority=task.priority,
                    pending_state="runnable" if runnable else "scheduled",
                    trigger=task.data.get("trigger") if isinstance(task.data.get("trigger"), str) else None,
                    created_at=task.created_at,
                    available_at=task.available_at,
                    age_seconds=max(now - task.created_at, 0.0),
                    ready_in_seconds=0.0 if runnable else max(task.available_at - now, 0.0),
                    overdue_seconds=max(now - task.available_at, 0.0) if runnable else 0.0,
                )
            )
        return diagnostics

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
                    last_result_metadata=_extract_run_result_metadata(summary.last_result),
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
            for summary in self._provider_usage.summarize_usage(workspace_id=self._workspace_id)
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
                result_metadata=_extract_run_result_metadata(_decode_run_result(row["result_json"])),
            )
            for row in rows
        ]

    def _build_graph_topology(self) -> GraphTopologyPayload:
        memory_rows = self._list_scoped_memories()
        memory_ids = {str(row["id"]) for row in memory_rows}
        if not memory_ids:
            return GraphTopologyPayload()

        link_rows = self._list_scoped_links(memory_ids)
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

    def _build_memory_lifecycle(self) -> MemoryLifecyclePayload:
        memory_rows = self._list_scoped_memories()
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

    def _build_search_quality(
        self,
        *,
        graph_topology: GraphTopologyPayload,
        memory_lifecycle: MemoryLifecyclePayload,
    ) -> SearchQualityPayload:
        health = self._build_search_health()
        return SearchQualityPayload(
            semantic_enabled=health.semantic_enabled,
            degraded=health.degraded,
            fallback_count=health.fallback_count,
            rebuild_count=health.rebuild_count,
            graph_supported_count=graph_topology.graph_supported_count,
            graph_supported_rate=graph_topology.graph_supported_rate,
            never_surfaced_count=memory_lifecycle.never_surfaced_count,
            last_error=health.last_error,
        )

    def _build_nerd_alerts(
        self,
        *,
        queue_snapshot: QueueSnapshotPayload,
        graph_topology: GraphTopologyPayload,
        memory_lifecycle: MemoryLifecyclePayload,
        search_quality: SearchQualityPayload,
        route_audit: list[TaskRouteAuditPayload],
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

    def _build_route_audit(self) -> list[TaskRouteAuditPayload]:
        task_names = [*TRIGGERABLE_BACKGROUND_TASK_NAMES, SUMMARIZE_MEMORY_TASK_NAME]
        usage_by_task: dict[str, tuple[int, int]] = {}
        top_provider_by_task: dict[str, tuple[str, str]] = {}
        top_provider_rank_by_task: dict[str, tuple[int, int]] = {}
        for summary in self._provider_usage.summarize_usage(workspace_id=self._workspace_id):
            task_name = summary.task_name
            if task_name is None:
                continue
            success_count, failure_count = usage_by_task.get(task_name, (0, 0))
            usage_by_task[task_name] = (
                success_count + summary.calls_last_day - summary.failures_last_day,
                failure_count + summary.failures_last_day,
            )
            rank = (summary.calls_last_day, -summary.failures_last_day)
            previous_rank = top_provider_rank_by_task.get(task_name)
            if previous_rank is None or rank > previous_rank:
                top_provider_by_task[task_name] = (summary.provider_key, summary.model_name)
                top_provider_rank_by_task[task_name] = rank

        audits: list[TaskRouteAuditPayload] = []
        for task_name in task_names:
            task_class = task_class_for_task(self._config, task_name)
            execution_kind = (
                "agentic"
                if task_class in {"cheap_agentic", "premium_agentic"}
                else "deterministic" if task_class == "deterministic" else "json"
            )
            configured_routes = self._configured_routes_for_task(task_name, execution_kind=execution_kind)
            selected = self._select_provider_for_audit(task_name)
            selected_provider_key = None if selected is None else getattr(selected, "_provider_key", None)
            selected_model_name = None if selected is None else getattr(selected, "_model_name", None)
            underlying_provider = None if selected is None else getattr(selected, "_provider", None)
            supports_agentic = None
            if selected is not None:
                supports_agentic_method = getattr(selected, "supports_agentic", None)
                if callable(supports_agentic_method):
                    supports_agentic = bool(supports_agentic_method())
            recent = self._provider_usage.list_conversations(task_name=task_name, limit=1)
            recent_record = recent[0] if recent else None
            recent_success_count, recent_failure_count = usage_by_task.get(task_name, (0, 0))
            usage_fallback = top_provider_by_task.get(task_name)

            audits.append(
                TaskRouteAuditPayload(
                    task_name=task_name,
                    task_class=task_class,
                    execution_kind=execution_kind,
                    low_priority=task_name in set((self._config.provider_routing.low_priority_task_names if self._config is not None else []) or DEFAULT_LOW_PRIORITY_TASK_NAMES),
                    configured_primary_route=configured_routes[0] if configured_routes else None,
                    configured_fallback_routes=configured_routes[1:] if len(configured_routes) > 1 else [],
                    resolved_provider_key=selected_provider_key,
                    resolved_model_name=selected_model_name,
                    resolved_provider_type=None if underlying_provider is None else type(underlying_provider).__name__,
                    resolved_supports_agentic=supports_agentic,
                    recent_provider_key=(None if recent_record is None else recent_record.provider_key) or (None if usage_fallback is None else usage_fallback[0]),
                    recent_model_name=(None if recent_record is None else recent_record.model_name) or (None if usage_fallback is None else usage_fallback[1]),
                    recent_status=None if recent_record is None else recent_record.status,
                    recent_success_count=recent_success_count,
                    recent_failure_count=recent_failure_count,
                    on_primary_route=None if selected_provider_key is None or not configured_routes else selected_provider_key == configured_routes[0] or str(selected_provider_key).startswith(configured_routes[0]),
                )
            )
        return audits

    def _configured_routes_for_task(self, task_name: str, *, execution_kind: str) -> list[str]:
        if self._config is None:
            return []
        routing = self._config.provider_routing
        if execution_kind == "deterministic":
            return []
        if task_name in routing.task_routes:
            return list(routing.task_routes[task_name])
        if execution_kind == "agentic":
            return list(routing.default_agentic_route)
        return list(routing.default_json_route)

    def _select_provider_for_audit(self, task_name: str):
        task = TaskRecord(
            id=f"audit:{task_name}",
            task_name=task_name,
            data={"workspace_id": self._workspace_id},
            workspace_id=self._workspace_id,
            status="pending",
            priority=task_priority(task_name),
            retries_count=0,
            max_retries=0,
            created_at=0.0,
            updated_at=0.0,
            available_at=0.0,
            claimed_at=None,
            started_at=None,
            completed_at=None,
            last_error=None,
        )
        audit_ctx = ApplicationContext(
            config=self._config,
            workspace_id=self._workspace_id,
            ai_provider_registry=self._ai_provider_registry,
        )
        return select_provider_for_task(
            audit_ctx,
            self._ai_json_provider,
            self._ai_agent_provider,
            task_name,
            task,
            agentic_task_names=set(DEFAULT_AGENTIC_TASK_NAMES),
        )

    def _list_scoped_memories(self):
        if self._db_manager is None:
            return []
        conn = self._db_manager.get_connection()
        query = (
            "SELECT memories.id, memories.type, memories.status, memories.last_accessed_at, memories.last_surfaced_at, "
            "LENGTH(COALESCE(memories.content, '')) AS content_bytes FROM memories"
        )
        params: list[object] = []
        if self._workspace_id is not None:
            query += (
                " WHERE EXISTS (SELECT 1 FROM memory_workspaces WHERE memory_workspaces.memory_id = memories.id "
                "AND memory_workspaces.workspace_id = ?)"
            )
            params.append(self._workspace_id)
        return conn.execute(query, params).fetchall()

    def _list_scoped_links(self, memory_ids: set[str]):
        if self._db_manager is None or not memory_ids:
            return []
        conn = self._db_manager.get_connection()
        if self._workspace_id is None:
            return conn.execute("SELECT source_id, target_id, type FROM links").fetchall()

        placeholders = ",".join("?" for _ in memory_ids)
        params = [*memory_ids, *memory_ids]
        query = (
            f"SELECT source_id, target_id, type FROM links WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})"
        )
        return conn.execute(query, params).fetchall()


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
        "requested_strategy",
        "strategy_used",
        "strategy_fallback_reason",
        "candidate_count",
        "sampled_memory_ids",
        "requested_grouping_strategy",
        "grouping_strategy_used",
        "grouping_fallback_reason",
        "group_count",
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


def _extract_run_result_metadata(result: dict[str, object]) -> RunResultMetadataPayload:
    sampled_memory_ids = result.get("sampled_memory_ids")
    return RunResultMetadataPayload(
        requested_strategy=_coerce_str(result.get("requested_strategy")),
        strategy_used=_coerce_str(result.get("strategy_used")),
        strategy_fallback_reason=_coerce_str(result.get("strategy_fallback_reason")),
        candidate_count=_coerce_int(result.get("candidate_count")),
        sampled_memory_ids=[str(item) for item in sampled_memory_ids] if isinstance(sampled_memory_ids, list) else [],
        requested_grouping_strategy=_coerce_str(result.get("requested_grouping_strategy")),
        grouping_strategy_used=_coerce_str(result.get("grouping_strategy_used")),
        grouping_fallback_reason=_coerce_str(result.get("grouping_fallback_reason")),
        group_count=_coerce_int(result.get("group_count")),
    )


def _coerce_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _coerce_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _decode_run_result(raw_result: object) -> dict[str, object]:
    if not isinstance(raw_result, str) or not raw_result.strip():
        return {}
    try:
        decoded = json.loads(raw_result)
    except ValueError:
        return {}
    return decoded if isinstance(decoded, dict) else {}


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


def _percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = max(math.ceil(len(sorted_values) * ratio) - 1, 0)
    return float(sorted_values[index])


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
