from __future__ import annotations

from dataclasses import asdict
from collections import Counter
from datetime import UTC, datetime, timedelta
import logging
import os
from pathlib import Path
import time
from time import perf_counter
from typing import cast
from uuid import uuid4

from mcp_memory.context import ApplicationContext
from mcp_memory.core import MemoryPipeline
from mcp_memory.core.journal_operations import RecordThoughtOperation
from mcp_memory.core.task_handlers import TRIGGERABLE_BACKGROUND_TASK_NAMES
from mcp_memory.core.task_handlers import task_priority
from mcp_memory.management.agent_run_reporting import build_recent_agent_runs
from mcp_memory.management.analytics_reporting import build_nerd_metrics
from mcp_memory.management.health_reporting import build_embedding_status, build_execution_attempt_health, build_search_health
from mcp_memory.management.operator_health_reporting import (
    build_operator_health_snapshot_payload,
    summarize_memory_tool_latency,
    summarize_provider_policy,
)
from mcp_memory.management.overview_reporting import build_overview
from mcp_memory.management.scope_policy import ScopePolicyKind, resolve_workspace_id_for_policy
from mcp_memory.management.selector_stats_reporting import build_selector_stats_payload
from mcp_memory.management.task_sampling_summary import build_task_sampling_summary
from mcp_memory.management.models import (
    AgentRunHistoryListPayload,
    AIConversationListPayload,
    AIConversationPayload,
    CacheHealthPayload,
    CacheMetricsPayload,
    HealthPayload,
    MemoryListPayload,
    MemoryDetailPayload,
    MemorySearchResultPayload,
    MemorySearchPayload,
    MemoryToolLatencyPayload,
    NerdMetricsPayload,
    OperatorHealthSnapshotPayload,
    RuntimeLogListPayload,
    RuntimeLogPrunePayload,
    RuntimeLogPayload,
    RuntimeLogSummaryPayload,
    SelectorStatsPayload,
    TaskDetailPayload,
    TaskSamplingSummaryPayload,
    TaskListPayload,
    TransportDiagnosticsPayload,
)
from mcp_memory.management.reporting_queries import count_recent_conversation_statuses, count_recent_memory_updates
from mcp_memory.provider_usage_store import ProviderUsageRepository
from mcp_memory.process_termination import send_process_signal as _send_process_signal
from mcp_memory.process_termination import terminate_process as _terminate_process_with_scope
from mcp_memory.process_termination import wait_for_process_exit as _wait_for_process_exit
from mcp_memory.runtime_log_store import RuntimeLogRepository, _AllWorkspacesSentinel
from mcp_memory.retrieval_telemetry_store import RetrievalTelemetryRepository
from mcp_memory.serialization import (
    compact_memory_record_payload,
    link_payload,
    memory_record_payload,
    search_result_payload,
    task_payload,
)
from mcp_memory.storage.noop import NoopProviderUsageRepository
from mcp_memory.storage.postgres_runtime_log_store import PostgresRuntimeLogRepository


_USE_SERVICE_WORKSPACE = object()
_SLOW_MEMORY_TOOL_WARNING_MS = 2_000.0


logger = logging.getLogger(__name__)


def _build_default_provider_usage(ctx: ApplicationContext):
    if (ctx.storage_backend or "sqlite") == "postgres":
        return NoopProviderUsageRepository(workspace_id=ctx.workspace_id)
    if not hasattr(ctx.db_manager, "get_connection"):
        return NoopProviderUsageRepository(workspace_id=ctx.workspace_id)
    return ProviderUsageRepository(ctx.db_manager, workspace_id=ctx.workspace_id)


def _build_default_runtime_logs(ctx: ApplicationContext):
    config = None if ctx.config is None else ctx.config.logging
    if (ctx.storage_backend or "sqlite") == "postgres":
        return PostgresRuntimeLogRepository(
            ctx.db_manager,
            workspace_id=ctx.workspace_id,
            config=config,
        )
    return RuntimeLogRepository(
        ctx.db_manager,
        workspace_id=ctx.workspace_id,
        config=config,
    )


def _resolve_log_workspace_id(
    service_workspace_id: str | None,
    workspace_id: str | None | object,
) -> str | None | _AllWorkspacesSentinel:
    if workspace_id is _USE_SERVICE_WORKSPACE:
        return service_workspace_id
    return cast(str | None | _AllWorkspacesSentinel, workspace_id)


def _resolve_service_workspace_id(
    service_workspace_id: str | None,
    workspace_id: str | None | object,
) -> str | None:
    if workspace_id is _USE_SERVICE_WORKSPACE:
        return service_workspace_id
    return cast(str | None, workspace_id)


class ManagementService:
    def __init__(self, ctx: ApplicationContext, controller) -> None:
        pipeline = MemoryPipeline.from_context(ctx, controller)
        self._controller = controller
        self._db_manager = ctx.db_manager
        self._storage_backend = ctx.storage_backend or "sqlite"
        self._workspace_id = ctx.workspace_id
        self._runtime_info = pipeline.runtime_info
        self._journal = pipeline.journal
        self._task_queue = pipeline.task_queue
        self._memory_queries = pipeline.memory_queries
        self._repository = ctx.repository
        self._provider_usage = ctx.provider_usage or _build_default_provider_usage(ctx)
        self._runtime_logs = ctx.runtime_logs or _build_default_runtime_logs(ctx)
        self._retrieval_telemetry = ctx.retrieval_telemetry
        if self._retrieval_telemetry is None:
            self._retrieval_telemetry = RetrievalTelemetryRepository(self._db_manager, workspace_id=self._workspace_id)
            ctx.retrieval_telemetry = self._retrieval_telemetry
        self._read_cache = getattr(ctx, "read_cache", None)
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

    @property
    def dashboard_static_root(self) -> Path:
        return self._dashboard_static_root

    @property
    def workspace_id(self) -> str | None:
        return self._workspace_id

    def get_health(self):
        embedder_status = build_embedding_status(self._embedder)
        return HealthPayload(
            status="ok",
            storage_backend=self._storage_backend,
            workspace_root=str(self._runtime_info.workspace_root) if self._runtime_info.workspace_root is not None else None,
            memory_path=str(self._runtime_info.memory_path) if self._runtime_info.memory_path is not None else None,
            db_path=str(self._runtime_info.db_path) if self._runtime_info.db_path is not None else None,
            runtime_active=bool(getattr(self._controller, "has_runtime", self._runtime_info.runtime_active)),
            client_count=int(getattr(self._controller, "client_count", self._runtime_info.client_count)),
            task_queue_enabled=self._runtime_info.task_queue_enabled,
            embeddings=embedder_status,
            search=build_search_health(self._relational_search),
            cache=self._build_cache_health(),
            transport_diagnostics=self._build_transport_diagnostics(),
            execution_attempts=build_execution_attempt_health(self._db_manager),
        )

    def _build_transport_diagnostics(self) -> TransportDiagnosticsPayload:
        snapshot = getattr(self._controller, "transport_diagnostics", None)
        if snapshot is None:
            return TransportDiagnosticsPayload()
        if isinstance(snapshot, TransportDiagnosticsPayload):
            return snapshot
        if isinstance(snapshot, dict):
            return TransportDiagnosticsPayload(**snapshot)
        return TransportDiagnosticsPayload()

    def _build_cache_health(self) -> CacheHealthPayload:
        cache_config = None if self._config is None else self._config.storage.cache
        cache_enabled = bool(getattr(cache_config, "enabled", False))
        cache_mode = getattr(cache_config, "mode", None) if cache_enabled else None
        metrics = self._build_cache_metrics_payload()
        if not cache_enabled:
            return CacheHealthPayload(enabled=False, mode=None, state="disabled", path=None, metrics=metrics)

        if self._storage_backend != "postgres":
            return CacheHealthPayload(enabled=True, mode=cache_mode, state="unsupported_backend", path=None, metrics=metrics)

        if cache_mode != "readonly":
            return CacheHealthPayload(enabled=True, mode=cache_mode, state="reserved_unimplemented", path=None, metrics=metrics)

        configured_path = self._configured_cache_path()
        if self._read_cache is None:
            return CacheHealthPayload(
                enabled=True,
                mode=cache_mode,
                state="inactive",
                path=str(configured_path) if configured_path is not None else None,
                metrics=metrics,
            )

        active_path = getattr(self._read_cache, "_db_path", None)
        resolved_path = active_path if isinstance(active_path, Path) else configured_path
        return CacheHealthPayload(
            enabled=True,
            mode=cache_mode,
            state="active",
            path=str(resolved_path) if resolved_path is not None else None,
            metrics=metrics,
        )

    def _build_cache_metrics_payload(self) -> CacheMetricsPayload:
        if self._read_cache is None:
            return CacheMetricsPayload()
        try:
            snapshot = self._read_cache.get_metrics_snapshot()
        except Exception:
            logger.warning("Failed to read shared cache metrics snapshot", exc_info=True)
            return CacheMetricsPayload()
        return CacheMetricsPayload(**asdict(snapshot))

    def _configured_cache_path(self) -> Path | None:
        if self._runtime_info.memory_path is None:
            return None
        return self._runtime_info.memory_path / "cache" / "shared_read_cache.sqlite3"

    def get_operator_health_snapshot(
        self,
        *,
        log_window_minutes: int = 15,
        recent_error_limit: int = 10,
        recent_warning_limit: int = 10,
        recent_run_limit: int = 10,
        conversation_window_hours: int = 24,
        conversation_limit: int = 10,
        recent_memory_limit: int = 10,
    ) -> OperatorHealthSnapshotPayload:
        generated_at = time.time()
        logs_after = generated_at - (max(log_window_minutes, 0) * 60)
        conversations_after = generated_at - (max(conversation_window_hours, 0) * 3600)

        health = self.get_health()
        log_summary = self.summarize_logs(workspace_id=None, after=logs_after)
        recent_errors = self.list_logs(workspace_id=None, level="ERROR", after=logs_after, limit=recent_error_limit).logs
        recent_warnings = self.list_logs(workspace_id=None, level="WARNING", after=logs_after, limit=recent_warning_limit).logs
        recent_run_rows = build_recent_agent_runs(
            self._db_manager,
            None,
            limit=max(recent_run_limit * 5, recent_run_limit),
            detail_level="compact",
        )
        recent_runs = recent_run_rows[:recent_run_limit]
        recent_run_status_counts = dict(sorted(Counter(run.status for run in recent_runs).items()))
        recent_failures = [run for run in recent_run_rows if run.status == "failed"][:recent_run_limit]
        recent_retries = [run for run in recent_run_rows if run.status == "retry"][:recent_run_limit]
        conversation_counts = count_recent_conversation_statuses(
            self._provider_usage,
            after=conversations_after,
            workspace_id=None,
        )
        recent_conversations = self.list_ai_conversations(workspace_id=None, limit=conversation_limit).conversations
        recent_memories = self.list_memories(workspace_id=None, limit=recent_memory_limit).records
        tool_latency = summarize_memory_tool_latency(
            self._db_manager,
            workspace_id=None,
            window_minutes=log_window_minutes,
            slow_threshold_ms=_SLOW_MEMORY_TOOL_WARNING_MS,
        )
        provider_policy = summarize_provider_policy(
            self._db_manager,
            provider_usage_repo=self._provider_usage,
            workspace_id=None,
            window_minutes=log_window_minutes,
        )
        updated_last_15_minutes = count_recent_memory_updates(
            self._repository,
            cutoff=datetime.now(UTC) - timedelta(minutes=15),
            workspace_id=None,
        )
        updated_last_hour = count_recent_memory_updates(
            self._repository,
            cutoff=datetime.now(UTC) - timedelta(minutes=60),
            workspace_id=None,
        )
        updated_last_day = count_recent_memory_updates(
            self._repository,
            cutoff=datetime.now(UTC) - timedelta(minutes=24 * 60),
            workspace_id=None,
        )

        return build_operator_health_snapshot_payload(
            generated_at=generated_at,
            log_window_minutes=log_window_minutes,
            conversation_window_hours=conversation_window_hours,
            health=health,
            log_summary=log_summary,
            recent_errors=recent_errors,
            recent_warnings=recent_warnings,
            recent_run_status_counts=recent_run_status_counts,
            recent_runs=recent_runs,
            recent_failures=recent_failures,
            recent_retries=recent_retries,
            conversation_counts=conversation_counts,
            recent_conversations=recent_conversations,
            updated_last_15_minutes=updated_last_15_minutes,
            updated_last_hour=updated_last_hour,
            updated_last_day=updated_last_day,
            recent_memories=recent_memories,
            tool_latency=tool_latency,
            provider_policy=provider_policy,
        )

    def get_overview(
        self,
        *,
        recent_limit: int = 10,
        failed_limit: int = 10,
    ):
        return build_overview(
            memory_queries=self._memory_queries,
            repository=self._repository,
            task_queue=self._task_queue,
            runtime_info=self._runtime_info,
            db_manager=self._db_manager,
            provider_usage_repo=self._provider_usage,
            runtime_logs_repo=self._runtime_logs,
            embedder=self._embedder,
            relational_search=self._relational_search,
            cache=self._build_cache_health(),
            recent_limit=recent_limit,
            failed_limit=failed_limit,
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
        workspace_id: str | None | object = _USE_SERVICE_WORKSPACE,
        request_id: str | None = None,
        task_name: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> AIConversationListPayload:
        effective_workspace_id = _resolve_log_workspace_id(self._workspace_id, workspace_id)
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
                    reason_category=record.reason_category,
                    reason_code=record.reason_code,
                    retry_delay_seconds=record.retry_delay_seconds,
                    started_at=record.started_at,
                    completed_at=record.completed_at,
                    duration_seconds=record.duration_seconds,
                )
                for record in self._provider_usage.list_conversations(
                    workspace_id=effective_workspace_id,
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

    def list_recent_agent_runs(self, *, limit: int = 20, detail_level: str = "compact") -> AgentRunHistoryListPayload:
        return AgentRunHistoryListPayload(
            runs=build_recent_agent_runs(
                self._db_manager,
                self._workspace_id,
                limit=limit,
                detail_level=detail_level,
            )
        )

    def get_task_sampling_summary(self, *, limit: int = 50) -> TaskSamplingSummaryPayload:
        return build_task_sampling_summary(self.list_recent_agent_runs(limit=limit).runs)

    def get_selector_stats(
        self,
        *,
        scope: str | None = None,
        workspace_id: str | None = None,
        window_hours: int = 24,
        limit: int = 200,
        now: float | None = None,
    ) -> SelectorStatsPayload:
        effective_workspace_id = self._resolve_scoped_workspace_id(
            scope=scope,
            workspace_id=workspace_id,
        )
        runs = build_recent_agent_runs(
            self._db_manager,
            effective_workspace_id,
            limit=limit,
            detail_level="compact",
        )
        return build_selector_stats_payload(
            runs,
            window_hours=window_hours,
            run_limit=limit,
            now=now,
        )

    def get_task_detail(self, task_id: str) -> TaskDetailPayload:
        task = self._task_queue.get_task(task_id)
        runs = self._task_queue.list_task_runs(task_id=task_id, limit=50)
        from mcp_memory.management.agent_run_reporting import build_agent_run_history_payload

        return TaskDetailPayload(
            task=task_payload(task),
            runs=[
                build_agent_run_history_payload(
                    task_id=run.task_id,
                    task_name=run.task_name,
                    status=run.status,
                    started_at=run.started_at,
                    completed_at=run.completed_at,
                    duration_seconds=run.duration_seconds,
                    error_text=run.error_text,
                    result=run.result,
                    detail_level="full",
                )
                for run in runs
            ],
        )

    def get_nerd_metrics(
        self,
        *,
        scope: str | None = None,
        workspace_id: str | None = None,
        window_hours: int = 24,
        bucket_minutes: int = 60,
        now: float | None = None,
    ) -> NerdMetricsPayload:
        effective_workspace_id = self._resolve_scoped_workspace_id(
            scope=scope,
            workspace_id=workspace_id,
        )
        return build_nerd_metrics(
            db_manager=self._db_manager,
            workspace_id=effective_workspace_id,
            task_queue=self._task_queue,
            provider_usage_repo=self._provider_usage,
            config=self._config,
            ai_json_provider=self._ai_json_provider,
            ai_agent_provider=self._ai_agent_provider,
            ai_provider_registry=self._ai_provider_registry,
            relational_search=self._relational_search,
            window_hours=window_hours,
            bucket_minutes=bucket_minutes,
            now=now,
        )

    def _resolve_scoped_workspace_id(
        self,
        *,
        scope: str | None,
        workspace_id: str | None,
    ) -> str | None:
        return resolve_workspace_id_for_policy(
            ScopePolicyKind.SERVICE_SCOPED_DEFAULT,
            scope=scope,
            workspace_id=workspace_id,
            current_workspace_id=self._workspace_id,
        )

    def record_thought(
        self,
        content: str,
        *,
        workspace_id: str | None | object = _USE_SERVICE_WORKSPACE,
    ) -> dict[str, object]:
        if self._journal.journal is None:
            raise ValueError("journal_not_initialized")
        effective_workspace_id = _resolve_service_workspace_id(self._workspace_id, workspace_id)
        return RecordThoughtOperation(
            self._journal.journal,
            self._task_queue.task_queue,
            effective_workspace_id,
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
        started_at = perf_counter()
        results = self._relational_search.search_memories(
            query,
            workspace_id=workspace_id,
            limit=limit,
            memory_type=memory_type,
            status=status,
            include_superseded=include_superseded,
            debug=debug,
        )
        duration_ms = (perf_counter() - started_at) * 1000.0
        self._retrieval_telemetry.record_search(
            invocation_id=str(uuid4()),
            caller_kind="operator",
            query=query,
            surfaced_memory_ids=[result.memory_id for result in results],
            duration_ms=duration_ms,
        )
        self._log_slow_memory_tool_operation(
            tool_name="management.search_memories",
            duration_ms=duration_ms,
            data={
                "query": query,
                "result_count": len(results),
                "storage_backend": self._storage_backend,
                "workspace_id": workspace_id,
            },
        )
        return MemorySearchPayload(
            results=[
                MemorySearchResultPayload(**search_result_payload(result))
                for result in results
            ]
        )

    def _summarize_memory_tool_latency(self, *, window_minutes: int) -> MemoryToolLatencyPayload:
        return summarize_memory_tool_latency(
            self._db_manager,
            workspace_id=self._workspace_id,
            window_minutes=window_minutes,
            slow_threshold_ms=_SLOW_MEMORY_TOOL_WARNING_MS,
        )

    def _log_slow_memory_tool_operation(
        self,
        *,
        tool_name: str,
        duration_ms: float,
        data: dict[str, object],
    ) -> None:
        if duration_ms < _SLOW_MEMORY_TOOL_WARNING_MS:
            return
        self._runtime_logs.write_log(
            source="memory-tool",
            logger_name=__name__,
            level="WARNING",
            message=f"Slow {tool_name} operation",
            created_at=time.time(),
            data={"tool_name": tool_name, "duration_ms": round(duration_ms, 3)} | data,
        )

    def list_logs(
        self,
        *,
        workspace_id: str | None | object = _USE_SERVICE_WORKSPACE,
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
        effective_workspace_id = _resolve_log_workspace_id(self._workspace_id, workspace_id)
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
                    workspace_id=effective_workspace_id,
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
        workspace_id: str | None | object = _USE_SERVICE_WORKSPACE,
        level: str | None = None,
        logger_name: str | None = None,
        source: str | None = None,
        query: str | None = None,
        after: float | None = None,
        before: float | None = None,
    ) -> RuntimeLogSummaryPayload:
        if self._db_manager is None:
            return RuntimeLogSummaryPayload()
        effective_workspace_id = _resolve_log_workspace_id(self._workspace_id, workspace_id)
        summary = self._runtime_logs.summarize_logs(
            workspace_id=effective_workspace_id,
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
        workspace_id: str | None | object = _USE_SERVICE_WORKSPACE,
        max_runtime_logs: int | None = None,
        max_log_age_days: int | None = None,
    ) -> RuntimeLogPrunePayload:
        config = self._runtime_logs.retention_policy
        effective_workspace_id = _resolve_log_workspace_id(self._workspace_id, workspace_id)
        deleted = self._runtime_logs.prune_logs(
            workspace_id=effective_workspace_id,
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

    def _summarize_provider_policy(self, *, window_minutes: int):
        return summarize_provider_policy(
            self._db_manager,
            provider_usage_repo=self._provider_usage,
            workspace_id=self._workspace_id,
            window_minutes=window_minutes,
        )



def _terminate_process(pid: int) -> bool:
    termination = _terminate_process_with_scope(
        pid,
        deadline=time.monotonic() + 1.0,
        poll_interval_seconds=0.05,
        is_process_running=_is_process_alive,
        send_signal=lambda process_id, sig: _send_process_signal(
            process_id,
            sig,
            scope="pid",
            suppress_permission_errors=True,
        ),
        wait_for_exit=lambda process_id, *, deadline, poll_interval_seconds: _wait_for_process_exit(
            process_id,
            deadline=deadline,
            poll_interval_seconds=poll_interval_seconds,
            is_process_running=_is_process_alive,
        ),
    )
    return termination.signal_sent


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
