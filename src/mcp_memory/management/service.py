from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Any, cast

from mcp_memory.application.memory_embedding_maintenance import MemoryEmbeddingMaintenance
from mcp_memory.context import ManagementContext, ManagementRuntimeCapabilities
from mcp_memory.core import MemoryPipeline
from mcp_memory.management.analytics_service import AnalyticsService, AnalyticsServiceDependencies
from mcp_memory.management.context_resources import (
    ManagementContextResources,
    _build_default_embedding_integrity_events as _context_build_default_embedding_integrity_events,
    _build_default_provider_usage as _context_build_default_provider_usage,
    ensure_management_context_resources,
)
from mcp_memory.management.capabilities import ManagementCapabilities
from mcp_memory.management.overview_service import OverviewService, OverviewServiceDependencies
from mcp_memory.management.runtime_health_service import (
    RuntimeHealthService,
    RuntimeHealthServiceDependencies,
    _coerce_transport_diagnostics_payload,  # noqa: F401 - retained for management helper compatibility
    _embedding_integrity_snapshot_payload,  # noqa: F401 - retained for management helper compatibility
)
from mcp_memory.management.runtime_log_service import (
    RuntimeLogService,
    RuntimeLogServiceDependencies,
    _ensure_dashboard_base_href,  # noqa: F401 - retained for management helper compatibility
    _resolve_log_workspace_id,  # noqa: F401 - retained for management helper compatibility
)
from mcp_memory.management.mutation_history_service import (
    MutationHistoryService,
    MutationHistoryServiceDependencies,
)
from mcp_memory.management.memory_service import (
    MemoryService,
    MemoryServiceDependencies,
    _USE_SERVICE_WORKSPACE,
    _resolve_service_workspace_id,  # noqa: F401 - retained for management helper compatibility
)
from mcp_memory.management.task_reporting_service import (
    TaskReportingService,
    TaskReportingServiceDependencies,
)
from mcp_memory.management.task_administration import (
    TaskAdministrationService,
    TaskAdministrationServiceDependencies,
)
from mcp_memory.integrations.memory_retrieval import build_memory_retrieval_facade
from mcp_memory.integrations.federation_source import MemoryFederationSource
from mcp_memory.management.models import (
    AgentRunHistoryListPayload,
    AIConversationListPayload,
    MemoryListPayload,
    MemorySearchPayload,
    MutationHistoryDetailPayload,
    MutationHistoryDiffPayload,
    MutationHistoryListPayload,
    NerdMetricsPayload,
    OperatorHealthSnapshotPayload,
    ProtectionListPayload,
    ProtectionMutationPayload,
    QualityCleanupCandidatesPayload,
    RuntimeLogListPayload,
    RuntimeLogPrunePayload,
    RuntimeLogSummaryPayload,
    RestoreEligibilityPayload,
    RestoreRequestPayload,
    SelectorStatsPayload,
    TaskDetailPayload,
    TaskSamplingSummaryPayload,
    TaskListPayload,
)
from mcp_memory.process_termination import send_process_signal as _send_process_signal
from mcp_memory.process_termination import terminate_process as _terminate_process_with_scope
from mcp_memory.process_termination import wait_for_process_exit as _wait_for_process_exit
from mcp_memory.management.query_runner import PostgresManagementQueryAdapter, SQLiteManagementQueryAdapter
from mcp_memory.serialization import (
    task_payload,
)


_build_default_provider_usage = _context_build_default_provider_usage
_build_default_embedding_integrity_events = _context_build_default_embedding_integrity_events


class ManagementService:
    def __init__(
        self,
        ctx: ManagementRuntimeCapabilities | ManagementContext,
        controller,
    ) -> None:
        is_composed_capabilities = isinstance(ctx, ManagementRuntimeCapabilities)
        capabilities = ctx if is_composed_capabilities else ManagementRuntimeCapabilities.from_context(ctx)
        memory = capabilities.memory
        mutation = capabilities.mutation
        provider = capabilities.provider
        pipeline = MemoryPipeline.from_context(memory, controller, mutation=mutation)
        resources = (
            ManagementContextResources.from_capabilities(capabilities)
            if is_composed_capabilities
            else ensure_management_context_resources(capabilities)
        )
        self._controller = controller
        self._db_manager = cast(Any, memory.db_manager)
        self._storage_backend = capabilities.storage_backend or "sqlite"
        query_adapter = (
            PostgresManagementQueryAdapter()
            if self._storage_backend == "postgres"
            else SQLiteManagementQueryAdapter()
        )
        self._workspace_id = memory.workspace_id
        self._config = memory.config
        self._runtime_info = pipeline.runtime_info
        self._journal = pipeline.journal
        self._task_queue = pipeline.task_queue
        self._memory_queries = pipeline.memory_queries
        self._repository = cast(Any, mutation.repository)
        self._mutation_history = cast(Any, resources.mutation_history)
        self._curation = cast(Any, mutation.curation)
        self._action_store = cast(Any, mutation.curation_action_store)
        self._provider_usage = resources.provider_usage
        self._runtime_logs = resources.runtime_logs
        self._embedding_integrity_events = resources.embedding_integrity_events
        self._retrieval_telemetry = resources.retrieval_telemetry
        self._read_cache = cast(Any, memory.read_cache)
        self._embedder = cast(Any, memory.embedder)
        self._vector_store = cast(Any, memory.vector_store)
        self._relational_search = cast(Any, memory.relational_search)
        self._search_health = memory.search_health
        self._embedding_maintenance = cast(Any, (
            memory.embedding_maintenance
            or getattr(self._relational_search, "_embedding_maintenance", None)
            or MemoryEmbeddingMaintenance.from_context(ctx)
        ))
        retrieval = (
            build_memory_retrieval_facade(
                self._repository,
                config=memory.config,
                vector_store=self._vector_store,
                embedder=self._embedder,
                embedding_maintenance=self._embedding_maintenance,
                native_search=self._relational_search,
            )
            if self._repository is not None
            else None
        )
        self._retrieval = retrieval
        self.federation_source = (
            MemoryFederationSource(
                retrieval,
                self._repository,
                health_provider=self._search_health or self._relational_search,
            )
            if retrieval is not None
            else None
        )
        self._task_reporting_service = TaskReportingService(
            TaskReportingServiceDependencies(
                db_manager=self._db_manager,
                workspace_id=self._workspace_id,
                query_adapter=query_adapter,
            )
        )
        self._ai_json_provider = provider.ai_json_provider
        self._ai_agent_provider = provider.ai_agent_provider
        self._ai_provider_registry = provider.ai_provider_registry or {}
        self._analytics_service = AnalyticsService(
            AnalyticsServiceDependencies(
                db_manager=self._db_manager,
                workspace_id=self._workspace_id,
                storage_backend=self._storage_backend,
                task_queue=self._task_queue,
                provider_usage_repo=self._provider_usage,
                config=self._config,
                ai_json_provider=self._ai_json_provider,
                ai_agent_provider=self._ai_agent_provider,
                ai_provider_registry=self._ai_provider_registry,
                relational_search=self._relational_search,
                search_health=self._search_health,
            )
        )
        self._task_administration_service = TaskAdministrationService(
            TaskAdministrationServiceDependencies(
                task_queue=self._task_queue,
                terminate_process=_terminate_process,
                task_payload=task_payload,
            )
        )
        self._mutation_history_service = MutationHistoryService(
            MutationHistoryServiceDependencies(
                mutation_history=self._mutation_history,
                memory_queries=self._memory_queries,
                curation=self._curation,
                action_store=self._action_store,
            )
        )
        self._memory_service = MemoryService(
            MemoryServiceDependencies(
                workspace_id=self._workspace_id,
                journal=self._journal,
                task_queue=self._task_queue,
                config=self._config,
                storage_backend=self._storage_backend,
                read_cache=self._read_cache,
                memory_queries=self._memory_queries,
                repository=self._repository,
                relational_search=self._relational_search,
                retrieval=retrieval,
                provider_usage_repo=self._provider_usage,
                retrieval_telemetry=self._retrieval_telemetry,
                runtime_logs=self._runtime_logs,
            )
        )
        self._dashboard_static_root = Path(__file__).with_name("static")
        self._dashboard_static_path = self._dashboard_static_root / "index.html"
        self._dashboard_dist_path = self._dashboard_static_root / "dist" / "index.html"
        self._dashboard_asset_root = self._dashboard_static_root / "dist" / "assets"
        self._runtime_log_service = RuntimeLogService(
            RuntimeLogServiceDependencies(
                db_manager=self._db_manager,
                runtime_logs=self._runtime_logs,
                workspace_id=self._workspace_id,
                dashboard_static_root=self._dashboard_static_root,
                dashboard_static_path=self._dashboard_static_path,
                dashboard_dist_path=self._dashboard_dist_path,
                dashboard_asset_root=self._dashboard_asset_root,
            )
        )
        self._runtime_health_service = RuntimeHealthService(
            RuntimeHealthServiceDependencies(
                runtime_info=self._runtime_info,
                controller=self._controller,
                db_manager=self._db_manager,
                storage_backend=self._storage_backend,
                config=self._config,
                read_cache=self._read_cache,
                embedder=self._embedder,
                vector_store=self._vector_store,
                relational_search=self._relational_search,
                embedding_integrity_events=self._embedding_integrity_events,
                embedding_maintenance=self._embedding_maintenance,
                repository=self._repository,
                provider_usage=self._provider_usage,
                retrieval_telemetry=self._retrieval_telemetry,
                runtime_log_service=self._runtime_log_service,
                list_ai_conversations=self._memory_service.list_ai_conversations,
                list_memories=self._memory_service.list_memories,
                search_health=self._search_health,
            )
        )
        self._overview_service = OverviewService(
            OverviewServiceDependencies(
                memory_queries=self._memory_queries,
                repository=self._repository,
                task_queue=self._task_queue,
                runtime_info=self._runtime_info,
                db_manager=self._db_manager,
                provider_usage_repo=self._provider_usage,
                runtime_logs_repo=self._runtime_logs,
                embedder=self._embedder,
                storage_backend=self._storage_backend,
                vector_store=self._vector_store,
                relational_search=self._relational_search,
                retrieval_telemetry=self._retrieval_telemetry,
                cache_health=self._runtime_health_service._build_cache_health,
                search_health=self._search_health,
                embedding_integrity_summary=lambda: self._runtime_health_service._embedding_integrity_summary(
                    workspace_id=None
                ),
            )
        )
        self.capabilities = ManagementCapabilities.from_service(self)

    @property
    def dashboard_static_root(self) -> Path:
        return self._runtime_log_service.dashboard_static_root

    @property
    def workspace_id(self) -> str | None:
        return self._workspace_id

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
        self._refresh_runtime_health_service()
        return self._runtime_health_service.get_operator_health_snapshot(
            log_window_minutes=log_window_minutes,
            recent_error_limit=recent_error_limit,
            recent_warning_limit=recent_warning_limit,
            recent_run_limit=recent_run_limit,
            conversation_window_hours=conversation_window_hours,
            conversation_limit=conversation_limit,
            recent_memory_limit=recent_memory_limit,
        )

    def get_health(self):
        self._refresh_runtime_health_service()
        return self._runtime_health_service.get_health()

    def repair_search_index(self) -> dict[str, int | bool | str | None]:
        self._refresh_runtime_health_service()
        return self._runtime_health_service.repair_search_index()

    def get_overview(
        self,
        *,
        recent_limit: int = 10,
        failed_limit: int = 10,
    ):
        self._refresh_runtime_health_service()
        return self._overview_service.get_overview(
            recent_limit=recent_limit,
            failed_limit=failed_limit,
        )

    def _refresh_runtime_health_service(self) -> None:
        self._runtime_health_service.update_runtime_state(
            storage_backend=self._storage_backend,
            config=self._config,
            read_cache=self._read_cache,
        )

    def enqueue_background_task(
        self,
        task_name: str,
        *,
        force: bool = False,
    ) -> dict:
        return self._task_administration_service.enqueue_background_task(task_name, force=force)

    def enqueue_all_background_tasks(self, *, force: bool = False) -> list[dict]:
        return self._task_administration_service.enqueue_all_background_tasks(force=force)

    def cancel_task(
        self,
        task_id: str,
        *,
        cancelled_by: str = "cli",
        reason: str = "cancelled_by_user",
    ) -> dict:
        return self._task_administration_service.cancel_task(
            task_id,
            cancelled_by=cancelled_by,
            reason=reason,
        )

    def list_ai_conversations(
        self,
        *,
        workspace_id: str | None | object = _USE_SERVICE_WORKSPACE,
        request_id: str | None = None,
        task_name: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> AIConversationListPayload:
        return self._memory_service.list_ai_conversations(
            workspace_id=workspace_id,
            request_id=request_id,
            task_name=task_name,
            status=status,
            limit=limit,
        )

    def get_memory_detail(self, memory_id: str):
        return self._memory_service.get_memory_detail(memory_id)

    def list_mutation_history(
        self,
        *,
        memory_id: str | None = None,
        actor_kind: str | None = None,
        family: str | None = None,
        operation: str | None = None,
        after: float | None = None,
        before: float | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> MutationHistoryListPayload:
        return self._mutation_history_service.list_mutation_history(
            memory_id=memory_id,
            actor_kind=actor_kind,
            family=family,
            operation=operation,
            after=after,
            before=before,
            limit=limit,
            offset=offset,
        )

    def get_mutation_history_event(self, event_id: str) -> MutationHistoryDetailPayload:
        return self._mutation_history_service.get_mutation_history_event(event_id)

    def get_mutation_history_diff(self, event_id: str) -> MutationHistoryDiffPayload:
        return self._mutation_history_service.get_mutation_history_diff(event_id)

    def list_protections(self, memory_id: str) -> ProtectionListPayload:
        return self._mutation_history_service.list_protections(memory_id)

    def set_protection(
        self,
        *,
        memory_id: str,
        mode: str,
        reason: str,
        actor_id: str | None = None,
        expires_at: str | None = None,
    ) -> ProtectionMutationPayload:
        return self._mutation_history_service.set_protection(
            memory_id=memory_id,
            mode=mode,
            reason=reason,
            actor_id=actor_id,
            expires_at=expires_at,
        )

    def remove_protection(self, *, memory_id: str, mode: str) -> ProtectionMutationPayload:
        return self._mutation_history_service.remove_protection(memory_id=memory_id, mode=mode)

    def get_restore_eligibility(self, event_id: str) -> RestoreEligibilityPayload:
        return self._mutation_history_service.get_restore_eligibility(event_id)

    def request_restore(
        self,
        *,
        event_id: str,
        scope: str = "all",
        expected_record_tokens: object = None,
        expected_link_tokens: object = None,
        actor_id: str | None = None,
        reason: str,
        idempotency_key: str,
        confirmation: bool = False,
    ) -> RestoreRequestPayload:
        return self._mutation_history_service.request_restore(
            event_id=event_id,
            scope=scope,
            expected_record_tokens=expected_record_tokens,
            expected_link_tokens=expected_link_tokens,
            actor_id=actor_id,
            reason=reason,
            idempotency_key=idempotency_key,
            confirmation=confirmation,
        )

    def create_memory_link(
        self,
        *,
        source_id: str,
        target_id: str,
        link_type: str,
        context: str = "",
    ) -> dict:
        return self._memory_service.create_memory_link(
            source_id=source_id,
            target_id=target_id,
            link_type=link_type,
            context=context,
        )

    def delete_memory_link(
        self,
        *,
        source_id: str,
        target_id: str,
        link_type: str,
    ) -> dict:
        return self._memory_service.delete_memory_link(
            source_id=source_id,
            target_id=target_id,
            link_type=link_type,
        )

    def list_tasks(
        self,
        status: str | None = None,
        workspace_id: str | None = None,
        limit: int = 20,
    ) -> TaskListPayload:
        return self._task_administration_service.list_tasks(
            status=status,
            workspace_id=workspace_id,
            limit=limit,
        )

    def list_recent_agent_runs(self, *, limit: int = 20, detail_level: str = "compact") -> AgentRunHistoryListPayload:
        return self._task_reporting_service.list_recent_agent_runs(
            limit=limit,
            detail_level=detail_level,
        )

    def get_task_sampling_summary(self, *, limit: int = 50) -> TaskSamplingSummaryPayload:
        return self._task_reporting_service.get_task_sampling_summary(limit=limit)

    def get_selector_stats(
        self,
        *,
        scope: str | None = None,
        workspace_id: str | None = None,
        window_hours: int = 24,
        limit: int = 200,
        now: float | None = None,
    ) -> SelectorStatsPayload:
        return self._analytics_service.get_selector_stats(
            scope=scope,
            workspace_id=workspace_id,
            window_hours=window_hours,
            limit=limit,
            now=now,
        )

    def get_task_detail(self, task_id: str) -> TaskDetailPayload:
        return self._task_administration_service.get_task_detail(task_id)

    def get_nerd_metrics(
        self,
        *,
        scope: str | None = None,
        workspace_id: str | None = None,
        window_hours: int = 24,
        bucket_minutes: int = 60,
        now: float | None = None,
    ) -> NerdMetricsPayload:
        return self._analytics_service.get_nerd_metrics(
            scope=scope,
            workspace_id=workspace_id,
            window_hours=window_hours,
            bucket_minutes=bucket_minutes,
            now=now,
        )

    def list_quality_cleanup_candidates(
        self,
        *,
        scope: str | None = None,
        workspace_id: str | None = None,
        window_hours: int = 24,
        bucket_minutes: int = 60,
        now: float | None = None,
        limit: int = 25,
        low_conversion_min_search_count: int = 3,
        low_conversion_max_conversion_rate: float = 0.25,
    ) -> QualityCleanupCandidatesPayload:
        return self._analytics_service.list_quality_cleanup_candidates(
            scope=scope,
            workspace_id=workspace_id,
            window_hours=window_hours,
            bucket_minutes=bucket_minutes,
            now=now,
            limit=limit,
            low_conversion_min_search_count=low_conversion_min_search_count,
            low_conversion_max_conversion_rate=low_conversion_max_conversion_rate,
        )

    def _resolve_scoped_workspace_id(
        self,
        *,
        scope: str | None,
        workspace_id: str | None,
    ) -> str | None:
        return self._analytics_service._resolve_scoped_workspace_id(
            scope=scope,
            workspace_id=workspace_id,
        )

    def record_thought(
        self,
        content: str,
        *,
        workspace_id: str | None | object = _USE_SERVICE_WORKSPACE,
    ) -> dict[str, object]:
        return self._memory_service.record_thought(content, workspace_id=workspace_id)

    def list_memories(
        self,
        workspace_id: str | None = None,
        memory_type: str | None = None,
        status: str | None = None,
        limit: int = 20,
    ) -> MemoryListPayload:
        return self._memory_service.list_memories(
            workspace_id=workspace_id,
            memory_type=memory_type,
            status=status,
            limit=limit,
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
        return self._memory_service.search_memories(
            query=query,
            workspace_id=workspace_id,
            memory_type=memory_type,
            status=status,
            include_superseded=include_superseded,
            limit=limit,
            debug=debug,
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
        return self._runtime_log_service.list_logs(
            workspace_id=workspace_id,
            level=level,
            logger_name=logger_name,
            source=source,
            query=query,
            after=after,
            before=before,
            limit=limit,
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
        return self._runtime_log_service.summarize_logs(
            workspace_id=workspace_id,
            level=level,
            logger_name=logger_name,
            source=source,
            query=query,
            after=after,
            before=before,
        )

    def prune_logs(
        self,
        *,
        workspace_id: str | None | object = _USE_SERVICE_WORKSPACE,
        max_runtime_logs: int | None = None,
        max_log_age_days: int | None = None,
    ) -> RuntimeLogPrunePayload:
        return self._runtime_log_service.prune_logs(
            workspace_id=workspace_id,
            max_runtime_logs=max_runtime_logs,
            max_log_age_days=max_log_age_days,
        )

    def load_dashboard_html(self):
        return self._runtime_log_service.load_dashboard_html()

    def resolve_dashboard_asset_path(self, asset_path: str) -> Path | None:
        return self._runtime_log_service.resolve_dashboard_asset_path(asset_path)

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
