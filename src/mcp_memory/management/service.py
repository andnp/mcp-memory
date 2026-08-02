from __future__ import annotations

from dataclasses import asdict
from collections import Counter
from datetime import UTC, datetime, timedelta
import logging
import os
from pathlib import Path
import time
from typing import Any, cast

from mcp_memory.application.memory_embedding_maintenance import MemoryEmbeddingMaintenance
from mcp_memory.context import ManagementContext, ManagementRuntimeCapabilities
from mcp_memory.core import MemoryPipeline
from mcp_memory.management.agent_run_reporting import build_recent_agent_runs
from mcp_memory.management.analytics_reporting import build_nerd_metrics
from mcp_memory.management.context_resources import (
    ManagementContextResources,
    _build_default_embedding_integrity_events as _context_build_default_embedding_integrity_events,
    _build_default_provider_usage as _context_build_default_provider_usage,
    ensure_management_context_resources,
)
from mcp_memory.management.capabilities import ManagementCapabilities
from mcp_memory.management.health_reporting import build_embedding_status, build_execution_attempt_health, build_search_health
from mcp_memory.management.operator_health_reporting import (
    build_operator_health_snapshot_payload,
    summarize_memory_tool_latency,
    summarize_provider_policy,
)
from mcp_memory.management.overview_service import OverviewService, OverviewServiceDependencies
from mcp_memory.management.mutation_history_service import (
    MutationHistoryService,
    MutationHistoryServiceDependencies,
)
from mcp_memory.management.memory_service import (
    MemoryService,
    MemoryServiceDependencies,
    _SLOW_MEMORY_TOOL_WARNING_MS,
    _USE_SERVICE_WORKSPACE,
    _resolve_log_workspace_id,
    _resolve_service_workspace_id,  # noqa: F401 - retained for management helper compatibility
)
from mcp_memory.management.query_runner import PostgresManagementQueryAdapter, SQLiteManagementQueryAdapter
from mcp_memory.management.scope_policy import ScopePolicyKind, resolve_workspace_id_for_policy
from mcp_memory.management.task_reporting_service import (
    TaskReportingService,
    TaskReportingServiceDependencies,
)
from mcp_memory.management.task_administration import (
    TaskAdministrationService,
    TaskAdministrationServiceDependencies,
)
from mcp_memory.management.selector_stats_reporting import build_selector_stats_payload
from mcp_memory.integrations.memory_retrieval import build_memory_retrieval_facade
from mcp_memory.management.models import (
    AgentRunHistoryListPayload,
    AIConversationListPayload,
    CacheHealthPayload,
    CacheMetricsPayload,
    EmbeddingIntegrityEventSnapshotPayload,
    EmbeddingIntegrityEventSummaryPayload,
    HealthPayload,
    MemoryListPayload,
    MemorySearchPayload,
    MemoryToolLatencyPayload,
    MutationHistoryDetailPayload,
    MutationHistoryDiffPayload,
    MutationHistoryListPayload,
    NerdMetricsPayload,
    OperatorHealthSnapshotPayload,
    ProtectionListPayload,
    ProtectionMutationPayload,
    QualityCleanupCandidatePayload,
    QualityCleanupCandidatesPayload,
    QualityCleanupCriterionPayload,
    QualityCleanupRecommendationPayload,
    RuntimeLogListPayload,
    RuntimeLogPrunePayload,
    RuntimeLogPayload,
    RuntimeLogSummaryPayload,
    RestoreEligibilityPayload,
    RestoreRequestPayload,
    SelectorStatsPayload,
    TaskDetailPayload,
    TaskSamplingSummaryPayload,
    TaskListPayload,
    TransportDiagnosticsPayload,
)
from mcp_memory.management.reporting_rows import count_recent_conversation_statuses, count_recent_memory_updates
from mcp_memory.process_termination import send_process_signal as _send_process_signal
from mcp_memory.process_termination import terminate_process as _terminate_process_with_scope
from mcp_memory.process_termination import wait_for_process_exit as _wait_for_process_exit
from mcp_memory.serialization import (
    task_payload,
)
from mcp_memory.storage.shared_mode_cache import resolve_shared_mode_cache_state


_LOW_CONVERSION_DEFAULT_MIN_SEARCH_COUNT = 3
_LOW_CONVERSION_DEFAULT_MAX_RATE = 0.25
_QUALITY_CLEANUP_CRITERIA: dict[str, tuple[str, str, int]] = {
    "trace_like_memory_count": (
        "Trace-like memory",
        "Title matches an operational trace/journal pattern that should usually be curated or archived.",
        60,
    ),
    "generic_summary_count": (
        "Generic summary",
        "Summary starts with a generic 'Covers ...' pattern that hides the concrete takeaway.",
        35,
    ),
    "untagged_observation_count": (
        "Untagged observation",
        "Observation memory has no tags, which makes targeted retrieval and cleanup harder.",
        30,
    ),
    "oversized_memory_count": (
        "Oversized memory",
        "Memory content is at least 4 KB, making it a good cleanup or split candidate.",
        20,
    ),
}
_QUALITY_CLEANUP_RECOMMENDATIONS: dict[str, tuple[str, str, int]] = {
    "trace_like_memory_count": (
        "Archive or curate",
        "This looks trace-like enough that it may belong in archival/cleanup flow rather than as a durable canonical memory.",
        60,
    ),
    "generic_summary_count": (
        "Resummarize",
        "Rewrite the summary to state the concrete takeaway so retrieval value is obvious before opening the record.",
        35,
    ),
    "untagged_observation_count": (
        "Retag",
        "Add concrete subsystem or topic tags so the memory is easier to retrieve and maintain.",
        30,
    ),
    "oversized_memory_count": (
        "Split or trim",
        "Break the memory into narrower focused records or trim excess detail if the size is obscuring the durable point.",
        20,
    ),
    "low_conversion": (
        "Review title and summary",
        "The memory is surfaced often but rarely opened, so its title/summary or overall relevance signal likely needs improvement.",
        50,
    ),
}

logger = logging.getLogger(__name__)

_build_default_provider_usage = _context_build_default_provider_usage
_build_default_embedding_integrity_events = _context_build_default_embedding_integrity_events


def _coerce_transport_diagnostics_payload(snapshot: object) -> TransportDiagnosticsPayload:
    if snapshot is None:
        return TransportDiagnosticsPayload()
    if isinstance(snapshot, TransportDiagnosticsPayload):
        return snapshot
    if isinstance(snapshot, dict):
        return TransportDiagnosticsPayload(**snapshot)
    return TransportDiagnosticsPayload()


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
                cache_health=self._build_cache_health,
                embedding_integrity_summary=lambda: self._embedding_integrity_summary(workspace_id=None),
            )
        )
        self._task_reporting_service = TaskReportingService(
            TaskReportingServiceDependencies(
                db_manager=self._db_manager,
                workspace_id=self._workspace_id,
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
        self._ai_json_provider = provider.ai_json_provider
        self._ai_agent_provider = provider.ai_agent_provider
        self._ai_provider_registry = provider.ai_provider_registry or {}
        self._dashboard_static_root = Path(__file__).with_name("static")
        self._dashboard_static_path = self._dashboard_static_root / "index.html"
        self._dashboard_dist_path = self._dashboard_static_root / "dist" / "index.html"
        self._dashboard_asset_root = self._dashboard_static_root / "dist" / "assets"
        self.capabilities = ManagementCapabilities.from_service(self)

    @property
    def dashboard_static_root(self) -> Path:
        return self._dashboard_static_root

    @property
    def workspace_id(self) -> str | None:
        return self._workspace_id

    def get_health(self):
        embedding_integrity_summary = self._embedding_integrity_summary(workspace_id=None)
        embedder_status = build_embedding_status(
            self._embedder,
            storage_backend=self._storage_backend,
            vector_store=self._vector_store,
            integrity_event_summary=embedding_integrity_summary,
        )
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
        return _coerce_transport_diagnostics_payload(snapshot)

    def _build_cache_health(self) -> CacheHealthPayload:
        cache_state = resolve_shared_mode_cache_state(
            self._config,
            storage_backend=self._storage_backend,
            read_cache=self._read_cache,
        )
        metrics = self._build_cache_metrics_payload()
        if not cache_state.enabled:
            return CacheHealthPayload(enabled=False, mode=None, state="disabled", path=None, metrics=metrics)

        if not cache_state.backend_supported:
            return CacheHealthPayload(enabled=True, mode=cache_state.mode, state="unsupported_backend", path=None, metrics=metrics)

        configured_path = self._configured_cache_path()
        if not cache_state.active:
            return CacheHealthPayload(
                enabled=True,
                mode=cache_state.mode,
                state="inactive",
                path=str(configured_path) if configured_path is not None else None,
                metrics=metrics,
            )

        active_path = getattr(cache_state.read_cache, "_db_path", None)
        resolved_path = active_path if isinstance(active_path, Path) else configured_path
        return CacheHealthPayload(
            enabled=True,
            mode=cache_state.mode,
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
        return self._overview_service.get_overview(
            recent_limit=recent_limit,
            failed_limit=failed_limit,
        )

    def _embedding_integrity_summary(self, *, workspace_id: str | None) -> EmbeddingIntegrityEventSummaryPayload:
        repository = self._embedding_integrity_events
        if repository is None:
            return EmbeddingIntegrityEventSummaryPayload()
        summary = repository.summarize_events(workspace_id=workspace_id)
        return EmbeddingIntegrityEventSummaryPayload(
            total=summary.total,
            by_kind=summary.by_kind,
            last_scan=_embedding_integrity_snapshot_payload(summary.last_scan),
            last_blocked_fallback_write=_embedding_integrity_snapshot_payload(summary.last_blocked_fallback_write),
        )

    def repair_search_index(self) -> dict[str, int | bool | str | None]:
        if self._embedding_maintenance is None:
            raise ValueError("search_not_initialized")
        return self._embedding_maintenance.rebuild_semantic_index()

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
        effective_workspace_id = self._resolve_scoped_workspace_id(
            scope=scope,
            workspace_id=workspace_id,
        )
        query_adapter = (
            PostgresManagementQueryAdapter()
            if self._storage_backend == "postgres"
            else SQLiteManagementQueryAdapter()
        )
        return build_nerd_metrics(
            db_manager=self._db_manager,
            query_adapter=query_adapter,
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

    def list_quality_cleanup_candidates(
        self,
        *,
        scope: str | None = None,
        workspace_id: str | None = None,
        window_hours: int = 24,
        bucket_minutes: int = 60,
        now: float | None = None,
        limit: int = 25,
        low_conversion_min_search_count: int = _LOW_CONVERSION_DEFAULT_MIN_SEARCH_COUNT,
        low_conversion_max_conversion_rate: float = _LOW_CONVERSION_DEFAULT_MAX_RATE,
    ) -> QualityCleanupCandidatesPayload:
        nerd_metrics = self.get_nerd_metrics(
            scope=scope,
            workspace_id=workspace_id,
            window_hours=window_hours,
            bucket_minutes=bucket_minutes,
            now=now,
        )
        generated_at = nerd_metrics.generated_at
        candidates_by_id: dict[str, QualityCleanupCandidatePayload] = {}

        for signal in nerd_metrics.quality_drilldown.signals:
            criteria_config = _QUALITY_CLEANUP_CRITERIA.get(signal.key)
            if criteria_config is None or signal.count <= 0:
                continue
            label, rationale, weight = criteria_config
            for record in signal.records:
                candidate = candidates_by_id.setdefault(
                    record.memory_id,
                    QualityCleanupCandidatePayload(
                        memory_id=record.memory_id,
                        title=record.title,
                        summary=record.summary,
                        memory_type=record.memory_type,
                        status=record.status,
                        updated_at=record.updated_at,
                        tags=list(record.tags),
                    ),
                )
                if candidate.updated_at is None:
                    candidate.updated_at = record.updated_at
                if candidate.summary is None:
                    candidate.summary = record.summary
                if not candidate.tags:
                    candidate.tags = list(record.tags)
                _append_quality_cleanup_criterion(
                    candidate,
                    QualityCleanupCriterionPayload(
                        key=signal.key,
                        label=label,
                        rationale=rationale,
                        weight=weight,
                    ),
                )
                _append_quality_cleanup_recommendation(candidate, signal.key)

        for record in nerd_metrics.retrieval.low_conversion_memories:
            if record.search_count < low_conversion_min_search_count:
                continue
            if record.conversion_rate > low_conversion_max_conversion_rate:
                continue
            candidate = candidates_by_id.setdefault(
                record.memory_id,
                QualityCleanupCandidatePayload(
                    memory_id=record.memory_id,
                    title=record.title,
                    memory_type=record.memory_type,
                    status=record.status,
                    tags=list(record.tags),
                ),
            )
            if not candidate.tags:
                candidate.tags = list(record.tags)
            candidate.search_count = record.search_count
            candidate.read_count = record.read_count
            candidate.converted_search_count = record.converted_search_count
            candidate.conversion_rate = record.conversion_rate
            candidate.last_search_at = record.last_search_at
            candidate.last_read_at = record.last_read_at
            _append_quality_cleanup_criterion(
                candidate,
                QualityCleanupCriterionPayload(
                    key="low_conversion",
                    label="Low conversion",
                    rationale=(
                        f"Surfaced at least {low_conversion_min_search_count} times but converts at or below "
                        f"{low_conversion_max_conversion_rate:.2f}."
                    ),
                    weight=50,
                    search_count=record.search_count,
                    read_count=record.read_count,
                    converted_search_count=record.converted_search_count,
                    conversion_rate=record.conversion_rate,
                ),
            )
            _append_quality_cleanup_recommendation(candidate, "low_conversion")

        all_candidates = sorted(
            candidates_by_id.values(),
            key=lambda candidate: (
                -candidate.priority_score,
                -(candidate.search_count or 0),
                candidate.title.lower(),
                candidate.memory_id,
            ),
        )
        candidates = all_candidates[: max(limit, 0)]
        return QualityCleanupCandidatesPayload(
            generated_at=generated_at,
            window_hours=window_hours,
            bucket_minutes=bucket_minutes,
            low_conversion_min_search_count=low_conversion_min_search_count,
            low_conversion_max_conversion_rate=low_conversion_max_conversion_rate,
            total_candidates=len(all_candidates),
            candidates=candidates,
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

    def _summarize_memory_tool_latency(self, *, window_minutes: int) -> MemoryToolLatencyPayload:
        return summarize_memory_tool_latency(
            self._db_manager,
            workspace_id=self._workspace_id,
            window_minutes=window_minutes,
            slow_threshold_ms=_SLOW_MEMORY_TOOL_WARNING_MS,
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
        return _ensure_dashboard_base_href(html_path.read_text(encoding="utf-8"))

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


def _ensure_dashboard_base_href(html: str) -> str:
    if "<base " in html:
        return html
    if "</head>" not in html:
        return html
    return html.replace("</head>", '    <base href="/dashboard/">\n  </head>', 1)


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


def _embedding_integrity_snapshot_payload(record) -> EmbeddingIntegrityEventSnapshotPayload | None:
    if record is None:
        return None
    return EmbeddingIntegrityEventSnapshotPayload(
        created_at=record.created_at,
        model_name=record.model_name,
        source_kind=record.source_kind,
        source_id=record.source_id,
        scanned_row_count=record.scanned_row_count,
        invalid_row_count=record.invalid_row_count,
        mixed_dimension_group_count=record.mixed_dimension_group_count,
    )


def _append_quality_cleanup_criterion(
    candidate: QualityCleanupCandidatePayload,
    criterion: QualityCleanupCriterionPayload,
) -> None:
    if any(existing.key == criterion.key for existing in candidate.criteria):
        return
    candidate.criteria.append(criterion)
    candidate.criteria.sort(key=lambda item: (-item.weight, item.key))
    candidate.priority_score = sum(item.weight for item in candidate.criteria)


def _append_quality_cleanup_recommendation(
    candidate: QualityCleanupCandidatePayload,
    criterion_key: str,
) -> None:
    recommendation = _QUALITY_CLEANUP_RECOMMENDATIONS.get(criterion_key)
    if recommendation is None:
        return
    label, rationale, weight = recommendation
    if any(existing.key == criterion_key for existing in candidate.recommendations):
        return
    candidate.recommendations.append(
        QualityCleanupRecommendationPayload(
            key=criterion_key,
            label=label,
            rationale=rationale,
            weight=weight,
        )
    )
    candidate.recommendations.sort(key=lambda item: (-item.weight, item.key))
