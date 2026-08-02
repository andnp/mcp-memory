from __future__ import annotations

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
from mcp_memory.management.operator_health_reporting import (
    summarize_memory_tool_latency,
    summarize_provider_policy,
)
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
    _SLOW_MEMORY_TOOL_WARNING_MS,
    _USE_SERVICE_WORKSPACE,
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
from mcp_memory.serialization import (
    task_payload,
)


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
                runtime_log_service=self._runtime_log_service,
                list_ai_conversations=self._memory_service.list_ai_conversations,
                list_memories=self._memory_service.list_memories,
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
                cache_health=self._runtime_health_service._build_cache_health,
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
