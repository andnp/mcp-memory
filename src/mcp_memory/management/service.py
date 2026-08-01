from __future__ import annotations

from dataclasses import asdict, dataclass
from collections import Counter
from datetime import UTC, datetime, timedelta
import hashlib
import json
import logging
import os
from pathlib import Path
import time
from time import perf_counter
from typing import Any, cast
from uuid import UUID, uuid4

from mcp_memory.application.memory_embedding_maintenance import MemoryEmbeddingMaintenance
from mcp_memory.context import ManagementContext, ManagementRuntimeCapabilities
from mcp_memory.core import MemoryPipeline
from mcp_memory.core.curation_identity import link_token, record_token
from mcp_memory.core.journal_operations import RecordThoughtOperation
from mcp_memory.core.mutation_restore import (
    InverseDescription,
    RestoreConflict,
    RestoreConflictCode,
    RestoreExecutor,
    build_inverse,
)
from mcp_memory.core.task_handlers import (
    CONFLICT_DETECTOR_TASK_NAME,
    CURATOR_TASK_NAME,
    DEDUPLICATOR_TASK_NAME,
    DEFRAGMENTER_TASK_NAME,
    FACT_CHECKER_TASK_NAME,
    GRAPH_LINKER_TASK_NAME,
    PROJECT_MANAGER_TASK_NAME,
    TAXONOMIST_TASK_NAME,
    TRIGGERABLE_BACKGROUND_TASK_NAMES,
    task_priority,
)
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
from mcp_memory.management.overview_reporting import build_overview
from mcp_memory.management.scope_policy import ScopePolicyKind, resolve_workspace_id_for_policy
from mcp_memory.mutation_history import (
    LinkRevision,
    MutationEvent,
    Protection,
    ProtectionMode,
    RecordRevision,
    RestoreRequest,
    RestoreResult,
    RestoreResultStatus,
    RestoreScope,
)
from mcp_memory.management.selector_stats_reporting import build_selector_stats_payload
from mcp_memory.management.task_sampling_summary import build_task_sampling_summary
from mcp_memory.integrations.memory_retrieval import build_memory_retrieval_facade
from mcp_memory.relational.operations import SearchMemoryRecordsOperation
from mcp_memory.management.models import (
    AgentRunHistoryListPayload,
    AIConversationListPayload,
    AIConversationPayload,
    CacheHealthPayload,
    CacheMetricsPayload,
    EmbeddingIntegrityEventSnapshotPayload,
    EmbeddingIntegrityEventSummaryPayload,
    HealthPayload,
    MemoryListPayload,
    MemoryDetailPayload,
    MemorySearchResultPayload,
    MemorySearchPayload,
    MemoryToolLatencyPayload,
    MutationHistoryDetailPayload,
    MutationHistoryDiffPayload,
    MutationHistoryListPayload,
    ProtectionListPayload,
    ProtectionMutationPayload,
    RestoreEligibilityPayload,
    RestoreRequestPayload,
    NerdMetricsPayload,
    OperatorHealthSnapshotPayload,
    QualityCleanupCandidatePayload,
    QualityCleanupCandidatesPayload,
    QualityCleanupCriterionPayload,
    QualityCleanupRecommendationPayload,
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
from mcp_memory.management.reporting_rows import count_recent_conversation_statuses, count_recent_memory_updates
from mcp_memory.process_termination import send_process_signal as _send_process_signal
from mcp_memory.process_termination import terminate_process as _terminate_process_with_scope
from mcp_memory.process_termination import wait_for_process_exit as _wait_for_process_exit
from mcp_memory.runtime_log_store import _AllWorkspacesSentinel
from mcp_memory.serialization import (
    compact_memory_record_payload,
    link_payload,
    memory_record_payload,
    search_result_payload,
    task_payload,
)
from mcp_memory.storage.shared_mode_cache import resolve_shared_mode_cache_state


_USE_SERVICE_WORKSPACE = object()
_SLOW_MEMORY_TOOL_WARNING_MS = 2_000.0
_LOW_CONVERSION_DEFAULT_MIN_SEARCH_COUNT = 3
_LOW_CONVERSION_DEFAULT_MAX_RATE = 0.25
_CURATOR_CAMPAIGN_ALIAS_TASK_NAMES = frozenset({
    CONFLICT_DETECTOR_TASK_NAME,
    DEDUPLICATOR_TASK_NAME,
    DEFRAGMENTER_TASK_NAME,
    FACT_CHECKER_TASK_NAME,
    GRAPH_LINKER_TASK_NAME,
    PROJECT_MANAGER_TASK_NAME,
    TAXONOMIST_TASK_NAME,
})
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

_MAX_HISTORY_LIST_LIMIT = 100
_MAX_HISTORY_OFFSET = 10_000
_MAX_HISTORY_REVISION_ROWS = 100
_MAX_HISTORY_VALUE_CHARS = 16_000


logger = logging.getLogger(__name__)

_build_default_provider_usage = _context_build_default_provider_usage
_build_default_embedding_integrity_events = _context_build_default_embedding_integrity_events


def _parse_uuid(value: str, *, field: str) -> UUID:
    try:
        return UUID(value)
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"{field}_invalid") from exc


def _history_time(value: float | None) -> datetime | None:
    return None if value is None else datetime.fromtimestamp(value, tz=UTC)


def _bounded_history_value(value: object) -> object:
    if value is None:
        return None
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    if len(encoded) <= _MAX_HISTORY_VALUE_CHARS:
        return value
    return {
        "truncated": True,
        "sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        "original_characters": len(encoded),
    }


@dataclass(frozen=True)
class _RestoreAnalysis:
    event: MutationEvent
    inverse: InverseDescription | None
    conflict: RestoreConflict | None
    current_record_tokens: dict[str, str]
    current_link_tokens: dict[str, str]
    protections: dict[str, list[Protection]]
    conflict_code: str | None = None
    conflict_reason: str | None = None


def _restore_risk(operation: str) -> str:
    if operation == "normalize_memory":
        return "low"
    if operation == "create_link":
        return "moderate"
    return "unsupported"


def _required_token(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field}_must_contain_tokens")
    return value


def _parse_record_tokens(value: object) -> dict[UUID, str]:
    if not isinstance(value, dict):
        raise ValueError("expected_record_tokens_must_be_object")
    return {
        _parse_uuid(str(memory_id), field="memory_id"): _required_token(token, "expected_record_tokens")
        for memory_id, token in value.items()
    }


def _parse_link_tokens(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError("expected_link_tokens_must_be_object")
    return {str(key): _required_token(token, "expected_link_tokens") for key, token in value.items()}


def _restore_conflict(target_event_id: UUID, code: str, reason: str) -> RestoreResult:
    return RestoreResult(
        status=RestoreResultStatus.CONFLICT,
        target_event_id=target_event_id,
        conflict_reason=reason,
        conflict_details={"code": code},
    )


def _restore_rejected(target_event_id: UUID, code: str, reason: str) -> RestoreResult:
    return RestoreResult(
        status=RestoreResultStatus.REJECTED,
        target_event_id=target_event_id,
        conflict_reason=reason,
        conflict_details={"code": code},
    )


def _restore_payload(result: RestoreResult) -> RestoreRequestPayload:
    return RestoreRequestPayload(
        status=result.status.value,
        target_event_id=str(result.target_event_id),
        request_id=None if result.request_id is None else str(result.request_id),
        event_id=None if result.event_id is None else str(result.event_id),
        conflict_reason=result.conflict_reason,
        conflict_details=result.conflict_details,
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


def _coerce_transport_diagnostics_payload(snapshot: object) -> TransportDiagnosticsPayload:
    if snapshot is None:
        return TransportDiagnosticsPayload()
    if isinstance(snapshot, TransportDiagnosticsPayload):
        return snapshot
    if isinstance(snapshot, dict):
        return TransportDiagnosticsPayload(**snapshot)
    return TransportDiagnosticsPayload()


def _resolve_manual_maintenance_task_name(task_name: str) -> tuple[str, str | None]:
    if task_name in _CURATOR_CAMPAIGN_ALIAS_TASK_NAMES:
        return CURATOR_TASK_NAME, task_name
    return task_name, None


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
        self._retrieval = (
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
        self._config = memory.config
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
        return build_overview(
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
            cache=self._build_cache_health(),
            embedding_integrity_summary=self._embedding_integrity_summary(workspace_id=None),
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
        canonical_task_name, redirected_from_task_name = _resolve_manual_maintenance_task_name(task_name)
        if canonical_task_name not in TRIGGERABLE_BACKGROUND_TASK_NAMES:
            raise ValueError(f"unknown_background_task:{task_name}")

        payload = {"workspace_id": None}
        if force:
            task = self._task_queue.enqueue(
                task_name=canonical_task_name,
                workspace_id=None,
                data=payload,
                priority=task_priority(canonical_task_name),
            )
            result = {"status": "enqueued", "created": True, "task": task_payload(task)}
        else:
            task, created = self._task_queue.enqueue_unique(
                task_name=canonical_task_name,
                workspace_id=None,
                data=payload,
                priority=task_priority(canonical_task_name),
            )
            result = {
                "status": "enqueued" if created else "already_pending",
                "created": created,
                "task": task_payload(task),
            }

        if redirected_from_task_name is not None:
            result["redirected_from_task_name"] = redirected_from_task_name
        return result

    def enqueue_all_background_tasks(self, *, force: bool = False) -> list[dict]:
        grouped_task_names: dict[str, list[str]] = {}
        for task_name in TRIGGERABLE_BACKGROUND_TASK_NAMES:
            canonical_task_name, redirected_from_task_name = _resolve_manual_maintenance_task_name(task_name)
            grouped_task_names.setdefault(canonical_task_name, [])
            if redirected_from_task_name is not None:
                grouped_task_names[canonical_task_name].append(redirected_from_task_name)

        results: list[dict] = []
        for canonical_task_name, redirected_from_task_names in grouped_task_names.items():
            result = self.enqueue_background_task(canonical_task_name, force=force)
            if redirected_from_task_names:
                result["redirected_from_task_names"] = redirected_from_task_names
            results.append(result)
        return results

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
        store = self._require_mutation_history()
        bounded_limit = min(limit, _MAX_HISTORY_LIST_LIMIT)
        if bounded_limit < 1:
            raise ValueError("limit_out_of_range")
        if offset < 0 or offset > _MAX_HISTORY_OFFSET:
            raise ValueError("offset_out_of_range")
        parsed_memory_id = _parse_uuid(memory_id, field="memory_id") if memory_id is not None else None
        events = store.list_events(
            memory_id=parsed_memory_id,
            actor_kind=actor_kind,
            family=family,
            operation=operation,
            created_after=_history_time(after),
            created_before=_history_time(before),
            limit=bounded_limit + 1,
            offset=offset,
        )
        has_more = len(events) > bounded_limit
        visible_events = events[:bounded_limit]
        return MutationHistoryListPayload(
            events=[self._history_event_payload(event) for event in visible_events],
            limit=bounded_limit,
            offset=offset,
            has_more=has_more,
            next_offset=offset + bounded_limit if has_more else None,
        )

    def get_mutation_history_event(self, event_id: str) -> MutationHistoryDetailPayload:
        event, records, links = self._get_history_parts(event_id)
        return MutationHistoryDetailPayload(
            event=self._history_event_payload(event),
            receipt=self._receipt_payload(event),
            curation_run=self._curation_run_payload(event),
            records=[self._record_revision_payload(revision) for revision in records],
            links=[self._link_revision_payload(revision) for revision in links],
            truncated=len(records) >= _MAX_HISTORY_REVISION_ROWS or len(links) >= _MAX_HISTORY_REVISION_ROWS,
        )

    def get_mutation_history_diff(self, event_id: str) -> MutationHistoryDiffPayload:
        event, records, links = self._get_history_parts(event_id)
        return MutationHistoryDiffPayload(
            event_id=str(event.id),
            records=[self._record_diff_payload(revision) for revision in records],
            links=[self._link_diff_payload(revision) for revision in links],
            truncated=len(records) >= _MAX_HISTORY_REVISION_ROWS or len(links) >= _MAX_HISTORY_REVISION_ROWS,
        )

    def list_protections(self, memory_id: str) -> ProtectionListPayload:
        parsed_memory_id = _parse_uuid(memory_id, field="memory_id")
        return ProtectionListPayload(
            memory_id=str(parsed_memory_id),
            protections=[
                protection.model_dump(mode="json")
                for protection in self._require_mutation_history().get_protections(parsed_memory_id)
            ],
        )

    def set_protection(
        self,
        *,
        memory_id: str,
        mode: str,
        reason: str,
        actor_id: str | None = None,
        expires_at: str | None = None,
    ) -> ProtectionMutationPayload:
        parsed_memory_id = _parse_uuid(memory_id, field="memory_id")
        try:
            protection = Protection(
                memory_id=parsed_memory_id,
                mode=ProtectionMode(mode),
                reason=reason,
                actor_id=actor_id,
                expires_at=None if expires_at is None else datetime.fromisoformat(expires_at),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("protection_invalid") from exc
        stored = self._require_mutation_history().set_protection(protection)
        return ProtectionMutationPayload(
            status="applied",
            memory_id=str(parsed_memory_id),
            mode=str(stored.mode),
            protection=stored.model_dump(mode="json"),
        )

    def remove_protection(self, *, memory_id: str, mode: str) -> ProtectionMutationPayload:
        parsed_memory_id = _parse_uuid(memory_id, field="memory_id")
        try:
            parsed_mode = ProtectionMode(mode)
        except ValueError as exc:
            raise ValueError("mode_invalid") from exc
        self._require_mutation_history().remove_protection(parsed_memory_id, parsed_mode)
        return ProtectionMutationPayload(
            status="removed",
            memory_id=str(parsed_memory_id),
            mode=str(parsed_mode),
        )

    def get_restore_eligibility(self, event_id: str) -> RestoreEligibilityPayload:
        analysis = self._analyze_restore(event_id)
        requires_confirmation = self._restore_requires_confirmation(analysis)
        conflict_code = analysis.conflict_code
        conflict_reason = analysis.conflict_reason
        if analysis.conflict is not None:
            conflict_code = analysis.conflict.code.value
            conflict_reason = analysis.conflict.reason
        if conflict_code is None:
            blocked = self._restore_protection_conflict(analysis)
            if blocked is not None:
                conflict_code, conflict_reason = blocked
            elif requires_confirmation:
                conflict_code = "confirmation_required"
                conflict_reason = "explicit confirmation is required by the current policy"
        return RestoreEligibilityPayload(
            event_id=str(analysis.event.id),
            eligible=conflict_code is None,
            operation=analysis.event.operation,
            inverse_operation=None if analysis.inverse is None else analysis.inverse.inverse_operation,
            risk=_restore_risk(analysis.event.operation),
            requires_confirmation=requires_confirmation,
            current_record_tokens=analysis.current_record_tokens,
            current_link_tokens=analysis.current_link_tokens,
            protections={
                memory_id: [protection.model_dump(mode="json") for protection in protections]
                for memory_id, protections in analysis.protections.items()
            },
            conflict_code=conflict_code,
            conflict_reason=conflict_reason,
        )

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
        parsed_event_id = _parse_uuid(event_id, field="event_id")
        try:
            parsed_scope = RestoreScope(scope)
        except ValueError as exc:
            raise ValueError("scope_invalid") from exc
        if not idempotency_key.strip():
            raise ValueError("idempotency_key_required")
        if not reason.strip():
            raise ValueError("reason_required")
        request = RestoreRequest(
            target_event_id=parsed_event_id,
            scope=parsed_scope,
            expected_record_tokens=_parse_record_tokens(expected_record_tokens or {}),
            expected_link_tokens=_parse_link_tokens(expected_link_tokens or {}),
            actor_id=actor_id,
            reason=reason,
            idempotency_key=idempotency_key,
            confirmation=confirmation,
        )
        analysis = self._analyze_restore(event_id)
        result: RestoreResult | None = None
        if analysis.conflict_code is not None:
            result = _restore_conflict(parsed_event_id, analysis.conflict_code, analysis.conflict_reason or "restore is not eligible")
        elif analysis.conflict is not None:
            result = _restore_conflict(parsed_event_id, analysis.conflict.code.value, analysis.conflict.reason)
        else:
            blocked = self._restore_protection_conflict(analysis)
            if blocked is not None:
                result = _restore_rejected(parsed_event_id, blocked[0], blocked[1])
            elif self._restore_requires_confirmation(analysis) and not confirmation:
                result = _restore_rejected(
                    parsed_event_id,
                    "confirmation_required",
                    "explicit confirmation is required by the current policy",
                )
            else:
                result = self._restore_token_conflict(request, analysis)
        history = self._require_mutation_history()
        if result is not None:
            registered = history.request_restore(request)
            if registered.status is RestoreResultStatus.ALREADY_APPLIED or registered.event_id is not None:
                return _restore_payload(registered)
            if registered.request_id is None:
                raise RuntimeError("restore request store returned no request id")
            finalized = history.terminalize_restore_request(registered.request_id, result)
            return _restore_payload(finalized)
        if self._action_store is None:
            raise ValueError("curation_action_store_unavailable")
        executed = RestoreExecutor(
            self._action_store,
            history,
            curation_store=self._curation,
        ).execute(request)
        return _restore_payload(executed)

    def _analyze_restore(self, event_id: str) -> _RestoreAnalysis:
        event, records, links = self._get_history_parts(event_id)
        inverse = build_inverse(event, records, links)
        protections: dict[str, list[Protection]] = {}
        current_record_tokens: dict[str, str] = {}
        current_link_tokens: dict[str, str] = {}
        if isinstance(inverse, RestoreConflict):
            return _RestoreAnalysis(event, None, inverse, current_record_tokens, current_link_tokens, protections)
        for change in inverse.record_changes:
            memory_id = str(change.memory_id)
            protections[memory_id] = self._require_mutation_history().get_protections(change.memory_id)
            record = self._memory_queries.get_memory(memory_id) if self._memory_queries is not None else None
            if record is None:
                return _RestoreAnalysis(
                    event, inverse, None, current_record_tokens, current_link_tokens, protections,
                    RestoreConflictCode.STALE_STATE.value, f"target memory {memory_id} is missing",
                )
            current_record_tokens[memory_id] = record_token(record)
            if change.expected_current_token != current_record_tokens[memory_id]:
                return _RestoreAnalysis(
                    event, inverse, None, current_record_tokens, current_link_tokens, protections,
                    RestoreConflictCode.STALE_STATE.value, f"current record token is stale for {memory_id}",
                )
        for change in inverse.link_changes:
            source_id = str(change.source_id)
            target_id = str(change.target_id)
            key = f"{source_id}:{target_id}:{change.link_type}"
            for memory_id, parsed_memory_id in ((source_id, change.source_id), (target_id, change.target_id)):
                protections.setdefault(memory_id, self._require_mutation_history().get_protections(parsed_memory_id))
                record = self._memory_queries.get_memory(memory_id) if self._memory_queries is not None else None
                if record is None:
                    return _RestoreAnalysis(
                        event, inverse, None, current_record_tokens, current_link_tokens, protections,
                        RestoreConflictCode.STALE_STATE.value, f"target memory {memory_id} is missing",
                    )
                current_record_tokens[memory_id] = record_token(record)
            matching = []
            if self._memory_queries is not None:
                matching = [
                    link for link in self._memory_queries.get_links(source_id, direction="outgoing")
                    if str(link.target_id) == target_id and link.link_type == change.link_type
                ]
            current = matching[0] if matching else None
            current_link_tokens[key] = link_token(
                source_id,
                target_id,
                change.link_type,
                None if current is None else current.context,
                exists=current is not None,
            )
            if change.exists != (current is not None) or (current is not None and current.context != change.context):
                return _RestoreAnalysis(
                    event, inverse, None, current_record_tokens, current_link_tokens, protections,
                    RestoreConflictCode.STALE_STATE.value, f"current link state is stale for {key}",
                )
        return _RestoreAnalysis(event, inverse, None, current_record_tokens, current_link_tokens, protections)

    @staticmethod
    def _restore_requires_confirmation(analysis: _RestoreAnalysis) -> bool:
        return any(
            ProtectionMode.MANUAL_REVIEW_REQUIRED in {protection.mode for protection in protections}
            for protections in analysis.protections.values()
        )

    @staticmethod
    def _restore_protection_conflict(analysis: _RestoreAnalysis) -> tuple[str, str] | None:
        if any(
            ProtectionMode.NO_AUTONOMOUS_MUTATION in {protection.mode for protection in protections}
            for protections in analysis.protections.values()
        ):
            return "protection_denied", "current protection does not allow autonomous restore"
        return None

    @staticmethod
    def _restore_token_conflict(request: RestoreRequest, analysis: _RestoreAnalysis) -> RestoreResult | None:
        for memory_id, token in analysis.current_record_tokens.items():
            if request.expected_record_tokens.get(UUID(memory_id)) != token:
                return _restore_conflict(request.target_event_id, "stale_state", f"current record token is stale for {memory_id}")
        for key, token in analysis.current_link_tokens.items():
            if request.expected_link_tokens.get(key) != token:
                return _restore_conflict(request.target_event_id, "stale_state", f"current link token is stale for {key}")
        return None

    def _require_mutation_history(self):
        if self._mutation_history is None:
            raise ValueError("mutation_history_unavailable")
        return self._mutation_history

    def _get_history_parts(self, event_id: str) -> tuple[MutationEvent, list[RecordRevision], list[LinkRevision]]:
        store = self._require_mutation_history()
        parsed_event_id = _parse_uuid(event_id, field="event_id")
        event = store.get_event(parsed_event_id)
        if event is None:
            raise ValueError("mutation_event_not_found")
        return (
            event,
            store.get_record_revisions(parsed_event_id, limit=_MAX_HISTORY_REVISION_ROWS),
            store.get_link_revisions(parsed_event_id, limit=_MAX_HISTORY_REVISION_ROWS),
        )

    def _history_event_payload(self, event: MutationEvent) -> dict[str, object]:
        payload = event.model_dump(mode="json")
        # Provider rationale is not an audit fact. Persisted revisions and
        # tokens remain the only source for before/after history.
        payload.pop("rationale", None)
        payload["receipt"] = self._receipt_payload(event)
        return payload

    def _receipt_payload(self, event: MutationEvent) -> dict[str, object] | None:
        if self._curation is None or event.curation_run_id is None or event.action_id is None:
            return None
        receipt = self._curation.get_receipt(event.curation_run_id, event.action_id)
        return None if receipt is None else receipt.model_dump(mode="json")

    def _curation_run_payload(self, event: MutationEvent) -> dict[str, object] | None:
        if self._curation is None or event.curation_run_id is None:
            return None
        run = self._curation.get_run(event.curation_run_id)
        return None if run is None else run.model_dump(mode="json")

    def _record_revision_payload(self, revision: RecordRevision) -> dict[str, object]:
        return {
            "memory_id": str(revision.memory_id),
            "role": str(revision.role),
            "before_exists": revision.before_exists,
            "after_exists": revision.after_exists,
            "before_snapshot": _bounded_history_value(revision.before_snapshot),
            "after_snapshot": _bounded_history_value(revision.after_snapshot),
            "before_token": revision.before_token,
            "after_token": revision.after_token,
        }

    def _link_revision_payload(self, revision: LinkRevision) -> dict[str, object]:
        return {
            "source_id": str(revision.source_id),
            "target_id": str(revision.target_id),
            "link_type": revision.link_type,
            "context": _bounded_history_value(revision.context),
            "before_exists": revision.before_exists,
            "after_exists": revision.after_exists,
        }

    def _record_diff_payload(self, revision: RecordRevision) -> dict[str, object]:
        return {
            "memory_id": str(revision.memory_id),
            "role": str(revision.role),
            "before": _bounded_history_value(revision.before_snapshot),
            "after": _bounded_history_value(revision.after_snapshot),
            "before_exists": revision.before_exists,
            "after_exists": revision.after_exists,
            "before_token": revision.before_token,
            "after_token": revision.after_token,
        }

    def _link_diff_payload(self, revision: LinkRevision) -> dict[str, object]:
        return {
            "source_id": str(revision.source_id),
            "target_id": str(revision.target_id),
            "link_type": revision.link_type,
            "before_context": _bounded_history_value(revision.context) if revision.before_exists else None,
            "after_context": _bounded_history_value(revision.context) if revision.after_exists else None,
            "before_exists": revision.before_exists,
            "after_exists": revision.after_exists,
        }

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
        if self._journal.journal is None:
            raise ValueError("journal_not_initialized")
        effective_workspace_id = _resolve_service_workspace_id(self._workspace_id, workspace_id)
        suppression_config = None if self._config is None else self._config.ingest_suppression
        cache_state = resolve_shared_mode_cache_state(
            self._config,
            storage_backend=self._storage_backend,
            read_cache=self._read_cache,
        )
        return RecordThoughtOperation(
            self._journal.journal,
            self._task_queue.task_queue,
            effective_workspace_id,
            suppression_config,
            writeback_cache=cache_state.writeback_cache,
            max_outbox_entries=cache_state.max_outbox_entries,
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
        if self._retrieval is None:
            return MemorySearchPayload()
        started_at = perf_counter()
        results = SearchMemoryRecordsOperation(self._retrieval).execute(
            query=query,
            # Management search is global; workspace context must not filter results.
            workspace_id=None,
            limit=limit,
            adaptive_limit=False,
            ranking_workspace_id=workspace_id,
            memory_type=memory_type,
            status=status,
            include_superseded=include_superseded,
            debug=debug,
        )
        surfaced_memory_ids = [result.memory_id for result in results]
        touch_last_surfaced = getattr(self._repository, "touch_last_surfaced", None)
        if surfaced_memory_ids and callable(touch_last_surfaced):
            touch_last_surfaced(
                surfaced_memory_ids,
                datetime.now(UTC).isoformat(),
                best_effort=True,
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
