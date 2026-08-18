from __future__ import annotations

import logging
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

from mcp_memory.application.ports import MemorySearchPort
from mcp_memory.core.ports import EmbeddingMaintenancePort, SearchHealthPort
from mcp_memory.management.agent_run_reporting import build_recent_agent_runs
from mcp_memory.management.health_reporting import (
    build_embedding_status,
    build_execution_attempt_health,
    build_search_health,
)
from mcp_memory.management.memory_service import _SLOW_MEMORY_TOOL_WARNING_MS
from mcp_memory.management.models import (
    CacheHealthPayload,
    CacheMetricsPayload,
    EmbeddingIntegrityEventSnapshotPayload,
    EmbeddingIntegrityEventSummaryPayload,
    HealthPayload,
    OperatorHealthSnapshotPayload,
    TransportDiagnosticsPayload,
)
from mcp_memory.management.operator_health_reporting import (
    build_operator_health_snapshot_payload,
    summarize_memory_tool_latency,
    summarize_provider_policy,
)
from mcp_memory.management.reporting_rows import count_recent_conversation_statuses, count_recent_memory_updates
from mcp_memory.management.runtime_log_service import RuntimeLogService
from mcp_memory.mcp.telemetry import search_diagnostics_snapshot
from mcp_memory.storage.shared_mode_cache import resolve_shared_mode_cache_state

logger = logging.getLogger(__name__)


class _RuntimeController(Protocol):
    has_runtime: bool
    client_count: int


@dataclass
class RuntimeHealthServiceDependencies:
    runtime_info: Any
    controller: _RuntimeController
    db_manager: Any
    storage_backend: str | None
    config: Any
    read_cache: Any
    embedder: Any
    vector_store: Any
    relational_search: MemorySearchPort | None
    embedding_integrity_events: Any
    embedding_maintenance: EmbeddingMaintenancePort | None
    repository: Any
    provider_usage: Any
    retrieval_telemetry: Any
    runtime_log_service: RuntimeLogService
    list_ai_conversations: Callable[..., Any]
    list_memories: Callable[..., Any]
    search_health: SearchHealthPort | None = None


def _coerce_transport_diagnostics_payload(snapshot: object) -> TransportDiagnosticsPayload:
    if snapshot is None:
        return TransportDiagnosticsPayload()
    if isinstance(snapshot, TransportDiagnosticsPayload):
        return snapshot
    if isinstance(snapshot, dict):
        return TransportDiagnosticsPayload(**snapshot)
    return TransportDiagnosticsPayload()


class RuntimeHealthService:
    def __init__(self, dependencies: RuntimeHealthServiceDependencies) -> None:
        self._dependencies = dependencies

    def update_runtime_state(self, *, storage_backend: str | None, config: Any, read_cache: Any) -> None:
        self._dependencies.storage_backend = storage_backend
        self._dependencies.config = config
        self._dependencies.read_cache = read_cache

    def get_health(self) -> HealthPayload:
        dependencies = self._dependencies
        embedding_integrity_summary = self._embedding_integrity_summary(workspace_id=None)
        embedder_status = build_embedding_status(
            dependencies.embedder,
            storage_backend=dependencies.storage_backend,
            vector_store=dependencies.vector_store,
            integrity_event_summary=embedding_integrity_summary,
        )
        return HealthPayload(
            status="ok",
            storage_backend=dependencies.storage_backend,
            workspace_root=(
                str(dependencies.runtime_info.workspace_root)
                if dependencies.runtime_info.workspace_root is not None
                else None
            ),
            memory_path=(
                str(dependencies.runtime_info.memory_path)
                if dependencies.runtime_info.memory_path is not None
                else None
            ),
            db_path=(
                str(dependencies.runtime_info.db_path)
                if dependencies.runtime_info.db_path is not None
                else None
            ),
            runtime_active=dependencies.controller.has_runtime,
            client_count=dependencies.controller.client_count,
            task_queue_enabled=dependencies.runtime_info.task_queue_enabled,
            embeddings=embedder_status,
            search=build_search_health(
                dependencies.relational_search,
                search_health=dependencies.search_health,
                search_diagnostics=search_diagnostics_snapshot(
                    dependencies.retrieval_telemetry
                ),
            ),
            cache=self._build_cache_health(),
            transport_diagnostics=self._build_transport_diagnostics(),
            execution_attempts=build_execution_attempt_health(dependencies.db_manager),
        )

    def _build_transport_diagnostics(self) -> TransportDiagnosticsPayload:
        snapshot = getattr(self._dependencies.controller, "transport_diagnostics", None)
        return _coerce_transport_diagnostics_payload(snapshot)

    def _build_cache_health(self) -> CacheHealthPayload:
        dependencies = self._dependencies
        cache_state = resolve_shared_mode_cache_state(
            dependencies.config,
            storage_backend=dependencies.storage_backend,
            read_cache=dependencies.read_cache,
        )
        metrics = self._build_cache_metrics_payload()
        if not cache_state.enabled:
            return CacheHealthPayload(enabled=False, mode=None, state="disabled", path=None, metrics=metrics)

        if not cache_state.backend_supported:
            return CacheHealthPayload(
                enabled=True,
                mode=cache_state.mode,
                state="unsupported_backend",
                path=None,
                metrics=metrics,
            )

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
        read_cache = self._dependencies.read_cache
        if read_cache is None:
            return CacheMetricsPayload()
        try:
            snapshot = read_cache.get_metrics_snapshot()
        except Exception:
            logger.warning("Failed to read shared cache metrics snapshot", exc_info=True)
            return CacheMetricsPayload()
        return CacheMetricsPayload(**asdict(snapshot))

    def _configured_cache_path(self) -> Path | None:
        memory_path = self._dependencies.runtime_info.memory_path
        if memory_path is None:
            return None
        return memory_path / "cache" / "shared_read_cache.sqlite3"

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
        dependencies = self._dependencies
        generated_at = time.time()
        logs_after = generated_at - (max(log_window_minutes, 0) * 60)
        conversations_after = generated_at - (max(conversation_window_hours, 0) * 3600)

        health = self.get_health()
        log_summary = dependencies.runtime_log_service.summarize_logs(
            workspace_id=None,
            after=logs_after,
        )
        recent_errors = dependencies.runtime_log_service.list_logs(
            workspace_id=None,
            level="ERROR",
            after=logs_after,
            limit=recent_error_limit,
        ).logs
        recent_warnings = dependencies.runtime_log_service.list_logs(
            workspace_id=None,
            level="WARNING",
            after=logs_after,
            limit=recent_warning_limit,
        ).logs
        recent_run_rows = build_recent_agent_runs(
            dependencies.db_manager,
            None,
            limit=max(recent_run_limit * 5, recent_run_limit),
            detail_level="compact",
        )
        recent_runs = recent_run_rows[:recent_run_limit]
        recent_run_status_counts = dict(sorted(Counter(run.status for run in recent_runs).items()))
        recent_failures = [run for run in recent_run_rows if run.status == "failed"][:recent_run_limit]
        recent_retries = [run for run in recent_run_rows if run.status == "retry"][:recent_run_limit]
        conversation_counts = count_recent_conversation_statuses(
            dependencies.provider_usage,
            after=conversations_after,
            workspace_id=None,
        )
        recent_conversations = dependencies.list_ai_conversations(
            workspace_id=None,
            limit=conversation_limit,
        ).conversations
        recent_memories = dependencies.list_memories(
            workspace_id=None,
            limit=recent_memory_limit,
        ).records
        tool_latency = summarize_memory_tool_latency(
            dependencies.db_manager,
            workspace_id=None,
            window_minutes=log_window_minutes,
            slow_threshold_ms=_SLOW_MEMORY_TOOL_WARNING_MS,
        )
        provider_policy = summarize_provider_policy(
            dependencies.db_manager,
            provider_usage_repo=dependencies.provider_usage,
            workspace_id=None,
            window_minutes=log_window_minutes,
        )
        updated_last_15_minutes = count_recent_memory_updates(
            dependencies.repository,
            cutoff=datetime.now(UTC) - timedelta(minutes=15),
            workspace_id=None,
        )
        updated_last_hour = count_recent_memory_updates(
            dependencies.repository,
            cutoff=datetime.now(UTC) - timedelta(minutes=60),
            workspace_id=None,
        )
        updated_last_day = count_recent_memory_updates(
            dependencies.repository,
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

    def _embedding_integrity_summary(
        self,
        *,
        workspace_id: str | None,
    ) -> EmbeddingIntegrityEventSummaryPayload:
        repository = self._dependencies.embedding_integrity_events
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
        embedding_maintenance = self._dependencies.embedding_maintenance
        if embedding_maintenance is None:
            raise ValueError("search_not_initialized")
        return dict(embedding_maintenance.rebuild_semantic_index())


def _embedding_integrity_snapshot_payload(record: Any) -> EmbeddingIntegrityEventSnapshotPayload | None:
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
