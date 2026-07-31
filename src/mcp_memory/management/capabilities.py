from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class _ManagementService(Protocol):
    @property
    def workspace_id(self) -> str | None: ...

    @property
    def dashboard_static_root(self) -> Any: ...


@dataclass(frozen=True)
class ManagementReportingService:
    """Reporting, health, and analytics operations exposed to management transports."""

    _service: _ManagementService

    def __getattr__(self, name: str) -> Any:
        if name in {
            "get_health",
            "get_operator_health_snapshot",
            "get_overview",
            "get_nerd_metrics",
            "get_selector_stats",
            "list_quality_cleanup_candidates",
        }:
            return getattr(self._service, name)
        raise AttributeError(name)


@dataclass(frozen=True)
class ManagementMaintenanceService:
    """Background task administration and maintenance operations."""

    _service: _ManagementService

    def __getattr__(self, name: str) -> Any:
        if name in {
            "enqueue_background_task",
            "enqueue_all_background_tasks",
            "cancel_task",
            "list_tasks",
            "list_recent_agent_runs",
            "get_task_sampling_summary",
            "get_task_detail",
        }:
            return getattr(self._service, name)
        raise AttributeError(name)


@dataclass(frozen=True)
class ManagementMutationService:
    """Mutation history, protection, and restore operations."""

    _service: _ManagementService

    def __getattr__(self, name: str) -> Any:
        if name in {
            "list_mutation_history",
            "get_mutation_history_event",
            "get_mutation_history_diff",
            "list_protections",
            "set_protection",
            "remove_protection",
            "get_restore_eligibility",
            "request_restore",
        }:
            return getattr(self._service, name)
        raise AttributeError(name)


@dataclass(frozen=True)
class ManagementRuntimeService:
    """Runtime logs, search repair, dashboard, and lifecycle operations."""

    _service: _ManagementService

    def __getattr__(self, name: str) -> Any:
        if name in {
            "repair_search_index",
            "list_logs",
            "summarize_logs",
            "prune_logs",
            "load_dashboard_html",
            "resolve_dashboard_asset_path",
        }:
            return getattr(self._service, name)
        raise AttributeError(name)


@dataclass(frozen=True)
class ManagementMemoryService:
    """Memory administration operations kept separate from reporting."""

    _service: _ManagementService

    def __getattr__(self, name: str) -> Any:
        if name in {
            "record_thought",
            "list_memories",
            "search_memories",
            "get_memory_detail",
            "create_memory_link",
            "delete_memory_link",
            "list_ai_conversations",
        }:
            return getattr(self._service, name)
        raise AttributeError(name)


@dataclass(frozen=True)
class ManagementCapabilities:
    """Typed capability bundle used by transport adapters."""

    reporting: ManagementReportingService
    maintenance: ManagementMaintenanceService
    mutation: ManagementMutationService
    runtime: ManagementRuntimeService
    memory: ManagementMemoryService
    _service: _ManagementService

    @classmethod
    def from_service(cls, service: _ManagementService) -> ManagementCapabilities:
        return cls(
            reporting=ManagementReportingService(service),
            maintenance=ManagementMaintenanceService(service),
            mutation=ManagementMutationService(service),
            runtime=ManagementRuntimeService(service),
            memory=ManagementMemoryService(service),
            _service=service,
        )

    @property
    def workspace_id(self) -> str | None:
        return self._service.workspace_id

    @property
    def dashboard_static_root(self) -> Any:
        return self._service.dashboard_static_root

    def __getattr__(self, name: str) -> Any:
        for capability in (
            self.reporting,
            self.maintenance,
            self.mutation,
            self.runtime,
            self.memory,
        ):
            try:
                return getattr(capability, name)
            except AttributeError:
                continue
        raise AttributeError(name)
