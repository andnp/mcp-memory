from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, cast

from mcp_memory.management.memory_service import _USE_SERVICE_WORKSPACE
from mcp_memory.management.models import MemoryListPayload, MutationHistoryListPayload, RuntimeLogListPayload


class _ReportingService(Protocol):
    def get_health(self) -> object: ...
    def get_operator_health_snapshot(self, **kwargs: object) -> object: ...
    def repair_search_index(self) -> object: ...


class _OverviewService(Protocol):
    def get_overview(self, **kwargs: object) -> object: ...


class _AnalyticsService(Protocol):
    def get_nerd_metrics(self, **kwargs: object) -> object: ...
    def get_selector_stats(self, **kwargs: object) -> object: ...
    def list_quality_cleanup_candidates(self, **kwargs: object) -> object: ...


class _TaskAdministrationService(Protocol):
    def enqueue_background_task(self, task_name: str, *, force: bool = False) -> dict: ...
    def enqueue_all_background_tasks(self, *, force: bool = False) -> list[dict]: ...
    def cancel_task(self, task_id: str, *, cancelled_by: str = "cli", reason: str = "cancelled_by_user") -> dict: ...
    def list_tasks(self, status: str | None = None, workspace_id: str | None = None, limit: int = 20) -> object: ...
    def get_task_detail(self, task_id: str) -> object: ...


class _TaskReportingService(Protocol):
    def list_recent_agent_runs(self, *, limit: int = 20, detail_level: str = "compact") -> object: ...
    def get_task_sampling_summary(self, *, limit: int = 50) -> object: ...


class _MutationService(Protocol):
    def list_mutation_history(self, **kwargs: object) -> object: ...
    def get_mutation_history_event(self, event_id: str) -> object: ...
    def get_mutation_history_diff(self, event_id: str) -> object: ...
    def list_protections(self, memory_id: str) -> object: ...
    def set_protection(self, **kwargs: object) -> object: ...
    def remove_protection(self, *, memory_id: str, mode: str) -> object: ...
    def get_restore_eligibility(self, event_id: str) -> object: ...
    def request_restore(self, **kwargs: object) -> object: ...


class _RuntimeLogService(Protocol):
    @property
    def dashboard_static_root(self) -> Path: ...

    def list_logs(self, **kwargs: object) -> object: ...
    def summarize_logs(self, **kwargs: object) -> object: ...
    def prune_logs(self, **kwargs: object) -> object: ...
    def load_dashboard_html(self) -> str: ...
    def resolve_dashboard_asset_path(self, asset_path: str) -> Path | None: ...


class _MemoryService(Protocol):
    def record_thought(self, content: str, *, workspace_id: object = None) -> dict[str, object]: ...
    def list_memories(self, **kwargs: object) -> object: ...
    def search_memories(self, **kwargs: object) -> object: ...
    def get_memory_detail(self, memory_id: str) -> object: ...
    def create_memory_link(self, **kwargs: object) -> dict: ...
    def delete_memory_link(self, **kwargs: object) -> dict: ...
    def list_ai_conversations(self, **kwargs: object) -> object: ...


class _ManagementService(Protocol):
    @property
    def workspace_id(self) -> str | None: ...

    @property
    def dashboard_static_root(self) -> Any: ...


class _LegacyManagementService(_ManagementService, _ReportingService, _OverviewService, _AnalyticsService,
                                _TaskAdministrationService, _TaskReportingService, _MutationService,
                                _RuntimeLogService, _MemoryService, Protocol):
    """Compatibility shape for callers that still provide one management object."""


@dataclass(frozen=True)
class ManagementReportingService:
    health: _ReportingService
    overview: _OverviewService
    analytics: _AnalyticsService
    refresh: Callable[[], None] | None = None

    def get_health(self) -> object:
        if self.refresh is not None:
            self.refresh()
        return self.health.get_health()

    def get_operator_health_snapshot(self, **kwargs: object) -> object:
        if self.refresh is not None:
            self.refresh()
        return self.health.get_operator_health_snapshot(**kwargs)

    def get_overview(self, **kwargs: object) -> object:
        return self.overview.get_overview(**kwargs)

    def get_nerd_metrics(self, **kwargs: object) -> object:
        return self.analytics.get_nerd_metrics(**kwargs)

    def get_selector_stats(self, **kwargs: object) -> object:
        return self.analytics.get_selector_stats(**kwargs)

    def list_quality_cleanup_candidates(self, **kwargs: object) -> object:
        return self.analytics.list_quality_cleanup_candidates(**kwargs)


@dataclass(frozen=True)
class ManagementMaintenanceService:
    administration: _TaskAdministrationService
    reporting: _TaskReportingService

    def enqueue_background_task(self, task_name: str, *, force: bool = False) -> dict:
        return self.administration.enqueue_background_task(task_name, force=force)

    def enqueue_all_background_tasks(self, *, force: bool = False) -> list[dict]:
        return self.administration.enqueue_all_background_tasks(force=force)

    def cancel_task(self, task_id: str, *, cancelled_by: str = "cli", reason: str = "cancelled_by_user") -> dict:
        return self.administration.cancel_task(task_id, cancelled_by=cancelled_by, reason=reason)

    def list_tasks(self, status: str | None = None, workspace_id: str | None = None, limit: int = 20) -> object:
        return self.administration.list_tasks(status=status, workspace_id=workspace_id, limit=limit)

    def list_recent_agent_runs(self, *, limit: int = 20, detail_level: str = "compact") -> object:
        return self.reporting.list_recent_agent_runs(limit=limit, detail_level=detail_level)

    def get_task_sampling_summary(self, *, limit: int = 50) -> object:
        return self.reporting.get_task_sampling_summary(limit=limit)

    def get_task_detail(self, task_id: str) -> object:
        return self.administration.get_task_detail(task_id)


@dataclass(frozen=True)
class ManagementMutationService:
    service: _MutationService

    def list_mutation_history(self, **kwargs: object) -> MutationHistoryListPayload:
        return cast(MutationHistoryListPayload, self.service.list_mutation_history(**kwargs))

    def get_mutation_history_event(self, event_id: str) -> object:
        return self.service.get_mutation_history_event(event_id)

    def get_mutation_history_diff(self, event_id: str) -> object:
        return self.service.get_mutation_history_diff(event_id)

    def list_protections(self, memory_id: str) -> object:
        return self.service.list_protections(memory_id)

    def set_protection(self, **kwargs: object) -> object:
        return self.service.set_protection(**kwargs)

    def remove_protection(self, *, memory_id: str, mode: str) -> object:
        return self.service.remove_protection(memory_id=memory_id, mode=mode)

    def get_restore_eligibility(self, event_id: str) -> object:
        return self.service.get_restore_eligibility(event_id)

    def request_restore(self, **kwargs: object) -> object:
        return self.service.request_restore(**kwargs)


@dataclass(frozen=True)
class ManagementRuntimeService:
    health: _ReportingService
    logs: _RuntimeLogService
    refresh: Callable[[], None] | None = None

    def repair_search_index(self) -> object:
        if self.refresh is not None:
            self.refresh()
        return self.health.repair_search_index()

    def list_logs(self, **kwargs: object) -> RuntimeLogListPayload:
        return cast(RuntimeLogListPayload, self.logs.list_logs(**kwargs))

    def summarize_logs(self, **kwargs: object) -> object:
        return self.logs.summarize_logs(**kwargs)

    def prune_logs(self, **kwargs: object) -> object:
        return self.logs.prune_logs(**kwargs)

    def load_dashboard_html(self) -> str:
        return self.logs.load_dashboard_html()

    def resolve_dashboard_asset_path(self, asset_path: str) -> Path | None:
        return self.logs.resolve_dashboard_asset_path(asset_path)


@dataclass(frozen=True)
class ManagementMemoryService:
    service: _MemoryService

    def record_thought(
        self,
        content: str,
        *,
        workspace_id: object = _USE_SERVICE_WORKSPACE,
    ) -> dict[str, object]:
        return self.service.record_thought(content, workspace_id=workspace_id)

    def list_memories(self, **kwargs: object) -> MemoryListPayload:
        return cast(MemoryListPayload, self.service.list_memories(**kwargs))

    def search_memories(self, **kwargs: object) -> object:
        return self.service.search_memories(**kwargs)

    def get_memory_detail(self, memory_id: str) -> object:
        return self.service.get_memory_detail(memory_id)

    def create_memory_link(self, **kwargs: object) -> dict:
        return self.service.create_memory_link(**kwargs)

    def delete_memory_link(self, **kwargs: object) -> dict:
        return self.service.delete_memory_link(**kwargs)

    def list_ai_conversations(self, **kwargs: object) -> object:
        return self.service.list_ai_conversations(**kwargs)


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
        service_object = cast(Any, service)
        if all(hasattr(service_object, name) for name in (
            "_runtime_health_service",
            "_overview_service",
            "_analytics_service",
            "_task_administration_service",
            "_task_reporting_service",
            "_mutation_history_service",
            "_runtime_log_service",
            "_memory_service",
        )):
            return cls(
                reporting=ManagementReportingService(
                    cast(_ReportingService, service_object._runtime_health_service),
                    cast(_OverviewService, service_object._overview_service),
                    cast(_AnalyticsService, service_object._analytics_service),
                    refresh=service_object._refresh_runtime_health_service,
                ),
                maintenance=ManagementMaintenanceService(
                    cast(_TaskAdministrationService, service_object._task_administration_service),
                    cast(_TaskReportingService, service_object._task_reporting_service),
                ),
                mutation=ManagementMutationService(cast(_MutationService, service_object._mutation_history_service)),
                runtime=ManagementRuntimeService(
                    cast(_ReportingService, service_object._runtime_health_service),
                    cast(_RuntimeLogService, service_object._runtime_log_service),
                    refresh=service_object._refresh_runtime_health_service,
                ),
                memory=ManagementMemoryService(cast(_MemoryService, service_object._memory_service)),
                _service=service,
            )

        # Existing callers may pass a service-shaped object instead of ManagementService.
        legacy = cast(_LegacyManagementService, service)
        return cls(
            reporting=ManagementReportingService(legacy, legacy, legacy),
            maintenance=ManagementMaintenanceService(legacy, legacy),
            mutation=ManagementMutationService(legacy),
            runtime=ManagementRuntimeService(legacy, legacy),
            memory=ManagementMemoryService(legacy),
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
