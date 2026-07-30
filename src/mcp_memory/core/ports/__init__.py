"""Public port compatibility exports.

Submodules are loaded only when an export is requested.  This keeps importing
one port independent from the other port modules.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .curation import CurationActionStore, CurationRepository, CurationTransaction, MutationResult
    from .maintenance import MaintenanceReadRepositoryLike
    from .planner import CurationPlanner, CurationReadTools, PlannerExecutionEnvelope, PlannerExecutionStatus
    from .providers import ProviderUsagePort, TaskExecutionAttemptPort, TaskExecutionAttemptRecordLike
    from .work_items import WorkItemRecordLike, WorkItemRepository

_EXPORTS = {
    "CurationActionStore": ("curation", "CurationActionStore"),
    "CurationRepository": ("curation", "CurationRepository"),
    "CurationTransaction": ("curation", "CurationTransaction"),
    "MutationResult": ("curation", "MutationResult"),
    "MaintenanceReadRepositoryLike": ("maintenance", "MaintenanceReadRepositoryLike"),
    "CurationPlanner": ("planner", "CurationPlanner"),
    "CurationReadTools": ("planner", "CurationReadTools"),
    "PlannerExecutionEnvelope": ("planner", "PlannerExecutionEnvelope"),
    "PlannerExecutionStatus": ("planner", "PlannerExecutionStatus"),
    "ProviderUsagePort": ("providers", "ProviderUsagePort"),
    "TaskExecutionAttemptPort": ("providers", "TaskExecutionAttemptPort"),
    "TaskExecutionAttemptRecordLike": ("providers", "TaskExecutionAttemptRecordLike"),
    "WorkItemRecordLike": ("work_items", "WorkItemRecordLike"),
    "WorkItemRepository": ("work_items", "WorkItemRepository"),
}

__all__ = (
    "CurationActionStore",
    "CurationRepository",
    "CurationTransaction",
    "MutationResult",
    "MaintenanceReadRepositoryLike",
    "CurationPlanner",
    "CurationReadTools",
    "PlannerExecutionEnvelope",
    "PlannerExecutionStatus",
    "ProviderUsagePort",
    "TaskExecutionAttemptPort",
    "TaskExecutionAttemptRecordLike",
    "WorkItemRecordLike",
    "WorkItemRepository",
)


def __getattr__(name: str) -> Any:
    try:
        module_name, symbol_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    return getattr(import_module(f"{__name__}.{module_name}"), symbol_name)
