"""Public port compatibility exports.

Submodules are loaded only when an export is requested.  This keeps importing
one port independent from the other port modules.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .curation import CurationActionStore, CurationRepository, CurationTransaction, MutationResult
    from .maintenance import MaintenanceReadRepositoryLike
    from .memory import (
        MemoryLink,
        MemoryMaintenanceReadPort,
        MemoryMutationPort,
        MemoryReadContext,
        MemoryReadPort,
        MemoryRecord,
        MemoryRepositoryPort,
        RankedMemoryCandidate,
    )
    from .providers import (
        NullProviderUsagePort,
        ProviderConversationLike,
        ProviderPolicyEventPort,
        ProviderUsagePort,
        TaskExecutionAttemptPort,
        TaskExecutionAttemptRecordLike,
    )
    from .work_items import WorkItemRecord, WorkItemRecordLike, WorkItemRepository
    from .tasks import TaskQueue, TaskRecord, TaskRunRecord, TaskRunSummary

_EXPORTS = {
    "CurationActionStore": ("curation", "CurationActionStore"),
    "CurationRepository": ("curation", "CurationRepository"),
    "CurationTransaction": ("curation", "CurationTransaction"),
    "MutationResult": ("curation", "MutationResult"),
    "MaintenanceReadRepositoryLike": ("maintenance", "MaintenanceReadRepositoryLike"),
    "MemoryLink": ("memory", "MemoryLink"),
    "MemoryMaintenanceReadPort": ("memory", "MemoryMaintenanceReadPort"),
    "MemoryMutationPort": ("memory", "MemoryMutationPort"),
    "MemoryReadContext": ("memory", "MemoryReadContext"),
    "MemoryReadPort": ("memory", "MemoryReadPort"),
    "MemoryRecord": ("memory", "MemoryRecord"),
    "MemoryRepositoryPort": ("memory", "MemoryRepositoryPort"),
    "EmbeddingMaintenancePort": ("search", "EmbeddingMaintenancePort"),
    "MemoryIDResolutionPort": ("search", "MemoryIDResolutionPort"),
    "ReadCacheValidationPort": ("search", "ReadCacheValidationPort"),
    "SearchHealthPort": ("search", "SearchHealthPort"),
    "StartupHealthPort": ("search", "StartupHealthPort"),
    "RankedMemoryCandidate": ("memory", "RankedMemoryCandidate"),
    "ProviderUsagePort": ("providers", "ProviderUsagePort"),
    "NullProviderUsagePort": ("providers", "NullProviderUsagePort"),
    "ProviderConversationLike": ("providers", "ProviderConversationLike"),
    "ProviderPolicyEventPort": ("providers", "ProviderPolicyEventPort"),
    "TaskExecutionAttemptPort": ("providers", "TaskExecutionAttemptPort"),
    "TaskExecutionAttemptRecordLike": ("providers", "TaskExecutionAttemptRecordLike"),
    "WorkItemRecordLike": ("work_items", "WorkItemRecordLike"),
    "WorkItemRecord": ("work_items", "WorkItemRecord"),
    "WorkItemRepository": ("work_items", "WorkItemRepository"),
    "TaskQueue": ("tasks", "TaskQueue"),
    "TaskRecord": ("tasks", "TaskRecord"),
    "TaskRunRecord": ("tasks", "TaskRunRecord"),
    "TaskRunSummary": ("tasks", "TaskRunSummary"),
}

__all__ = (
    "CurationActionStore",
    "CurationRepository",
    "CurationTransaction",
    "MutationResult",
    "MaintenanceReadRepositoryLike",
    "MemoryLink",
    "MemoryMaintenanceReadPort",
    "MemoryMutationPort",
    "MemoryReadContext",
    "MemoryReadPort",
    "MemoryRecord",
    "MemoryRepositoryPort",
    "RankedMemoryCandidate",
    "EmbeddingMaintenancePort",
    "MemoryIDResolutionPort",
    "ReadCacheValidationPort",
    "SearchHealthPort",
    "StartupHealthPort",
    "ProviderUsagePort",
    "NullProviderUsagePort",
    "ProviderConversationLike",
    "ProviderPolicyEventPort",
    "TaskExecutionAttemptPort",
    "TaskExecutionAttemptRecordLike",
    "WorkItemRecordLike",
    "WorkItemRecord",
    "WorkItemRepository",
    "TaskQueue",
    "TaskRecord",
    "TaskRunRecord",
    "TaskRunSummary",
)


def __getattr__(name: str) -> Any:
    try:
        module_name, symbol_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    return getattr(import_module(f"{__name__}.{module_name}"), symbol_name)
