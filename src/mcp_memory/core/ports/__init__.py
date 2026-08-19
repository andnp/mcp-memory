"""Public port compatibility exports.

Submodules are loaded only when an export is requested.  This keeps importing
one port independent from the other port modules.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .curation import CurationActionStore, CurationRepository, CurationTransaction, MutationResult
    from .maintenance import (
        ArchivedMemoryGcResult,
        DanglingLinkReconciliationResult,
        ExternalLinkRecord,
        LineageMemoryRecord,
        MaintenanceHousekeepingPort,
        MaintenanceHousekeepingTransaction,
        MaintenanceReadRepositoryLike,
    )
    from .memory import (
        MemoryLink,
        MemoryMaintenanceReadPort,
        MemoryMutationPort,
        MemoryQueriesPort,
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
    from .search import (
        EmbeddingMaintenancePort,
        MemoryIDResolutionPort,
        ReadCacheValidationPort,
        SearchHealthPort,
        SearchHealthSnapshot,
        StartupHealthPort,
    )
    from .tasks import (
        TaskCancellationPort,
        TaskDataMutationPort,
        TaskLifecyclePort,
        TaskProcessSupervisionPort,
        TaskQueue,
        TaskRecord,
        TaskReportingPort,
        TaskRunRecord,
        TaskRunSummary,
        TaskSubmissionPort,
    )
    from .work_items import WorkItemRecord, WorkItemRecordLike, WorkItemRepository

_EXPORTS = {
    "CurationActionStore": ("curation", "CurationActionStore"),
    "CurationRepository": ("curation", "CurationRepository"),
    "CurationTransaction": ("curation", "CurationTransaction"),
    "MutationResult": ("curation", "MutationResult"),
    "MaintenanceReadRepositoryLike": ("maintenance", "MaintenanceReadRepositoryLike"),
    "ArchivedMemoryGcResult": ("maintenance", "ArchivedMemoryGcResult"),
    "DanglingLinkReconciliationResult": ("maintenance", "DanglingLinkReconciliationResult"),
    "ExternalLinkRecord": ("maintenance", "ExternalLinkRecord"),
    "LineageMemoryRecord": ("maintenance", "LineageMemoryRecord"),
    "MaintenanceHousekeepingPort": ("maintenance", "MaintenanceHousekeepingPort"),
    "MaintenanceHousekeepingTransaction": ("maintenance", "MaintenanceHousekeepingTransaction"),
    "MemoryLink": ("memory", "MemoryLink"),
    "MemoryMaintenanceReadPort": ("memory", "MemoryMaintenanceReadPort"),
    "MemoryMutationPort": ("memory", "MemoryMutationPort"),
    "MemoryReadContext": ("memory", "MemoryReadContext"),
    "MemoryReadPort": ("memory", "MemoryReadPort"),
    "MemoryQueriesPort": ("memory", "MemoryQueriesPort"),
    "MemoryRecord": ("memory", "MemoryRecord"),
    "MemoryRepositoryPort": ("memory", "MemoryRepositoryPort"),
    "EmbeddingMaintenancePort": ("search", "EmbeddingMaintenancePort"),
    "MemoryIDResolutionPort": ("search", "MemoryIDResolutionPort"),
    "ReadCacheValidationPort": ("search", "ReadCacheValidationPort"),
    "SearchHealthPort": ("search", "SearchHealthPort"),
    "SearchHealthSnapshot": ("search", "SearchHealthSnapshot"),
    "StartupHealthPort": ("search", "StartupHealthPort"),
    "RankedMemoryCandidate": ("memory", "RankedMemoryCandidate"),
    "ProviderUsagePort": ("providers", "ProviderUsagePort"),
    "NullProviderUsagePort": ("providers", "NullProviderUsagePort"),
    "ProviderConversationLike": ("providers", "ProviderConversationLike"),
    "ProviderPolicyEventPort": ("providers", "ProviderPolicyEventPort"),
    "TaskExecutionAttemptPort": ("providers", "TaskExecutionAttemptPort"),
    "TaskExecutionAttemptRecordLike": ("providers", "TaskExecutionAttemptRecordLike"),
    "TaskCancellationPort": ("tasks", "TaskCancellationPort"),
    "TaskDataMutationPort": ("tasks", "TaskDataMutationPort"),
    "TaskLifecyclePort": ("tasks", "TaskLifecyclePort"),
    "TaskProcessSupervisionPort": ("tasks", "TaskProcessSupervisionPort"),
    "WorkItemRecordLike": ("work_items", "WorkItemRecordLike"),
    "WorkItemRecord": ("work_items", "WorkItemRecord"),
    "WorkItemRepository": ("work_items", "WorkItemRepository"),
    "TaskQueue": ("tasks", "TaskQueue"),
    "TaskRecord": ("tasks", "TaskRecord"),
    "TaskReportingPort": ("tasks", "TaskReportingPort"),
    "TaskRunRecord": ("tasks", "TaskRunRecord"),
    "TaskRunSummary": ("tasks", "TaskRunSummary"),
    "TaskSubmissionPort": ("tasks", "TaskSubmissionPort"),
}

__all__ = (
    "ArchivedMemoryGcResult",
    "CurationActionStore",
    "CurationRepository",
    "CurationTransaction",
    "DanglingLinkReconciliationResult",
    "EmbeddingMaintenancePort",
    "ExternalLinkRecord",
    "LineageMemoryRecord",
    "MaintenanceHousekeepingPort",
    "MaintenanceHousekeepingTransaction",
    "MaintenanceReadRepositoryLike",
    "MemoryIDResolutionPort",
    "MemoryLink",
    "MemoryMaintenanceReadPort",
    "MemoryMutationPort",
    "MemoryQueriesPort",
    "MemoryReadContext",
    "MemoryReadPort",
    "MemoryRecord",
    "MemoryRepositoryPort",
    "MutationResult",
    "NullProviderUsagePort",
    "ProviderConversationLike",
    "ProviderPolicyEventPort",
    "ProviderUsagePort",
    "RankedMemoryCandidate",
    "ReadCacheValidationPort",
    "SearchHealthPort",
    "SearchHealthSnapshot",
    "StartupHealthPort",
    "TaskCancellationPort",
    "TaskDataMutationPort",
    "TaskExecutionAttemptPort",
    "TaskExecutionAttemptRecordLike",
    "TaskLifecyclePort",
    "TaskProcessSupervisionPort",
    "TaskQueue",
    "TaskRecord",
    "TaskReportingPort",
    "TaskRunRecord",
    "TaskRunSummary",
    "TaskSubmissionPort",
    "WorkItemRecord",
    "WorkItemRecordLike",
    "WorkItemRepository",
)


def __getattr__(name: str) -> Any:
    try:
        module_name, symbol_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    return getattr(import_module(f"{__name__}.{module_name}"), symbol_name)
