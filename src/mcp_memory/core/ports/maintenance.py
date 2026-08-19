"""Provider-neutral maintenance ports."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, TypedDict, TypeVar

from mcp_memory.core.ports.memory import MemoryMaintenanceReadPort

MaintenanceReadRepositoryLike = MemoryMaintenanceReadPort


@dataclass(frozen=True)
class ExternalLinkRecord:
    memory_id: str
    status: str | None
    target_id: str
    workspace_id: str | None


@dataclass(frozen=True)
class LineageMemoryRecord:
    memory_id: str
    status: str
    metadata: object
    relationship_count: int


class DanglingLinkReconciliationResult(TypedDict):
    scanned: int
    deleted: int
    deleted_link_examples: list[dict[str, str]]


class ArchivedMemoryGcResult(TypedDict):
    mode: str
    cutoff: str
    scanned: int
    eligible: int
    skipped_protected: int
    skipped_linked: int
    deleted: int
    eligible_memory_examples: list[str]
    deleted_memory_examples: list[str]


class MaintenanceHousekeepingTransaction(Protocol):
    def mark_stale_plans(self, cutoff: str, workspace_id: str | None) -> int: ...

    def list_external_links(self) -> list[ExternalLinkRecord]: ...

    def update_memory_statuses(
        self,
        memory_ids: Sequence[str],
        *,
        status: str,
        current_status: str | None = None,
    ) -> None: ...

    def list_lineage_memories(self) -> list[LineageMemoryRecord]: ...

    def reconcile_dangling_links(self, *, batch_size: int) -> DanglingLinkReconciliationResult: ...

    def gc_archived_memories(
        self,
        *,
        cutoff: datetime,
        batch_size: int,
        mode: str,
    ) -> ArchivedMemoryGcResult: ...

    def delete_completed_tasks(self, cutoff_timestamp: float) -> int: ...

    def purge_recoverable_journal_entries(self, cutoff_timestamp: float) -> list[int]: ...

    def purge_processed_journal_entries(self, cutoff_timestamp: float) -> int: ...

    def gc_dead_metadata_keys(self, keys: Sequence[str]) -> int: ...


ResultT = TypeVar("ResultT")


class MaintenanceHousekeepingPort(Protocol):
    def execute(
        self,
        operation: Callable[[MaintenanceHousekeepingTransaction], ResultT],
    ) -> ResultT: ...


__all__ = [
    "ArchivedMemoryGcResult",
    "DanglingLinkReconciliationResult",
    "ExternalLinkRecord",
    "LineageMemoryRecord",
    "MaintenanceHousekeepingPort",
    "MaintenanceHousekeepingTransaction",
    "MaintenanceReadRepositoryLike",
]
