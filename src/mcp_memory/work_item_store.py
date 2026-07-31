"""Compatibility exports for work-item application and storage boundaries."""

from __future__ import annotations

from typing import Any

from mcp_memory.core.ports.work_items import (
    COMPATIBILITY_GROUP_LIGHTWEIGHT_REVIEW,
    COMPATIBILITY_GROUP_STRUCTURAL_REVIEW,
    DEFAULT_WORK_ITEM_LEASE_TTL_SECONDS,
    EXECUTION_LANE_AGENTIC,
    EXECUTION_LANE_DETERMINISTIC,
    WORK_FAMILY_CONFLICT_REVIEW,
    WORK_FAMILY_GRAPH_LINK_REVIEW,
    WORK_FAMILY_MEMORY_CURATION_REVIEW,
    WORK_FAMILY_MEMORY_DEDUP_REVIEW,
    WORK_FAMILY_MEMORY_EMBEDDING_REPAIR,
    WORK_FAMILY_MEMORY_TAGGING,
    WORK_FAMILY_OPERATOR_REVIEW,
    WORK_ITEM_STATUS_COMPLETED,
    WORK_ITEM_STATUS_DEFERRED,
    WORK_ITEM_STATUS_PENDING,
    WORK_ITEM_STATUS_RUNNING,
    WorkItemRecord,
    compatibility_group_families,
)
class SQLiteWorkItemRepository:
    """Compatibility constructor for the storage-owned SQLite repository."""

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        from mcp_memory.storage.sqlite_work_item_store import (
            SQLiteWorkItemRepository as StorageWorkItemRepository,
        )

        return StorageWorkItemRepository(*args, **kwargs)


__all__ = [
    "COMPATIBILITY_GROUP_LIGHTWEIGHT_REVIEW",
    "COMPATIBILITY_GROUP_STRUCTURAL_REVIEW",
    "DEFAULT_WORK_ITEM_LEASE_TTL_SECONDS",
    "EXECUTION_LANE_AGENTIC",
    "EXECUTION_LANE_DETERMINISTIC",
    "SQLiteWorkItemRepository",
    "WORK_FAMILY_CONFLICT_REVIEW",
    "WORK_FAMILY_GRAPH_LINK_REVIEW",
    "WORK_FAMILY_MEMORY_CURATION_REVIEW",
    "WORK_FAMILY_MEMORY_DEDUP_REVIEW",
    "WORK_FAMILY_MEMORY_EMBEDDING_REPAIR",
    "WORK_FAMILY_MEMORY_TAGGING",
    "WORK_FAMILY_OPERATOR_REVIEW",
    "WORK_ITEM_STATUS_COMPLETED",
    "WORK_ITEM_STATUS_DEFERRED",
    "WORK_ITEM_STATUS_PENDING",
    "WORK_ITEM_STATUS_RUNNING",
    "WorkItemRecord",
    "compatibility_group_families",
]


def __getattr__(name: str) -> Any:
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
