"""Provider-neutral work-item application contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


WORK_ITEM_STATUS_PENDING = "pending"
WORK_ITEM_STATUS_RUNNING = "running"
WORK_ITEM_STATUS_COMPLETED = "completed"
WORK_ITEM_STATUS_DEFERRED = "deferred"
DEFAULT_WORK_ITEM_LEASE_TTL_SECONDS = 1800.0


EXECUTION_LANE_DETERMINISTIC = "deterministic"
EXECUTION_LANE_AGENTIC = "agentic"
WORK_FAMILY_MEMORY_EMBEDDING_REPAIR = "memory_embedding_repair"
WORK_FAMILY_CONFLICT_REVIEW = "conflict_review"
WORK_FAMILY_MEMORY_CURATION_REVIEW = "memory_curation_review"
WORK_FAMILY_MEMORY_DEDUP_REVIEW = "memory_dedup_review"
WORK_FAMILY_GRAPH_LINK_REVIEW = "graph_link_review"
WORK_FAMILY_MEMORY_TAGGING = "memory_tagging"
WORK_FAMILY_OPERATOR_REVIEW = "operator_review"
COMPATIBILITY_GROUP_LIGHTWEIGHT_REVIEW = "lightweight_review"
COMPATIBILITY_GROUP_STRUCTURAL_REVIEW = "structural_review"

_WORK_ITEM_COMPATIBILITY_GROUPS: dict[str, tuple[str, ...]] = {
    COMPATIBILITY_GROUP_LIGHTWEIGHT_REVIEW: (
        WORK_FAMILY_MEMORY_TAGGING,
        WORK_FAMILY_GRAPH_LINK_REVIEW,
        WORK_FAMILY_CONFLICT_REVIEW,
    ),
    COMPATIBILITY_GROUP_STRUCTURAL_REVIEW: (
        WORK_FAMILY_MEMORY_CURATION_REVIEW,
        WORK_FAMILY_MEMORY_DEDUP_REVIEW,
        WORK_FAMILY_CONFLICT_REVIEW,
    ),
}


@dataclass(frozen=True)
class WorkItemRecord:
    id: str
    family_key: str
    execution_lane: str
    workspace_id: str | None
    payload: dict[str, Any]
    status: str
    priority: int
    attempt_count: int
    available_at: float
    created_at: float
    updated_at: float
    claimed_at: float | None
    completed_at: float | None
    lease_owner: str | None
    lease_expires_at: float | None
    idempotency_key: str | None
    last_error: str | None


def compatibility_group_families(
    group: str,
    *,
    allowed_families: list[str] | None = None,
) -> tuple[str, ...]:
    """Return the application-defined families compatible with a group."""

    families = _WORK_ITEM_COMPATIBILITY_GROUPS.get(group)
    if families is None:
        raise ValueError(f"Unknown work-item compatibility group: {group}")
    if not allowed_families:
        return families
    normalized_allowed = tuple(dict.fromkeys(str(family) for family in allowed_families if str(family).strip()))
    if not normalized_allowed:
        raise ValueError("allowed_families must include at least one family when provided")
    invalid = [family for family in normalized_allowed if family not in families]
    if invalid:
        raise ValueError(f"Families {invalid} are not compatible with group {group}")
    return normalized_allowed


class WorkItemRecordLike(Protocol):
    id: str
    family_key: str
    execution_lane: str
    workspace_id: str | None
    payload: dict[str, Any]
    status: str
    priority: int
    attempt_count: int
    available_at: float
    created_at: float
    updated_at: float
    claimed_at: float | None
    completed_at: float | None
    lease_owner: str | None
    lease_expires_at: float | None
    idempotency_key: str | None
    last_error: str | None


class WorkItemRepository(Protocol):
    def enqueue_unique(
        self,
        *,
        family_key: str,
        execution_lane: str,
        payload: dict[str, Any] | None = None,
        workspace_id: str | None = None,
        priority: int = 100,
        available_at: float | None = None,
        idempotency_key: str | None = None,
    ) -> tuple[WorkItemRecordLike, bool]: ...

    def complete_item(self, item_id: str, *, completed_at: float | None = None) -> WorkItemRecordLike: ...

    def defer_item(
        self,
        item_id: str,
        *,
        error: str,
        retry_delay_seconds: float,
        deferred_at: float | None = None,
    ) -> WorkItemRecordLike: ...

    def claim_batch(
        self,
        *,
        family_key: str,
        execution_lane: str,
        lease_owner: str,
        limit: int,
        workspace_id: str | None = None,
        now: float | None = None,
        lease_ttl_seconds: float = DEFAULT_WORK_ITEM_LEASE_TTL_SECONDS,
    ) -> list[WorkItemRecordLike]: ...

    def claim_compatible_batch(
        self,
        *,
        family_keys: list[str] | tuple[str, ...],
        execution_lane: str,
        lease_owner: str,
        limit: int,
        workspace_id: str | None = None,
        now: float | None = None,
        lease_ttl_seconds: float = DEFAULT_WORK_ITEM_LEASE_TTL_SECONDS,
    ) -> list[WorkItemRecordLike]: ...

    def heartbeat_item(
        self,
        item_id: str,
        *,
        lease_owner: str,
        lease_ttl_seconds: float = DEFAULT_WORK_ITEM_LEASE_TTL_SECONDS,
        heartbeated_at: float | None = None,
    ) -> WorkItemRecordLike: ...

    def release_item(self, item_id: str, *, released_at: float | None = None) -> WorkItemRecordLike: ...
    def get_item(self, item_id: str) -> WorkItemRecordLike: ...
    def list_items(self, *, status: str | None = None, lease_owner: str | None = None, limit: int = 100) -> list[WorkItemRecordLike]: ...
    def find_by_idempotency_key(self, idempotency_key: str | None) -> WorkItemRecordLike | None: ...


__all__ = [
    "COMPATIBILITY_GROUP_LIGHTWEIGHT_REVIEW",
    "COMPATIBILITY_GROUP_STRUCTURAL_REVIEW",
    "DEFAULT_WORK_ITEM_LEASE_TTL_SECONDS",
    "EXECUTION_LANE_AGENTIC",
    "EXECUTION_LANE_DETERMINISTIC",
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
    "WorkItemRecordLike",
    "WorkItemRepository",
    "compatibility_group_families",
]
