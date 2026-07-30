"""Provider-neutral work-item application contracts."""

from __future__ import annotations

from typing import Any, Protocol


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
    @property
    def id(self) -> str: ...

    @property
    def payload(self) -> dict[str, Any]: ...

    @property
    def family_key(self) -> str: ...

    @property
    def execution_lane(self) -> str: ...

    @property
    def workspace_id(self) -> str | None: ...

    @property
    def status(self) -> str: ...

    @property
    def priority(self) -> int: ...

    @property
    def attempt_count(self) -> int: ...

    @property
    def available_at(self) -> float: ...

    @property
    def created_at(self) -> float: ...

    @property
    def updated_at(self) -> float: ...

    @property
    def claimed_at(self) -> float | None: ...

    @property
    def completed_at(self) -> float | None: ...

    @property
    def lease_owner(self) -> str | None: ...

    @property
    def lease_expires_at(self) -> float | None: ...

    @property
    def idempotency_key(self) -> str | None: ...

    @property
    def last_error(self) -> str | None: ...


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


__all__ = [
    "COMPATIBILITY_GROUP_LIGHTWEIGHT_REVIEW",
    "COMPATIBILITY_GROUP_STRUCTURAL_REVIEW",
    "EXECUTION_LANE_AGENTIC",
    "EXECUTION_LANE_DETERMINISTIC",
    "WORK_FAMILY_CONFLICT_REVIEW",
    "WORK_FAMILY_GRAPH_LINK_REVIEW",
    "WORK_FAMILY_MEMORY_CURATION_REVIEW",
    "WORK_FAMILY_MEMORY_DEDUP_REVIEW",
    "WORK_FAMILY_MEMORY_EMBEDDING_REPAIR",
    "WORK_FAMILY_MEMORY_TAGGING",
    "WORK_FAMILY_OPERATOR_REVIEW",
    "WorkItemRecordLike",
    "WorkItemRepository",
    "compatibility_group_families",
]
