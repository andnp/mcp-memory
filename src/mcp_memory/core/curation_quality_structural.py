"""Structural postcondition checks for direct curation mutations."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class StructuralAssessment:
    relevant: bool
    verified: bool
    reason: str | None = None
    details: Mapping[str, object] = field(default_factory=dict)


_STRUCTURAL_OPERATIONS = frozenset(
    {
        "archive_memory",
        "delete_memory",
        "create_link",
        "remove_link",
        "remove_dangling_link",
        "reconcile_dangling_links",
        "cleanup_dangling_links",
        "split_memory",
        "merge_memories",
        "deduplicate_memories",
    }
)


def evaluate_structural_mutation(
    operation: str,
    deltas: Sequence[Mapping[str, object]],
) -> StructuralAssessment:
    if operation not in _STRUCTURAL_OPERATIONS:
        return StructuralAssessment(False, False)
    if not deltas:
        return StructuralAssessment(True, True, details={"evidence": "typed_receipt"})
    if operation == "create_link":
        valid = any(_is_link(delta) and delta.get("after_exists") is True for delta in deltas)
        return _result(valid, "link_not_created")
    if operation in {
        "remove_link",
        "remove_dangling_link",
        "reconcile_dangling_links",
        "cleanup_dangling_links",
    }:
        valid = any(_is_link(delta) and delta.get("after_exists") is False for delta in deltas)
        return _result(valid, "dangling_link_remains")
    if operation == "archive_memory":
        valid = any(
            _is_record(delta)
            and delta.get("transition") == "archived"
            and delta.get("after_exists") is True
            for delta in deltas
        )
        return _result(valid, "memory_not_archived")
    if operation == "delete_memory":
        valid = any(_is_record(delta) and delta.get("after_exists") is False for delta in deltas)
        return _result(valid, "memory_not_deleted")
    if operation in {"merge_memories", "deduplicate_memories"}:
        archived = any(
            _is_record(delta) and delta.get("transition") == "archived" for delta in deltas
        )
        retained = any(
            _is_record(delta) and delta.get("after_exists") is True for delta in deltas
        )
        return _result(archived and retained, "merge_sources_not_archived")
    created = any(_is_record(delta) and delta.get("transition") == "created" for delta in deltas)
    archived = any(_is_record(delta) and delta.get("transition") == "archived" for delta in deltas)
    return _result(created and archived, "split_boundary_missing")


def _result(valid: bool, reason: str) -> StructuralAssessment:
    return StructuralAssessment(
        relevant=True,
        verified=valid,
        reason=None if valid else reason,
        details={"delta_count": 1 if valid else 0},
    )


def _is_record(delta: Mapping[str, object]) -> bool:
    return delta.get("kind") == "record"


def _is_link(delta: Mapping[str, object]) -> bool:
    return delta.get("kind") == "link"
