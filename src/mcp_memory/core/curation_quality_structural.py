"""Structural postcondition checks for direct curation mutations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field


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
    expected_entity_set: Mapping[str, str] | None = None,
) -> StructuralAssessment:
    if operation not in _STRUCTURAL_OPERATIONS:
        return StructuralAssessment(False, False)
    if not deltas:
        return StructuralAssessment(True, True, details={"evidence": "typed_receipt"})
    if operation == "create_link":
        matched = sum(
            _is_link(delta) and delta.get("after_exists") is True for delta in deltas
        )
        return _result(matched > 0, "link_not_created", matched)
    if operation in {
        "remove_link",
        "remove_dangling_link",
        "reconcile_dangling_links",
        "cleanup_dangling_links",
    }:
        matched = sum(
            _is_link(delta) and delta.get("after_exists") is False for delta in deltas
        )
        return _result(matched > 0, "dangling_link_remains", matched)
    if operation == "archive_memory":
        matched = sum(
            _is_record(delta)
            and delta.get("transition") == "archived"
            and delta.get("after_exists") is True
            for delta in deltas
        )
        return _result(matched > 0, "memory_not_archived", matched)
    if operation == "delete_memory":
        matched = sum(
            _is_record(delta) and delta.get("after_exists") is False for delta in deltas
        )
        return _result(matched > 0, "memory_not_deleted", matched)
    if operation in {"merge_memories", "deduplicate_memories"}:
        if expected_entity_set:
            matched = _matched_expected_records(deltas, expected_entity_set)
            return _result(
                matched == len(expected_entity_set), "merge_sources_not_archived", matched
            )
        archived = any(
            _is_record(delta) and delta.get("transition") == "archived" for delta in deltas
        )
        retained = any(
            _is_record(delta) and delta.get("after_exists") is True for delta in deltas
        )
        matched = sum(
            _is_record(delta)
            and (
                delta.get("transition") == "archived"
                or delta.get("after_exists") is True
            )
            for delta in deltas
        )
        return _result(archived and retained, "merge_sources_not_archived", matched)
    if expected_entity_set:
        matched = _matched_expected_records(deltas, expected_entity_set)
        return _result(matched == len(expected_entity_set), "split_boundary_missing", matched)
    created = any(_is_record(delta) and delta.get("transition") == "created" for delta in deltas)
    archived = any(_is_record(delta) and delta.get("transition") == "archived" for delta in deltas)
    matched = sum(
        _is_record(delta)
        and delta.get("transition") in {"created", "archived"}
        for delta in deltas
    )
    return _result(created and archived, "split_boundary_missing", matched)


def _result(valid: bool, reason: str, matched_delta_count: int) -> StructuralAssessment:
    return StructuralAssessment(
        relevant=True,
        verified=valid,
        reason=None if valid else reason,
        details={"delta_count": matched_delta_count},
    )


def _matched_expected_records(
    deltas: Sequence[Mapping[str, object]], expected_entity_set: Mapping[str, str]
) -> int:
    expected = {str(entity_id): str(transition) for entity_id, transition in expected_entity_set.items()}
    return len({
        entity_id
        for delta in deltas
        if _is_record(delta)
        and (entity_id := str(delta.get("entity_id"))) in expected
        and delta.get("transition") == expected[entity_id]
        and delta.get("after_exists") is True
    })


def _is_record(delta: Mapping[str, object]) -> bool:
    return delta.get("kind") == "record"


def _is_link(delta: Mapping[str, object]) -> bool:
    return delta.get("kind") == "link"
