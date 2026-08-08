from __future__ import annotations

import pytest

from mcp_memory.core.curation_quality_structural import evaluate_structural_mutation


pytestmark = pytest.mark.small


def _record_delta(entity_id: str, transition: str) -> dict[str, object]:
    return {
        "kind": "record",
        "entity_id": entity_id,
        "transition": transition,
        "after_exists": True,
    }


def test_complete_split_requires_every_expected_record() -> None:
    """A split verifies when the archived source and all children are observed."""
    assessment = evaluate_structural_mutation(
        "split_memory",
        (
            _record_delta("source", "archived"),
            _record_delta("child-a", "created"),
            _record_delta("child-b", "created"),
        ),
        expected_entity_set={
            "source": "archived",
            "child-a": "created",
            "child-b": "created",
        },
    )

    assert assessment.verified is True
    assert assessment.details["delta_count"] == 3


def test_partial_split_is_rejected() -> None:
    """A split stays unverified when one expected child is absent."""
    assessment = evaluate_structural_mutation(
        "split_memory",
        (_record_delta("source", "archived"), _record_delta("child-a", "created")),
        expected_entity_set={
            "source": "archived",
            "child-a": "created",
            "child-b": "created",
        },
    )

    assert assessment.verified is False
    assert assessment.reason == "split_boundary_missing"
    assert assessment.details["delta_count"] == 2


def test_complete_merge_requires_every_expected_record() -> None:
    """A merge verifies when the canonical record and every source transition match."""
    assessment = evaluate_structural_mutation(
        "merge_memories",
        (
            _record_delta("canonical", "retained"),
            _record_delta("source-a", "archived"),
            _record_delta("source-b", "archived"),
        ),
        expected_entity_set={
            "canonical": "retained",
            "source-a": "archived",
            "source-b": "archived",
        },
    )

    assert assessment.verified is True
    assert assessment.details["delta_count"] == 3


def test_partial_merge_is_rejected() -> None:
    """A merge stays unverified when one expected source is absent."""
    assessment = evaluate_structural_mutation(
        "merge_memories",
        (_record_delta("canonical", "retained"), _record_delta("source-a", "archived")),
        expected_entity_set={
            "canonical": "retained",
            "source-a": "archived",
            "source-b": "archived",
        },
    )

    assert assessment.verified is False
    assert assessment.reason == "merge_sources_not_archived"
    assert assessment.details["delta_count"] == 2


def test_legacy_single_delta_behavior_is_preserved() -> None:
    """A legacy caller without an expected set retains single-delta verification."""
    assessment = evaluate_structural_mutation(
        "archive_memory", (_record_delta("memory", "archived"),)
    )

    assert assessment.verified is True
    assert assessment.details["delta_count"] == 1
