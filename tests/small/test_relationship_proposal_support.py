from __future__ import annotations

from mcp_memory.core.task_handlers.relationship_proposals import (
    normalize_conflict_proposals,
    normalize_graph_link_proposals,
)


def test_normalize_graph_link_proposals_applies_defaults_and_filters_invalid_rows() -> None:
    proposals = normalize_graph_link_proposals(
        [
            {"source_id": "a", "target_id": "b", "link_type": "", "context": ""},
            {"source_id": "", "target_id": "b"},
        ]
    )

    assert proposals == [("a", "b", "DEPENDS_ON", "Auto-linked by graph linker.")]


def test_normalize_conflict_proposals_applies_default_context() -> None:
    proposals = normalize_conflict_proposals([
        {"left_id": "left", "right_id": "right", "context": ""},
    ])

    assert proposals == [("left", "right", "Potential contradiction detected.")]