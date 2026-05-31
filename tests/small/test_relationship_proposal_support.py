from __future__ import annotations

from mcp_memory.context import ApplicationContext
from mcp_memory.core.task_handlers.relationship_proposals import (
    _fallback_graph_links,
    normalize_conflict_proposals,
    normalize_graph_link_proposals,
)
from mcp_memory.relational.repository import RelationalMemoryRepository


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


def test_fallback_graph_links_skips_broad_tag_only_matches(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    left = repository.create_memory(
        title="Curator housekeeping cadence",
        content="Curator schedule cleanup",
        workspace_ids=["workspace-a"],
        memory_type="plan",
        tags=["maintenance"],
    )
    right = repository.create_memory(
        title="Dashboard cache metrics",
        content="Operator dashboard analytics",
        workspace_ids=["workspace-a"],
        memory_type="plan",
        tags=["maintenance"],
    )
    assert left is not None
    assert right is not None

    proposals = _fallback_graph_links(
        ApplicationContext(repository=repository),
        [left, right],
    )

    assert proposals == []


def test_fallback_graph_links_skips_relationship_dense_records(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    dense = repository.create_memory(
        title="Graph linker retry policy",
        content="Dense memory used for relationship guardrail testing",
        workspace_ids=["workspace-a"],
        memory_type="plan",
        tags=["graph-linker"],
    )
    candidate = repository.create_memory(
        title="Graph linker retry strategy",
        content="Candidate memory for fallback auto-link testing",
        workspace_ids=["workspace-a"],
        memory_type="plan",
        tags=["graph-linker"],
    )
    assert dense is not None
    assert candidate is not None

    for index in range(9):
        related = repository.create_memory(
            title=f"Related graph linker note {index}",
            content="Related memory",
            workspace_ids=["workspace-a"],
            memory_type="fact",
            tags=["graph-linker"],
        )
        assert related is not None
        repository.add_link(dense.id, related.id, "DEPENDS_ON", "Existing relationship")

    proposals = _fallback_graph_links(
        ApplicationContext(repository=repository),
        [dense, candidate],
    )

    assert proposals == []


def test_fallback_graph_links_keeps_high_confidence_title_similarity(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    newer = repository.create_memory(
        title="Graph linker retry strategy",
        content="Fallback linker review",
        workspace_ids=["workspace-a"],
        memory_type="plan",
        tags=[],
        updated_at="2026-01-02T00:00:00+00:00",
    )
    older = repository.create_memory(
        title="Graph linker retry policy",
        content="Fallback linker policy",
        workspace_ids=["workspace-a"],
        memory_type="plan",
        tags=[],
        updated_at="2026-01-01T00:00:00+00:00",
    )
    assert newer is not None
    assert older is not None

    proposals = _fallback_graph_links(
        ApplicationContext(repository=repository),
        [older, newer],
    )

    assert len(proposals) == 1
    assert proposals[0][0:3] == (newer.id, older.id, "AMENDS")
    assert proposals[0][3] == "Auto-linked from title similarity."