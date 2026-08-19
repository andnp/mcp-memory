from __future__ import annotations

import pytest

from mcp_memory.relational.record_helpers import (
    NormalizedMemoryRecord,
    hydrate_memory_record,
    normalize_link_type,
    normalize_values,
    serialize_metadata,
    validate_memory_status,
    validate_memory_type,
    validate_required_text,
)

pytestmark = pytest.mark.small


def test_record_helpers_normalize_repository_values() -> None:
    """Preserve ordering while removing blank and duplicate persistence values."""
    assert normalize_values([" alpha ", "", "alpha", " beta "]) == ["alpha", "beta"]
    assert normalize_link_type(" Depends-On ") == "DEPENDS_ON"
    assert serialize_metadata({"priority": "high", "phase": 2}) == '{"phase": 2, "priority": "high"}'
    assert validate_memory_type(" fact ") == "fact"
    assert validate_memory_status(" archived ") == "archived"

    with pytest.raises(ValueError, match="title must be non-empty"):
        validate_required_text("title", "")


def test_hydrate_memory_record_preserves_persisted_record_fields() -> None:
    """Build the public record shape from all normalized persistence fields."""
    record = hydrate_memory_record(
        NormalizedMemoryRecord(
            id="memory-id",
            title="Archived fact",
            content="The persisted content.",
            summary="The persisted summary.",
            type="fact",
            status="archived",
            created_at="2026-08-18T00:00:00+00:00",
            updated_at="2026-08-18T01:00:00+00:00",
            read_count=3,
            access_score=1.5,
            last_accessed_at="2026-08-18T02:00:00+00:00",
            last_surfaced_at="2026-08-18T03:00:00+00:00",
            metadata={"source": "test"},
            workspace_ids=["workspace-a", "workspace-b"],
            tags=["archived", "fact"],
            memory_ref=7,
            archived_at="2026-08-18T04:00:00+00:00",
        )
    )

    assert record.id == "memory-id"
    assert record.memory_ref == 7
    assert record.archived_at == "2026-08-18T04:00:00+00:00"
    assert record.metadata == {"source": "test"}
    assert record.workspace_ids == ["workspace-a", "workspace-b"]
    assert record.tags == ["archived", "fact"]
