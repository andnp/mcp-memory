from mcp_memory.application.payloads import (
    build_read_payload,
    build_search_result_payloads,
)
from mcp_memory.core.ports.memory import MemoryLink, MemoryRecord
from mcp_memory.mcp.payloads import (
    build_read_payload as compatibility_build_read_payload,
    build_search_result_payloads as compatibility_build_search_result_payloads,
)
from mcp_memory.relational.search import RelationalReadResult, RelationalSearchResult


def _record(*, memory_id: str = "id-1", memory_ref: int = 7) -> MemoryRecord:
    return MemoryRecord(
        id=memory_id,
        title="Title",
        content="Content",
        summary="Summary",
        type="fact",
        status="active",
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-02T00:00:00+00:00",
        read_count=1,
        access_score=0.5,
        last_accessed_at=None,
        last_surfaced_at=None,
        metadata={"source": "test"},
        workspace_ids=["workspace-1"],
        tags=["tag-1"],
        memory_ref=memory_ref,
    )


def test_build_search_result_payloads_preserves_compact_and_debug_shapes() -> None:
    result = RelationalSearchResult(
        memory_id="id-1",
        memory_ref=7,
        title="Title",
        summary="Summary",
        memory_type="fact",
        status="active",
        tags=["tag-1"],
        workspace_ids=["workspace-1"],
        score=0.9,
        ranking_debug={"source": "test"},
    )

    assert build_search_result_payloads([result], debug_enabled=False) == [
        {
            "memory_ref": "mem-7",
            "title": "Title",
            "summary": "Summary",
            "status": "active",
            "created_at": None,
            "updated_at": None,
        }
    ]
    assert build_search_result_payloads([result], debug_enabled=True) == [
        {
            "memory_id": "id-1",
            "memory_ref": "mem-7",
            "title": "Title",
            "summary": "Summary",
            "memory_type": "fact",
            "status": "active",
            "created_at": None,
            "updated_at": None,
            "tags": ["tag-1"],
            "workspace_ids": ["workspace-1"],
            "score": 0.9,
            "ranking_debug": {"source": "test"},
        }
    ]
    assert compatibility_build_search_result_payloads is build_search_result_payloads


def test_search_result_payloads_expose_temporal_metadata() -> None:
    result = RelationalSearchResult(
        memory_id="id-1",
        title="Title",
        summary="Summary",
        memory_type="fact",
        status="active",
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-02T00:00:00+00:00",
    )

    compact = build_search_result_payloads([result], debug_enabled=False)[0]
    full = build_search_result_payloads([result], debug_enabled=True)[0]

    assert compact["status"] == full["status"] == "active"
    assert compact["created_at"] == full["created_at"] == "2026-01-01T00:00:00+00:00"
    assert compact["updated_at"] == full["updated_at"] == "2026-01-02T00:00:00+00:00"


def test_build_read_payload_applies_requested_relationship_and_metadata_flags() -> None:
    result = RelationalReadResult(
        record=_record(),
        relationships={
            "outgoing": [MemoryLink("id-1", "id-2", "supports", "context")],
            "incoming": [],
        },
        superseded=[_record(memory_id="id-0", memory_ref=6)],
    )

    assert build_read_payload(
        result,
        include_relationships=True,
        include_superseded=True,
        include_metadata=True,
    ) == {
        "status": "ok",
        "record": {
            "memory_ref": "mem-7",
            "title": "Title",
            "content": "Content",
            "metadata": {"source": "test"},
            "workspace_ids": ["workspace-1"],
        },
        "relationships": {
            "outgoing": [
                {
                    "source_id": "id-1",
                    "target_id": "id-2",
                    "link_type": "supports",
                    "context": "context",
                }
            ],
            "incoming": [],
        },
        "superseded": [
            {
                "memory_ref": "mem-6",
                "title": "Title",
                "content": "Content",
                "metadata": {"source": "test"},
                "workspace_ids": ["workspace-1"],
            }
        ],
    }
    assert compatibility_build_read_payload is build_read_payload
