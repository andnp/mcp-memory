import json
from types import SimpleNamespace

from mcp_memory.application.payloads import (
    build_read_payload,
    build_search_result_payloads,
    MAX_SEARCH_EVIDENCE_EXCERPT_CHARS,
)
from mcp_memory.core.ports.memory import MemoryLink, MemoryRecord
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


def _raw_search_result(
    *,
    chunks: object = (),
    strategies: object = ("keyword", "vector"),
    normalized_score: object = 0.87654321,
    community_boost: object = 1.25,
    project_uplift: object = 1.1,
) -> SimpleNamespace:
    return SimpleNamespace(
        record=SimpleNamespace(source_id="id-1"),
        normalized_score=normalized_score,
        provenance=SimpleNamespace(
            strategies=strategies,
            community_boost=community_boost,
            project_uplift=project_uplift,
        ),
        chunk_matches=chunks,
    )


def _evidence(payload: dict[str, object]) -> dict[str, object]:
    ranking_debug = payload["ranking_debug"]
    assert isinstance(ranking_debug, dict)
    evidence = ranking_debug["evidence"]
    assert isinstance(evidence, dict)
    return evidence


def test_compact_search_payload_omits_opt_in_evidence() -> None:
    """Keep default search results summary-first without debug evidence."""
    result = RelationalSearchResult(
        memory_id="id-1",
        title="Title",
        summary="Summary",
        memory_type="fact",
        status="active",
        ranking_debug={"source": "test"},
    )

    payload = build_search_result_payloads(
        [result],
        debug_enabled=False,
        raw_results=[_raw_search_result()],
    )[0]

    assert payload == {
        "memory_ref": "id-1",
        "title": "Title",
        "summary": "Summary",
        "status": "active",
        "created_at": None,
        "updated_at": None,
    }


def test_debug_search_payload_includes_bounded_evidence() -> None:
    """Expose deterministic lanes, scores, adjustments, and three short excerpts."""
    result = RelationalSearchResult(
        memory_id="id-1",
        title="Title",
        summary="Summary",
        memory_type="fact",
        status="active",
        ranking_debug={"source": "test"},
    )
    chunks = [
        SimpleNamespace(score=0.2, content=" low "),
        SimpleNamespace(score=0.9, content="high"),
        SimpleNamespace(score=0.5, content="middle"),
        SimpleNamespace(score=0.8, content="upper"),
    ]

    payload = build_search_result_payloads(
        [result],
        debug_enabled=True,
        raw_results=[_raw_search_result(chunks=chunks)],
    )[0]

    assert _evidence(payload) == {
        "lanes": ["keyword", "semantic"],
        "normalized_score": 0.876543,
        "score_adjustments": {"community_boost": 1.25, "project_uplift": 1.1},
        "excerpts": [
            {"content": "high", "score": 0.9},
            {"content": "upper", "score": 0.8},
            {"content": "middle", "score": 0.5},
        ],
    }


def test_debug_search_payload_redacts_malformed_chunk_details() -> None:
    """Ignore malformed chunks and provider metadata while retaining safe lanes."""
    result = RelationalSearchResult(
        memory_id="id-1",
        title="Title",
        summary="Summary",
        memory_type="fact",
        status="active",
        ranking_debug={"source": "test"},
    )
    raw = _raw_search_result(
        chunks=[
            SimpleNamespace(
                content={"not": "text"},
                score=0.9,
                chunk_id="private-chunk-id",
                parent_content="private parent content",
                metadata={"secret": "private metadata"},
            ),
            SimpleNamespace(content="safe", score=float("nan")),
            "not-a-chunk",
        ],
        strategies=("keyword", "provider-private"),
        normalized_score=float("nan"),
        community_boost=float("nan"),
        project_uplift=101.0,
    )

    evidence = _evidence(
        build_search_result_payloads([result], debug_enabled=True, raw_results=[raw])[0]
    )

    assert evidence == {"lanes": ["keyword"], "excerpts": [{"content": "safe"}]}
    assert "private" not in json.dumps(evidence)


def test_debug_search_payload_omits_unavailable_chunk_evidence() -> None:
    """Represent missing SearchKernel chunks as unavailable rather than empty data."""
    result = RelationalSearchResult(
        memory_id="id-1",
        title="Title",
        summary="Summary",
        memory_type="fact",
        status="active",
        ranking_debug={"source": "test"},
    )

    payload = build_search_result_payloads(
        [result],
        debug_enabled=True,
        raw_results=[
            _raw_search_result(
                chunks=None,
                strategies=("provider-private",),
                normalized_score=float("nan"),
                community_boost=None,
                project_uplift=None,
            )
        ],
    )[0]

    assert payload["ranking_debug"] == {"source": "test"}


def test_debug_search_payload_serialization_is_deterministic() -> None:
    """Serialize equivalent evidence identically despite provider input ordering."""
    result = RelationalSearchResult(
        memory_id="id-1",
        title="Title",
        summary="Summary",
        memory_type="fact",
        status="active",
        ranking_debug={"source": "test"},
    )
    raw_a = _raw_search_result(
        chunks=[SimpleNamespace(score=0.2, content="b"), SimpleNamespace(score=0.8, content="a")],
        strategies=("vector", "keyword"),
    )
    raw_b = _raw_search_result(
        chunks=[SimpleNamespace(score=0.8, content="a"), SimpleNamespace(score=0.2, content="b")],
        strategies=("keyword", "vector"),
    )

    payload_a = build_search_result_payloads(
        [result], debug_enabled=True, raw_results=[raw_a]
    )
    payload_b = build_search_result_payloads(
        [result], debug_enabled=True, raw_results=[raw_b]
    )

    assert json.dumps(payload_a, sort_keys=True) == json.dumps(payload_b, sort_keys=True)
    evidence = _evidence(payload_a[0])
    excerpts = evidence["excerpts"]
    assert isinstance(excerpts, list)
    assert len(excerpts[0]["content"]) <= MAX_SEARCH_EVIDENCE_EXCERPT_CHARS


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
