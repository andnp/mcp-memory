from uuid import UUID

from mcp_memory.core.curation_identity import (
    action_id,
    canonical_json,
    context_fingerprint,
    frontier_fingerprint,
    graph_token,
    record_token,
)
from mcp_memory.core.curation_models import NormalizeMemoryAction

MEMORY_ID = UUID("00000000-0000-0000-0000-000000000001")
OTHER_ID = UUID("00000000-0000-0000-0000-000000000002")


def test_canonical_json_normalizes_unicode_newlines_and_maps() -> None:
    assert canonical_json({"b": "e\u0301\r", "a": "x"}) == b'{"a":"x","b":"\xc3\xa9\\n"}'


def test_record_token_sorts_semantic_sets_and_excludes_access_telemetry() -> None:
    record = {
        "id": MEMORY_ID,
        "title": "Title",
        "content": "Content",
        "summary": None,
        "type": " observation ",
        "status": "active",
        "tags": ["b", "a", "a"],
        "workspace_ids": [OTHER_ID, MEMORY_ID],
        "read_count": 1,
        "last_accessed_at": "before",
    }
    changed_telemetry = {**record, "read_count": 99, "last_accessed_at": "after"}
    assert record_token(record) == record_token(changed_telemetry)
    assert record_token(record) == record_token({**record, "tags": ["a", "b"]})
    assert record_token(record) != record_token({**record, "content": "Changed"})


def test_record_token_includes_durable_metadata_from_metadata() -> None:
    record = {
        "id": MEMORY_ID,
        "title": "Title",
        "content": "Content",
        "summary": None,
        "type": "observation",
        "status": "active",
        "metadata": {"lineage": {"parent_id": "parent"}, "mutation_metadata": {"reason": "initial"}},
    }
    assert record_token(record) != record_token(
        {**record, "metadata": {**record["metadata"], "lineage": {"parent_id": "other"}}}
    )
    assert record_token(record) != record_token(
        {**record, "metadata": {**record["metadata"], "mutation_metadata": {"reason": "changed"}}}
    )


def test_graph_token_deduplicates_and_sorts_edges() -> None:
    edges = [
        {"source_id": OTHER_ID, "target_id": MEMORY_ID, "type": " relates ", "context": "c"},
        {"source_id": OTHER_ID, "target_id": MEMORY_ID, "type": "relates", "context": "c"},
    ]
    assert graph_token(MEMORY_ID, edges) == graph_token(MEMORY_ID, list(reversed(edges)))
    assert graph_token(MEMORY_ID, edges) != graph_token(MEMORY_ID, [{**edges[0], "context": "different"}])


def test_frontier_and_context_fingerprints_have_distinct_inputs() -> None:
    assert frontier_fingerprint("curator", "recent", [OTHER_ID, MEMORY_ID]) == frontier_fingerprint(
        "curator", "recent", [MEMORY_ID, OTHER_ID, MEMORY_ID]
    )
    packet = {"frontier_fingerprint": "v1:x", "seeds": ["a"], "support": ["b"], "limits": {"records": 2}}
    assert context_fingerprint({**packet, "request_id": "volatile"}) == context_fingerprint(packet)
    assert context_fingerprint({**packet, "support": ["c"]}) != context_fingerprint(packet)


def test_action_id_changes_with_intent_but_not_advisory_fields() -> None:
    action = NormalizeMemoryAction(
        action_id=UUID(int=9), target_id=MEMORY_ID, confidence=0.2, rationale="first", title="Title"
    )
    revised = action.model_copy(update={"confidence": 0.9, "rationale": "different"})
    changed = action.model_copy(update={"title": "Changed"})
    assert action_id(MEMORY_ID, 0, action) == action_id(MEMORY_ID, 0, revised)
    assert action_id(MEMORY_ID, 0, action) != action_id(MEMORY_ID, 0, changed)
    assert action_id(MEMORY_ID, 0, action) != action_id(MEMORY_ID, 1, action)
