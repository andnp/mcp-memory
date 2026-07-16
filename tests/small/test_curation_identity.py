from uuid import UUID

from mcp_memory.core.curation_identity import (
    action_id,
    action_intent_token,
    canonical_json,
    context_fingerprint,
    frontier_fingerprint,
    graph_token,
    record_token,
)
from mcp_memory.core.curation_models import ActionPreconditions, NormalizeMemoryAction

MEMORY_ID = UUID("00000000-0000-0000-0000-000000000001")
OTHER_ID = UUID("00000000-0000-0000-0000-000000000002")


def test_action_identity_normalizes_set_like_permutations_but_not_positional_lists() -> None:
    first = {
        "operation": "merge_memories",
        "action_id": str(UUID(int=3)),
        "confidence": 0.8,
        "rationale": "advisory",
        "canonical_id": str(UUID(int=4)),
        "source_ids": [str(UUID(int=2)), str(UUID(int=1))],
        "tags": ["tag-b", "tag-a"],
        "evidence": [{"memory_id": str(UUID(int=6))}, {"memory_id": str(UUID(int=5))}],
        "preconditions": {
            "required_statuses": {str(UUID(int=2)): "active", str(UUID(int=1)): "active"},
            "required_links": [
                {"source_id": str(UUID(int=2)), "target_id": str(UUID(int=1)), "link_type": "related"},
                {"source_id": str(UUID(int=1)), "target_id": str(UUID(int=2)), "link_type": "related"},
            ],
            "absent_links": [{"source_id": str(UUID(int=1)), "target_id": str(UUID(int=2)), "link_type": "blocked"}],
        },
    }
    second = {**first, "source_ids": list(reversed(first["source_ids"])), "tags": list(reversed(first["tags"]))}
    second["evidence"] = list(reversed(first["evidence"]))
    second["preconditions"] = {**first["preconditions"], "required_links": list(reversed(first["preconditions"]["required_links"]))}
    assert action_id("plan", 0, first) == action_id("plan", 0, second)

    ordered = {**first, "claim_manifest": {"preserved_claims": ["one", "two"]}}
    reordered = {**ordered, "claim_manifest": {"preserved_claims": ["two", "one"]}}
    assert action_id("plan", 0, ordered) != action_id("plan", 0, reordered)


def test_intent_hash_normalizes_set_like_permutations_and_detects_changes() -> None:
    common = {
        "operation": "rewrite_memory",
        "target_ids": ["b", "a"],
        "expected_tokens": {"b": "token-b", "a": "token-a"},
        "preconditions": {
            "required_statuses": {"b": "active", "a": "active"},
            "absent_links": [
                {"source_id": "b", "target_id": "a", "link_type": "related"},
                {"source_id": "a", "target_id": "b", "link_type": "related"},
            ],
        },
        "payload": {"tags": ["b", "a"]},
    }
    permutation = {**common, "target_ids": ["a", "b"], "expected_tokens": {"a": "token-a", "b": "token-b"}}
    permutation["preconditions"] = {**common["preconditions"], "absent_links": list(reversed(common["preconditions"]["absent_links"]))}
    permutation["payload"] = {"tags": ["a", "b"]}
    assert action_intent_token(**common) == action_intent_token(**permutation)
    assert action_intent_token(**common) != action_intent_token(**{**permutation, "payload": {"tags": ["a", "c"]}})


def test_intent_hash_canonicalizes_typed_preconditions() -> None:
    preconditions = ActionPreconditions(record_tokens={MEMORY_ID: "token"})
    assert action_intent_token(
        operation="normalize_memory",
        target_ids=[str(MEMORY_ID)],
        expected_tokens={str(MEMORY_ID): "token"},
        preconditions=preconditions,
        payload={"summary": "specific"},
    ) == action_intent_token(
        operation="normalize_memory",
        target_ids=[str(MEMORY_ID)],
        expected_tokens={str(MEMORY_ID): "token"},
        preconditions=preconditions.model_dump(mode="json"),
        payload={"summary": "specific"},
    )


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
