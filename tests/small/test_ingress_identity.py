from uuid import UUID

import pytest

from mcp_memory.core.ingress_identity import (
    PayloadComparison,
    SourceEntryIdentity,
    action_identity,
    batch_id,
    compare_action_payload,
    normalize_execution_identity,
    quality_identity,
    source_fingerprint,
)


def test_execution_identity_normalizes_identifiers_and_rejects_invalid_epochs() -> None:
    """Normalize task identity while rejecting an invalid execution epoch."""
    identity = normalize_execution_identity("  ta\u0301sk-1 ", 0)

    assert identity.task_id == "tásk-1"
    assert normalize_execution_identity(UUID(int=1), 2).task_id == "00000000-0000-0000-0000-000000000001"
    with pytest.raises(ValueError, match="non-negative"):
        normalize_execution_identity("task-1", -1)


def test_batch_identity_is_stable_for_retry_order_and_execution_changes() -> None:
    """Keep a batch identity stable across entry ordering and retries."""
    entries = [
        SourceEntryIdentity("entry-b", "2026-01-02T00:00:00Z", ("workspace-b",), "digest-b"),
        SourceEntryIdentity("entry-a", "2026-01-01T00:00:00Z", ("workspace-a",), "digest-a"),
    ]
    fingerprint = source_fingerprint(list(reversed(entries)))

    assert batch_id(["entry-b", "entry-a", "entry-a"], fingerprint) == batch_id(
        ["entry-a", "entry-b"], fingerprint
    )


def test_action_identity_excludes_payload_batch_and_execution_epoch() -> None:
    """Keep action identity independent of execution and batch metadata."""
    first = action_identity(" append ", ["entry-b", "entry-a"], ["target-b", "target-a"])
    retry = action_identity("append", ["entry-a", "entry-b", "entry-a"], ["target-a", "target-b"])

    assert first == retry
    assert first.action_id != action_identity("create", first.entry_ids, first.target_ids).action_id


def test_divergent_payload_returns_collision_metadata_without_new_action_id() -> None:
    """Report divergent payloads as receipt collisions for one action."""
    identity = action_identity("create", ["entry-1"], [])
    stored = compare_action_payload(identity, {"title": "Original"}, "v1:stored")
    replay = compare_action_payload(identity, {"title": "Changed"}, stored.requested_payload_digest)

    assert replay.action_id == identity.action_id
    assert replay.result is PayloadComparison.COLLISION
    assert replay.stored_payload_digest != replay.requested_payload_digest


def test_quality_identity_changes_with_evidence_or_action() -> None:
    """Key quality identity by mutation evidence and logical action."""
    action = action_identity("append", ["entry-1"], ["memory-1"])

    first = quality_identity("evidence-1", action.action_id)
    retry = quality_identity("evidence-1", action.action_id)

    assert first == retry
    assert first.quality_id != quality_identity("evidence-2", action.action_id).quality_id
    assert first.quality_id != quality_identity("evidence-1", action_identity("create", ["entry-1"], []).action_id).quality_id
