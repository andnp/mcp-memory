from __future__ import annotations

import json

import pytest

from mcp_memory.core.ingress_identity import SourceEntryIdentity, source_fingerprint
from mcp_memory.core.ingress_source import (
    SourceSnapshotTooLargeError,
    normalize_source_snapshot,
    serialize_source_snapshot,
    source_snapshot_fingerprint,
)


def _entry(entry_id: str, timestamp: str = "2026-01-01T00:00:00Z") -> SourceEntryIdentity:
    return SourceEntryIdentity(entry_id, timestamp, ("workspace-b", "workspace-a"), "digest")


def test_normalization_orders_entries_and_workspace_ids() -> None:
    """Canonical snapshots use bytewise ordering independent of input order."""
    snapshot = normalize_source_snapshot([_entry("b"), _entry("a")])

    assert snapshot["entries"] == [
        {"entry_id": "a", "timestamp": "2026-01-01T00:00:00Z", "workspace_ids": ["workspace-a", "workspace-b"], "content_digest": "digest"},
        {"entry_id": "b", "timestamp": "2026-01-01T00:00:00Z", "workspace_ids": ["workspace-a", "workspace-b"], "content_digest": "digest"},
    ]


def test_normalization_rejects_duplicate_ids_after_normalization() -> None:
    """Distinct input spellings cannot bypass source-entry uniqueness."""
    with pytest.raises(ValueError, match="duplicate source entry"):
        normalize_source_snapshot([_entry("entry"), _entry(" entry ")])


def test_normalization_applies_unicode_nfc() -> None:
    """Equivalent decomposed and composed Unicode forms produce one snapshot."""
    composed = normalize_source_snapshot([_entry("café", "é")])
    decomposed = normalize_source_snapshot([_entry("cafe\u0301", "e\u0301")])

    assert composed == decomposed
    assert source_snapshot_fingerprint(composed) == source_snapshot_fingerprint(decomposed)


def test_snapshot_normalization_round_trips_through_canonical_json() -> None:
    """Parsing canonical bytes preserves the normalized snapshot exactly."""
    snapshot = normalize_source_snapshot([_entry("entry")])

    assert normalize_source_snapshot(json.loads(serialize_source_snapshot(snapshot))) == snapshot


def test_serialization_rejects_oversized_snapshot_explicitly() -> None:
    """A size bound fails closed instead of claiming replayable evidence."""
    snapshot = normalize_source_snapshot([_entry("entry")])

    with pytest.raises(SourceSnapshotTooLargeError, match="configured limit"):
        serialize_source_snapshot(snapshot, max_bytes=len(serialize_source_snapshot(snapshot)) - 1)


def test_snapshot_fingerprint_reuses_existing_source_identity_contract() -> None:
    """The new complete snapshot identity matches the established ingress hash."""
    entries = [_entry("b"), _entry("a")]

    assert source_snapshot_fingerprint(entries) == source_fingerprint(entries)
