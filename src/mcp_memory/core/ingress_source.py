"""Pure canonical source snapshots for ingress replay evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast
from unicodedata import normalize
from uuid import UUID

from mcp_memory.core.curation_identity import canonical_json, canonical_token
from mcp_memory.core.ingress_identity import INGRESS_IDENTITY_VERSION, SourceEntryIdentity


class SourceSnapshotTooLargeError(ValueError):
    """Raised when a complete source snapshot exceeds its configured bound."""


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, (str, UUID)):
        raise TypeError(f"{name} must be a string or UUID")
    result = normalize("NFC", str(value)).strip()
    if not result:
        raise ValueError(f"{name} must be non-empty")
    return result


def _workspace_ids(value: object) -> list[str]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError("workspace_ids must be a sequence")
    return sorted({_text(item, name="workspace_id") for item in value}, key=lambda item: item.encode("utf-8"))


def _entry_values(entry: SourceEntryIdentity | Mapping[str, object]) -> tuple[object, object, object, object]:
    if isinstance(entry, SourceEntryIdentity):
        return entry.entry_id, entry.timestamp, entry.workspace_ids, entry.content_digest
    return (
        entry.get("entry_id"),
        entry.get("timestamp"),
        entry.get("workspace_ids", ()),
        entry.get("content_digest"),
    )


def normalize_source_snapshot(
    source: Mapping[str, object] | Sequence[SourceEntryIdentity | Mapping[str, object]],
) -> dict[str, object]:
    """Return the complete, deterministic, JSON-shaped source snapshot."""
    if isinstance(source, Mapping):
        version = source.get("schema_version", INGRESS_IDENTITY_VERSION)
        entries = source.get("entries")
        if isinstance(entries, (str, bytes, bytearray)) or not isinstance(entries, Sequence):
            raise TypeError("source snapshot entries must be a sequence")
    else:
        version = INGRESS_IDENTITY_VERSION
        entries = source
    if isinstance(version, bool) or version != INGRESS_IDENTITY_VERSION:
        raise ValueError(f"unsupported source snapshot schema_version: {version!r}")

    normalized_entries: list[dict[str, object]] = []
    seen: set[str] = set()
    for entry in entries:
        entry_id, timestamp, workspace_ids, content_digest = _entry_values(entry)
        normalized_id = _text(entry_id, name="entry_id")
        if normalized_id in seen:
            raise ValueError(f"duplicate source entry: {normalized_id}")
        seen.add(normalized_id)
        normalized_timestamp = _text(timestamp, name="source entry timestamp")
        normalized_digest = _text(content_digest, name="source entry content_digest")
        normalized_entries.append(
            {
                "entry_id": normalized_id,
                "timestamp": normalized_timestamp,
                "workspace_ids": _workspace_ids(workspace_ids),
                "content_digest": normalized_digest,
            }
        )
    normalized_entries.sort(key=lambda item: cast(str, item["entry_id"]).encode("utf-8"))
    return {"schema_version": INGRESS_IDENTITY_VERSION, "entries": normalized_entries}


def serialize_source_snapshot(
    source: Mapping[str, object] | Sequence[SourceEntryIdentity | Mapping[str, object]],
    *,
    max_bytes: int | None = None,
) -> bytes:
    """Serialize a normalized source snapshot, rejecting oversized evidence."""
    serialized = canonical_json(normalize_source_snapshot(source))
    if max_bytes is not None:
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 0:
            raise ValueError("max_bytes must be a non-negative integer")
        if len(serialized) > max_bytes:
            raise SourceSnapshotTooLargeError(
                f"source snapshot is {len(serialized)} bytes; configured limit is {max_bytes}"
            )
    return serialized


def source_snapshot_fingerprint(
    source: Mapping[str, object] | Sequence[SourceEntryIdentity | Mapping[str, object]],
) -> str:
    """Return the stable identity derived from the normalized snapshot."""
    return canonical_token(normalize_source_snapshot(source))
