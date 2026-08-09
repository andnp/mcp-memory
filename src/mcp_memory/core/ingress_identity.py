"""Pure identities for the ingress execution, mutation, and quality boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from collections.abc import Sequence
from unicodedata import normalize
from uuid import UUID

from mcp_memory.core.curation_identity import canonical_token

INGRESS_IDENTITY_VERSION = 1


def _identifier(value: str | UUID, *, name: str) -> str:
    normalized = normalize("NFC", str(value)).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty")
    return normalized


def canonical_ids(values: Sequence[str | UUID]) -> tuple[str, ...]:
    """Normalize, deduplicate, and sort a set-like identity collection."""
    result = {_identifier(value, name="identity") for value in values}
    return tuple(sorted(result, key=lambda value: value.encode("utf-8")))


@dataclass(frozen=True, slots=True)
class ExecutionIdentity:
    """The worker-owned identity for one task execution attempt."""

    task_id: str
    execution_epoch: int


def normalize_execution_identity(task_id: str | UUID, execution_epoch: int) -> ExecutionIdentity:
    """Validate and normalize the execution identity carried by ingress evidence."""
    if isinstance(execution_epoch, bool) or not isinstance(execution_epoch, int):
        raise TypeError("execution_epoch must be an integer")
    if execution_epoch < 0:
        raise ValueError("execution_epoch must be non-negative")
    return ExecutionIdentity(_identifier(task_id, name="task_id"), execution_epoch)


@dataclass(frozen=True, slots=True)
class SourceEntryIdentity:
    """Replay-relevant source fields used to calculate a batch fingerprint."""

    entry_id: str | UUID
    timestamp: str
    workspace_ids: tuple[str | UUID, ...] = ()
    content_digest: str = ""


def source_fingerprint(entries: Sequence[SourceEntryIdentity]) -> str:
    """Return a stable fingerprint for the immutable source-entry snapshot."""
    normalized: list[dict[str, object]] = []
    seen: set[str] = set()
    for entry in entries:
        entry_id = _identifier(entry.entry_id, name="entry_id")
        if entry_id in seen:
            raise ValueError(f"duplicate source entry: {entry_id}")
        seen.add(entry_id)
        timestamp = normalize("NFC", entry.timestamp).strip()
        content_digest = normalize("NFC", entry.content_digest).strip()
        if not timestamp:
            raise ValueError("source entry timestamp must be non-empty")
        if not content_digest:
            raise ValueError("source entry content_digest must be non-empty")
        normalized.append(
            {
                "entry_id": entry_id,
                "timestamp": timestamp,
                "workspace_ids": canonical_ids(list(entry.workspace_ids)),
                "content_digest": content_digest,
            }
        )
    normalized.sort(key=lambda item: str(item["entry_id"]).encode("utf-8"))
    return canonical_token({"schema_version": INGRESS_IDENTITY_VERSION, "entries": normalized})


def batch_id(
    claimed_entry_ids: Sequence[str | UUID],
    source_fingerprint_value: str,
) -> str:
    """Return the stable batch identity, independent of execution telemetry."""
    return canonical_token(
        {
            "schema_version": INGRESS_IDENTITY_VERSION,
            "claimed_entry_ids": canonical_ids(claimed_entry_ids),
            "source_fingerprint": _identifier(source_fingerprint_value, name="source_fingerprint"),
        }
    )


@dataclass(frozen=True, slots=True)
class ActionIdentity:
    """Stable logical mutation identity; payload is deliberately absent."""

    action_id: str
    operation: str
    entry_ids: tuple[str, ...]
    target_ids: tuple[str, ...]


def action_identity(
    operation: str,
    entry_ids: Sequence[str | UUID],
    target_ids: Sequence[str | UUID],
) -> ActionIdentity:
    """Build an action identity without epoch, batch, or payload inputs."""
    normalized_operation = _identifier(operation, name="operation")
    normalized_entries = canonical_ids(entry_ids)
    normalized_targets = canonical_ids(target_ids)
    identity = canonical_token(
        {
            "schema_version": INGRESS_IDENTITY_VERSION,
            "operation": normalized_operation,
            "entry_ids": normalized_entries,
            "target_ids": normalized_targets,
        }
    )
    return ActionIdentity(identity, normalized_operation, normalized_entries, normalized_targets)


def canonical_payload_digest(payload: object) -> str:
    """Return receipt collision data for a canonical mutation payload."""
    return canonical_token({"schema_version": INGRESS_IDENTITY_VERSION, "payload": payload})


class PayloadComparison(StrEnum):
    MATCH = "match"
    COLLISION = "collision"


@dataclass(frozen=True, slots=True)
class PayloadCollision:
    """Comparison metadata for reusing an action receipt safely."""

    action_id: str
    stored_payload_digest: str
    requested_payload_digest: str
    result: PayloadComparison


def compare_action_payload(
    identity: ActionIdentity, payload: object, stored_payload_digest: str
) -> PayloadCollision:
    """Describe whether a replay payload matches the existing action receipt."""
    requested = canonical_payload_digest(payload)
    stored = _identifier(stored_payload_digest, name="stored_payload_digest")
    return PayloadCollision(
        action_id=identity.action_id,
        stored_payload_digest=stored,
        requested_payload_digest=requested,
        result=PayloadComparison.MATCH if requested == stored else PayloadComparison.COLLISION,
    )


@dataclass(frozen=True, slots=True)
class QualityIdentity:
    """Identity joining mutation evidence to its logical action."""

    quality_id: str
    mutation_evidence_id: str
    action_id: str


def quality_identity(mutation_evidence_id: str | UUID, action_id_value: str) -> QualityIdentity:
    """Build quality identity from mutation evidence and action identity only."""
    evidence = _identifier(mutation_evidence_id, name="mutation_evidence_id")
    action = _identifier(action_id_value, name="action_id")
    quality = canonical_token(
        {
            "schema_version": INGRESS_IDENTITY_VERSION,
            "mutation_evidence_id": evidence,
            "action_id": action,
        }
    )
    return QualityIdentity(quality, evidence, action)
