"""Mutation observations accepted by the independent quality evaluator."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Mapping
from uuid import UUID, NAMESPACE_URL, uuid5


@dataclass(frozen=True, slots=True)
class CurationQualityMutation:
    """Provider-neutral mutation evidence for one quality observation."""

    mutation_id: UUID
    operation: str
    affected_memory_ids: tuple[UUID, ...] = ()
    applied: bool = True
    verified: bool = False
    mutation_event_id: UUID | None = None
    applied_at: datetime | None = None
    payload: Mapping[str, Any] = field(default_factory=dict)
    before_entities: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    after_entities: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    structural_deltas: tuple[Mapping[str, Any], ...] = ()
    affected_memory_clusters: tuple[tuple[UUID, ...], ...] = ()
    evidence_id: str | None = None
    action_identity_required: bool = False


def mutation_from_receipt(receipt: Any) -> CurationQualityMutation:
    """Adapt a typed receipt without making the evaluator receipt-dependent."""
    applied_at = getattr(receipt, "applied_at", None)
    status = getattr(receipt, "status", "")
    status_value = str(getattr(status, "value", status))
    return CurationQualityMutation(
        mutation_id=UUID(str(receipt.action_id)),
        operation=str(receipt.operation),
        affected_memory_ids=tuple(UUID(str(value)) for value in receipt.affected_ids),
        applied=status_value == "verified",
        verified=status_value == "verified",
        mutation_event_id=_optional_uuid(getattr(receipt, "mutation_event_id", None)),
        applied_at=_as_datetime(applied_at),
    )


def mutations_from_direct_evidence(evidence: Any) -> CurationQualityMutation:
    """Adapt persisted direct evidence while retaining its independent fields."""
    evidence_id = _optional_identity(getattr(evidence, "evidence_id", None))
    payload = getattr(evidence, "payload", {})
    if not isinstance(payload, Mapping):
        payload = {}
    deltas: tuple[dict[str, Any], ...] = tuple(
        {
            "kind": str(getattr(delta, "kind", "")),
            "entity_id": str(getattr(delta, "entity_id", "")),
            "before_exists": bool(getattr(delta, "before_exists", False)),
            "after_exists": bool(getattr(delta, "after_exists", False)),
            "transition": str(getattr(delta, "transition", "updated")),
            "before_revision": getattr(delta, "before_revision", None),
            "after_revision": getattr(delta, "after_revision", None),
            "snapshot": dict(getattr(delta, "snapshot", {}) or {}),
        }
        for delta in getattr(evidence, "deltas", ())
    )
    before_entities = payload.get("before_entities", {})
    raw_clusters = payload.get("affected_memory_clusters", ())
    clusters = tuple(
        tuple(UUID(str(value)) for value in cluster if _is_uuid(str(value)))
        for cluster in raw_clusters
        if isinstance(cluster, (list, tuple))
    ) if isinstance(raw_clusters, (list, tuple)) else ()
    after_entities: dict[str, Mapping[str, Any]] = {
        str(delta["entity_id"]): delta["snapshot"]
        for delta in deltas
        if delta["kind"] == "record" and isinstance(delta["snapshot"], Mapping)
    }
    return CurationQualityMutation(
        mutation_id=direct_action_id(evidence_id) if evidence_id else _legacy_direct_action_id(evidence),
        operation=str(getattr(evidence, "operation", "unknown")),
        affected_memory_ids=tuple(
            UUID(str(delta["entity_id"]))
            for delta in deltas
            if delta["kind"] == "record" and _is_uuid(str(delta["entity_id"]))
        ),
        applied=str(getattr(evidence, "outcome", ""))
        in {"applied_verified", "applied_unverified"},
        verified=str(getattr(evidence, "outcome", "")) == "applied_verified",
        applied_at=_as_datetime(getattr(evidence, "completed_at", None)),
        payload=payload,
        before_entities=(
            {str(key): dict(value) for key, value in before_entities.items() if isinstance(value, Mapping)}
            if isinstance(before_entities, Mapping)
            else {}
        ),
        after_entities=after_entities,
        structural_deltas=deltas,
        affected_memory_clusters=clusters,
        evidence_id=evidence_id,
        action_identity_required=True,
    )


def direct_action_id(evidence_id: str) -> UUID:
    """Return the stable action identity bridged from direct evidence."""
    return uuid5(NAMESPACE_URL, f"mcp-memory:direct-quality:{evidence_id}")


def _legacy_direct_action_id(evidence: Any) -> UUID:
    identity = ":".join(
        str(getattr(evidence, field, ""))
        for field in (
            "task_id",
            "execution_epoch",
            "session_id",
            "call_id",
            "sequence",
            "tool_name",
            "operation",
            "idempotency_key",
        )
    )
    return uuid5(NAMESPACE_URL, f"mcp-memory:direct-quality:missing:{identity}")


def _optional_uuid(value: Any) -> UUID | None:
    if value is None:
        return None
    try:
        return UUID(str(value))
    except (TypeError, ValueError):
        return None


def _optional_identity(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _is_uuid(value: str) -> bool:
    return _optional_uuid(value) is not None


def _as_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
    return None
