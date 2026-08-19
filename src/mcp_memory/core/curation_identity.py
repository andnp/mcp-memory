"""Pure, versioned identities for curation plans and semantic state."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any
from uuid import UUID

from pydantic import BaseModel

from ..utils.canonical_identity import canonicalize, normalize, normalize_text, stable_hash, stable_uuid5
from .curation_models import CurationAction

SCHEMA_VERSION = 1
# Stable namespace; changing this is an identity schema change.
CURATION_ACTION_NAMESPACE = UUID("2f5b4bb8-4f1b-5e2f-8f9d-6a7d8e9c0b1a")

_normalize = normalize


def canonical_json(value: Any) -> bytes:
    """Return the exact UTF-8 bytes used by all curation identities."""
    return canonicalize(value).encode("utf-8")


def canonical_token(value: Any) -> str:
    return stable_hash(value)


def _identifier(value: Any) -> str:
    if not isinstance(value, (str, UUID)):
        raise TypeError("identity identifiers must be strings or UUIDs")
    return normalize_text(str(value), collapse=True)


def _set_values(values: Sequence[Any]) -> list[str]:
    return sorted({_identifier(value) for value in values}, key=lambda item: item.encode("utf-8"))


_ACTION_SET_LIST_KEYS = frozenset(
    {"tags", "evidence", "references", "target_ids", "source_ids", "required_links", "absent_links"}
)
_ACTION_IDENTIFIER_KEYS = frozenset(
    {
        "id",
        "memory_id",
        "source_id",
        "target_id",
        "canonical_id",
        "link_type",
        "status",
        "tags",
        "target_ids",
        "source_ids",
        "references",
    }
)


def _canonical_action_value(value: Any, *, key: str | None = None) -> Any:
    if isinstance(value, BaseModel):
        return _canonical_action_value(value.model_dump(mode="json"), key=key)
    if isinstance(value, str):
        return _identifier(value) if key in _ACTION_IDENTIFIER_KEYS else normalize_text(value)
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, Mapping):
        return {
            normalize_text(str(name)): _canonical_action_value(item, key=normalize_text(str(name)))
            for name, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        items = [_canonical_action_value(item, key=key) for item in value]
        if key in _ACTION_SET_LIST_KEYS:
            unique = {canonical_json(item): item for item in items}
            return [unique[item] for item in sorted(unique)]
        return items
    return _normalize(value)


def canonicalize_action_value(value: Any) -> Any:
    """Normalize action identity values while preserving semantic list order."""
    return _canonical_action_value(value)


def action_intent_token(
    *,
    operation: str,
    target_ids: Sequence[Any],
    expected_tokens: Mapping[Any, Any],
    preconditions: Any | None,
    payload: Any | None,
) -> str:
    return canonical_token(
        canonicalize_action_value(
            {
                "schema_version": SCHEMA_VERSION,
                "operation": operation,
                "target_ids": list(target_ids),
                "expected_tokens": dict(expected_tokens),
                "preconditions": preconditions,
                "payload": payload,
            }
        )
    )


def _mapping_value(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _durable_metadata_value(record: Any, name: str, override: Mapping[str, Any] | None) -> Any:
    if override is not None:
        return override
    metadata = _mapping_value(record, "metadata", {})
    if isinstance(metadata, Mapping):
        return metadata.get(name, {}) or {}
    return {}


def record_snapshot(
    record: Mapping[str, Any] | Any,
    *,
    workspace_ids: Sequence[Any] | None = None,
    lineage: Mapping[str, Any] | None = None,
    mutation_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the allowlisted semantic record snapshot from a mapping or model."""
    tags = _mapping_value(record, "tags", []) or []
    workspaces = workspace_ids if workspace_ids is not None else (_mapping_value(record, "workspace_ids", []) or [])
    snapshot = {
        "schema_version": SCHEMA_VERSION,
        "record": {
            "id": _identifier(_mapping_value(record, "id")),
            "title": _normalize(_mapping_value(record, "title")),
            "content": _normalize(_mapping_value(record, "content")),
            "summary": _normalize(_mapping_value(record, "summary")),
            "type": _identifier(_mapping_value(record, "type")),
            "status": _identifier(_mapping_value(record, "status")),
            "tags": _set_values(tags),
            "workspace_ids": _set_values(workspaces),
            "lineage": _normalize(_durable_metadata_value(record, "lineage", lineage)),
            "mutation_metadata": _normalize(_durable_metadata_value(record, "mutation_metadata", mutation_metadata)),
        },
    }
    return snapshot


def record_token(record: Mapping[str, Any] | Any, **kwargs: Any) -> str:
    return canonical_token(record_snapshot(record, **kwargs))


def graph_snapshot(memory_id: Any, edges: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    normalized: set[tuple[str, str, str, str | None]] = set()
    for edge in edges:
        normalized.add(
            (
                _identifier(edge["source_id"]),
                _identifier(edge["target_id"]),
                _identifier(edge["type"]),
                None if edge.get("context") is None else _normalize(edge["context"]),
            )
        )
    ordered = sorted(normalized, key=lambda item: tuple((part or "").encode("utf-8") for part in item))
    return {
        "schema_version": SCHEMA_VERSION,
        "memory_id": _identifier(memory_id),
        "edges": [
            {"source_id": source, "target_id": target, "type": link_type, "context": context}
            for source, target, link_type, context in ordered
        ],
    }


def graph_token(memory_id: Any, edges: Sequence[Mapping[str, Any]]) -> str:
    return canonical_token(graph_snapshot(memory_id, edges))


def candidate_revision_token(record_revision_token: str, graph_revision_token: str) -> str:
    """Combine semantic record and adjacency identities for candidate state."""
    return canonical_token(
        {
            "record_revision_token": record_revision_token,
            "graph_revision_token": graph_revision_token,
        }
    )


def link_token(source_id: Any, target_id: Any, link_type: Any, context: Any, *, exists: bool = True) -> str:
    """Return the canonical identity of one relationship state."""
    return canonical_token(
        {
            "schema_version": SCHEMA_VERSION,
            "source_id": _identifier(source_id),
            "target_id": _identifier(target_id),
            "type": _identifier(link_type),
            "context": None if context is None else _normalize(str(context)),
            "exists": exists,
        }
    )


def frontier_fingerprint(family: str, strategy: str, seed_ids: Sequence[Any]) -> str:
    return canonical_token(
        {"schema_version": SCHEMA_VERSION, "family": _identifier(family), "strategy": _identifier(strategy), "seed_ids": _set_values(seed_ids)}
    )


def context_fingerprint(packet: Mapping[str, Any]) -> str:
    allowed = {
        "frontier_fingerprint",
        "seeds",
        "support",
        "record_tokens",
        "graph_tokens",
        "disclosure",
        "omissions",
        "limits",
        "campaign_hypothesis",
    }
    return canonical_token(
        {"schema_version": SCHEMA_VERSION, **{key: packet[key] for key in sorted(allowed) if key in packet}}
    )


def action_id(plan_id: Any, position: int, action: CurationAction | Mapping[str, Any]) -> UUID:
    """Derive the action UUID from intent, never from provider advisory fields."""
    data = canonicalize_action_value(action.model_dump(mode="json") if isinstance(action, BaseModel) else dict(action))
    operation = _identifier(data["operation"])
    base = {"action_id", "confidence", "rationale", "operation", "evidence", "preconditions"}
    target_keys = ("target_id", "source_id", "canonical_id", "source_ids")
    target_ids = [data[key] for key in target_keys if key in data and key != "source_ids"]
    if "source_ids" in data:
        target_ids.extend(data["source_ids"])
    arguments = {key: data[key] for key in data if key not in base and key not in target_keys}
    arguments["operation"] = operation
    preconditions = data.get("preconditions", {})
    evidence = data.get("evidence", [])
    name = {
        "schema_version": SCHEMA_VERSION,
        "plan_id": _identifier(plan_id),
        "position": position,
        "operation": operation,
        "target_ids": _set_values(target_ids),
        "arguments": canonicalize_action_value(arguments),
        "preconditions": canonicalize_action_value(preconditions),
        "evidence": canonicalize_action_value(evidence),
    }
    return stable_uuid5(CURATION_ACTION_NAMESPACE, name)


# Descriptive aliases make the contract convenient without duplicating logic.
canonicalize_json = canonical_json
deterministic_action_id = action_id
