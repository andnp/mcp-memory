"""Shared dialect-independent unit-of-work pieces for curation actions.

``SQLiteCurationActionStore`` and ``PostgresCurationActionStore`` apply one
curation action atomically on their respective backends.  Both share the
same error taxonomy, callback protocol, identity/replay rules, and
record-preparation rules for ``update_memory``/``create_memory``; only the
SQL bodies that touch the backend connection differ (placeholder style,
casts, optimistic-concurrency guard shape, and the lexical projection
table). This module holds every piece that does not touch a connection or
cursor, so the two backend modules only carry the SQL that is genuinely
dialect-specific.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, cast
from uuid import UUID

from mcp_memory.core.curation_identity import action_intent_token, canonical_token
from mcp_memory.core.curation_models import CurationVerificationDescriptor
from mcp_memory.core.ports.memory import MemoryLink, MemoryRecord
from mcp_memory.curation_store import CurationActionReceipt

_SUMMARY_UNSET = object()
VALID_MEMORY_TYPES = frozenset({"journal", "plan", "fact", "observation", "reflection"})
VALID_MEMORY_STATUSES = frozenset({"active", "stale", "degraded", "archived"})


class CurationActionError(RuntimeError):
    """Base class for action-transaction failures."""


class CurationActionStaleError(CurationActionError):
    """The action preconditions no longer describe the authoritative state."""


class CurationActionTransientError(CurationActionError):
    """The backend failed in a way that is safe to retry with the same action."""


class CurationActionFatalError(CurationActionError):
    """The action or transaction callback violated a non-retryable contract."""


class CurationActionContractError(CurationActionFatalError):
    """The curation agent supplied an action that can be repaired before execution."""

    def __init__(self, message: str, *, code: str = "action_contract") -> None:
        super().__init__(message)
        self.code = code


class CurationActionInjectedFailure(CurationActionFatalError):
    """Failure raised by the test-only transaction stage hook."""


@dataclass(frozen=True)
class MutationResult:
    """The compact normalized result returned by a transaction callback."""

    operation: str
    affected_ids: tuple[str, ...] = ()
    verification_descriptor: CurationVerificationDescriptor | None = None

    def __init__(
        self,
        operation: str,
        affected_ids: Sequence[str | UUID] = (),
        verification_descriptor: CurationVerificationDescriptor | None = None,
    ) -> None:
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "affected_ids", tuple(str(value) for value in affected_ids))
        object.__setattr__(self, "verification_descriptor", verification_descriptor)


class CurationTransaction(Protocol):
    """The callback-only, transaction-scoped domain mutation surface."""

    def get_memory(self, memory_id: str | UUID) -> MemoryRecord | None: ...

    def get_links(
        self,
        memory_id: str | UUID,
        *,
        direction: str = "outgoing",
        link_type: str | None = None,
    ) -> list[MemoryLink]: ...

    def get_protections(self, memory_id: str | UUID) -> list[object]: ...

    def update_memory(self, memory_id: str | UUID, **changes: object) -> MemoryRecord: ...

    def create_memory(self, **values: object) -> MemoryRecord: ...

    def delete_memory(self, memory_id: str | UUID) -> MemoryRecord: ...

    def add_link(
        self,
        source_id: str | UUID,
        target_id: str | UUID,
        link_type: str,
        context: str = "",
    ) -> MemoryLink: ...

    def remove_link(self, source_id: str | UUID, target_id: str | UUID, link_type: str) -> bool: ...

    def set_lineage(self, memory_id: str | UUID, lineage: Mapping[str, object]) -> MemoryRecord: ...


class CurationActionStore(Protocol):
    def execute_action(
        self,
        *,
        run_id: UUID,
        action_id: UUID,
        target_ids: Sequence[str],
        expected_tokens: Mapping[str, str],
        apply: Callable[[CurationTransaction], MutationResult],
        preconditions: Any | None = None,
        operation: str | None = None,
        payload: Any | None = None,
    ) -> CurationActionReceipt: ...


@dataclass
class _MutationResultData:
    operation: str
    affected_ids: list[str] = field(default_factory=list)
    verification_descriptor: CurationVerificationDescriptor | None = None


def normalize_result(result: MutationResult | Mapping[str, object] | object) -> _MutationResultData:
    if isinstance(result, MutationResult):
        return _MutationResultData(result.operation, list(result.affected_ids), result.verification_descriptor)
    operation = field_of(result, "operation", None)
    affected_ids = field_of(result, "affected_ids", [])
    is_valid_ids = isinstance(affected_ids, Sequence) and not isinstance(affected_ids, (str, bytes))
    if not isinstance(operation, str) or not is_valid_ids:
        raise CurationActionFatalError("callback must return MutationResult")
    descriptor = field_of(result, "verification_descriptor", None)
    if descriptor is not None and not isinstance(descriptor, CurationVerificationDescriptor):
        raise CurationActionFatalError("mutation result verification descriptor is invalid")
    normalized_ids = cast(Sequence[object], affected_ids)
    return _MutationResultData(operation, [str(value) for value in normalized_ids], descriptor)


def canonical_target_ids(values: Sequence[str | UUID]) -> list[str]:
    normalized = {canonical_id(value) for value in values}
    if "" in normalized:
        raise CurationActionFatalError("target IDs must be non-empty")
    return sorted(normalized, key=lambda value: value.encode("utf-8"))


def local_memory_ids(values: Sequence[str | UUID]) -> list[str]:
    return [value for value in canonical_target_ids(values) if not value.startswith("ext:")]


def canonical_id(value: object) -> str:
    if value is None:
        raise CurationActionFatalError("identity must be non-null")
    return unicodedata.normalize("NFC", str(value)).strip()


def field_of(value: object, name: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def text_or(value: object, fallback: str) -> str:
    return fallback if value is None else str(value).strip()


def normalized_values(value: object, fallback: Sequence[str]) -> list[str]:
    source = fallback if value is None else value
    if not isinstance(source, Sequence) or isinstance(source, (str, bytes)):
        raise ValueError("collection value expected")
    return sorted({str(item).strip() for item in source if str(item).strip()}, key=lambda item: item.encode("utf-8"))


def normalize_link_type(link_type: str) -> str:
    normalized = re.sub(r"[\s-]+", "_", link_type.strip()).upper()
    if not normalized:
        raise ValueError("link_type must be non-empty")
    return normalized


def link_token_key(key: str) -> tuple[str, str, str] | None:
    raw = key.removeprefix("link:") if key.startswith("link:") else key
    parts = raw.split(":")
    if len(parts) != 3 or any(not part for part in parts):
        return None
    return parts[0], parts[1], parts[2]


def link_mapping(link: MemoryLink) -> dict[str, str | None]:
    return {"source_id": link.source_id, "target_id": link.target_id, "type": link.link_type, "context": link.context}


def build_summary(title: str, content: str, memory_type: str) -> str:
    # Keep the transaction primitive independent from the public repository
    # while retaining its established summary shape.
    from mcp_memory.core.summaries import build_deterministic_summary

    return build_deterministic_summary(title=title, content=content, memory_type=memory_type)


def snapshot(record: MemoryRecord) -> dict[str, Any]:
    from mcp_memory.core.curation_identity import record_snapshot

    return record_snapshot(record)


def semantic_token(record: MemoryRecord) -> str:
    from mcp_memory.core.curation_identity import record_token

    return record_token(record)


def state_token(records: Mapping[str, MemoryRecord], ids: Sequence[str]) -> str | None:
    if not records:
        return None
    sorted_keys = sorted(records, key=lambda value: value.encode("utf-8"))
    return canonical_token(
        {"targets": list(ids), "records": {key: snapshot(records[key]) for key in sorted_keys}}
    )


def action_intent_hash(
    *,
    operation: str,
    target_ids: Sequence[str],
    expected_tokens: Mapping[str, str],
    preconditions: Any | None,
    payload: Any | None,
) -> str:
    try:
        return action_intent_token(
            operation=operation,
            target_ids=target_ids,
            expected_tokens=expected_tokens,
            preconditions=preconditions,
            payload=payload,
        )
    except (TypeError, ValueError) as exc:
        raise CurationActionFatalError("action intent is not canonically representable") from exc


def check_replay_identity(
    receipt: CurationActionReceipt,
    *,
    run_id: UUID,
    action_id: UUID,
    operation: str | None,
    target_ids: Sequence[str],
    expected_tokens: Mapping[str, str],
    preconditions: Any | None,
    payload: Any | None,
    verification_descriptor: CurationVerificationDescriptor | None,
) -> None:
    if receipt.intent_hash is None:
        raise CurationActionFatalError(
            f"curation action identity collision: legacy-unverifiable receipt for {run_id}/{action_id}"
        )
    if operation is None:
        raise CurationActionFatalError(
            f"curation action identity collision: operation is required for {run_id}/{action_id}"
        )
    candidate = action_intent_hash(
        operation=operation,
        target_ids=target_ids,
        expected_tokens=expected_tokens,
        preconditions=preconditions,
        payload=payload,
    )
    if candidate != receipt.intent_hash:
        raise CurationActionFatalError(f"curation action identity collision for {run_id}/{action_id}")
    if (
        receipt.verification_descriptor is not None
        and verification_descriptor is not None
        and receipt.verification_descriptor != verification_descriptor
    ):
        raise CurationActionFatalError(f"curation action identity collision for {run_id}/{action_id}")


_FAIL_STAGE_ALIASES: dict[str, set[str]] = {
    "domain_mutation": {"domain", "after_domain", "domain_mutation", "after_domain_mutation"},
    "projection_update": {"projection", "after_projection", "projection_update", "after_projection_update"},
    "repair_intent": {"repair", "after_repair", "repair_intent", "after_repair_intent"},
    "history": {"history", "after_history"},
    "receipt_preparation": {
        "receipt",
        "before_receipt",
        "receipt_preparation",
        "after_receipt_preparation",
    },
}


def fail_stage(
    stage: str,
    *,
    fault_stage: str | None,
    fault_injector: Callable[[str], None] | None,
) -> None:
    if fault_injector is not None:
        fault_injector(stage)
    normalized_stage = None if fault_stage is None else fault_stage.replace("-", "_")
    if normalized_stage in _FAIL_STAGE_ALIASES.get(stage, {stage}):
        raise CurationActionInjectedFailure(f"injected curation transaction failure at {stage}")


@dataclass(frozen=True)
class PreparedMemoryUpdate:
    """Validated, normalized values for one ``update_memory`` call."""

    title: str
    content: str
    memory_type: str
    status: str
    summary: str | None
    workspace_ids: list[str]
    tags: list[str]
    metadata: dict[str, object]
    optional_columns: dict[str, object]


def prepare_memory_update(existing: MemoryRecord, changes: Mapping[str, object]) -> PreparedMemoryUpdate:
    title = text_or(changes.get("title"), existing.title)
    content = text_or(changes.get("content"), existing.content)
    if not title or not content:
        raise ValueError("title and content must be non-empty")
    memory_type = str(changes.get("memory_type", changes.get("type", existing.type)))
    status = str(changes.get("status", existing.status))
    if memory_type not in VALID_MEMORY_TYPES:
        raise ValueError(f"invalid memory_type: {memory_type!r}")
    if status not in VALID_MEMORY_STATUSES:
        raise ValueError(f"invalid status: {status!r}")
    summary_value = changes.get("summary", _SUMMARY_UNSET)
    summary = (
        build_summary(title, content, memory_type)
        if summary_value is _SUMMARY_UNSET
        and ("title" in changes or "content" in changes or "type" in changes or "memory_type" in changes)
        else None
        if summary_value is None
        else str(summary_value)
        if summary_value is not _SUMMARY_UNSET
        else existing.summary
    )
    workspace_ids = normalized_values(changes.get("workspace_ids"), existing.workspace_ids)
    tags = normalized_values(changes.get("tags"), existing.tags)
    metadata = changes.get("metadata", existing.metadata)
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a mapping")
    optional_columns = {
        name: changes[name] for name in ("access_score", "last_accessed_at", "last_surfaced_at") if name in changes
    }
    return PreparedMemoryUpdate(
        title=title,
        content=content,
        memory_type=memory_type,
        status=status,
        summary=summary,
        workspace_ids=workspace_ids,
        tags=tags,
        metadata=dict(metadata),
        optional_columns=optional_columns,
    )


@dataclass(frozen=True)
class PreparedMemoryCreate:
    """Validated, normalized values for one ``create_memory`` call."""

    memory_id: str
    title: str
    content: str
    memory_type: str
    status: str
    workspace_ids: list[str]
    tags: list[str]
    summary: str
    created_at: str
    updated_at: str
    metadata: dict[str, object]


def prepare_memory_create(values: Mapping[str, object]) -> PreparedMemoryCreate:
    from uuid import uuid4

    from mcp_memory.storage.ingress_mutation_uow import now_text

    memory_id = canonical_id(values.get("memory_id", uuid4()))
    title = str(values.get("title", "")).strip()
    content = str(values.get("content", "")).strip()
    if not title or not content:
        raise ValueError("title and content must be non-empty")
    memory_type = str(values.get("memory_type", values.get("type", "journal")))
    status = str(values.get("status", "active"))
    if memory_type not in VALID_MEMORY_TYPES or status not in VALID_MEMORY_STATUSES:
        raise ValueError("invalid memory type or status")
    workspace_ids = normalized_values(values.get("workspace_ids", []), [])
    if not workspace_ids:
        raise ValueError("workspace_ids must contain at least one non-empty value")
    tags = normalized_values(values.get("tags", []), [])
    summary = str(values.get("summary") or build_summary(title, content, memory_type))
    created_at = str(values.get("created_at", now_text()))
    updated_at = str(values.get("updated_at", created_at))
    metadata = values.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a mapping")
    return PreparedMemoryCreate(
        memory_id=memory_id,
        title=title,
        content=content,
        memory_type=memory_type,
        status=status,
        workspace_ids=workspace_ids,
        tags=tags,
        summary=summary,
        created_at=created_at,
        updated_at=updated_at,
        metadata=dict(metadata),
    )


__all__ = [
    "VALID_MEMORY_STATUSES",
    "VALID_MEMORY_TYPES",
    "CurationActionContractError",
    "CurationActionError",
    "CurationActionFatalError",
    "CurationActionInjectedFailure",
    "CurationActionStaleError",
    "CurationActionStore",
    "CurationActionTransientError",
    "CurationTransaction",
    "MutationResult",
    "PreparedMemoryCreate",
    "PreparedMemoryUpdate",
    "action_intent_hash",
    "build_summary",
    "canonical_id",
    "canonical_target_ids",
    "check_replay_identity",
    "fail_stage",
    "field_of",
    "link_mapping",
    "link_token_key",
    "local_memory_ids",
    "normalize_link_type",
    "normalize_result",
    "normalized_values",
    "prepare_memory_create",
    "prepare_memory_update",
    "semantic_token",
    "snapshot",
    "state_token",
    "text_or",
]
