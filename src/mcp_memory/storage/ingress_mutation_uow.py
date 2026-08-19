"""Shared dialect-independent unit-of-work pieces for ingress mutations.

``SQLiteIngressMutationStore`` and ``PostgresIngressMutationStore`` apply one
create/append ingress mutation atomically on their respective backends.  The
two backends share the same identity normalization, coverage computation,
receipt/history shape, and record-preparation rules; only the SQL bodies that
touch the backend connection differ (placeholder style, casts, and the
lexical projection table). This module holds every piece that does not touch
a connection or cursor, so the two backend modules only carry the SQL that is
genuinely dialect-specific.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol
from unicodedata import normalize
from uuid import uuid4

from mcp_memory.core.ingress_evidence import (
    IngressActionReceipt,
    SourceCoverage,
    SourceCoverageOutcome,
)
from mcp_memory.core.ports.ingress import IngressActionReceiptIdentityConflictError
from mcp_memory.core.ports.memory import MemoryRecord


class IngressMutationInjectedFailure(RuntimeError):
    """Failure raised by a test-only transaction stage hook."""


class IngressMutationResult:
    """Normalized result returned by an ingress domain callback."""

    def __init__(self, operation: str, affected_ids: Sequence[str] = ()) -> None:
        self.operation = operation
        self.affected_ids = tuple(str(value) for value in affected_ids)


class IngressDomainTransaction(Protocol):
    """Connection-scoped create/append domain surface."""

    touched_ids: list[str]

    def get_memory(self, memory_id: str) -> MemoryRecord | None: ...

    def create_memory(self, **values: object) -> MemoryRecord: ...

    def append_memory(
        self,
        memory_id: str,
        content: str,
        *,
        summary: str | object | None = None,
        tags: Sequence[str] | None = None,
        workspace_ids: Sequence[str] | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> MemoryRecord: ...


_FAIL_STAGE_ALIASES: dict[str, set[str]] = {
    "action_reserved": {"action_reserved", "reserve"},
    "after_domain_mutation": {"domain", "after_domain_mutation"},
    "repair_intent": {"repair", "repair_intent"},
    "history": {"history"},
    "before_receipt_coverage_commit": {"receipt", "coverage", "before_receipt_coverage_commit"},
    "journal_reconcile": {"journal", "journal_reconcile"},
}


def fail_stage(
    stage: str,
    *,
    fault_stage: str | None,
    fault_injector: Callable[[str], None] | None,
) -> None:
    """Raise ``IngressMutationInjectedFailure`` when ``stage`` is the configured fault point.

    Shared by both backends so their rollback-boundary tests exercise identical
    stage names regardless of dialect.
    """
    if fault_injector is not None:
        fault_injector(stage)
    normalized = None if fault_stage is None else fault_stage.replace("-", "_")
    if normalized in _FAIL_STAGE_ALIASES.get(stage, {stage}):
        raise IngressMutationInjectedFailure(f"injected ingress transaction failure at {stage}")


@dataclass(frozen=True)
class PreparedCreate:
    """Validated, normalized values for one ``create_memory`` call."""

    memory_id: str
    title: str
    content: str
    memory_type: str
    status: str
    workspace_ids: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    created_at: str = ""
    summary: str = ""
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PreparedAppend:
    """Validated, normalized values for one ``append_memory`` call."""

    content: str
    summary: str | None
    tags: list[str]
    workspace_ids: list[str]
    metadata: dict[str, object]


def prepare_create(values: Mapping[str, object]) -> PreparedCreate:
    memory_id = identifier(str(values.get("memory_id", uuid4())), "memory_id")
    title = required_text(values.get("title"), "title")
    content = required_text(values.get("content"), "content")
    memory_type = str(values.get("memory_type", values.get("type", "observation")))
    status = str(values.get("status", "active"))
    workspace_ids = normalized_values(values.get("workspace_ids", ()))
    if not workspace_ids:
        raise ValueError("workspace_ids must contain at least one non-empty value")
    tags = normalized_values(values.get("tags", ()), normalize_tags=True)
    created_at = str(values.get("created_at", now_text()))
    summary = str(values.get("summary") or build_summary(title, content, memory_type))
    metadata = values.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a mapping")
    return PreparedCreate(
        memory_id=memory_id,
        title=title,
        content=content,
        memory_type=memory_type,
        status=status,
        workspace_ids=workspace_ids,
        tags=tags,
        created_at=created_at,
        summary=summary,
        metadata=dict(metadata),
    )


def prepare_append(
    existing: MemoryRecord,
    content: str,
    *,
    summary: str | object | None = None,
    tags: Sequence[str] | None = None,
    workspace_ids: Sequence[str] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> PreparedAppend:
    addition = required_text(content, "content")
    merged_content = existing.content if addition in existing.content else f"{existing.content.rstrip()}\n\n{addition}"
    merged_tags = normalized_values(tags if tags is not None else existing.tags, normalize_tags=True)
    merged_workspaces = normalized_values(workspace_ids if workspace_ids is not None else existing.workspace_ids)
    merged_metadata = dict(existing.metadata)
    if metadata is not None:
        merged_metadata.update(metadata)
    resolved_summary = existing.summary if summary is None else str(summary)
    return PreparedAppend(
        content=merged_content.strip(),
        summary=resolved_summary,
        tags=merged_tags,
        workspace_ids=merged_workspaces,
        metadata=merged_metadata,
    )


def coverage_for_action(
    coverage: Sequence[SourceCoverage] | None,
    *,
    entry_ids: Sequence[str],
    action_id: str,
    operation: str,
) -> tuple[SourceCoverage, ...]:
    if coverage is None:
        outcome = SourceCoverageOutcome.CREATED if operation == "create" else SourceCoverageOutcome.APPENDED
        return tuple(SourceCoverage(entry_id, outcome, action_id) for entry_id in entry_ids)
    normalized = tuple(coverage)
    if canonical_ids([item.entry_id for item in normalized]) != canonical_ids(entry_ids):
        raise ValueError("source coverage must contain exactly the action entry IDs")
    result: list[SourceCoverage] = []
    for item in normalized:
        if item.action_id not in {None, action_id}:
            raise ValueError("source coverage action ID does not match the mutation")
        if item.outcome is SourceCoverageOutcome.UNOBSERVED:
            raise ValueError("create and append require terminal handled source coverage")
        result.append(SourceCoverage(item.entry_id, item.outcome, action_id, item.reason))
    return tuple(sorted(result, key=lambda value: value.entry_id.encode("utf-8")))


def check_digest(receipt: IngressActionReceipt, requested_digest: str) -> None:
    if receipt.canonical_payload_digest != requested_digest:
        raise IngressActionReceiptIdentityConflictError(
            receipt.action_id, receipt.canonical_payload_digest, requested_digest
        )


def canonical_ids(values: Sequence[str]) -> tuple[str, ...]:
    normalized = {identifier(value, "identity") for value in values}
    return tuple(sorted(normalized, key=lambda value: value.encode("utf-8")))


def journal_entry_ids(values: Sequence[str]) -> tuple[int, ...]:
    if not values:
        raise ValueError("journal entry IDs must contain only positive integer IDs")
    result: list[int] = []
    for value in values:
        normalized = str(value).strip()
        if not normalized.isdigit() or int(normalized) <= 0:
            raise ValueError("journal entry IDs must be positive integer IDs")
        result.append(int(normalized))
    return tuple(result)


def identifier(value: object, name: str) -> str:
    normalized = normalize("NFC", str(value)).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty")
    return normalized


def required_text(value: object, name: str) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty")
    return text


def normalized_values(value: object, *, normalize_tags: bool = False) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("collection value expected")
    result: set[str] = set()
    for item in value:
        text = str(item).strip()
        if normalize_tags:
            text = text.lower().replace("_", "-")
        if text:
            result.add(text)
    return sorted(result, key=lambda item: item.encode("utf-8"))


def json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def now_text() -> str:
    return datetime.now(UTC).isoformat()


def build_summary(title: str, content: str, memory_type: str) -> str:
    from mcp_memory.core.summaries import build_deterministic_summary

    return build_deterministic_summary(title=title, content=content, memory_type=memory_type)


__all__ = [
    "IngressDomainTransaction",
    "IngressMutationInjectedFailure",
    "IngressMutationResult",
    "PreparedAppend",
    "PreparedCreate",
    "build_summary",
    "canonical_ids",
    "check_digest",
    "coverage_for_action",
    "fail_stage",
    "identifier",
    "journal_entry_ids",
    "json_text",
    "normalized_values",
    "now_text",
    "prepare_append",
    "prepare_create",
    "required_text",
]
