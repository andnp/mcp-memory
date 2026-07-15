"""Pure, non-executing descriptions of safe mutation inverses."""

from __future__ import annotations

from copy import deepcopy
from enum import StrEnum
import json
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from mcp_memory.core.curation_identity import SCHEMA_VERSION, canonical_json
from mcp_memory.mutation_history import (
    LinkRevision,
    MutationEvent,
    MutationEventStatus,
    RecordRevision,
)


class RestoreConflictCode(StrEnum):
    MALFORMED_HISTORY = "malformed_history"
    INCOMPLETE_HISTORY = "incomplete_history"
    UNSUPPORTED_OPERATION = "unsupported_operation"


class RestoreConflict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: RestoreConflictCode
    operation: str
    reason: str


class InverseRecordChange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_id: UUID
    exists: bool
    snapshot: dict[str, Any] | None = None
    expected_current_token: str | None = None
    resulting_token: str | None = None


class InverseLinkChange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: UUID
    target_id: UUID
    link_type: str
    context: str | None = None
    exists: bool


class InverseDescription(BaseModel):
    """A deterministic plan for an inverse; this model performs no writes."""

    model_config = ConfigDict(extra="forbid")

    source_operation: str
    inverse_operation: str
    record_changes: list[InverseRecordChange] = Field(default_factory=list)
    link_changes: list[InverseLinkChange] = Field(default_factory=list)


_UNSUPPORTED = {"merge_memories", "split_memory", "delete_memory", "delete"}
_RECORD_OPERATIONS = {
    "normalize_memory",
    "rewrite_memory",
    "archive_memory",
    "status_memory",
    "update_status",
    "set_status",
}


def build_inverse(
    event: MutationEvent,
    record_revisions: list[RecordRevision] | tuple[RecordRevision, ...] = (),
    link_revisions: list[LinkRevision] | tuple[LinkRevision, ...] = (),
) -> InverseDescription | RestoreConflict:
    """Build a safe inverse description, or a typed conflict when history is unsafe."""
    operation = event.operation
    if operation in _UNSUPPORTED:
        return RestoreConflict(
            code=RestoreConflictCode.UNSUPPORTED_OPERATION,
            operation=operation,
            reason="merge, split, and delete inverses are not supported",
        )
    if event.status is not MutationEventStatus.APPLIED:
        return _conflict(RestoreConflictCode.MALFORMED_HISTORY, operation, "event is not applied")

    records = list(record_revisions)
    links = list(link_revisions)
    if operation in _RECORD_OPERATIONS:
        if links or len(records) != 1:
            return _conflict(RestoreConflictCode.INCOMPLETE_HISTORY, operation, "expected exactly one record revision")
        revision = records[0]
        snapshot_conflict = _record_conflict(event, revision)
        if snapshot_conflict is not None:
            return snapshot_conflict
        if revision.before_snapshot is None or revision.after_snapshot is None:
            return _conflict(RestoreConflictCode.INCOMPLETE_HISTORY, operation, "record snapshots are required")
        return InverseDescription(
            source_operation=operation,
            inverse_operation="restore_record",
            record_changes=[
                InverseRecordChange(
                    memory_id=revision.memory_id,
                    exists=revision.before_exists,
                    snapshot=_semantic_snapshot(revision.before_snapshot),
                    expected_current_token=revision.after_token,
                    resulting_token=revision.before_token,
                )
            ],
        )

    if operation in {"create_link", "remove_link"}:
        if records or len(links) != 1:
            return _conflict(RestoreConflictCode.INCOMPLETE_HISTORY, operation, "expected exactly one link revision")
        revision = links[0]
        if (
            revision.event_id != event.id
            or not revision.link_type.strip()
            or revision.before_exists == revision.after_exists
        ):
            return _conflict(RestoreConflictCode.MALFORMED_HISTORY, operation, "link existence transition is invalid")
        return InverseDescription(
            source_operation=operation,
            inverse_operation="remove_link" if operation == "create_link" else "create_link",
            link_changes=[
                InverseLinkChange(
                    source_id=revision.source_id,
                    target_id=revision.target_id,
                    link_type=revision.link_type,
                    context=revision.context,
                    exists=revision.before_exists,
                )
            ],
        )

    return _conflict(RestoreConflictCode.UNSUPPORTED_OPERATION, operation, "operation has no safe inverse")


def _record_conflict(event: MutationEvent, revision: RecordRevision) -> RestoreConflict | None:
    if revision.event_id != event.id or revision.role.value != "target":
        return _conflict(RestoreConflictCode.MALFORMED_HISTORY, event.operation, "revision belongs to another event")
    if revision.before_exists != (revision.before_snapshot is not None) or revision.after_exists != (
        revision.after_snapshot is not None
    ):
        return _conflict(RestoreConflictCode.INCOMPLETE_HISTORY, event.operation, "existence flags do not match snapshots")
    if revision.before_exists and not revision.before_token:
        return _conflict(RestoreConflictCode.INCOMPLETE_HISTORY, event.operation, "before token is missing")
    if revision.after_exists and not revision.after_token:
        return _conflict(RestoreConflictCode.INCOMPLETE_HISTORY, event.operation, "after token is missing")
    try:
        if revision.before_snapshot is not None:
            _semantic_snapshot(revision.before_snapshot)
        if revision.after_snapshot is not None:
            _semantic_snapshot(revision.after_snapshot)
    except (TypeError, ValueError) as error:
        return _conflict(RestoreConflictCode.MALFORMED_HISTORY, event.operation, str(error))
    return None


def _semantic_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Validate and re-canonicalize a stored snapshot, dropping telemetry fields."""
    required = {"schema_version", "record"}
    if set(snapshot) != required or not isinstance(snapshot["record"], dict):
        raise ValueError("snapshot is not a complete semantic snapshot")
    if snapshot["schema_version"] != SCHEMA_VERSION:
        raise ValueError("snapshot schema version is unsupported")
    record = snapshot["record"]
    fields = {"id", "title", "content", "summary", "type", "status", "tags", "workspace_ids", "lineage", "mutation_metadata"}
    if not fields.issubset(record):
        raise ValueError("snapshot record is incomplete")
    record = {key: record[key] for key in fields}
    # canonical_json validates values and makes the returned description independent of caller mutation.
    return deepcopy(json.loads(canonical_json({"schema_version": snapshot["schema_version"], "record": record})))


def _conflict(code: RestoreConflictCode, operation: str, reason: str) -> RestoreConflict:
    return RestoreConflict(code=code, operation=operation, reason=reason)


describe_inverse = build_inverse
build_inverse_operation = build_inverse
build_inverse_description = build_inverse
