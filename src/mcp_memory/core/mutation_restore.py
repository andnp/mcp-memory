"""Pure, non-executing descriptions of safe mutation inverses."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Callable, Protocol
from uuid import UUID, uuid4, uuid5

from pydantic import BaseModel, ConfigDict, Field

from mcp_memory.core.curation_identity import SCHEMA_VERSION, canonical_json
from mcp_memory.core.curation_models import ActionPreconditions, LinkAssertion
from mcp_memory.curation_action_store import CurationActionStaleError, MutationResult
from mcp_memory.curation_store import (
    CurationReceiptState,
    CurationRepository,
    CurationRun,
    CurationRunState,
)
from mcp_memory.mutation_history import (
    LinkRevision,
    MutationEvent,
    MutationEventStatus,
    RecordRevision,
    RestoreRequest,
    RestoreResult,
    RestoreResultStatus,
    RestoreScope,
)


class RestoreConflictCode(StrEnum):
    MALFORMED_HISTORY = "malformed_history"
    INCOMPLETE_HISTORY = "incomplete_history"
    UNSUPPORTED_OPERATION = "unsupported_operation"
    STALE_STATE = "stale_state"


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


_UNSUPPORTED = {
    "rewrite_memory",
    "remove_link",
    "archive_memory",
    "merge_memories",
    "split_memory",
    "delete_memory",
    "delete",
}
_RECORD_OPERATIONS = {"normalize_memory"}


class RestoreActionStore(Protocol):
    def execute_action(
        self,
        *,
        run_id: UUID,
        action_id: UUID,
        target_ids: list[str],
        expected_tokens: Mapping[str, str],
        apply: Callable[[Any], MutationResult],
        preconditions: Any | None = None,
        operation: str | None = None,
        payload: Any | None = None,
        actor_kind: str = "restore",
        restores_event_id: UUID | None = None,
        idempotency_key: str | None = None,
    ) -> Any: ...


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
            reason="this operation does not have a supported low-risk restore",
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

    if operation == "create_link":
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


class RestoreExecutor:
    """Execute the supported inverse through the normal action transaction.

    Restore requests are registered in the history store first.  The action
    store then owns the domain, projection, repair, history, and receipt
    transaction, so a restore cannot leave a partial compensating mutation.
    ``expected_link_tokens`` accepts graph-token keys (``memory_id`` or
    ``graph:memory_id``); exact edge context is additionally guarded by the
    required-link precondition.
    """

    def __init__(
        self,
        action_store: RestoreActionStore,
        history_store: Any,
        *,
        curation_store: CurationRepository | None = None,
    ) -> None:
        self._action_store = action_store
        self._history_store = history_store
        self._curation_store = curation_store

    def execute(
        self,
        request: RestoreRequest,
        *,
        run_id: UUID | None = None,
        action_id: UUID | None = None,
    ) -> RestoreResult:
        registered = self._history_store.request_restore(request)
        if registered.status is not RestoreResultStatus.APPLIED:
            # A request can be left in its initial, non-terminal APPLIED state
            # by a process crash.  Reuse the deterministic action identity and
            # finish it; completed requests remain a cheap idempotent replay.
            if registered.status is not RestoreResultStatus.ALREADY_APPLIED or registered.event_id is not None:
                return registered
            request_id = registered.request_id
        else:
            request_id = registered.request_id

        event = self._history_store.get_event(request.target_event_id)
        if event is None:
            return self._finish(
                request,
                request_id,
                self._conflict_result(
                    request,
                    RestoreConflictCode.MALFORMED_HISTORY,
                    "target mutation event was not found",
                ),
            )
        inverse = build_inverse(
            event,
            self._history_store.get_record_revisions(event.id),
            self._history_store.get_link_revisions(event.id),
        )
        if isinstance(inverse, RestoreConflict):
            return self._finish(request, request_id, self._conflict_result(request, inverse.code, inverse.reason, inverse))
        scope_conflict = _scope_conflict(request.scope, inverse)
        if scope_conflict is not None:
            return self._finish(request, request_id, self._conflict_result(request, *scope_conflict))

        target_ids = _inverse_target_ids(inverse)
        expected_tokens = {str(memory_id): token for memory_id, token in request.expected_record_tokens.items()}
        for key, token in request.expected_link_tokens.items():
            normalized_key = str(key)
            if normalized_key.startswith("graph:"):
                expected_tokens[normalized_key] = token
            elif normalized_key in target_ids:
                expected_tokens[f"graph:{normalized_key}"] = token
            else:
                expected_tokens[f"link:{normalized_key}"] = token
        if any(memory_id not in expected_tokens for memory_id in _record_target_ids(inverse)):
            return self._finish(
                request,
                request_id,
                self._conflict_result(
                    request,
                    RestoreConflictCode.INCOMPLETE_HISTORY,
                    "restore requires current record tokens for every affected record",
                ),
            )

        restore_run_id = run_id or event.curation_run_id or uuid4()
        if self._curation_store is not None and self._curation_store.get_run(restore_run_id) is None:
            self._curation_store.create_run(
                CurationRun(
                    run_id=restore_run_id,
                    frontier_key=f"restore:{event.id}",
                    context_fingerprint=request.idempotency_key,
                    state=CurationRunState.EXECUTING,
                )
            )
        restore_action_id = action_id or uuid5(event.id, f"restore:{request.idempotency_key}")
        preconditions = _inverse_preconditions(inverse)

        def apply(transaction) -> MutationResult:
            for change in inverse.record_changes:
                if not change.exists or change.snapshot is None:
                    raise ValueError("restoring a missing record is not supported")
                record = change.snapshot["record"]
                current = transaction.get_memory(str(change.memory_id))
                if current is None:
                    raise CurationActionStaleError(f"target memory {change.memory_id} is missing")
                metadata = dict(current.metadata)
                lineage = record.get("lineage")
                if isinstance(lineage, Mapping):
                    metadata["lineage"] = dict(lineage)
                mutation_metadata = record.get("mutation_metadata")
                if isinstance(mutation_metadata, Mapping):
                    metadata["mutation_metadata"] = dict(mutation_metadata)
                transaction.update_memory(
                    str(change.memory_id),
                    title=record["title"],
                    content=record["content"],
                    summary=record["summary"],
                    memory_type=record["type"],
                    status=record["status"],
                    tags=record["tags"],
                    workspace_ids=record["workspace_ids"],
                    metadata=metadata,
                )
            for change in inverse.link_changes:
                if change.exists:
                    transaction.add_link(
                        str(change.source_id),
                        str(change.target_id),
                        change.link_type,
                        change.context or "",
                    )
                else:
                    transaction.remove_link(str(change.source_id), str(change.target_id), change.link_type)
            return MutationResult("restore", target_ids)

        try:
            receipt = self._action_store.execute_action(
                run_id=restore_run_id,
                action_id=restore_action_id,
                target_ids=target_ids,
                expected_tokens=expected_tokens,
                apply=apply,
                preconditions=preconditions,
                operation="restore",
                payload={"target_event_id": str(event.id), "request_id": request.idempotency_key},
                actor_kind="restore",
                restores_event_id=event.id,
                idempotency_key=request.idempotency_key,
            )
        except CurationActionStaleError as error:
            return self._finish(
                request,
                request_id,
                self._conflict_result(request, RestoreConflictCode.STALE_STATE, str(error)),
            )
        result = RestoreResult(
            status=RestoreResultStatus.APPLIED,
            request_id=request_id,
            event_id=receipt.mutation_event_id,
            target_event_id=request.target_event_id,
        )
        if self._curation_store is not None:
            self._curation_store.transition_receipt(
                restore_run_id,
                restore_action_id,
                CurationReceiptState.APPLIED_UNVERIFIED,
                receipt.model_copy(
                    update={
                        "status": CurationReceiptState.VERIFIED,
                        "verified_at": datetime.now(UTC),
                    }
                ),
            )
        return self._finish(request, request_id, result)

    def execute_restore(
        self,
        request: RestoreRequest,
        *,
        run_id: UUID | None = None,
        action_id: UUID | None = None,
    ) -> RestoreResult:
        """Descriptive alias for :meth:`execute`."""
        return self.execute(request, run_id=run_id, action_id=action_id)

    def _finish(self, request: RestoreRequest, request_id: UUID | None, result: RestoreResult) -> RestoreResult:
        if request_id is None:
            return result
        return self._history_store.terminalize_restore_request(request_id, result)

    @staticmethod
    def _conflict_result(
        request: RestoreRequest,
        code: RestoreConflictCode,
        reason: str,
        conflict: RestoreConflict | None = None,
    ) -> RestoreResult:
        details: dict[str, Any] = {"code": code.value}
        if conflict is not None:
            details["operation"] = conflict.operation
        return RestoreResult(
            status=RestoreResultStatus.CONFLICT,
            target_event_id=request.target_event_id,
            conflict_reason=reason,
            conflict_details=details,
        )


def execute_restore(
    action_store: RestoreActionStore,
    history_store: Any,
    request: RestoreRequest,
    *,
    curation_store: CurationRepository | None = None,
    run_id: UUID | None = None,
    action_id: UUID | None = None,
) -> RestoreResult:
    """Functional entry point for token-guarded low-risk restore execution."""
    return RestoreExecutor(action_store, history_store, curation_store=curation_store).execute(
        request,
        run_id=run_id,
        action_id=action_id,
    )


def _record_target_ids(inverse: InverseDescription) -> list[str]:
    return [str(change.memory_id) for change in inverse.record_changes]


def _inverse_target_ids(inverse: InverseDescription) -> list[str]:
    values = _record_target_ids(inverse)
    for change in inverse.link_changes:
        values.extend([str(change.source_id), str(change.target_id)])
    return sorted(set(values), key=lambda value: value.encode("utf-8"))


def _inverse_preconditions(inverse: InverseDescription) -> ActionPreconditions | None:
    if not inverse.link_changes:
        return None
    required_links = [
        LinkAssertion(
            source_id=change.source_id,
            target_id=change.target_id,
            link_type=change.link_type,
            context=change.context,
        )
        for change in inverse.link_changes
        if not change.exists
    ]
    return ActionPreconditions(required_links=required_links)


def _scope_conflict(scope: RestoreScope, inverse: InverseDescription) -> tuple[RestoreConflictCode, str] | None:
    if scope is RestoreScope.ALL:
        return None
    if scope is RestoreScope.RECORDS and inverse.record_changes and not inverse.link_changes:
        return None
    if scope is RestoreScope.LINKS and inverse.link_changes and not inverse.record_changes:
        return None
    return RestoreConflictCode.UNSUPPORTED_OPERATION, "requested restore scope does not match the supported inverse"
