from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Callable
from uuid import UUID

from mcp_memory.core.curation_identity import link_token, record_token
from mcp_memory.core.mutation_restore import (
    InverseDescription,
    RestoreConflict,
    RestoreConflictCode,
    RestoreExecutor,
    build_inverse,
)
from mcp_memory.management.models import (
    MutationHistoryDetailPayload,
    MutationHistoryDiffPayload,
    MutationHistoryListPayload,
    ProtectionListPayload,
    ProtectionMutationPayload,
    RestoreEligibilityPayload,
    RestoreRequestPayload,
)
from mcp_memory.mutation_history import (
    LinkRevision,
    MutationEvent,
    Protection,
    ProtectionMode,
    RecordRevision,
    RestoreRequest,
    RestoreResult,
    RestoreResultStatus,
    RestoreScope,
)


@dataclass(frozen=True)
class MutationHistoryServiceDependencies:
    mutation_history: Any
    memory_queries: Any
    curation: Any
    action_store: Any
    restore_executor: type[RestoreExecutor] = RestoreExecutor
    inverse_builder: Callable[..., InverseDescription | RestoreConflict] = build_inverse


_MAX_HISTORY_LIST_LIMIT = 100
_MAX_HISTORY_OFFSET = 10_000
_MAX_HISTORY_REVISION_ROWS = 100
_MAX_HISTORY_VALUE_CHARS = 16_000


def _parse_uuid(value: str, *, field: str) -> UUID:
    try:
        return UUID(value)
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"{field}_invalid") from exc


def _history_time(value: float | None) -> datetime | None:
    return None if value is None else datetime.fromtimestamp(value, tz=UTC)


def _bounded_history_value(value: object) -> object:
    if value is None:
        return None
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    if len(encoded) <= _MAX_HISTORY_VALUE_CHARS:
        return value
    return {
        "truncated": True,
        "sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        "original_characters": len(encoded),
    }


@dataclass(frozen=True)
class _RestoreAnalysis:
    event: MutationEvent
    inverse: InverseDescription | None
    conflict: RestoreConflict | None
    current_record_tokens: dict[str, str]
    current_link_tokens: dict[str, str]
    protections: dict[str, list[Protection]]
    conflict_code: str | None = None
    conflict_reason: str | None = None


def _restore_risk(operation: str) -> str:
    if operation == "normalize_memory":
        return "low"
    if operation == "create_link":
        return "moderate"
    return "unsupported"


def _required_token(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field}_must_contain_tokens")
    return value


def _parse_record_tokens(value: object) -> dict[UUID, str]:
    if not isinstance(value, dict):
        raise ValueError("expected_record_tokens_must_be_object")
    return {
        _parse_uuid(str(memory_id), field="memory_id"): _required_token(token, "expected_record_tokens")
        for memory_id, token in value.items()
    }


def _parse_link_tokens(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError("expected_link_tokens_must_be_object")
    return {str(key): _required_token(token, "expected_link_tokens") for key, token in value.items()}


def _restore_conflict(target_event_id: UUID, code: str, reason: str) -> RestoreResult:
    return RestoreResult(
        status=RestoreResultStatus.CONFLICT,
        target_event_id=target_event_id,
        conflict_reason=reason,
        conflict_details={"code": code},
    )


def _restore_rejected(target_event_id: UUID, code: str, reason: str) -> RestoreResult:
    return RestoreResult(
        status=RestoreResultStatus.REJECTED,
        target_event_id=target_event_id,
        conflict_reason=reason,
        conflict_details={"code": code},
    )


def _restore_payload(result: RestoreResult) -> RestoreRequestPayload:
    return RestoreRequestPayload(
        status=result.status.value,
        target_event_id=str(result.target_event_id),
        request_id=None if result.request_id is None else str(result.request_id),
        event_id=None if result.event_id is None else str(result.event_id),
        conflict_reason=result.conflict_reason,
        conflict_details=result.conflict_details,
    )


class MutationHistoryService:
    def __init__(self, dependencies: MutationHistoryServiceDependencies) -> None:
        self._dependencies = dependencies

    def list_mutation_history(
        self,
        *,
        memory_id: str | None = None,
        actor_kind: str | None = None,
        family: str | None = None,
        operation: str | None = None,
        after: float | None = None,
        before: float | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> MutationHistoryListPayload:
        store = self._require_mutation_history()
        bounded_limit = min(limit, _MAX_HISTORY_LIST_LIMIT)
        if bounded_limit < 1:
            raise ValueError("limit_out_of_range")
        if offset < 0 or offset > _MAX_HISTORY_OFFSET:
            raise ValueError("offset_out_of_range")
        parsed_memory_id = _parse_uuid(memory_id, field="memory_id") if memory_id is not None else None
        events = store.list_events(
            memory_id=parsed_memory_id,
            actor_kind=actor_kind,
            family=family,
            operation=operation,
            created_after=_history_time(after),
            created_before=_history_time(before),
            limit=bounded_limit + 1,
            offset=offset,
        )
        has_more = len(events) > bounded_limit
        visible_events = events[:bounded_limit]
        return MutationHistoryListPayload(
            events=[self._history_event_payload(event) for event in visible_events],
            limit=bounded_limit,
            offset=offset,
            has_more=has_more,
            next_offset=offset + bounded_limit if has_more else None,
        )

    def get_mutation_history_event(self, event_id: str) -> MutationHistoryDetailPayload:
        event, records, links = self._get_history_parts(event_id)
        return MutationHistoryDetailPayload(
            event=self._history_event_payload(event),
            receipt=self._receipt_payload(event),
            curation_run=self._curation_run_payload(event),
            records=[self._record_revision_payload(revision) for revision in records],
            links=[self._link_revision_payload(revision) for revision in links],
            truncated=len(records) >= _MAX_HISTORY_REVISION_ROWS or len(links) >= _MAX_HISTORY_REVISION_ROWS,
        )

    def get_mutation_history_diff(self, event_id: str) -> MutationHistoryDiffPayload:
        event, records, links = self._get_history_parts(event_id)
        return MutationHistoryDiffPayload(
            event_id=str(event.id),
            records=[self._record_diff_payload(revision) for revision in records],
            links=[self._link_diff_payload(revision) for revision in links],
            truncated=len(records) >= _MAX_HISTORY_REVISION_ROWS or len(links) >= _MAX_HISTORY_REVISION_ROWS,
        )

    def list_protections(self, memory_id: str) -> ProtectionListPayload:
        parsed_memory_id = _parse_uuid(memory_id, field="memory_id")
        return ProtectionListPayload(
            memory_id=str(parsed_memory_id),
            protections=[
                protection.model_dump(mode="json")
                for protection in self._require_mutation_history().get_protections(parsed_memory_id)
            ],
        )

    def set_protection(
        self,
        *,
        memory_id: str,
        mode: str,
        reason: str,
        actor_id: str | None = None,
        expires_at: str | None = None,
    ) -> ProtectionMutationPayload:
        parsed_memory_id = _parse_uuid(memory_id, field="memory_id")
        try:
            protection = Protection(
                memory_id=parsed_memory_id,
                mode=ProtectionMode(mode),
                reason=reason,
                actor_id=actor_id,
                expires_at=None if expires_at is None else datetime.fromisoformat(expires_at),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("protection_invalid") from exc
        stored = self._require_mutation_history().set_protection(protection)
        return ProtectionMutationPayload(
            status="applied",
            memory_id=str(parsed_memory_id),
            mode=str(stored.mode),
            protection=stored.model_dump(mode="json"),
        )

    def remove_protection(self, *, memory_id: str, mode: str) -> ProtectionMutationPayload:
        parsed_memory_id = _parse_uuid(memory_id, field="memory_id")
        try:
            parsed_mode = ProtectionMode(mode)
        except ValueError as exc:
            raise ValueError("mode_invalid") from exc
        self._require_mutation_history().remove_protection(parsed_memory_id, parsed_mode)
        return ProtectionMutationPayload(
            status="removed",
            memory_id=str(parsed_memory_id),
            mode=str(parsed_mode),
        )

    def get_restore_eligibility(self, event_id: str) -> RestoreEligibilityPayload:
        analysis = self._analyze_restore(event_id)
        requires_confirmation = self._restore_requires_confirmation(analysis)
        conflict_code = analysis.conflict_code
        conflict_reason = analysis.conflict_reason
        if analysis.conflict is not None:
            conflict_code = analysis.conflict.code.value
            conflict_reason = analysis.conflict.reason
        if conflict_code is None:
            blocked = self._restore_protection_conflict(analysis)
            if blocked is not None:
                conflict_code, conflict_reason = blocked
            elif requires_confirmation:
                conflict_code = "confirmation_required"
                conflict_reason = "explicit confirmation is required by the current policy"
        return RestoreEligibilityPayload(
            event_id=str(analysis.event.id),
            eligible=conflict_code is None,
            operation=analysis.event.operation,
            inverse_operation=None if analysis.inverse is None else analysis.inverse.inverse_operation,
            risk=_restore_risk(analysis.event.operation),
            requires_confirmation=requires_confirmation,
            current_record_tokens=analysis.current_record_tokens,
            current_link_tokens=analysis.current_link_tokens,
            protections={
                memory_id: [protection.model_dump(mode="json") for protection in protections]
                for memory_id, protections in analysis.protections.items()
            },
            conflict_code=conflict_code,
            conflict_reason=conflict_reason,
        )

    def request_restore(
        self,
        *,
        event_id: str,
        scope: str = "all",
        expected_record_tokens: object = None,
        expected_link_tokens: object = None,
        actor_id: str | None = None,
        reason: str,
        idempotency_key: str,
        confirmation: bool = False,
    ) -> RestoreRequestPayload:
        dependencies = self._dependencies
        parsed_event_id = _parse_uuid(event_id, field="event_id")
        try:
            parsed_scope = RestoreScope(scope)
        except ValueError as exc:
            raise ValueError("scope_invalid") from exc
        if not idempotency_key.strip():
            raise ValueError("idempotency_key_required")
        if not reason.strip():
            raise ValueError("reason_required")
        request = RestoreRequest(
            target_event_id=parsed_event_id,
            scope=parsed_scope,
            expected_record_tokens=_parse_record_tokens(expected_record_tokens or {}),
            expected_link_tokens=_parse_link_tokens(expected_link_tokens or {}),
            actor_id=actor_id,
            reason=reason,
            idempotency_key=idempotency_key,
            confirmation=confirmation,
        )
        analysis = self._analyze_restore(event_id)
        result: RestoreResult | None = None
        if analysis.conflict_code is not None:
            result = _restore_conflict(parsed_event_id, analysis.conflict_code, analysis.conflict_reason or "restore is not eligible")
        elif analysis.conflict is not None:
            result = _restore_conflict(parsed_event_id, analysis.conflict.code.value, analysis.conflict.reason)
        else:
            blocked = self._restore_protection_conflict(analysis)
            if blocked is not None:
                result = _restore_rejected(parsed_event_id, blocked[0], blocked[1])
            elif self._restore_requires_confirmation(analysis) and not confirmation:
                result = _restore_rejected(
                    parsed_event_id,
                    "confirmation_required",
                    "explicit confirmation is required by the current policy",
                )
            else:
                result = self._restore_token_conflict(request, analysis)
        history = self._require_mutation_history()
        if result is not None:
            registered = history.request_restore(request)
            if registered.status is RestoreResultStatus.ALREADY_APPLIED or registered.event_id is not None:
                return _restore_payload(registered)
            if registered.request_id is None:
                raise RuntimeError("restore request store returned no request id")
            finalized = history.terminalize_restore_request(registered.request_id, result)
            return _restore_payload(finalized)
        if dependencies.action_store is None:
            raise ValueError("curation_action_store_unavailable")
        executed = dependencies.restore_executor(
            dependencies.action_store,
            history,
            curation_store=dependencies.curation,
        ).execute(request)
        return _restore_payload(executed)

    def _analyze_restore(self, event_id: str) -> _RestoreAnalysis:
        dependencies = self._dependencies
        event, records, links = self._get_history_parts(event_id)
        inverse = dependencies.inverse_builder(event, records, links)
        protections: dict[str, list[Protection]] = {}
        current_record_tokens: dict[str, str] = {}
        current_link_tokens: dict[str, str] = {}
        if isinstance(inverse, RestoreConflict):
            return _RestoreAnalysis(event, None, inverse, current_record_tokens, current_link_tokens, protections)
        for change in inverse.record_changes:
            memory_id = str(change.memory_id)
            protections[memory_id] = self._require_mutation_history().get_protections(change.memory_id)
            record = dependencies.memory_queries.get_memory(memory_id) if dependencies.memory_queries is not None else None
            if record is None:
                return _RestoreAnalysis(
                    event, inverse, None, current_record_tokens, current_link_tokens, protections,
                    RestoreConflictCode.STALE_STATE.value, f"target memory {memory_id} is missing",
                )
            current_record_tokens[memory_id] = record_token(record)
            if change.expected_current_token != current_record_tokens[memory_id]:
                return _RestoreAnalysis(
                    event, inverse, None, current_record_tokens, current_link_tokens, protections,
                    RestoreConflictCode.STALE_STATE.value, f"current record token is stale for {memory_id}",
                )
        for change in inverse.link_changes:
            source_id = str(change.source_id)
            target_id = str(change.target_id)
            key = f"{source_id}:{target_id}:{change.link_type}"
            for memory_id, parsed_memory_id in ((source_id, change.source_id), (target_id, change.target_id)):
                protections.setdefault(memory_id, self._require_mutation_history().get_protections(parsed_memory_id))
                record = dependencies.memory_queries.get_memory(memory_id) if dependencies.memory_queries is not None else None
                if record is None:
                    return _RestoreAnalysis(
                        event, inverse, None, current_record_tokens, current_link_tokens, protections,
                        RestoreConflictCode.STALE_STATE.value, f"target memory {memory_id} is missing",
                    )
                current_record_tokens[memory_id] = record_token(record)
            matching = []
            if dependencies.memory_queries is not None:
                matching = [
                    link for link in dependencies.memory_queries.get_links(source_id, direction="outgoing")
                    if str(link.target_id) == target_id and link.link_type == change.link_type
                ]
            current = matching[0] if matching else None
            current_link_tokens[key] = link_token(
                source_id,
                target_id,
                change.link_type,
                None if current is None else current.context,
                exists=current is not None,
            )
            if change.exists != (current is not None) or (current is not None and current.context != change.context):
                return _RestoreAnalysis(
                    event, inverse, None, current_record_tokens, current_link_tokens, protections,
                    RestoreConflictCode.STALE_STATE.value, f"current link state is stale for {key}",
                )
        return _RestoreAnalysis(event, inverse, None, current_record_tokens, current_link_tokens, protections)

    @staticmethod
    def _restore_requires_confirmation(analysis: _RestoreAnalysis) -> bool:
        return any(
            ProtectionMode.MANUAL_REVIEW_REQUIRED in {protection.mode for protection in protections}
            for protections in analysis.protections.values()
        )

    @staticmethod
    def _restore_protection_conflict(analysis: _RestoreAnalysis) -> tuple[str, str] | None:
        if any(
            ProtectionMode.NO_AUTONOMOUS_MUTATION in {protection.mode for protection in protections}
            for protections in analysis.protections.values()
        ):
            return "protection_denied", "current protection does not allow autonomous restore"
        return None

    @staticmethod
    def _restore_token_conflict(request: RestoreRequest, analysis: _RestoreAnalysis) -> RestoreResult | None:
        for memory_id, token in analysis.current_record_tokens.items():
            if request.expected_record_tokens.get(UUID(memory_id)) != token:
                return _restore_conflict(request.target_event_id, "stale_state", f"current record token is stale for {memory_id}")
        for key, token in analysis.current_link_tokens.items():
            if request.expected_link_tokens.get(key) != token:
                return _restore_conflict(request.target_event_id, "stale_state", f"current link token is stale for {key}")
        return None

    def _require_mutation_history(self):
        if self._dependencies.mutation_history is None:
            raise ValueError("mutation_history_unavailable")
        return self._dependencies.mutation_history

    def _get_history_parts(self, event_id: str) -> tuple[MutationEvent, list[RecordRevision], list[LinkRevision]]:
        store = self._require_mutation_history()
        parsed_event_id = _parse_uuid(event_id, field="event_id")
        event = store.get_event(parsed_event_id)
        if event is None:
            raise ValueError("mutation_event_not_found")
        return (
            event,
            store.get_record_revisions(parsed_event_id, limit=_MAX_HISTORY_REVISION_ROWS),
            store.get_link_revisions(parsed_event_id, limit=_MAX_HISTORY_REVISION_ROWS),
        )

    def _history_event_payload(self, event: MutationEvent) -> dict[str, object]:
        payload = event.model_dump(mode="json")
        # Provider rationale is not an audit fact. Persisted revisions and
        # tokens remain the only source for before/after history.
        payload.pop("rationale", None)
        payload["receipt"] = self._receipt_payload(event)
        return payload

    def _receipt_payload(self, event: MutationEvent) -> dict[str, object] | None:
        curation = self._dependencies.curation
        if curation is None or event.curation_run_id is None or event.action_id is None:
            return None
        receipt = curation.get_receipt(event.curation_run_id, event.action_id)
        return None if receipt is None else receipt.model_dump(mode="json")

    def _curation_run_payload(self, event: MutationEvent) -> dict[str, object] | None:
        curation = self._dependencies.curation
        if curation is None or event.curation_run_id is None:
            return None
        run = curation.get_run(event.curation_run_id)
        return None if run is None else run.model_dump(mode="json")

    def _record_revision_payload(self, revision: RecordRevision) -> dict[str, object]:
        return {
            "memory_id": str(revision.memory_id),
            "role": str(revision.role),
            "before_exists": revision.before_exists,
            "after_exists": revision.after_exists,
            "before_snapshot": _bounded_history_value(revision.before_snapshot),
            "after_snapshot": _bounded_history_value(revision.after_snapshot),
            "before_token": revision.before_token,
            "after_token": revision.after_token,
        }

    def _link_revision_payload(self, revision: LinkRevision) -> dict[str, object]:
        return {
            "source_id": str(revision.source_id),
            "target_id": str(revision.target_id),
            "link_type": revision.link_type,
            "context": _bounded_history_value(revision.context),
            "before_exists": revision.before_exists,
            "after_exists": revision.after_exists,
        }

    def _record_diff_payload(self, revision: RecordRevision) -> dict[str, object]:
        return {
            "memory_id": str(revision.memory_id),
            "role": str(revision.role),
            "before": _bounded_history_value(revision.before_snapshot),
            "after": _bounded_history_value(revision.after_snapshot),
            "before_exists": revision.before_exists,
            "after_exists": revision.after_exists,
            "before_token": revision.before_token,
            "after_token": revision.after_token,
        }

    def _link_diff_payload(self, revision: LinkRevision) -> dict[str, object]:
        return {
            "source_id": str(revision.source_id),
            "target_id": str(revision.target_id),
            "link_type": revision.link_type,
            "before_context": _bounded_history_value(revision.context) if revision.before_exists else None,
            "after_context": _bounded_history_value(revision.context) if revision.after_exists else None,
            "before_exists": revision.before_exists,
            "after_exists": revision.after_exists,
        }
