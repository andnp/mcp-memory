"""Independent evidence for direct internal mutation tool calls."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Iterable
from uuid import uuid4


class DirectMutationOutcome(StrEnum):
    APPLIED_VERIFIED = "applied_verified"
    APPLIED_UNVERIFIED = "applied_unverified"
    LEDGER_INVALID = "ledger_invalid"
    MUTATION_EVIDENCE_INVALID = "mutation_evidence_invalid"
    NO_OP = "no_op"


class EvidenceEntityKind(StrEnum):
    RECORD = "record"
    LINK = "link"


@dataclass(frozen=True)
class DirectMutationEntityDelta:
    kind: str
    entity_id: str
    before_revision: str | None = None
    after_revision: str | None = None
    before_exists: bool = False
    after_exists: bool = False
    transition: str = "updated"
    snapshot: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DirectMutationEvidence:
    evidence_id: str
    task_id: str | None
    execution_epoch: int | None
    session_id: str | None
    call_id: str | None
    sequence: int | None
    tool_name: str
    operation: str
    idempotency_key: str
    payload: dict[str, Any] = field(default_factory=dict)
    ledger_entry: dict[str, Any] = field(default_factory=dict)
    deltas: tuple[DirectMutationEntityDelta, ...] = ()
    outcome: str | None = None
    error_code: str | None = None
    started_at: str = ""
    completed_at: str | None = None
    finalized_at: str | None = None

    @classmethod
    def start(
        cls,
        *,
        task_id: str | None,
        execution_epoch: int | None,
        session_id: str | None,
        call_id: str | None,
        sequence: int | None,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> "DirectMutationEvidence":
        key = call_id or f"{task_id or 'missing'}:{execution_epoch or 0}:{tool_name}:{uuid4().hex}"
        now = _now()
        return cls(
            evidence_id=uuid4().hex,
            task_id=task_id,
            execution_epoch=execution_epoch,
            session_id=session_id,
            call_id=call_id,
            sequence=sequence,
            tool_name=tool_name,
            operation=tool_name.removeprefix("internal_").removesuffix("_service"),
            idempotency_key=key,
            payload={"arguments": _json_safe(arguments)},
            started_at=now,
        )

    def finish(
        self,
        *,
        payload: dict[str, Any],
        ledger_entry: dict[str, Any],
        deltas: Iterable[DirectMutationEntityDelta] = (),
        outcome: str | None = None,
        error_code: str | None = None,
    ) -> "DirectMutationEvidence":
        merged_payload = _json_safe(payload)
        if isinstance(merged_payload, dict) and isinstance(self.payload.get("before_entities"), dict):
            merged_payload = {**merged_payload, "before_entities": self.payload["before_entities"]}
        return replace(
            self,
            payload=merged_payload, ledger_entry=_json_safe(ledger_entry),
            deltas=tuple(deltas), outcome=outcome, error_code=error_code,
            completed_at=_now(), finalized_at=_now(),
        )


def reconcile_direct_mutation_evidence(
    evidence: DirectMutationEvidence,
    *,
    ledger_entry: dict[str, Any] | None = None,
    payload: dict[str, Any] | None = None,
    semantic_postcondition: bool | None = None,
) -> DirectMutationEvidence:
    """Classify evidence without trusting provider summaries.

    A verified result requires a matching successful ledger entry, a valid
    success payload, at least one independently observed delta, and a true
    semantic postcondition when one was supplied.
    """
    ledger = ledger_entry if ledger_entry is not None else evidence.ledger_entry
    result = payload if payload is not None else evidence.payload
    if not _ledger_matches(evidence, ledger):
        outcome = DirectMutationOutcome.LEDGER_INVALID
        error = "ledger_missing_or_mismatched"
    elif not isinstance(result, dict) or result.get("status", "ok") == "error":
        outcome = DirectMutationOutcome.NO_OP
        error = "mutation_not_applied"
    elif not _valid_deltas(evidence.deltas):
        outcome = DirectMutationOutcome.MUTATION_EVIDENCE_INVALID
        error = "missing_or_malformed_entity_delta"
    elif semantic_postcondition is False:
        outcome = DirectMutationOutcome.APPLIED_UNVERIFIED
        error = "semantic_postcondition_failed"
    elif semantic_postcondition is None:
        outcome = DirectMutationOutcome.APPLIED_UNVERIFIED
        error = "semantic_postcondition_missing"
    else:
        outcome = DirectMutationOutcome.APPLIED_VERIFIED
        error = None
    return replace(
        evidence,
        payload=_json_safe(result), ledger_entry=_json_safe(ledger),
        outcome=str(outcome), error_code=error,
        finalized_at=evidence.finalized_at or _now(),
    )


def productive_mutation_count(evidence: Iterable[DirectMutationEvidence]) -> int:
    return sum(item.outcome == DirectMutationOutcome.APPLIED_VERIFIED for item in evidence)


def entity_deltas_for_payload(
    payload: dict[str, Any],
    arguments: dict[str, Any],
    *,
    repository: Any = None,
    before_entities: dict[str, dict[str, Any]] | None = None,
) -> tuple[DirectMutationEntityDelta, ...]:
    """Extract record/link transitions from compact internal tool responses."""
    records = list(_record_payloads(payload))
    seen: set[str] = set()
    deltas: list[DirectMutationEntityDelta] = []
    for key, record in records:
        entity_id = record.get("id")
        if not isinstance(entity_id, str) or entity_id in seen:
            continue
        seen.add(entity_id)
        before_payload = (before_entities or {}).get(entity_id)
        get_memory = getattr(repository, "get_memory", None)
        before = get_memory(entity_id) if callable(get_memory) and before_payload is None else None
        before_revision = (before_payload or {}).get("updated_at") or getattr(before, "updated_at", None)
        after_revision = record.get("updated_at") if isinstance(record.get("updated_at"), str) else None
        transition = "created" if before is None else "updated"
        if key == "deleted":
            transition, after_revision = "deleted", None
        elif record.get("status") == "archived" and getattr(before, "status", None) != "archived":
            transition = "archived"
        deltas.append(DirectMutationEntityDelta(
            kind=EvidenceEntityKind.RECORD,
            entity_id=entity_id,
            before_revision=before_revision,
            after_revision=after_revision,
            before_exists=before_payload is not None or before is not None,
            after_exists=key != "deleted",
            transition=transition,
            snapshot=dict(record),
        ))
    link = payload.get("link")
    if not isinstance(link, dict):
        link = {
            "source_id": arguments.get("source_id"),
            "target_id": arguments.get("target_id"),
            "link_type": arguments.get("link_type"),
        }
    source_id, target_id, link_type = link.get("source_id"), link.get("target_id"), link.get("link_type")
    if all(isinstance(item, str) and item for item in (source_id, target_id, link_type)):
        entity_id = f"{source_id}:{target_id}:{link_type}"
        transition = "created" if "link" in payload else "deleted"
        deltas.append(DirectMutationEntityDelta(
            kind=EvidenceEntityKind.LINK, entity_id=entity_id,
            before_exists=transition == "deleted", after_exists=transition != "deleted",
            transition=transition, snapshot=dict(link),
        ))
    return tuple(deltas)


def _record_payloads(payload: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any]]]:
    for key, value in payload.items():
        if isinstance(value, dict) and isinstance(value.get("id"), str):
            yield key, value
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and isinstance(item.get("id"), str):
                    yield key, item


def _ledger_matches(evidence: DirectMutationEvidence, ledger: object) -> bool:
    if not isinstance(ledger, dict):
        return False
    return (
        ledger.get("status") == "success"
        and ledger.get("tool_name") == evidence.tool_name
        and (evidence.call_id is None or ledger.get("call_id") == evidence.call_id)
        and (evidence.task_id is None or ledger.get("task_id") == evidence.task_id)
        and (
            evidence.execution_epoch is None
            or ledger.get("execution_epoch") == evidence.execution_epoch
        )
    )


def _valid_deltas(deltas: Iterable[DirectMutationEntityDelta]) -> bool:
    values = tuple(deltas)
    return bool(values) and all(
        item.kind in {EvidenceEntityKind.RECORD, EvidenceEntityKind.LINK}
        and bool(item.entity_id)
        and item.before_exists != item.after_exists
        or (
            item.kind in {EvidenceEntityKind.RECORD, EvidenceEntityKind.LINK}
            and bool(item.entity_id)
            and item.before_revision != item.after_revision
        )
        for item in values
    )


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return str(value)
    return value


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
