"""Application-owned execution and verification for accepted curation actions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from uuid import UUID

from mcp_memory.core.curation_context import CurationContextPacket
from mcp_memory.core.curation_executor import CurationExecutor, CurationPolicyRejection
from mcp_memory.core.curation_identity import action_intent_token
from mcp_memory.core.curation_models import (
    ArchiveMemoryAction,
    CurationAction,
    CreateLinkAction,
    CurationRunOutcome,
    LinkAssertion,
    MergeMemoriesAction,
    NormalizeMemoryAction,
    RemoveLinkAction,
    RewriteMemoryAction,
    SplitMemoryAction,
)
from mcp_memory.core.curation_validation import CurationValidationResult
from mcp_memory.core.curation_verifier import CurationVerifier
from mcp_memory.core.ports.curation import (
    CurationActionFatalError,
    CurationActionReceipt,
    CurationReceiptState,
    CurationRepository,
    CurationRun,
    CurationRunState,
)

_ACTION_FATAL_ERROR_CODE = "action_fatal"
_POLICY_REJECTION_ERROR_CODE = "policy_rejected"


class CurationExecutionService:
    def __init__(
        self,
        *,
        curation_store: CurationRepository,
        executor: CurationExecutor | None,
        verifier: CurationVerifier | None,
        memory_types: Mapping[UUID, str] | None,
        contradictory_memory_ids: set[UUID],
        protections_by_memory: Mapping[UUID, Any] | None,
    ) -> None:
        self._store = curation_store
        self._executor = executor
        self._verifier = verifier
        self._memory_types = memory_types
        self._contradictory_memory_ids = contradictory_memory_ids
        self._protections = protections_by_memory

    def execute(
        self,
        *,
        run_id: UUID,
        executing: CurationRun,
        validation: CurationValidationResult,
        context: CurationContextPacket,
    ) -> tuple[CurationRunOutcome, str, CurationRunState, tuple[CurationActionReceipt, ...]]:
        if self._executor is None or self._verifier is None:
            raise RuntimeError("curation execution requires an executor and verifier")
        actions = [_hydrate_record_tokens(item.action, context) for item in validation.accepted_actions]
        if not actions:
            return CurationRunOutcome.DEFERRED, "unsupported_execution_action", CurationRunState.EXECUTING, ()
        memory_types = dict(self._memory_types or {})
        for record in (*context.seeds, *context.support):
            memory_id, memory_type = record.get("memory_id"), record.get("type")
            if memory_id is not None and isinstance(memory_type, str):
                memory_types[UUID(str(memory_id))] = memory_type
        receipts = [
            (self._execute_action_or_reject(action, run_id, memory_types, context), action)
            for action in actions
        ]
        rejection_codes = list(executing.rejection_codes)
        rejection_codes.extend(
            receipt.error_code
            for receipt, _ in receipts
            if receipt.status is CurationReceiptState.REJECTED
            and receipt.error_code is not None
            and receipt.error_code not in rejection_codes
        )
        verifying = executing.model_copy(
            update={
                "state": CurationRunState.VERIFYING,
                "rejection_codes": rejection_codes,
            }
        )
        if self._store.transition_run(run_id, CurationRunState.EXECUTING, verifying) is None:
            raise RuntimeError(f"curation run {run_id} could not enter verification")
        verified: list[CurationActionReceipt] = []
        for receipt, action in receipts:
            if receipt.status is CurationReceiptState.REJECTED:
                verified.append(receipt)
                continue
            result = self._verifier.verify(receipt, action)
            verified.append(result)
            if result.status.value != "verified":
                return CurationRunOutcome.VERIFICATION_FAILED, "verification_failed", CurationRunState.VERIFYING, tuple(verified)
        route_ids = {item.action_id for item in actions}
        rejected = any(receipt.status is CurationReceiptState.REJECTED for receipt in verified)
        if rejected and not any(receipt.status is not CurationReceiptState.REJECTED for receipt in verified):
            return CurationRunOutcome.DEFERRED, "all_actions_rejected", CurationRunState.VERIFYING, tuple(verified)
        partial = bool(
            rejected
            or validation.rejected_actions
            or any(route.action.action_id not in route_ids for route in validation.specialist_routes)
        )
        return (
            CurationRunOutcome.PARTIALLY_APPLIED if partial else CurationRunOutcome.APPLIED,
            "partially_applied" if partial else "verified_receipts",
            CurationRunState.VERIFYING,
            tuple(verified),
        )

    def _execute_action_or_reject(
        self,
        action: Any,
        run_id: UUID,
        memory_types: dict[UUID, str],
        context: CurationContextPacket,
    ) -> CurationActionReceipt:
        try:
            return self._execute_action(action, run_id, memory_types, context)
        except CurationPolicyRejection:
            return self._rejected_receipt(action, run_id, _POLICY_REJECTION_ERROR_CODE)
        except CurationActionFatalError:
            return self._rejected_receipt(action, run_id, _ACTION_FATAL_ERROR_CODE)

    def _rejected_receipt(
        self,
        action: Any,
        run_id: UUID,
        error_code: str,
    ) -> CurationActionReceipt:
        target_ids = sorted((str(value) for value in _action_memory_ids(action)), key=lambda value: value.encode("utf-8"))
        return self._store.put_receipt(
            CurationActionReceipt(
                run_id=run_id,
                action_id=action.action_id,
                operation=action.operation,
                affected_ids=[UUID(value) for value in target_ids],
                status=CurationReceiptState.REJECTED,
                intent_hash=action_intent_token(
                    operation=action.operation,
                    target_ids=target_ids,
                    expected_tokens={
                        str(key): str(value)
                        for key, value in action.preconditions.record_tokens.items()
                    },
                    preconditions=action.preconditions,
                    payload=action.model_dump(mode="json"),
                ),
                error_code=error_code,
            )
        )

    def _execute_action(self, action: Any, run_id: UUID, memory_types: dict[UUID, str], context: CurationContextPacket) -> CurationActionReceipt:
        executor = self._executor
        if executor is None:
            raise RuntimeError("curation execution requires an executor and verifier")
        protections = self._protections or {}
        if isinstance(action, NormalizeMemoryAction):
            target = action.target_id
            return executor.execute_normalize(action, run_id=run_id, memory_type=memory_types.get(target, ""), protections=protections.get(target, ()), expected_token=context.record_tokens.get(str(target)))
        if isinstance(action, CreateLinkAction):
            ids = (action.source_id, action.target_id)
            return executor.execute_create_link(action, run_id=run_id, memory_types=memory_types, protections={p for i in ids for p in protections.get(i, ())})
        if isinstance(action, RewriteMemoryAction):
            target = action.target_id
            return executor.execute_rewrite(action, run_id=run_id, memory_type=memory_types.get(target, ""), protections=protections.get(target, ()))
        if isinstance(action, RemoveLinkAction):
            ids = (action.source_id, action.target_id)
            return executor.execute_remove_link(action, run_id=run_id, memory_types=memory_types, protections={p for i in ids for p in protections.get(i, ())})
        if isinstance(action, MergeMemoriesAction):
            ids = (action.canonical_id, *action.source_ids)
            return executor.execute_merge(action, run_id=run_id, memory_types=memory_types, contradictory_memory_ids=self._contradictory_memory_ids, protections={p for i in ids for p in protections.get(i, ())})
        if isinstance(action, SplitMemoryAction):
            target = action.target_id
            return executor.execute_split(action, run_id=run_id, memory_type=memory_types.get(target, ""), protections=protections.get(target, ()))
        if isinstance(action, ArchiveMemoryAction):
            target = action.target_id
            return executor.execute_archive(action, run_id=run_id, memory_type=memory_types.get(target, ""), protections=protections.get(target, ()))
        raise RuntimeError(f"unsupported curation action: {type(action).__name__}")


def _hydrate_record_tokens(
    action: CurationAction,
    context: CurationContextPacket,
) -> CurationAction:
    """Fill safe omissions without changing provider-supplied preconditions."""
    record_tokens = dict(action.preconditions.record_tokens)
    for memory_id in _action_memory_ids(action):
        token = context.record_tokens.get(str(memory_id))
        if token is not None:
            record_tokens.setdefault(memory_id, token)
    preconditions = action.preconditions
    if isinstance(action, CreateLinkAction):
        absent_links = _hydrate_create_link_preconditions(action, preconditions.absent_links)
        if absent_links != preconditions.absent_links:
            preconditions = preconditions.model_copy(update={"absent_links": absent_links})
    if (
        record_tokens == preconditions.record_tokens
        and preconditions is action.preconditions
    ):
        return action
    preconditions = preconditions.model_copy(update={"record_tokens": record_tokens})
    return action.model_copy(update={"preconditions": preconditions})


def _hydrate_create_link_preconditions(
    action: CreateLinkAction,
    absent_links: list[LinkAssertion],
) -> list[LinkAssertion]:
    if action.context is None:
        return absent_links
    hydrated: list[LinkAssertion] = []
    for assertion in absent_links:
        if (
            assertion.context is None
            and assertion.source_id == action.source_id
            and assertion.target_id == action.target_id
            and assertion.link_type.strip() == action.link_type.strip()
        ):
            hydrated.append(assertion.model_copy(update={"context": action.context}))
        else:
            hydrated.append(assertion)
    return hydrated


def _action_memory_ids(action: CurationAction) -> set[UUID]:
    ids: set[UUID] = set()
    for attribute in ("target_id", "source_id", "canonical_id"):
        value = getattr(action, attribute, None)
        if isinstance(value, UUID):
            ids.add(value)
    ids.update(value for value in getattr(action, "source_ids", ()) if isinstance(value, UUID))
    return ids
