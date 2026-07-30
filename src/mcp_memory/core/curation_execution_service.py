"""Application-owned execution and verification for accepted curation actions."""

from __future__ import annotations

from typing import Any, Mapping
from uuid import UUID

from mcp_memory.core.curation_context import CurationContextPacket
from mcp_memory.core.curation_executor import CurationExecutor
from mcp_memory.core.curation_models import (
    ArchiveMemoryAction,
    CreateLinkAction,
    CurationRunOutcome,
    MergeMemoriesAction,
    NormalizeMemoryAction,
    RemoveLinkAction,
    RewriteMemoryAction,
    SplitMemoryAction,
)
from mcp_memory.core.curation_validation import CurationValidationResult
from mcp_memory.core.curation_verifier import CurationVerifier
from mcp_memory.curation_store import (
    CurationActionReceipt,
    CurationRepository,
    CurationRun,
    CurationRunState,
)


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
        actions = [item.action for item in validation.accepted_actions]
        if not actions:
            return CurationRunOutcome.DEFERRED, "unsupported_execution_action", CurationRunState.EXECUTING, ()
        memory_types = dict(self._memory_types or {})
        for record in (*context.seeds, *context.support):
            memory_id, memory_type = record.get("memory_id"), record.get("type")
            if memory_id is not None and isinstance(memory_type, str):
                memory_types[UUID(str(memory_id))] = memory_type
        receipts = [(self._execute_action(action, run_id, memory_types, context), action) for action in actions]
        verifying = executing.model_copy(update={"state": CurationRunState.VERIFYING})
        if self._store.transition_run(run_id, CurationRunState.EXECUTING, verifying) is None:
            raise RuntimeError(f"curation run {run_id} could not enter verification")
        verified: list[CurationActionReceipt] = []
        for receipt, action in receipts:
            result = self._verifier.verify(receipt, action)
            verified.append(result)
            if result.status.value != "verified":
                return CurationRunOutcome.VERIFICATION_FAILED, "verification_failed", CurationRunState.VERIFYING, tuple(verified)
        route_ids = {item.action_id for item in actions}
        partial = bool(validation.rejected_actions or any(route.action.action_id not in route_ids for route in validation.specialist_routes))
        return (
            CurationRunOutcome.PARTIALLY_APPLIED if partial else CurationRunOutcome.APPLIED,
            "partially_applied" if partial else "verified_receipts",
            CurationRunState.VERIFYING,
            tuple(verified),
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
