from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast
from uuid import UUID

import pytest

from mcp_memory.core.curation_context import (
    AcceptedMaintenanceRead,
    build_context_packet,
)
from mcp_memory.core.curation_disclosure import ProviderTrust, ProviderTrustClass
from mcp_memory.core.curation_execution_service import (
    CurationExecutionService,
    _hydrate_record_tokens,
)
from mcp_memory.core.curation_executor import CurationPolicyRejection
from mcp_memory.core.curation_models import (
    ActionPreconditions,
    ClaimManifest,
    CreateLinkAction,
    CurationRunOutcome,
    EvidenceRef,
    LinkAssertion,
    MergeMemoriesAction,
    NormalizeMemoryAction,
)
from mcp_memory.core.curation_policy import (
    OperationRisk,
    PolicyDecision,
    PolicyMode,
    RejectionCode,
)
from mcp_memory.core.curation_routing import MaintenanceFamily
from mcp_memory.core.curation_validation import (
    AcceptedCurationAction,
    CurationValidationResult,
)
from mcp_memory.core.ports.curation import CurationActionFatalError
from mcp_memory.curation_store import (
    CurationActionReceipt,
    CurationReceiptState,
    CurationRun,
    CurationRunState,
)

pytestmark = pytest.mark.small

CANONICAL_ID = UUID("00000000-0000-0000-0000-000000000001")
SOURCE_ID = UUID("00000000-0000-0000-0000-000000000002")
INVALID_ACTION_ID = UUID("00000000-0000-0000-0000-000000000005")
VALID_ACTION_ID = UUID("00000000-0000-0000-0000-000000000006")


def _record(memory_id: UUID) -> dict[str, object]:
    return {
        "id": memory_id,
        "title": "Title",
        "content": "Content",
        "summary": "Summary",
        "type": "observation",
        "status": "active",
        "tags": [],
        "workspace_ids": [],
    }


def _context():
    return build_context_packet(
        family="curator",
        strategy="recent",
        seed_reads=[
            AcceptedMaintenanceRead(_record(CANONICAL_ID)),
            AcceptedMaintenanceRead(_record(SOURCE_ID)),
        ],
        provider=ProviderTrust(ProviderTrustClass.LOCAL),
    )


def _normalize_action(action_id: UUID, target_id: UUID = CANONICAL_ID) -> NormalizeMemoryAction:
    return NormalizeMemoryAction(
        action_id=action_id,
        target_id=target_id,
        confidence=1,
        rationale="normalize metadata",
        title="Specific title",
    )


class _ExecutionStore:
    def __init__(self) -> None:
        self.receipts: list[CurationActionReceipt] = []
        self.transitioned: CurationRun | None = None

    def put_receipt(self, receipt: CurationActionReceipt) -> CurationActionReceipt:
        self.receipts.append(receipt)
        return receipt

    def transition_run(
        self,
        run_id: UUID,
        expected_state: CurationRunState,
        run: CurationRun,
    ) -> CurationRun:
        assert run_id == run.run_id
        assert expected_state is CurationRunState.EXECUTING
        self.transitioned = run
        return run


class _ExecutionExecutor:
    def __init__(self, failures: dict[UUID, BaseException] | None = None) -> None:
        self.failures = failures or {}
        self.executed: list[UUID] = []

    def execute_normalize(
        self,
        action: NormalizeMemoryAction,
        *,
        run_id: UUID,
        memory_type: str,
        protections: object,
        expected_token: str | None,
    ) -> CurationActionReceipt:
        self.executed.append(action.action_id)
        failure = self.failures.get(action.action_id)
        if failure is not None:
            raise failure
        return CurationActionReceipt(
            run_id=run_id,
            action_id=action.action_id,
            operation=action.operation,
            affected_ids=[action.target_id],
            status=CurationReceiptState.APPLIED_UNVERIFIED,
            intent_hash="valid-action",
        )


class _ExecutionVerifier:
    def __init__(self) -> None:
        self.verified: list[UUID] = []

    def verify(
        self,
        receipt: CurationActionReceipt,
        action: NormalizeMemoryAction,
    ) -> CurationActionReceipt:
        self.verified.append(action.action_id)
        return receipt.model_copy(
            update={
                "status": CurationReceiptState.VERIFIED,
                "verified_at": datetime.now(UTC),
            }
        )


def _validation(*actions: NormalizeMemoryAction) -> CurationValidationResult:
    return CurationValidationResult(
        plan=None,
        accepted_actions=tuple(
            AcceptedCurationAction(action=action, family=MaintenanceFamily.CURATOR)
            for action in actions
        ),
    )


def _run(run_id: UUID) -> CurationRun:
    return CurationRun(
        run_id=run_id,
        frontier_key="frontier",
        context_fingerprint="context",
        state=CurationRunState.EXECUTING,
    )


def _service(
    store: _ExecutionStore,
    executor: _ExecutionExecutor,
    verifier: _ExecutionVerifier,
) -> CurationExecutionService:
    return CurationExecutionService(
        curation_store=cast(Any, store),
        executor=cast(Any, executor),
        verifier=cast(Any, verifier),
        memory_types={CANONICAL_ID: "observation"},
        contradictory_memory_ids=set(),
        protections_by_memory={},
    )


def test_hydrates_missing_tokens_for_every_affected_memory() -> None:
    context = _context()
    action = MergeMemoriesAction(
        action_id=UUID("00000000-0000-0000-0000-000000000003"),
        canonical_id=CANONICAL_ID,
        source_ids=[SOURCE_ID],
        confidence=1,
        rationale="merge duplicate records",
        content="Merged content",
        claim_manifest=ClaimManifest(preserved_claims=["claim"]),
    )

    hydrated = _hydrate_record_tokens(action, context)

    assert hydrated.preconditions.record_tokens == {
        CANONICAL_ID: context.record_tokens[str(CANONICAL_ID)],
        SOURCE_ID: context.record_tokens[str(SOURCE_ID)],
    }


def test_preserves_provider_supplied_token_mismatch() -> None:
    context = _context()
    action = NormalizeMemoryAction(
        action_id=UUID("00000000-0000-0000-0000-000000000004"),
        target_id=CANONICAL_ID,
        confidence=1,
        rationale="normalize metadata",
        title="Specific title",
        preconditions=ActionPreconditions(record_tokens={CANONICAL_ID: "stale-provider-token"}),
    )

    hydrated = _hydrate_record_tokens(action, context)

    assert hydrated.preconditions.record_tokens[CANONICAL_ID] == "stale-provider-token"


def test_does_not_hydrate_create_link_context_for_absent_link() -> None:
    context = _context()
    action = CreateLinkAction(
        action_id=UUID("00000000-0000-0000-0000-000000000008"),
        source_id=CANONICAL_ID,
        target_id=SOURCE_ID,
        confidence=1,
        rationale="connect related records",
        link_type="RELATED",
        context="The records cover the same operational area.",
        evidence=[
            EvidenceRef(
                link=LinkAssertion(
                    source_id=CANONICAL_ID,
                    target_id=SOURCE_ID,
                    link_type="RELATED",
                    context="The records cover the same operational area.",
                )
            )
        ],
        preconditions=ActionPreconditions(
            absent_links=[
                LinkAssertion(
                    source_id=CANONICAL_ID,
                    target_id=SOURCE_ID,
                    link_type="RELATED",
                )
            ]
        ),
    )

    hydrated = _hydrate_record_tokens(action, context)

    assert hydrated.preconditions.absent_links[0].context is None
    assert set(hydrated.preconditions.record_tokens) == {CANONICAL_ID, SOURCE_ID}


def test_preserves_explicit_create_link_absent_context_without_hydration() -> None:
    context = _context()
    action = CreateLinkAction(
        action_id=UUID("00000000-0000-0000-0000-000000000009"),
        source_id=CANONICAL_ID,
        target_id=SOURCE_ID,
        confidence=1,
        rationale="connect related records",
        link_type="RELATED",
        context="The records cover the same operational area.",
        evidence=[
            EvidenceRef(
                link=LinkAssertion(
                    source_id=CANONICAL_ID,
                    target_id=SOURCE_ID,
                    link_type="RELATED",
                    context="The records cover the same operational area.",
                )
            )
        ],
        preconditions=ActionPreconditions(
            absent_links=[
                LinkAssertion(
                    source_id=CANONICAL_ID,
                    target_id=SOURCE_ID,
                    link_type="RELATED",
                    context="Different relationship evidence.",
                )
            ]
        ),
    )

    hydrated = _hydrate_record_tokens(action, context)

    assert hydrated.preconditions.absent_links[0].context == "Different relationship evidence."


def test_invalid_action_receipt_does_not_block_valid_action() -> None:
    run_id = UUID("00000000-0000-0000-0000-000000000007")
    store = _ExecutionStore()
    executor = _ExecutionExecutor({INVALID_ACTION_ID: CurationActionFatalError("missing evidence")})
    verifier = _ExecutionVerifier()

    outcome, reason, state, receipts = _service(store, executor, verifier).execute(
        run_id=run_id,
        executing=_run(run_id),
        validation=_validation(
            _normalize_action(INVALID_ACTION_ID),
            _normalize_action(VALID_ACTION_ID, SOURCE_ID),
        ),
        context=_context(),
    )

    assert (outcome, reason, state) == (
        CurationRunOutcome.PARTIALLY_APPLIED,
        "partially_applied",
        CurationRunState.VERIFYING,
    )
    assert [receipt.status for receipt in receipts] == [
        CurationReceiptState.REJECTED,
        CurationReceiptState.VERIFIED,
    ]
    assert receipts[0].error_code == "action_fatal"
    assert receipts[0].intent_hash is not None
    assert store.receipts == [receipts[0]]
    assert verifier.verified == [VALID_ACTION_ID]
    assert executor.executed == [INVALID_ACTION_ID, VALID_ACTION_ID]
    assert store.transitioned is not None
    assert store.transitioned.rejection_codes == ["action_fatal"]


def test_policy_rejection_is_durable_and_all_invalid_actions_defer() -> None:
    run_id = UUID("00000000-0000-0000-0000-000000000008")
    action = _normalize_action(INVALID_ACTION_ID)
    decision = PolicyDecision(
        operation=action.operation,
        risk=OperationRisk.LOW,
        mode=PolicyMode.ENABLED,
        rejection_codes=(RejectionCode.GENERIC_SUMMARY,),
    )
    store = _ExecutionStore()
    executor = _ExecutionExecutor({action.action_id: CurationPolicyRejection(decision)})
    verifier = _ExecutionVerifier()

    outcome, reason, _, receipts = _service(store, executor, verifier).execute(
        run_id=run_id,
        executing=_run(run_id),
        validation=_validation(action),
        context=_context(),
    )

    assert (outcome, reason) == (CurationRunOutcome.DEFERRED, "all_actions_rejected")
    assert receipts == tuple(store.receipts)
    assert receipts[0].status is CurationReceiptState.REJECTED
    assert receipts[0].error_code == "policy_rejected"
    assert verifier.verified == []


def test_unexpected_runtime_error_is_not_converted_to_rejection() -> None:
    run_id = UUID("00000000-0000-0000-0000-000000000009")
    store = _ExecutionStore()
    executor = _ExecutionExecutor({INVALID_ACTION_ID: RuntimeError("provider bug")})

    with pytest.raises(RuntimeError, match="provider bug"):
        _service(store, executor, _ExecutionVerifier()).execute(
            run_id=run_id,
            executing=_run(run_id),
            validation=_validation(_normalize_action(INVALID_ACTION_ID)),
            context=_context(),
        )

    assert store.receipts == []
