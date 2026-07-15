from uuid import uuid4

from mcp_memory.core.curation_models import (
    ClaimManifest,
    ClaimMapping,
    EvidenceRef,
    MergeMemoriesAction,
    NormalizeMemoryAction,
)
from mcp_memory.core.curation_policy import (
    OperationRisk,
    PolicyMode,
    RejectionCode,
    evaluate_curation_action,
)


def test_initial_policy_enables_only_low_risk_operations() -> None:
    action = NormalizeMemoryAction(action_id=uuid4(), target_id=uuid4(), confidence=1, rationale="specific", summary="specific")
    decision = evaluate_curation_action(action, memory_types={action.target_id: "observation"})
    assert decision.risk is OperationRisk.LOW
    assert decision.mode is PolicyMode.ENABLED
    assert decision.authorized


def test_risky_action_requires_evidence_and_complete_manifest() -> None:
    action = MergeMemoriesAction(
        action_id=uuid4(), canonical_id=uuid4(), source_ids=[uuid4()], confidence=1, rationale="same subject",
        content="claim", claim_manifest=ClaimManifest(),
    )
    decision = evaluate_curation_action(action, memory_types={action.canonical_id: "observation", action.source_ids[0]: "observation"})
    assert decision.mode is PolicyMode.SHADOW
    assert RejectionCode.EVIDENCE_REQUIRED in decision.rejection_codes
    assert RejectionCode.CLAIM_MANIFEST_INCOMPLETE in decision.rejection_codes
    assert not decision.authorized


def test_contradictory_facts_never_merge_without_explicit_tension() -> None:
    canonical, source = uuid4(), uuid4()
    action = MergeMemoriesAction(
        action_id=uuid4(), canonical_id=canonical, source_ids=[source], confidence=1, rationale="facts",
        content="both claims", evidence=[EvidenceRef(memory_id=canonical)],
        claim_manifest=ClaimManifest(preserved_claims=["both claims"], source_mapping=[ClaimMapping(output="content", source_memory_ids=[canonical, source])]),
    )
    decision = evaluate_curation_action(action, memory_types={canonical: "fact", source: "fact"}, contradictory_memory_ids={source})
    assert RejectionCode.CONTRADICTORY_FACTS in decision.rejection_codes


def test_delete_intent_is_disabled_and_typed() -> None:
    decision = evaluate_curation_action({"operation": "delete_memory"})
    assert decision.mode is PolicyMode.DISABLED
    assert RejectionCode.DELETE_NOT_SUPPORTED in decision.rejection_codes
