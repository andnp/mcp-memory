from uuid import uuid4

from mcp_memory.mutation_history import ProtectionMode

from mcp_memory.core.curation_models import (
    ArchiveMemoryAction,
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
    is_generic_summary,
    policy_mode,
)


def test_policy_enables_all_supported_operations() -> None:
    action = NormalizeMemoryAction(action_id=uuid4(), target_id=uuid4(), confidence=1, rationale="specific", summary="specific")
    decision = evaluate_curation_action(action, memory_types={action.target_id: "observation"})
    assert decision.risk is OperationRisk.LOW
    assert decision.mode is PolicyMode.ENABLED
    assert decision.authorized
    assert all(
        policy_mode(operation) is PolicyMode.ENABLED
        for operation in (
            "normalize_memory",
            "create_link",
            "rewrite_memory",
            "remove_link",
            "merge_memories",
            "split_memory",
            "archive_memory",
        )
    )


def test_generic_summary_predicate_covers_known_boilerplate() -> None:
    assert is_generic_summary("Covers several related findings.")
    assert is_generic_summary("Added several related findings.")
    assert not is_generic_summary("The daemon now reports the Ollama backend.")


def test_policy_rejects_empty_normalize_even_if_model_validation_is_bypassed() -> None:
    action = NormalizeMemoryAction.model_construct(
        action_id=uuid4(), target_id=uuid4(), confidence=1, rationale="empty"
    )
    decision = evaluate_curation_action(action, memory_types={action.target_id: "observation"})
    assert RejectionCode.EMPTY_NORMALIZE in decision.rejection_codes
    assert not decision.authorized


def test_risky_action_requires_evidence_and_complete_manifest() -> None:
    action = MergeMemoriesAction(
        action_id=uuid4(), canonical_id=uuid4(), source_ids=[uuid4()], confidence=1, rationale="same subject",
        content="claim", claim_manifest=ClaimManifest(),
    )
    decision = evaluate_curation_action(action, memory_types={action.canonical_id: "observation", action.source_ids[0]: "observation"})
    assert decision.mode is PolicyMode.ENABLED
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


def test_no_autonomous_mutation_rejects_even_enabled_actions() -> None:
    action = NormalizeMemoryAction(action_id=uuid4(), target_id=uuid4(), confidence=1, rationale="specific", summary="specific")
    decision = evaluate_curation_action(
        action,
        memory_types={action.target_id: "observation"},
        protections_by_memory={action.target_id: {ProtectionMode.NO_AUTONOMOUS_MUTATION}},
    )
    assert RejectionCode.NO_AUTONOMOUS_MUTATION in decision.rejection_codes
    assert not decision.authorized


def test_destructive_and_manual_review_protections_require_review() -> None:
    action = MergeMemoriesAction(
        action_id=uuid4(), canonical_id=uuid4(), source_ids=[uuid4()], confidence=1, rationale="same subject",
        content="claim", claim_manifest=ClaimManifest(preserved_claims=["claim"]),
    )
    protections = {ProtectionMode.NO_AUTONOMOUS_DESTRUCTIVE_CHANGE, ProtectionMode.MANUAL_REVIEW_REQUIRED}
    decision = evaluate_curation_action(
        action,
        memory_types={action.canonical_id: "observation", action.source_ids[0]: "observation"},
        protections_by_memory={action.canonical_id: protections},
    )
    assert RejectionCode.DESTRUCTIVE_CHANGE_REQUIRES_REVIEW in decision.rejection_codes
    assert RejectionCode.MANUAL_REVIEW_REQUIRED in decision.rejection_codes


def test_pinned_active_rejects_archive() -> None:
    action = ArchiveMemoryAction(
        action_id=uuid4(), target_id=uuid4(), confidence=1, rationale="old", claim_manifest=ClaimManifest(preserved_claims=["old"]),
    )
    decision = evaluate_curation_action(
        action,
        memory_types={action.target_id: "observation"},
        protections_by_memory={action.target_id: {ProtectionMode.PINNED_ACTIVE}},
    )
    assert RejectionCode.PINNED_ACTIVE in decision.rejection_codes
