from uuid import uuid4

import pytest
from pydantic import TypeAdapter, ValidationError

from mcp_memory.core.curation_models import (
    ArchiveMemoryAction,
    CampaignAcceptance,
    CampaignHypothesis,
    CampaignRetrievalProblem,
    CampaignTargetMode,
    ClaimManifest,
    ClaimMapping,
    CreateLinkAction,
    CurationAction,
    CurationBudgetUsage,
    EvidenceRef,
    LinkAssertion,
    MergeMemoriesAction,
    MutationReceipt,
    NormalizeMemoryAction,
    ReceiptStatus,
    RemoveLinkAction,
    RewriteMemoryAction,
    SplitMemoryAction,
    campaign_hypothesis_from_payload,
)


def test_action_union_rejects_unknown_and_delete_operations() -> None:
    adapter = TypeAdapter(CurationAction)
    with pytest.raises(ValidationError):
        adapter.validate_python({"operation": "delete_memory", "target_id": str(uuid4())})
    with pytest.raises(ValidationError):
        adapter.validate_python({"operation": "unknown", "target_id": str(uuid4())})


def test_defaults_are_isolated_and_contracts_are_pure() -> None:
    first = CurationBudgetUsage()
    second = CurationBudgetUsage()
    first.seed_records = 1
    assert second.seed_records == 0

    receipt = MutationReceipt(
        run_id=uuid4(), action_id=uuid4(), operation="archive_memory", status=ReceiptStatus.VERIFIED,
    )
    assert receipt.status == ReceiptStatus.VERIFIED


def test_campaign_hypothesis_is_typed_and_legacy_payloads_are_deterministic() -> None:
    expected = uuid4()
    hypothesis = CampaignHypothesis(
        retrieval_problem=CampaignRetrievalProblem.RETRIEVAL_QUALITY,
        expected_memory_ids=[expected],
        acceptance=CampaignAcceptance(
            target_mode=CampaignTargetMode.TOP_K,
            top_k=3,
            minimum_improvement=0.2,
        ),
    )

    assert hypothesis.expected_memory_ids == [expected]
    assert hypothesis.acceptance.top_k == 3
    assert campaign_hypothesis_from_payload({"campaign_hypothesis": hypothesis.model_dump(mode="json")}) == hypothesis
    assert campaign_hypothesis_from_payload({}) == CampaignHypothesis.legacy()


def test_archive_action_is_typed_and_has_no_delete_sibling() -> None:
    action = ArchiveMemoryAction(
        action_id=uuid4(), target_id=uuid4(), confidence=0.9, rationale="preserve lineage",
        claim_manifest=ClaimManifest(preserved_claims=["claim"]),
    )
    assert action.operation == "archive_memory"


def test_normalize_requires_at_least_one_metadata_field() -> None:
    with pytest.raises(ValidationError, match="at least one metadata field"):
        NormalizeMemoryAction(action_id=uuid4(), target_id=uuid4(), confidence=1, rationale="empty")

    action = NormalizeMemoryAction(
        action_id=uuid4(), target_id=uuid4(), confidence=1, rationale="clear tags", tags=[]
    )
    assert action.tags == []


@pytest.mark.parametrize("action_type", [CreateLinkAction, RemoveLinkAction])
@pytest.mark.parametrize("link_type", ["", "documents_cause", "1CAUSE", "CAUSE-HAS"])
def test_link_actions_require_canonical_link_type(
    action_type: type[CreateLinkAction] | type[RemoveLinkAction], link_type: str
) -> None:
    with pytest.raises(ValidationError):
        action_type(
            action_id=uuid4(),
            source_id=uuid4(),
            target_id=uuid4(),
            link_type=link_type,
            confidence=1,
            rationale="link memories",
        )


def test_link_actions_allow_project_defined_canonical_link_type() -> None:
    source_id = uuid4()
    target_id = uuid4()
    action = CreateLinkAction(
        action_id=uuid4(),
        source_id=source_id,
        target_id=target_id,
        link_type="PROJECT_CAUSES_V2",
        confidence=1,
        rationale="link memories",
        context="The source causes the target.",
        evidence=[
            EvidenceRef(
                link=LinkAssertion(
                    source_id=source_id,
                    target_id=target_id,
                    link_type="PROJECT_CAUSES_V2",
                    context="The source causes the target.",
                )
            )
        ],
    )
    assert action.link_type == "PROJECT_CAUSES_V2"


def test_create_link_rejects_label_only_context() -> None:
    source_id = uuid4()
    target_id = uuid4()

    with pytest.raises(ValidationError, match="descriptive relationship context"):
        CreateLinkAction(
            action_id=uuid4(),
            source_id=source_id,
            target_id=target_id,
            link_type="RELATED",
            confidence=1,
            rationale="link memories",
            context="workflow-policy",
            evidence=[
                EvidenceRef(
                    link=LinkAssertion(
                        source_id=source_id,
                        target_id=target_id,
                        link_type="RELATED",
                        context="workflow-policy",
                    )
                )
            ],
        )


def test_action_union_accepts_all_current_operations() -> None:
    target_id = uuid4()
    source_id = uuid4()
    canonical_id = uuid4()
    child_id = uuid4()
    adapter = TypeAdapter(CurationAction)
    samples = [
        NormalizeMemoryAction(action_id=uuid4(), target_id=target_id, confidence=1, rationale="normalize", title="Title"),
        RewriteMemoryAction(
            action_id=uuid4(),
            target_id=target_id,
            confidence=1,
            rationale="rewrite",
            content="Rewritten content.",
            claim_manifest=ClaimManifest(
                preserved_claims=["claim"],
                transformed_claims=["claim"],
                source_mapping=[ClaimMapping(output="claim", source_memory_ids=[target_id])],
            ),
            evidence=[EvidenceRef(memory_id=target_id)],
        ),
        CreateLinkAction(
            action_id=uuid4(),
            source_id=source_id,
            target_id=target_id,
            link_type="DEPENDS_ON",
            confidence=1,
            rationale="link",
            context="The source depends on the target.",
            evidence=[
                EvidenceRef(
                    link=LinkAssertion(
                        source_id=source_id,
                        target_id=target_id,
                        link_type="DEPENDS_ON",
                        context="The source depends on the target.",
                    )
                )
            ],
        ),
        RemoveLinkAction(
            action_id=uuid4(),
            source_id=source_id,
            target_id=target_id,
            link_type="DEPENDS_ON",
            confidence=1,
            rationale="unlink",
            evidence=[EvidenceRef(link=LinkAssertion(source_id=source_id, target_id=target_id, link_type="DEPENDS_ON"))],
        ),
        MergeMemoriesAction(
            action_id=uuid4(),
            canonical_id=canonical_id,
            source_ids=[source_id],
            confidence=1,
            rationale="merge",
            content="Merged content.",
            claim_manifest=ClaimManifest(
                preserved_claims=["claim"],
                transformed_claims=["claim"],
                source_mapping=[ClaimMapping(output="claim", source_memory_ids=[canonical_id, source_id])],
            ),
            evidence=[EvidenceRef(memory_id=canonical_id)],
        ),
        SplitMemoryAction(
            action_id=uuid4(),
            target_id=target_id,
            confidence=1,
            rationale="split",
            claim_manifest=ClaimManifest(
                preserved_claims=["claim"],
                transformed_claims=["claim"],
                source_mapping=[
                    ClaimMapping(output="child one", source_memory_ids=[target_id]),
                    ClaimMapping(output="child two", source_memory_ids=[target_id]),
                ],
            ),
            children=[
                ClaimMapping(output="child one", source_memory_ids=[target_id]),
                ClaimMapping(output="child two", source_memory_ids=[target_id]),
            ],
            evidence=[EvidenceRef(memory_id=target_id)],
        ),
        ArchiveMemoryAction(
            action_id=uuid4(),
            target_id=child_id,
            confidence=1,
            rationale="archive",
            claim_manifest=ClaimManifest(preserved_claims=["claim"]),
            evidence=[EvidenceRef(memory_id=child_id)],
        ),
    ]

    operations = {adapter.validate_python(sample.model_dump(mode="json")).operation for sample in samples}
    assert operations == {
        "archive_memory",
        "create_link",
        "merge_memories",
        "normalize_memory",
        "remove_link",
        "rewrite_memory",
        "split_memory",
    }
