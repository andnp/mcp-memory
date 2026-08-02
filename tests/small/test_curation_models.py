from uuid import uuid4

import pytest
from pydantic import TypeAdapter, ValidationError

from mcp_memory.core.curation_models import (
    ActionPreconditions,
    ArchiveMemoryAction,
    ClaimManifest,
    ClaimMapping,
    CreateLinkAction,
    CurationAction,
    CurationBudgetUsage,
    CurationContextPacket,
    CurationPlan,
    CurationRunOutcome,
    CurationRunResult,
    EvidenceRef,
    LinkAssertion,
    MergeMemoriesAction,
    MutationReceipt,
    NormalizeMemoryAction,
    ReceiptStatus,
    RemoveLinkAction,
    RetentionDecision,
    RetentionReason,
    RewriteMemoryAction,
    SplitMemoryAction,
)


def test_action_union_rejects_unknown_and_delete_operations() -> None:
    adapter = TypeAdapter(CurationAction)
    with pytest.raises(ValidationError):
        adapter.validate_python({"operation": "delete_memory", "target_id": str(uuid4())})
    with pytest.raises(ValidationError):
        adapter.validate_python({"operation": "unknown", "target_id": str(uuid4())})


def test_plan_requires_complete_seed_dispositions() -> None:
    seed = uuid4()
    with pytest.raises(ValidationError, match="missing seed dispositions"):
        CurationPlan(
            plan_id=uuid4(), run_id=uuid4(), frontier_key="frontier", context_fingerprint="context",
            rationale="no safe action", seed_memory_ids=[seed],
        )

    decision = RetentionDecision(memory_id=seed, reason=RetentionReason.ALREADY_FOCUSED, rationale="focused")
    plan = CurationPlan(
        plan_id=uuid4(), run_id=uuid4(), frontier_key="frontier", context_fingerprint="context",
        rationale="no safe action", seed_memory_ids=[seed], retained=[decision],
    )
    assert plan.retained == [decision]


def test_plan_rejects_overlapping_action_and_retention_dispositions() -> None:
    seed = uuid4()
    action = NormalizeMemoryAction(
        action_id=uuid4(),
        target_id=seed,
        confidence=1,
        rationale="normalize focused record",
        summary="Focused record",
        preconditions=ActionPreconditions(),
    )
    retained = RetentionDecision(
        memory_id=seed,
        reason=RetentionReason.ALREADY_FOCUSED,
        rationale="focused",
    )

    with pytest.raises(ValidationError, match="exactly one disposition"):
        CurationPlan(
            plan_id=uuid4(),
            run_id=uuid4(),
            frontier_key="frontier",
            context_fingerprint="context",
            rationale="invalid overlap",
            seed_memory_ids=[seed],
            actions=[action],
            retained=[retained],
        )


def test_defaults_are_isolated_and_contracts_are_pure() -> None:
    first = CurationBudgetUsage()
    second = CurationBudgetUsage()
    first.seed_records = 1
    assert second.seed_records == 0

    receipt = MutationReceipt(
        run_id=uuid4(), action_id=uuid4(), operation="archive_memory", status=ReceiptStatus.VERIFIED,
    )
    result = CurationRunResult(run_id=uuid4(), outcome=CurationRunOutcome.APPLIED, receipts=[receipt])
    assert result.budget_usage.read_tool_calls == 0


def test_context_packet_canonicalizes_visible_ids() -> None:
    seed = uuid4()
    support = uuid4()

    context = CurationContextPacket.from_visible_ids(
        seed_memory_ids=[str(seed).upper()],
        support_memory_ids=[str(support).upper()],
        context_fingerprint="context",
    )

    assert context.seed_memory_ids == [seed]
    assert context.support_memory_ids == [support]


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
