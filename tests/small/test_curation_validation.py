from dataclasses import replace
from typing import Any, cast
from uuid import uuid4

from mcp_memory.core.curation_context import CurationContextPacket, CurationReadCounters
from mcp_memory.core.curation_identity import action_id
from mcp_memory.core.curation_models import (
    CurationPlan,
    CurationPlanningRequest,
    NormalizeMemoryAction,
    RetentionDecision,
    RetentionReason,
)
from mcp_memory.core.curation_policy import RejectionCode
from mcp_memory.core.curation_routing import MaintenanceFamily
from mcp_memory.core.curation_validation import (
    CurationMutationBudget,
    validate_curation_plan,
)
from mcp_memory.mutation_history import ProtectionMode


def _request(run_id, plan_id, frontier, context):
    return CurationPlanningRequest(
        run_id=run_id, plan_id=plan_id, frontier_key=frontier, context_fingerprint=context
    )


def _context(memory_id, fingerprint):
    return CurationContextPacket(
        frontier_fingerprint="frontier-token",
        seeds=({"memory_id": str(memory_id)},),
        support=(),
        record_tokens={}, graph_tokens={}, disclosure=(), omissions=(), limits={},
        usage=CurationReadCounters(), context_fingerprint=fingerprint,
    )


def _plan(seed, run_id, plan_id, frontier, context, **kwargs):
    retained = [] if kwargs.get("actions") else [
        RetentionDecision(memory_id=seed, reason=RetentionReason.ALREADY_FOCUSED, rationale="focused")
    ]
    return CurationPlan(
        run_id=run_id, plan_id=plan_id, frontier_key=frontier, context_fingerprint=context,
        seed_memory_ids=[seed], rationale="retain", retained=retained, **kwargs
    )


def test_validation_binds_identities_and_context() -> None:
    run_id, plan_id, seed = uuid4(), uuid4(), uuid4()
    request = _request(run_id, plan_id, "frontier", "context")
    result = validate_curation_plan(
        _plan(seed, run_id, plan_id, "other", "context"), request=request, context=_context(seed, "context")
    )
    assert not result.valid
    assert {issue.code for issue in result.issues} == {"frontier_mismatch"}


def test_validation_rejects_hidden_targets_and_budgets() -> None:
    run_id, plan_id, seed, hidden = uuid4(), uuid4(), uuid4(), uuid4()
    request = _request(run_id, plan_id, "frontier", "context")
    action = {
        "operation": "create_link", "action_id": uuid4(), "source_id": seed, "target_id": hidden,
        "link_type": "RELATED", "confidence": 1, "rationale": "link",
        "context": "The source and target are related.",
        "evidence": [{
            "link": {
                "source_id": seed,
                "target_id": hidden,
                "link_type": "RELATED",
                "context": "The source and target are related.",
            }
        }],
    }
    plan = _plan(seed, run_id, plan_id, "frontier", "context", actions=[action])
    result = validate_curation_plan(
        plan, request=request, context=_context(seed, "context"),
        mutation_budget=CurationMutationBudget(max_proposed_actions=0, max_accepted_mutations=1),
    )
    assert {issue.code for issue in result.issues} == {"target_not_visible", "proposed_actions_budget"}


def test_validation_normalizes_action_ids_deterministically() -> None:
    run_id, plan_id, seed = uuid4(), uuid4(), uuid4()
    request = _request(run_id, plan_id, "frontier", "context")
    action = {
        "operation": "normalize_memory", "action_id": uuid4(), "target_id": seed,
        "confidence": 1, "rationale": "normalize", "tags": ["x"],
    }
    result = validate_curation_plan(
        _plan(seed, run_id, plan_id, "frontier", "context", actions=[action]),
        request=request, context=_context(seed, "context"),
    )
    assert result.valid
    assert result.plan is not None
    assert result.plan.actions[0].action_id == action_id(plan_id, 0, result.plan.actions[0])


def test_validation_allows_actions_on_explored_records() -> None:
    run_id, plan_id, seed, explored = uuid4(), uuid4(), uuid4(), uuid4()
    request = _request(run_id, plan_id, "frontier", "context")
    context = CurationContextPacket(
        frontier_fingerprint="frontier-token",
        seeds=({"memory_id": str(seed)},),
        support=(),
        exploratory=({"memory_id": str(explored)},),
        record_tokens={},
        graph_tokens={},
        disclosure=(),
        omissions=(),
        limits={},
        usage=CurationReadCounters(),
        context_fingerprint="context",
    )
    action = {
        "operation": "normalize_memory",
        "action_id": uuid4(),
        "target_id": explored,
        "confidence": 1,
        "rationale": "make the explored record specific",
        "summary": "Specific explored finding.",
    }
    plan = CurationPlan(
        run_id=run_id,
        plan_id=plan_id,
        frontier_key="frontier",
        context_fingerprint="context",
        seed_memory_ids=[seed],
        rationale="improve explored evidence",
        retained=[
            RetentionDecision(
                memory_id=seed,
                reason=RetentionReason.ALREADY_FOCUSED,
                rationale="seed remains focused",
            )
        ],
        actions=[NormalizeMemoryAction.model_validate(action)],
    )

    result = validate_curation_plan(plan, request=request, context=context)

    assert result.valid
    assert result.plan is not None
    assert isinstance(result.plan.actions[0], NormalizeMemoryAction)
    assert result.plan.actions[0].target_id == explored


def test_invalid_schema_returns_formatting_retry_feedback() -> None:
    run_id, plan_id, seed = uuid4(), uuid4(), uuid4()
    request = _request(run_id, plan_id, "frontier", "context")
    raw = _plan(seed, run_id, plan_id, "frontier", "context").model_dump(mode="json")
    raw.pop("rationale")
    result = validate_curation_plan(raw, request=request, context=_context(seed, "context"))
    assert not result.valid
    assert result.retry_feedback is not None
    assert result.retry_feedback.reason_code == "formatting_only"
    assert "rationale" in result.retry_feedback.message


def test_context_bearing_absent_link_returns_field_specific_retry_feedback() -> None:
    run_id, plan_id, source, target = uuid4(), uuid4(), uuid4(), uuid4()
    request = _request(run_id, plan_id, "frontier", "context")
    raw = {
        "run_id": str(run_id),
        "plan_id": str(plan_id),
        "frontier_key": "frontier",
        "context_fingerprint": "context",
        "seed_memory_ids": [str(source), str(target)],
        "rationale": "connect related records",
        "actions": [
            {
                "operation": "create_link",
                "action_id": str(uuid4()),
                "source_id": str(source),
                "target_id": str(target),
                "link_type": "RELATED",
                "context": "The records describe one relationship.",
                "confidence": 1,
                "rationale": "connect related records",
                "evidence": [
                    {
                        "link": {
                            "source_id": str(source),
                            "target_id": str(target),
                            "link_type": "RELATED",
                            "context": "The records describe one relationship.",
                        }
                    }
                ],
                "preconditions": {
                    "absent_links": [
                        {
                            "source_id": str(source),
                            "target_id": str(target),
                            "link_type": "RELATED",
                            "context": "The records describe one relationship.",
                        }
                    ]
                },
            }
        ],
    }
    context = replace(
        _context(source, "context"),
        seeds=({"memory_id": str(source)}, {"memory_id": str(target)}),
    )

    result = validate_curation_plan(raw, request=request, context=context)

    assert not result.valid
    assert result.retry_feedback is not None
    assert "actions.0.create_link.preconditions.absent_links.0.context" in result.retry_feedback.fields
    assert "Extra inputs are not permitted" in result.retry_feedback.message


def test_material_seed_ambiguity_is_not_retryable() -> None:
    seed = uuid4()
    raw = {
        "run_id": str(uuid4()), "plan_id": str(uuid4()), "frontier_key": "frontier",
        "context_fingerprint": "context", "rationale": "missing", "seed_memory_ids": [str(seed)],
    }
    result = validate_curation_plan(
        raw,
        request=_request(uuid4(), uuid4(), "frontier", "context"),
        context=_context(seed, "context"),
    )
    assert result.retry_feedback is None
    assert result.issues[0].code == "material_ambiguity"


def test_validation_classifies_actions_without_mutating_or_routing_work() -> None:
    run_id, plan_id, seed, linked = uuid4(), uuid4(), uuid4(), uuid4()
    request = _request(run_id, plan_id, "frontier", "context")
    normalize = {
        "operation": "normalize_memory", "action_id": uuid4(), "target_id": seed,
        "confidence": 1, "rationale": "specific", "summary": "specific",
    }
    create_link = {
        "operation": "create_link", "action_id": uuid4(), "source_id": seed, "target_id": linked,
        "link_type": "RELATED", "confidence": 1, "rationale": "evidence",
        "context": "The source and target cover related operational guidance.",
        "evidence": [{
            "link": {
                "source_id": seed,
                "target_id": linked,
                "link_type": "RELATED",
                "context": "The source and target cover related operational guidance.",
            }
        }],
    }
    result = validate_curation_plan(
        _plan(seed, run_id, plan_id, "frontier", _context(seed, "context").context_fingerprint,
              actions=[normalize, create_link]),
        request=request,
        context={"context_fingerprint": "context", "seeds": [{"memory_id": str(seed)}],
                 "support": [{"memory_id": str(linked)}]},
        memory_types={seed: "observation", linked: "observation"},
    )
    assert [item.family for item in result.accepted_actions] == [
        MaintenanceFamily.CURATOR,
        MaintenanceFamily.GRAPH_LINKER,
    ]
    assert not result.specialist_routes
    assert not result.rejected_actions


def test_validation_uses_canonical_disclosed_seed_ids() -> None:
    run_id, plan_id, seed = uuid4(), uuid4(), uuid4()
    request = _request(run_id, plan_id, "frontier", "context")
    result = validate_curation_plan(
        _plan(seed, run_id, plan_id, "frontier", "context"),
        request=request,
        context={
            "context_fingerprint": "context",
            "seeds": [{"id": str(seed).upper()}],
            "support": [],
        },
    )

    assert result.valid


def test_validation_rejects_support_records_as_seed_ids() -> None:
    run_id, plan_id, seed, support = uuid4(), uuid4(), uuid4(), uuid4()
    request = _request(run_id, plan_id, "frontier", "context")
    plan = _plan(seed, run_id, plan_id, "frontier", "context")
    plan = plan.model_copy(update={"seed_memory_ids": [support]})
    result = validate_curation_plan(
        plan,
        request=request,
        context={
            "context_fingerprint": "context",
            "seeds": [{"memory_id": str(seed)}],
            "support": [{"memory_id": str(support)}],
        },
    )

    assert not result.valid
    assert {issue.code for issue in result.issues} == {"seed_set_mismatch"}


def test_validation_rejects_hidden_seed_ids() -> None:
    run_id, plan_id, seed, hidden = uuid4(), uuid4(), uuid4(), uuid4()
    request = _request(run_id, plan_id, "frontier", "context")
    plan = _plan(seed, run_id, plan_id, "frontier", "context")
    plan = plan.model_copy(update={"seed_memory_ids": [hidden]})
    result = validate_curation_plan(
        plan,
        request=request,
        context={
            "context_fingerprint": "context",
            "seeds": [{"memory_id": str(seed)}],
            "support": [],
        },
    )

    assert not result.valid
    assert {issue.code for issue in result.issues} == {"seed_set_mismatch"}


def test_validation_accepts_all_policy_authorized_operation_families() -> None:
    run_id, plan_id = uuid4(), uuid4()
    seed, linked, source, target, remove_source, remove_target, canonical, merge_source, split_target, archive_target = (
        uuid4(),
        uuid4(),
        uuid4(),
        uuid4(),
        uuid4(),
        uuid4(),
        uuid4(),
        uuid4(),
        uuid4(),
        uuid4(),
    )
    request = _request(run_id, plan_id, "frontier", "context")
    actions = [
        {"operation": "normalize_memory", "action_id": uuid4(), "target_id": seed, "confidence": 1, "rationale": "normalize", "summary": "specific"},
        {
            "operation": "rewrite_memory",
            "action_id": uuid4(),
            "target_id": linked,
            "confidence": 1,
            "rationale": "rewrite",
            "content": "rewrite",
            "summary": "rewrite",
            "evidence": [{"memory_id": linked}],
            "claim_manifest": {
                "preserved_claims": ["keep"],
                "source_mapping": [{"output": "rewrite", "source_memory_ids": [linked]}],
            },
        },
        {
            "operation": "create_link",
            "action_id": uuid4(),
            "source_id": source,
            "target_id": target,
            "link_type": "RELATED",
            "confidence": 1,
            "rationale": "link",
            "context": "The source and target are related.",
            "evidence": [{
                "link": {
                    "source_id": source,
                    "target_id": target,
                    "link_type": "RELATED",
                    "context": "The source and target are related.",
                }
            }],
            "preconditions": {
                "record_tokens": {source: "tok-source", target: "tok-target"},
                "absent_links": [{"source_id": source, "target_id": target, "link_type": "RELATED"}],
            },
        },
        {
            "operation": "remove_link",
            "action_id": uuid4(),
            "source_id": remove_source,
            "target_id": remove_target,
            "link_type": "RELATED",
            "confidence": 1,
            "rationale": "remove",
            "evidence": [{"link": {"source_id": remove_source, "target_id": remove_target, "link_type": "RELATED"}}],
            "preconditions": {
                "record_tokens": {remove_source: "tok-remove-source", remove_target: "tok-remove-target"},
            },
        },
        {
            "operation": "merge_memories",
            "action_id": uuid4(),
            "canonical_id": canonical,
            "source_ids": [merge_source],
            "confidence": 1,
            "rationale": "merge",
            "content": "merged",
            "evidence": [{"memory_id": canonical}],
            "claim_manifest": {
                "preserved_claims": ["keep"],
                "unresolved_tensions": ["merge reviewed"],
                "source_mapping": [{"output": "merged", "source_memory_ids": [canonical, merge_source]}],
            },
            "preconditions": {
                "record_tokens": {canonical: "tok-canonical", merge_source: "tok-merge-source"},
            },
        },
        {
            "operation": "split_memory",
            "action_id": uuid4(),
            "target_id": split_target,
            "confidence": 1,
            "rationale": "split",
            "evidence": [{"memory_id": split_target}],
            "children": [{"output": "child", "source_memory_ids": [split_target]}],
            "claim_manifest": {
                "preserved_claims": ["keep"],
                "source_mapping": [{"output": "child", "source_memory_ids": [split_target]}],
            },
            "preconditions": {"record_tokens": {split_target: "tok-split"}},
        },
        {
            "operation": "archive_memory",
            "action_id": uuid4(),
            "target_id": archive_target,
            "confidence": 1,
            "rationale": "archive",
            "evidence": [{"memory_id": archive_target}],
            "claim_manifest": {"preserved_claims": ["keep"]},
            "preconditions": {"record_tokens": {archive_target: "tok-archive"}},
        },
    ]
    base_context = _context(seed, "context")
    context = CurationContextPacket(
        frontier_fingerprint=base_context.frontier_fingerprint,
        seeds=tuple({"memory_id": str(value)} for value in (seed, linked, source, target, remove_source, remove_target, canonical, merge_source, split_target, archive_target)),
        support=(),
        record_tokens={
            str(seed): "tok-seed",
            str(linked): "tok-linked",
            str(source): "tok-source",
            str(target): "tok-target",
            str(remove_source): "tok-remove-source",
            str(remove_target): "tok-remove-target",
            str(canonical): "tok-canonical",
            str(merge_source): "tok-merge-source",
            str(split_target): "tok-split",
            str(archive_target): "tok-archive",
        },
        graph_tokens={},
        disclosure=(),
        omissions=(),
        limits={},
        usage=base_context.usage,
        context_fingerprint="context",
    )
    plan = CurationPlan(
        run_id=run_id,
        plan_id=plan_id,
        frontier_key="frontier",
        context_fingerprint="context",
        seed_memory_ids=[
            seed,
            linked,
            source,
            target,
            remove_source,
            remove_target,
            canonical,
            merge_source,
            split_target,
            archive_target,
        ],
        actions=cast(Any, actions),
        retained=[],
        rationale="retain",
    )

    default_result = validate_curation_plan(
        plan,
        request=request,
        context=context,
        memory_types={value: "observation" for value in (seed, linked, source, target, remove_source, remove_target, canonical, merge_source, split_target, archive_target)},
    )

    assert [item.action.operation for item in default_result.accepted_actions] == [
        "normalize_memory",
        "rewrite_memory",
        "create_link",
        "remove_link",
        "merge_memories",
        "split_memory",
        "archive_memory",
    ]
    assert [item.family for item in default_result.accepted_actions] == [
        MaintenanceFamily.CURATOR,
        MaintenanceFamily.CURATOR,
        MaintenanceFamily.GRAPH_LINKER,
        MaintenanceFamily.GRAPH_LINKER,
        MaintenanceFamily.DEDUPLICATOR,
        MaintenanceFamily.CURATOR,
        MaintenanceFamily.CURATOR,
    ]
    assert not default_result.specialist_routes


def test_validation_returns_typed_policy_reasons_for_protected_actions() -> None:
    run_id, plan_id, seed = uuid4(), uuid4(), uuid4()
    request = _request(run_id, plan_id, "frontier", "context")
    action = {
        "operation": "normalize_memory", "action_id": uuid4(), "target_id": seed,
        "confidence": 1, "rationale": "specific", "summary": "specific",
    }
    result = validate_curation_plan(
        _plan(seed, run_id, plan_id, "frontier", "context", actions=[action]),
        request=request, context=_context(seed, "context"),
        memory_types={seed: "observation"},
        protections_by_memory={seed: {ProtectionMode.NO_AUTONOMOUS_MUTATION}},
    )
    assert not result.accepted_actions
    assert result.rejected_actions[0].reason_codes == (RejectionCode.NO_AUTONOMOUS_MUTATION,)


def test_validation_rejects_invalid_action_without_accepting_it() -> None:
    run_id, plan_id, seed = uuid4(), uuid4(), uuid4()
    request = _request(run_id, plan_id, "frontier", "context")
    action = {
        "operation": "rewrite_memory",
        "action_id": uuid4(),
        "target_id": seed,
        "confidence": 1,
        "rationale": "rewrite",
        "content": "replacement",
        "claim_manifest": {},
    }
    result = validate_curation_plan(
        _plan(seed, run_id, plan_id, "frontier", "context", actions=[action]),
        request=request,
        context=_context(seed, "context"),
        memory_types={seed: "observation"},
    )

    assert not result.accepted_actions
    assert result.rejected_actions[0].reason_codes == (
        RejectionCode.EVIDENCE_REQUIRED,
        RejectionCode.CLAIM_MANIFEST_INCOMPLETE,
    )
