from uuid import uuid4

from mcp_memory.core.curation_context import CurationContextPacket, CurationReadCounters
from mcp_memory.core.curation_identity import action_id
from mcp_memory.core.curation_models import (
    CurationPlan,
    CurationPlanningRequest,
    RetentionDecision,
    RetentionReason,
)
from mcp_memory.core.curation_policy import RejectionCode
from mcp_memory.core.curation_routing import MaintenanceFamily
from mcp_memory.core.curation_validation import CurationMutationBudget, validate_curation_plan
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


def test_invalid_schema_returns_formatting_retry_feedback() -> None:
    run_id, plan_id, seed = uuid4(), uuid4(), uuid4()
    request = _request(run_id, plan_id, "frontier", "context")
    raw = _plan(seed, run_id, plan_id, "frontier", "context").model_dump(mode="json")
    raw.pop("rationale")
    result = validate_curation_plan(raw, request=request, context=_context(seed, "context"))
    assert not result.valid
    assert result.retry_feedback is not None
    assert result.retry_feedback.reason_code == "formatting_only"


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
        "evidence": [{"memory_id": seed}],
    }
    result = validate_curation_plan(
        _plan(seed, run_id, plan_id, "frontier", _context(seed, "context").context_fingerprint,
              actions=[normalize, create_link]),
        request=request,
        context={"context_fingerprint": "context", "seeds": [{"memory_id": str(seed)}],
                 "support": [{"memory_id": str(linked)}]},
        memory_types={seed: "observation", linked: "observation"},
    )
    assert [item.family for item in result.accepted_actions] == [MaintenanceFamily.CURATOR]
    assert [item.family for item in result.specialist_routes] == [MaintenanceFamily.GRAPH_LINKER]
    assert not result.rejected_actions


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
