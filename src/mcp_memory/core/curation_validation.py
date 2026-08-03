"""Pure structural and context validation for typed curation plans.

This module deliberately does not evaluate policy, inspect storage, or apply
actions.  It only proves that a plan belongs to the request and context that
produced it and fits the harness' structural budgets.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal, Mapping
from uuid import UUID

from pydantic import TypeAdapter, ValidationError

from mcp_memory.core.curation_context import CurationContextPacket
from mcp_memory.core.curation_identity import action_id
from mcp_memory.core.curation_models import (
    CurationAction,
    CurationPlan,
    CurationPlanningRequest,
)
from mcp_memory.core.curation_policy import RejectionCode, evaluate_curation_action
from mcp_memory.core.curation_routing import (
    MaintenanceFamily,
    primary_family_for_operation,
)
from mcp_memory.mutation_history import ProtectionMode


@dataclass(frozen=True, slots=True)
class CurationMutationBudget:
    """Independent limits for proposed and eventually accepted mutations."""

    max_proposed_actions: int = 24
    max_accepted_mutations: int = 16

    def __post_init__(self) -> None:
        for name in ("max_proposed_actions", "max_accepted_mutations"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class CurationRetryFeedback:
    """Bounded feedback suitable for one planner retry."""

    reason_code: Literal[
        "formatting_only",
        "schema_invalid",
        "contract_invalid",
        "provider_failed",
    ]
    message: str
    fields: tuple[str, ...] = ()
    issue_codes: tuple[str, ...] = ()
    expected_fields: tuple[str, ...] = ()
    received_fields: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CurationValidationIssue:
    code: "CurationValidationReasonCode"
    message: str
    field: str | None = None


@dataclass(frozen=True, slots=True)
class CurationValidationResult:
    """The result of structural/context validation, with no policy decision."""

    plan: CurationPlan | None
    issues: tuple[CurationValidationIssue, ...] = ()
    retry_feedback: CurationRetryFeedback | None = None
    accepted_actions: tuple["AcceptedCurationAction", ...] = ()
    rejected_actions: tuple["RejectedCurationAction", ...] = ()
    specialist_routes: tuple["CurationSpecialistRoute", ...] = ()

    @property
    def valid(self) -> bool:
        return self.plan is not None and not self.issues


class CurationValidationReasonCode(StrEnum):
    SCHEMA_VERSION_MISMATCH = "schema_version_mismatch"
    RUN_ID_MISMATCH = "run_id_mismatch"
    PLAN_ID_MISMATCH = "plan_id_mismatch"
    FRONTIER_MISMATCH = "frontier_mismatch"
    CONTEXT_MISMATCH = "context_mismatch"
    CONTEXT_MISSING = "context_missing"
    SEED_SET_MISMATCH = "seed_set_mismatch"
    DUPLICATE_SEED = "duplicate_seed"
    DUPLICATE_RETENTION = "duplicate_retention"
    PROPOSED_ACTIONS_BUDGET = "proposed_actions_budget"
    ACCEPTED_MUTATIONS_BUDGET = "accepted_mutations_budget"
    TARGET_NOT_VISIBLE = "target_not_visible"
    DUPLICATE_ACTION_ID = "duplicate_action_id"
    MATERIAL_AMBIGUITY = "material_ambiguity"
    NEEDS_DIFFERENT_SPECIALIST = "needs_different_specialist"


@dataclass(frozen=True, slots=True)
class AcceptedCurationAction:
    action: CurationAction
    family: MaintenanceFamily


@dataclass(frozen=True, slots=True)
class RejectedCurationAction:
    action: CurationAction
    reason_codes: tuple[RejectionCode, ...]


@dataclass(frozen=True, slots=True)
class CurationSpecialistRoute:
    action: CurationAction
    family: MaintenanceFamily
    reason_code: CurationValidationReasonCode = CurationValidationReasonCode.NEEDS_DIFFERENT_SPECIALIST


def validate_curation_plan(
    plan: CurationPlan | Mapping[str, Any],
    *,
    request: CurationPlanningRequest,
    context: CurationContextPacket | Mapping[str, Any] | None = None,
    mutation_budget: CurationMutationBudget | None = None,
    memory_types: Mapping[UUID, str] | None = None,
    contradictory_memory_ids: set[UUID] | frozenset[UUID] = frozenset(),
    protections_by_memory: Mapping[UUID, set[ProtectionMode] | frozenset[ProtectionMode]] | None = None,
) -> CurationValidationResult:
    """Validate one plan and classify actions without storage or mutations."""

    typed_plan = _parse_plan(plan)
    if isinstance(typed_plan, CurationValidationResult):
        return typed_plan
    budget = mutation_budget or CurationMutationBudget()
    issues: list[CurationValidationIssue] = []

    if typed_plan.schema_version != request.schema_version:
        issues.append(_issue("schema_version_mismatch", "plan schema version does not match request"))
    if typed_plan.run_id != request.run_id:
        issues.append(_issue("run_id_mismatch", "plan run_id does not match request"))
    if typed_plan.plan_id != request.plan_id:
        issues.append(_issue("plan_id_mismatch", "plan plan_id does not match request"))
    if typed_plan.frontier_key != request.frontier_key:
        issues.append(_issue("frontier_mismatch", "plan frontier_key does not match request"))
    if typed_plan.context_fingerprint != request.context_fingerprint:
        issues.append(_issue("context_mismatch", "plan context_fingerprint does not match request"))

    visible_ids = _context_ids(context)
    if context is None:
        issues.append(_issue("context_missing", "an immutable context packet is required"))
    else:
        context_fingerprint = _context_value(context, "context_fingerprint")
        if context_fingerprint != request.context_fingerprint:
            issues.append(_issue("context_mismatch", "context fingerprint does not match request"))
        context_seeds = {_uuid_text(value) for value in _context_values(context, "seeds", "seed_memory_ids")}
        plan_seeds = {_uuid_text(value) for value in typed_plan.seed_memory_ids}
        if plan_seeds != context_seeds:
            issues.append(_issue("seed_set_mismatch", "plan seed_memory_ids do not match the context"))

    seed_ids = [_uuid_text(value) for value in typed_plan.seed_memory_ids]
    if len(seed_ids) != len(set(seed_ids)):
        issues.append(_issue("duplicate_seed", "seed_memory_ids must be unique"))
    retained_ids = [_uuid_text(decision.memory_id) for decision in typed_plan.retained]
    if len(retained_ids) != len(set(retained_ids)):
        issues.append(_issue("duplicate_retention", "retained decisions must be unique"))

    if len(typed_plan.actions) > budget.max_proposed_actions:
        issues.append(_issue("proposed_actions_budget", "plan exceeds the proposed action budget"))
    if len(typed_plan.actions) > budget.max_accepted_mutations:
        issues.append(_issue("accepted_mutations_budget", "plan exceeds the accepted mutation budget"))

    normalized_actions = []
    normalized_ids: set[UUID] = set()
    for position, action in enumerate(typed_plan.actions):
        targets = _action_targets(action)
        hidden = sorted(set(targets) - visible_ids, key=lambda value: value.encode("utf-8"))
        if hidden:
            issues.append(_issue("target_not_visible", f"action target(s) are absent from context: {hidden}"))
        expected_id = action_id(typed_plan.plan_id, position, action)
        if expected_id in normalized_ids:
            issues.append(_issue("duplicate_action_id", "normalized action IDs must be unique"))
        normalized_ids.add(expected_id)
        normalized_actions.append(action.model_copy(update={"action_id": expected_id}))

    if issues:
        return CurationValidationResult(
            plan=None,
            issues=tuple(issues),
            retry_feedback=CurationRetryFeedback(
                reason_code="contract_invalid",
                message=(
                    "Return a plan matching the supplied curation context. Fix: "
                    + "; ".join(
                        f"{issue.code}: {issue.message}" for issue in issues
                    )
                )[:1000],
                issue_codes=tuple(str(issue.code) for issue in issues),
            ),
        )
    normalized_plan = typed_plan.model_copy(update={"actions": normalized_actions})
    accepted: list[AcceptedCurationAction] = []
    rejected: list[RejectedCurationAction] = []
    routes: list[CurationSpecialistRoute] = []
    for action in normalized_plan.actions:
        decision = evaluate_curation_action(
            action,
            memory_types=memory_types,
            contradictory_memory_ids=contradictory_memory_ids,
            protections_by_memory=protections_by_memory,
        )
        if decision.rejection_codes:
            rejected.append(RejectedCurationAction(action=action, reason_codes=decision.rejection_codes))
            continue
        family = primary_family_for_operation(action.operation)
        if decision.authorized:
            accepted.append(AcceptedCurationAction(action=action, family=family))
        else:
            routes.append(CurationSpecialistRoute(action=action, family=family))
    return CurationValidationResult(
        plan=normalized_plan,
        accepted_actions=tuple(accepted),
        rejected_actions=tuple(rejected),
        specialist_routes=tuple(routes),
    )


def _parse_plan(plan: CurationPlan | Mapping[str, Any]) -> CurationPlan | CurationValidationResult:
    if isinstance(plan, CurationPlan):
        return plan
    try:
        return TypeAdapter(CurationPlan).validate_python(plan)
    except ValidationError as exc:
        if _is_formatting_only(exc):
            errors = tuple(
                (
                    ".".join(str(part) for part in error.get("loc", ())),
                    str(error.get("msg", "invalid value")),
                )
                for error in exc.errors()
            )
            fields = tuple(sorted({field for field, _message in errors}))
            return CurationValidationResult(
                plan=None,
                retry_feedback=CurationRetryFeedback(
                    reason_code="formatting_only",
                    message=(
                        "Return a schema-valid typed curation plan. Fix: "
                        + "; ".join(f"{field}: {message}" for field, message in errors)
                    )[:1000],
                    fields=fields,
                ),
            )
        return CurationValidationResult(
            plan=None,
            issues=(CurationValidationIssue(CurationValidationReasonCode.MATERIAL_AMBIGUITY, str(exc)),),
        )


def _is_formatting_only(error: ValidationError) -> bool:
    for item in error.errors():
        error_type = str(item.get("type", ""))
        location = item.get("loc", ())
        if error_type == "literal_error" and location == ("schema_version",):
            return False
        if error_type.startswith("value_error"):
            return False
    return True


def _context_ids(context: CurationContextPacket | Mapping[str, Any] | None) -> set[str]:
    if context is None:
        return set()
    values = list(_context_values(context, "seeds", "seed_memory_ids"))
    values.extend(_context_values(context, "support", "support_memory_ids"))
    values.extend(_context_values(context, "exploratory", "exploratory_memory_ids"))
    return {_uuid_text(value) for value in values}


def _context_values(context: CurationContextPacket | Mapping[str, Any], records: str, ids: str) -> list[Any]:
    value = _context_value(context, records)
    if value is not None:
        values: list[Any] = []
        for record in value:
            if isinstance(record, Mapping):
                memory_id = record.get("memory_id", record.get("id"))
                if memory_id is not None:
                    values.append(memory_id)
            else:
                values.append(record)
        return values
    value = _context_value(context, ids)
    return [] if value is None else list(value)


def _context_value(context: CurationContextPacket | Mapping[str, Any], name: str) -> Any:
    if isinstance(context, Mapping):
        return context.get(name)
    return getattr(context, name, None)


def _action_targets(action: Any) -> set[str]:
    values: list[Any] = []
    for name in ("target_id", "source_id", "canonical_id"):
        if hasattr(action, name):
            values.append(getattr(action, name))
    values.extend(getattr(action, "source_ids", []))
    return {_uuid_text(value) for value in values}


def _uuid_text(value: Any) -> str:
    return str(value if isinstance(value, UUID) else UUID(str(value)))


def _issue(code: CurationValidationReasonCode | str, message: str) -> CurationValidationIssue:
    return CurationValidationIssue(code=CurationValidationReasonCode(code), message=message)


# Short aliases for callers that use the contract's descriptive names.
validate_plan = validate_curation_plan
MutationBudget = CurationMutationBudget
