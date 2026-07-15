"""Deterministic, planning-only orchestration for one curation frontier."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any, Protocol, cast
from uuid import UUID, uuid4

from mcp_memory.core.curation_context import (
    AcceptedMaintenanceRead,
    CurationContextPacket as ImmutableCurationContextPacket,
    CurationReadBudget,
    build_context_packet,
)
from mcp_memory.core.curation_identity import candidate_revision_token
from mcp_memory.core.curation_disclosure import ProviderTrust, ProviderTrustClass
from mcp_memory.core.curation_models import (
    CurationBudgetUsage,
    CurationContextPacket,
    CurationPlan,
    CurationPlanningRequest,
    CurationRunOutcome,
    CurationRunResult,
    RetentionDecision,
)
from mcp_memory.core.curation_planner import (
    CurationPlanner,
    CurationPlannerCancelledError,
    CurationPlannerError,
    CurationPlannerSchemaError,
    PlannerExecutionEnvelope,
)
from mcp_memory.core.curation_validation import (
    CurationMutationBudget,
    CurationRetryFeedback,
    CurationValidationResult,
    validate_curation_plan,
)
from mcp_memory.curation_store import (
    CandidateDisposition,
    CurationCandidateState,
    CurationRepository,
    CurationRun,
    CurationRunState,
)
from mcp_memory.mutation_history import ProtectionMode
from mcp_memory.work_item_store import WorkItemRecord


class WorkItemAction(StrEnum):
    NONE = "none"
    COMPLETE = "complete"
    DEFER = "defer"


@dataclass(frozen=True, slots=True)
class CurationWorkItemDecision:
    """The lifecycle decision made from persisted harness evidence."""

    action: WorkItemAction
    reason_code: str
    retry_delay_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class CurationFrontier:
    """One direct or already-claimed frontier; the harness never claims it."""

    family: str
    strategy: str
    seed_reads: tuple[Mapping[str, Any] | AcceptedMaintenanceRead, ...]
    support_reads: tuple[Mapping[str, Any] | AcceptedMaintenanceRead, ...] = ()
    work_item_id: str | None = None
    task_id: UUID | None = None
    frontier_key: str | None = None

    @classmethod
    def direct(
        cls,
        *,
        family: str,
        strategy: str,
        seed_reads: Iterable[Mapping[str, Any] | AcceptedMaintenanceRead],
        support_reads: Iterable[Mapping[str, Any] | AcceptedMaintenanceRead] = (),
        task_id: UUID | None = None,
        frontier_key: str | None = None,
    ) -> CurationFrontier:
        return cls(
            family=family,
            strategy=strategy,
            seed_reads=tuple(seed_reads),
            support_reads=tuple(support_reads),
            task_id=task_id,
            frontier_key=frontier_key,
        )

    @classmethod
    def claimed(
        cls,
        work_item: WorkItemRecord | str,
        *,
        family: str,
        strategy: str,
        seed_reads: Iterable[Mapping[str, Any] | AcceptedMaintenanceRead],
        support_reads: Iterable[Mapping[str, Any] | AcceptedMaintenanceRead] = (),
        task_id: UUID | None = None,
        frontier_key: str | None = None,
    ) -> CurationFrontier:
        work_item_id = work_item.id if isinstance(work_item, WorkItemRecord) else str(work_item)
        return cls(
            family=family,
            strategy=strategy,
            seed_reads=tuple(seed_reads),
            support_reads=tuple(support_reads),
            work_item_id=work_item_id,
            task_id=task_id,
            frontier_key=frontier_key,
        )


@dataclass(frozen=True, slots=True)
class CurationPlannerTools:
    """Read-only planner input, including immutable context and retry feedback."""

    context: ImmutableCurationContextPacket
    retry_feedback: CurationRetryFeedback | None = None


@dataclass(frozen=True, slots=True)
class CurationDryRunResult:
    """Typed dry-run evidence returned after the run and lifecycle write."""

    run: CurationRun
    result: CurationRunResult
    context: ImmutableCurationContextPacket
    validation: CurationValidationResult | None
    work_item: CurationWorkItemDecision
    planner_attempts: int

    @property
    def outcome(self) -> CurationRunOutcome:
        return self.result.outcome


class WorkItemRepository(Protocol):
    def complete_item(self, item_id: str, *, completed_at: float | None = None) -> WorkItemRecord: ...

    def defer_item(
        self,
        item_id: str,
        *,
        error: str,
        retry_delay_seconds: float,
        deferred_at: float | None = None,
    ) -> WorkItemRecord: ...


@dataclass(slots=True)
class CurationHarnessConfig:
    provider: ProviderTrust = field(default_factory=lambda: ProviderTrust(ProviderTrustClass.LOCAL))
    read_budget: CurationReadBudget = field(default_factory=CurationReadBudget)
    mutation_budget: CurationMutationBudget = field(default_factory=CurationMutationBudget)
    no_op_cooldown_seconds: float = 3600.0


class CurationDryRunHarness:
    """Run one frontier through planning and validation without an executor."""

    def __init__(
        self,
        *,
        curation_store: CurationRepository,
        planner: CurationPlanner,
        work_items: WorkItemRepository | None = None,
        config: CurationHarnessConfig | None = None,
        memory_types: Mapping[UUID, str] | None = None,
        contradictory_memory_ids: set[UUID] | frozenset[UUID] = frozenset(),
        protections_by_memory: Mapping[UUID, set[ProtectionMode] | frozenset[ProtectionMode]] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._curation_store = curation_store
        self._planner = planner
        self._work_items = work_items
        self._config = config or CurationHarnessConfig()
        self._memory_types = memory_types
        self._contradictory_memory_ids = contradictory_memory_ids
        self._protections_by_memory = protections_by_memory
        self._clock = clock or (lambda: datetime.now(UTC))

    async def run(self, frontier: CurationFrontier) -> CurationDryRunResult:
        context = build_context_packet(
            family=frontier.family,
            strategy=frontier.strategy,
            seed_reads=frontier.seed_reads,
            support_reads=frontier.support_reads,
            provider=self._config.provider,
            budget=self._config.read_budget,
            protections_by_memory=cast(Any, self._protections_by_memory),
        )
        run_id = uuid4()
        plan_id = uuid4()
        frontier_key = frontier.frontier_key or context.frontier_fingerprint
        run = self._curation_store.create_run(
            CurationRun(
                run_id=run_id,
                task_id=frontier.task_id,
                work_item_id=_uuid_or_none(frontier.work_item_id),
                frontier_key=frontier_key,
                selector_strategy=frontier.strategy,
                context_fingerprint=context.context_fingerprint,
                plan_id=plan_id,
            )
        )
        planning = run.model_copy(update={"state": CurationRunState.PLANNING, "planner_id": type(self._planner).__name__})
        stored = self._curation_store.transition_run(run_id, CurationRunState.CREATED, planning)
        if stored is not None:
            run = stored

        request = CurationPlanningRequest(
            run_id=run_id,
            plan_id=plan_id,
            frontier_key=frontier_key,
            context_fingerprint=context.context_fingerprint,
            context=CurationContextPacket(
                seed_memory_ids=[UUID(value) for value in context.seed_memory_ids],
                support_memory_ids=[UUID(value) for value in context.support_memory_ids],
                context_fingerprint=context.context_fingerprint,
            ),
        )
        plan, validation, envelopes, retry_reason, failure = await self._plan(
            request=request,
            context=context,
        )

        outcome, reason_code = self._classify_outcome(plan, validation, failure)
        rejection_codes = _rejection_codes(validation)
        if not rejection_codes and isinstance(failure, CurationPlannerSchemaError):
            rejection_codes.append("schema_invalid")
        budget_usage = _budget_usage(context, envelopes, validation)
        latest = envelopes[-1] if envelopes else None
        executing = run.model_copy(
            update={
                "state": CurationRunState.EXECUTING,
                "planner_id": type(self._planner).__name__,
                "provider_id": None if latest is None else latest.provider_key,
                "model_id": None if latest is None else latest.model_name,
                "plan_id": plan_id,
                "rejection_codes": rejection_codes,
                "retry_reason": retry_reason,
                "budget_usage": budget_usage,
            }
        )
        transitioned = self._curation_store.transition_run(run_id, run.state, executing)
        if transitioned is not None:
            executing = transitioned
        terminal = self._curation_store.terminalize_run(run_id, CurationRunState.EXECUTING, outcome)
        if terminal is None:
            terminal = self._curation_store.get_run(run_id)
        if terminal is None:
            raise RuntimeError(f"curation run {run_id} was not persisted")

        if outcome is CurationRunOutcome.NO_OP and plan is not None:
            self._persist_no_op_dispositions(plan.retained, context, run_id, frontier_key)

        work_item = self._work_item_decision(outcome, reason_code, latest)
        self._apply_work_item_decision(frontier.work_item_id, work_item)
        run_result = CurationRunResult(
            run_id=run_id,
            outcome=outcome,
            plan_id=plan_id,
            rejection_codes=rejection_codes,
            retry_reason=retry_reason,
            budget_usage=budget_usage,
        )
        return CurationDryRunResult(
            run=terminal,
            result=run_result,
            context=context,
            validation=validation,
            work_item=work_item,
            planner_attempts=len(envelopes),
        )

    async def _plan(
        self,
        *,
        request: CurationPlanningRequest,
        context: ImmutableCurationContextPacket,
    ) -> tuple[
        CurationPlan | None,
        CurationValidationResult | None,
        list[PlannerExecutionEnvelope[Any]],
        str | None,
        CurationPlannerError | BaseException | None,
    ]:
        envelopes: list[PlannerExecutionEnvelope[Any]] = []
        feedback: CurationRetryFeedback | None = None
        retry_reason: str | None = None
        validation: CurationValidationResult | None = None
        failure: CurationPlannerError | BaseException | None = None
        for attempt in range(2):
            try:
                envelope = await self._planner.create_plan(
                    request,
                    cast(Any, CurationPlannerTools(context=context, retry_feedback=feedback)),
                )
                envelopes.append(cast(PlannerExecutionEnvelope[Any], envelope))
            except CurationPlannerSchemaError as error:
                failure = error
                if error.envelope is not None:
                    envelopes.append(cast(PlannerExecutionEnvelope[Any], error.envelope))
                if attempt == 0:
                    retry_reason = "schema_invalid"
                    feedback = CurationRetryFeedback(
                        reason_code="formatting_only",
                        message=str(error),
                    )
                    continue
                return None, None, envelopes, retry_reason, failure
            except CurationPlannerCancelledError as error:
                failure = error
                if error.envelope is not None:
                    envelopes.append(cast(PlannerExecutionEnvelope[Any], error.envelope))
                return None, None, envelopes, retry_reason, failure
            except asyncio.CancelledError as error:
                failure = error
                return None, None, envelopes, retry_reason, failure
            except CurationPlannerError as error:
                failure = error
                if error.envelope is not None:
                    envelopes.append(cast(PlannerExecutionEnvelope[Any], error.envelope))
                return None, None, envelopes, retry_reason, failure

            validation = validate_curation_plan(
                cast(Any, envelope.plan),
                request=request,
                context=context,
                mutation_budget=self._config.mutation_budget,
                memory_types=self._memory_types,
                contradictory_memory_ids=self._contradictory_memory_ids,
                protections_by_memory=self._protections_by_memory,
            )
            if validation.valid:
                return cast(CurationPlan, validation.plan), validation, envelopes, retry_reason, None
            if validation.retry_feedback is not None and attempt == 0:
                retry_reason = validation.retry_feedback.reason_code
                feedback = validation.retry_feedback
                continue
            return None, validation, envelopes, retry_reason, None
        return None, validation, envelopes, retry_reason, failure

    def _persist_no_op_dispositions(
        self,
        retained: Iterable[RetentionDecision],
        context: ImmutableCurationContextPacket,
        run_id: UUID,
        frontier_key: str,
    ) -> None:
        now = self._clock()
        for decision in retained:
            memory_id = decision.memory_id
            record_token = context.record_tokens.get(str(memory_id))
            graph_token = context.graph_tokens.get(str(memory_id))
            token = (
                None
                if record_token is None or graph_token is None
                else candidate_revision_token(record_token, graph_token)
            )
            previous = self._curation_store.get_candidate_state(memory_id)
            count = 1
            if previous is not None and previous.last_observed_revision_token == token:
                count = previous.consecutive_no_op_count + 1
            self._curation_store.put_candidate_state(
                CurationCandidateState(
                    memory_id=memory_id,
                    last_observed_revision_token=token,
                    disposition=CandidateDisposition.COOLDOWN,
                    consecutive_no_op_count=count,
                    cooldown_until=now + timedelta(seconds=self._config.no_op_cooldown_seconds),
                    last_disposition_reason=str(decision.reason),
                    last_frontier_key=frontier_key,
                    last_run_id=run_id,
                )
            )

    def _classify_outcome(
        self,
        plan: CurationPlan | None,
        validation: CurationValidationResult | None,
        failure: CurationPlannerError | BaseException | None,
    ) -> tuple[CurationRunOutcome, str]:
        if isinstance(failure, CurationPlannerCancelledError) or isinstance(failure, asyncio.CancelledError):
            return CurationRunOutcome.CANCELLED, "provider_cancelled"
        if failure is not None:
            if isinstance(failure, CurationPlannerSchemaError):
                return CurationRunOutcome.INVALID_PLAN, "invalid_plan"
            return CurationRunOutcome.PROVIDER_FAILED, getattr(failure, "reason_code", "provider_failed")
        if validation is None or not validation.valid or plan is None:
            return CurationRunOutcome.INVALID_PLAN, "invalid_plan"
        if not plan.actions:
            return CurationRunOutcome.NO_OP, "valid_no_op"
        if validation.rejected_actions and not validation.accepted_actions and not validation.specialist_routes:
            return CurationRunOutcome.DEFERRED, "policy_rejected"
        return CurationRunOutcome.DEFERRED, "dry_run_requires_executor"

    def _work_item_decision(
        self,
        outcome: CurationRunOutcome,
        reason_code: str,
        envelope: PlannerExecutionEnvelope[Any] | None,
    ) -> CurationWorkItemDecision:
        if outcome is CurationRunOutcome.NO_OP:
            return CurationWorkItemDecision(WorkItemAction.COMPLETE, reason_code)
        delay = None if envelope is None else envelope.retry_delay_seconds
        return CurationWorkItemDecision(WorkItemAction.DEFER, reason_code, delay)

    def _apply_work_item_decision(self, work_item_id: str | None, decision: CurationWorkItemDecision) -> None:
        if work_item_id is None or self._work_items is None:
            return
        if decision.action is WorkItemAction.COMPLETE:
            self._work_items.complete_item(work_item_id)
        elif decision.action is WorkItemAction.DEFER:
            self._work_items.defer_item(
                work_item_id,
                error=decision.reason_code,
                retry_delay_seconds=decision.retry_delay_seconds or 0.0,
            )


def _uuid_or_none(value: str | None) -> UUID | None:
    return None if value is None else UUID(str(value))


def _rejection_codes(validation: CurationValidationResult | None) -> list[str]:
    if validation is None:
        return []
    codes = [str(issue.code) for issue in validation.issues]
    codes.extend(str(code) for item in validation.rejected_actions for code in item.reason_codes)
    codes.extend(str(route.reason_code) for route in validation.specialist_routes)
    return list(dict.fromkeys(codes))


def _budget_usage(
    context: ImmutableCurationContextPacket,
    envelopes: list[PlannerExecutionEnvelope[Any]],
    validation: CurationValidationResult | None,
) -> CurationBudgetUsage:
    token_usage = [envelope.token_usage for envelope in envelopes if envelope.token_usage is not None]
    token_source = next(
        (envelope.token_usage_source for envelope in reversed(envelopes) if envelope.token_usage_source is not None),
        None,
    )
    return CurationBudgetUsage(
        seed_records=context.usage.seed_records,
        support_records=context.usage.support_records,
        context_characters=context.usage.context_characters,
        read_tool_calls=context.usage.read_tool_calls,
        records_returned=context.usage.records_returned,
        proposed_actions=0 if validation is None or validation.plan is None else len(validation.plan.actions),
        accepted_mutations=0 if validation is None else len(validation.accepted_actions),
        planner_attempts=len(envelopes),
        premium_requests=sum(1 for envelope in envelopes if envelope.premium_request),
        token_usage=sum(token_usage) if token_usage else None,
        token_usage_source=token_source,
    )
