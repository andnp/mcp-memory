"""Deterministic, planning-only orchestration for one curation frontier."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, cast
from uuid import UUID, uuid4

from mcp_memory.core.curation_context import (
    AcceptedMaintenanceRead,
    CurationBudgetExhausted,
    CurationContextPacket as ImmutableCurationContextPacket,
    CurationReadBudget,
    accept_maintenance_read,
    build_context_packet,
    disclosure_audit_manifest,
)
from mcp_memory.core.curation_identity import candidate_revision_token
from mcp_memory.core.curation_disclosure import ProviderTrust, ProviderTrustClass
from mcp_memory.core.curation_models import (
    ArchiveMemoryAction,
    CreateLinkAction,
    CurationBudgetUsage,
    CurationContextPacket,
    CurationPlan,
    CurationPlanningRequest,
    CurationRunOutcome,
    CurationRunResult,
    MergeMemoriesAction,
    NormalizeMemoryAction,
    RemoveLinkAction,
    RewriteMemoryAction,
    MutationReceipt,
    RetentionDecision,
    SplitMemoryAction,
)
from mcp_memory.core.curation_executor import CurationExecutor
from mcp_memory.core.curation_verifier import CurationVerifier
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
from mcp_memory.core.curation_work_items import (
    CurationWorkItemDecision,
    CurationWorkItemService,
    WorkItemRepository,
    WorkItemAction as _WorkItemAction,
)
from mcp_memory.curation_store import (
    CurationActionReceipt,
    CurationReceiptState,
    CandidateDisposition,
    CurationCandidateState,
    CurationRepository,
    CurationRun,
    CurationRunState,
)
from mcp_memory.mutation_history import ProtectionMode
from mcp_memory.work_item_store import WorkItemRecord

WorkItemAction = _WorkItemAction


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
    specialist_work_items: tuple[WorkItemRecord, ...] = ()

    @property
    def outcome(self) -> CurationRunOutcome:
        return self.result.outcome


def _context_record_counts(
    context: ImmutableCurationContextPacket,
    *,
    source_seed_count: int,
    source_support_count: int,
) -> dict[str, int]:
    return {
        "included_seed_count": len(context.seeds),
        "included_support_count": len(context.support),
        "omitted_seed_count": max(source_seed_count - len(context.seeds), 0),
        "omitted_support_count": max(source_support_count - len(context.support), 0),
    }


def _materialize_read(
    value: Mapping[str, Any] | AcceptedMaintenanceRead,
) -> AcceptedMaintenanceRead:
    return value if isinstance(value, AcceptedMaintenanceRead) else accept_maintenance_read(value)


def _assemble_context(
    *,
    family: str,
    strategy: str,
    seed_reads: tuple[AcceptedMaintenanceRead, ...],
    support_reads: tuple[AcceptedMaintenanceRead, ...],
    provider: ProviderTrust,
    budget: CurationReadBudget,
    protections_by_memory: Mapping[UUID | str, set[ProtectionMode] | frozenset[ProtectionMode]] | None,
    sensitive_fields_by_memory: Mapping[UUID | str, set[str] | frozenset[str]] | None,
    require_authoritative_disclosure_context: bool,
) -> ImmutableCurationContextPacket:
    """Build the largest deterministic packet that fits the read budget.

    Support is advisory, so it is removed from the trailing edge first.  If
    the seeds still do not fit, trailing seeds are removed while preserving
    the highest-priority seed.  A single seed that cannot fit remains a typed
    budget failure instead of being silently dropped.
    """

    selected_seeds = list(seed_reads)
    selected_support = list(support_reads)
    while True:
        try:
            return build_context_packet(
                family=family,
                strategy=strategy,
                seed_reads=selected_seeds,
                support_reads=selected_support,
                provider=provider,
                budget=budget,
                protections_by_memory=cast(Any, protections_by_memory),
                sensitive_fields_by_memory=cast(Any, sensitive_fields_by_memory),
                require_authoritative_disclosure_context=require_authoritative_disclosure_context,
            )
        except CurationBudgetExhausted:
            if selected_support:
                selected_support.pop()
                continue
            if len(selected_seeds) > 1:
                selected_seeds.pop()
                continue
            raise


@dataclass(slots=True)
class CurationHarnessConfig:
    provider: ProviderTrust = field(default_factory=lambda: ProviderTrust(ProviderTrustClass.LOCAL))
    read_budget: CurationReadBudget = field(default_factory=CurationReadBudget)
    mutation_budget: CurationMutationBudget = field(default_factory=CurationMutationBudget)
    no_op_cooldown_seconds: float = 3600.0
    execute_accepted_actions: bool = False
    require_authoritative_disclosure_context: bool = False


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
        sensitive_fields_by_memory: Mapping[UUID, set[str] | frozenset[str]] | None = None,
        clock: Callable[[], datetime] | None = None,
        executor: CurationExecutor | None = None,
        verifier: CurationVerifier | None = None,
        work_item_service: CurationWorkItemService | None = None,
    ) -> None:
        self._curation_store = curation_store
        self._planner = planner
        self._work_items = work_items
        self._config = config or CurationHarnessConfig()
        self._memory_types = memory_types
        self._contradictory_memory_ids = contradictory_memory_ids
        self._protections_by_memory = protections_by_memory
        self._sensitive_fields_by_memory = sensitive_fields_by_memory
        self._clock = clock or (lambda: datetime.now(UTC))
        self._executor = executor
        self._verifier = verifier
        self._work_item_service = work_item_service or CurationWorkItemService(
            work_items,
            no_op_cooldown_seconds=self._config.no_op_cooldown_seconds,
        )

    async def run(self, frontier: CurationFrontier) -> CurationDryRunResult:
        seed_reads = tuple(_materialize_read(value) for value in frontier.seed_reads)
        support_reads = tuple(_materialize_read(value) for value in frontier.support_reads)
        context_failure: CurationBudgetExhausted | None = None
        try:
            context = _assemble_context(
                family=frontier.family,
                strategy=frontier.strategy,
                seed_reads=seed_reads,
                support_reads=support_reads,
                provider=self._config.provider,
                budget=self._config.read_budget,
                protections_by_memory=cast(Any, self._protections_by_memory),
                sensitive_fields_by_memory=cast(Any, self._sensitive_fields_by_memory),
                require_authoritative_disclosure_context=self._config.require_authoritative_disclosure_context,
            )
        except CurationBudgetExhausted as error:
            context_failure = error
            context = build_context_packet(
                family=frontier.family,
                strategy=frontier.strategy,
                seed_reads=(),
                support_reads=(),
                provider=self._config.provider,
                budget=self._config.read_budget,
                protections_by_memory=cast(Any, self._protections_by_memory),
                sensitive_fields_by_memory=cast(Any, self._sensitive_fields_by_memory),
                require_authoritative_disclosure_context=self._config.require_authoritative_disclosure_context,
            )
        context_record_counts = _context_record_counts(
            context,
            source_seed_count=len(seed_reads),
            source_support_count=len(support_reads),
        )
        run_id = uuid4()
        plan_id = uuid4()
        frontier_key = context.frontier_fingerprint
        disclosure_audit = disclosure_audit_manifest(context, self._config.provider)
        disclosure_audit["record_counts"] = context_record_counts
        if context_failure is not None:
            disclosure_audit["budget_failure"] = {
                "reason_code": context_failure.reason_code,
                "dimension": context_failure.dimension.value,
                "used": context_failure.used,
                "requested": context_failure.requested,
                "limit": context_failure.limit,
            }
        run = self._curation_store.create_run(
            CurationRun(
                run_id=run_id,
                task_id=frontier.task_id,
                work_item_id=_uuid_or_none(frontier.work_item_id),
                frontier_key=frontier_key,
                selector_strategy=frontier.strategy,
                context_fingerprint=context.context_fingerprint,
                plan_id=plan_id,
                disclosure_audit=disclosure_audit,
            )
        )
        planning = run.model_copy(update={"state": CurationRunState.PLANNING, "planner_id": type(self._planner).__name__})
        stored = self._curation_store.transition_run(run_id, CurationRunState.CREATED, planning)
        if stored is not None:
            run = stored

        if context_failure is not None:
            outcome = CurationRunOutcome.BUDGET_EXHAUSTED
            reason_code = context_failure.reason_code
            rejection_codes: list[str] = []
            budget_usage = _budget_usage(context, [], None)
            retry_reason = context_failure.reason_code
            latest = None
            plan: CurationPlan | None = None
            validation: CurationValidationResult | None = None
            envelopes: list[PlannerExecutionEnvelope[Any]] = []
        else:
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
        terminal_state = CurationRunState.EXECUTING
        if (
            self._config.execute_accepted_actions
            and validation is not None
            and validation.valid
            and validation.plan is not None
            and validation.plan.actions
        ):
            outcome, reason_code, terminal_state, receipts = self._execute_accepted_actions(
                run_id=run_id,
                executing=executing,
                validation=validation,
                context=context,
            )
        else:
            receipts = ()
        terminal = self._curation_store.terminalize_run(run_id, terminal_state, outcome)
        if terminal is None:
            terminal = self._curation_store.get_run(run_id)
        if terminal is None:
            raise RuntimeError(f"curation run {run_id} was not persisted")

        if outcome is CurationRunOutcome.NO_OP and plan is not None:
            self._persist_no_op_dispositions(plan.retained, context, run_id, frontier_key)

        work_item = self._work_item_service.decide(outcome, reason_code, latest)
        specialist_work_items: tuple[WorkItemRecord, ...] = ()
        specialist_routes = () if validation is None else validation.specialist_routes
        if self._config.execute_accepted_actions:
            specialist_routes = ()
        if validation is not None and validation.valid and specialist_routes and self._work_items is not None:
            from mcp_memory.core.task_handlers.maintenance_work_items import enqueue_specialist_routes

            specialist_work_items = enqueue_specialist_routes(
                self._work_items,
                specialist_routes,
                context=context,
            )
        self._work_item_service.apply(frontier.work_item_id, work_item)
        receipt_models = [
            MutationReceipt.model_validate(
                receipt.model_dump(mode="json", exclude={"mutation_event_id", "intent_hash"})
            )
            for receipt in receipts
        ]
        run_result = CurationRunResult(
            run_id=run_id,
            outcome=outcome,
            plan_id=plan_id,
            receipts=receipt_models,
            rejection_codes=rejection_codes,
            retry_reason=retry_reason,
            budget_usage=budget_usage,
            context_record_counts=context_record_counts,
            verified_action_count=_count_verified_receipts(receipts),
            affected_memory_count=_count_affected_memory_ids(receipts),
        )
        return CurationDryRunResult(
            run=terminal,
            result=run_result,
            context=context,
            validation=validation,
            work_item=work_item,
            planner_attempts=len(envelopes),
            specialist_work_items=specialist_work_items,
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
                allow_verified_actions=self._config.execute_accepted_actions,
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

    def _execute_accepted_actions(
        self,
        *,
        run_id: UUID,
        executing: CurationRun,
        validation: CurationValidationResult,
        context: ImmutableCurationContextPacket,
    ) -> tuple[CurationRunOutcome, str, CurationRunState, tuple[CurationActionReceipt, ...]]:
        """Execute and verify accepted actions through the deterministic pipeline."""
        executor = self._executor
        verifier = self._verifier
        if executor is None or verifier is None:
            raise RuntimeError("curation execution requires an executor and verifier")

        actions = [
            item.action
            for item in validation.accepted_actions
        ]
        if not actions:
            return CurationRunOutcome.DEFERRED, "unsupported_execution_action", CurationRunState.EXECUTING, ()

        memory_types = dict(self._memory_types or {})
        for record in (*context.seeds, *context.support):
            memory_id = record.get("memory_id")
            memory_type = record.get("type")
            if memory_id is not None and isinstance(memory_type, str):
                memory_types[UUID(str(memory_id))] = memory_type

        receipts: list[tuple[CurationActionReceipt, Any]] = []
        for action in actions:
            receipt = self._execute_action(
                action,
                executor=executor,
                run_id=run_id,
                memory_types=memory_types,
                context=context,
            )
            receipts.append((receipt, action))

        verifying = executing.model_copy(update={"state": CurationRunState.VERIFYING})
        stored = self._curation_store.transition_run(run_id, CurationRunState.EXECUTING, verifying)
        if stored is None:
            raise RuntimeError(f"curation run {run_id} could not enter verification")

        verified_receipts: list[CurationActionReceipt] = []
        for receipt, action in receipts:
            verified = verifier.verify(receipt, action)
            verified_receipts.append(verified)
            if verified.status.value != "verified":
                return (
                    CurationRunOutcome.VERIFICATION_FAILED,
                    "verification_failed",
                    CurationRunState.VERIFYING,
                    tuple(verified_receipts),
                )

        remaining_specialist_routes = tuple(
            route for route in validation.specialist_routes if route.action.action_id not in {item.action_id for item in actions}
        )
        partial = bool(validation.rejected_actions or remaining_specialist_routes)
        return (
            CurationRunOutcome.PARTIALLY_APPLIED if partial else CurationRunOutcome.APPLIED,
            "partially_applied" if partial else "verified_receipts",
            CurationRunState.VERIFYING,
            tuple(verified_receipts),
        )

    def _execute_action(
        self,
        action: Any,
        *,
        executor: CurationExecutor,
        run_id: UUID,
        memory_types: dict[UUID, str],
        context: ImmutableCurationContextPacket,
    ) -> CurationActionReceipt:
        if isinstance(action, NormalizeMemoryAction):
            target_id = action.target_id
            token = context.record_tokens.get(str(target_id))
            return executor.execute_normalize(
                action,
                run_id=run_id,
                memory_type=memory_types.get(target_id, ""),
                protections=(self._protections_by_memory or {}).get(target_id, ()),
                expected_token=token,
            )
        if isinstance(action, CreateLinkAction):
            protections = set()
            for endpoint_id in (action.source_id, action.target_id):
                protections.update((self._protections_by_memory or {}).get(endpoint_id, ()))
            return executor.execute_create_link(
                action,
                run_id=run_id,
                memory_types=memory_types,
                protections=protections,
            )
        if isinstance(action, RewriteMemoryAction):
            target_id = action.target_id
            return executor.execute_rewrite(
                action,
                run_id=run_id,
                memory_type=memory_types.get(target_id, ""),
                protections=(self._protections_by_memory or {}).get(target_id, ()),
            )
        if isinstance(action, RemoveLinkAction):
            protections = set()
            for endpoint_id in (action.source_id, action.target_id):
                protections.update((self._protections_by_memory or {}).get(endpoint_id, ()))
            return executor.execute_remove_link(
                action,
                run_id=run_id,
                memory_types=memory_types,
                protections=protections,
            )
        if isinstance(action, MergeMemoriesAction):
            protections = set()
            for endpoint_id in (action.canonical_id, *action.source_ids):
                protections.update((self._protections_by_memory or {}).get(endpoint_id, ()))
            return executor.execute_merge(
                action,
                run_id=run_id,
                memory_types=memory_types,
                contradictory_memory_ids=self._contradictory_memory_ids,
                protections=protections,
            )
        if isinstance(action, SplitMemoryAction):
            target_id = action.target_id
            return executor.execute_split(
                action,
                run_id=run_id,
                memory_type=memory_types.get(target_id, ""),
                protections=(self._protections_by_memory or {}).get(target_id, ()),
            )
        if isinstance(action, ArchiveMemoryAction):
            target_id = action.target_id
            return executor.execute_archive(
                action,
                run_id=run_id,
                memory_type=memory_types.get(target_id, ""),
                protections=(self._protections_by_memory or {}).get(target_id, ()),
            )
        raise RuntimeError(f"unsupported curation action: {type(action).__name__}")


def _uuid_or_none(value: str | None) -> UUID | None:
    return None if value is None else UUID(str(value))


def _count_verified_receipts(receipts: tuple[CurationActionReceipt, ...]) -> int:
    return sum(1 for receipt in receipts if receipt.status is CurationReceiptState.VERIFIED)


def _count_affected_memory_ids(receipts: tuple[CurationActionReceipt, ...]) -> int:
    affected_ids = {
        str(memory_id)
        for receipt in receipts
        for memory_id in receipt.affected_ids
    }
    return len(affected_ids)


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
