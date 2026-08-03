"""Deterministic, planning-only orchestration for one curation frontier."""

from __future__ import annotations

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
    CampaignHypothesis,
    CurationBudgetUsage,
    CurationContextPacket,
    CurationPlan,
    CurationPlanningRequest,
    CurationRunOutcome,
    CurationRunResult,
)
from mcp_memory.core.curation_executor import CurationExecutor
from mcp_memory.core.curation_execution_service import CurationExecutionService
from mcp_memory.core.curation_verifier import CurationVerifier
from mcp_memory.core.curation_planner import (
    CurationPlanner,
    CurationPlannerError,
    PlannerExecutionEnvelope,
)
from mcp_memory.core.curation_validation import (
    CurationMutationBudget,
    CurationValidationResult,
)
from mcp_memory.core.curation_planning_service import (
    CurationPlannerTools,  # noqa: F401 - compatibility export
    CurationPlanningInput,
    plan_and_validate,
)
from mcp_memory.core.curation_quality import CurationQualitySampler
from mcp_memory.core.mutation_restore import RestoreWaveResult, restore_wave
from mcp_memory.core.curation_run_outcomes import (
    build_run_result,
    budget_usage as project_budget_usage,
    classify_outcome as project_classify_outcome,
    failure_rejection_codes as project_failure_rejection_codes,
    rejection_codes as project_rejection_codes,
)
from mcp_memory.core.curation_work_items import (
    CurationWorkItemDecision,
    CurationWorkItemService,
    WorkItemRepository,
    WorkItemAction as _WorkItemAction,
)
from mcp_memory.core.ports.curation import (
    CurationActionReceipt,
    CandidateDisposition,
    CurationCandidateState,
    CurationReceiptState,
    CurationRepository,
    CurationRun,
    CurationRunState,
)
from mcp_memory.mutation_history import MutationHistoryStore, ProtectionMode
from mcp_memory.core.ports.work_items import WorkItemRecordLike

WorkItemAction = _WorkItemAction


@dataclass(frozen=True, slots=True)
class CurationFrontier:
    """One direct or already-claimed frontier; the harness never claims it."""

    family: str
    strategy: str
    seed_reads: tuple[Mapping[str, Any] | AcceptedMaintenanceRead, ...]
    support_reads: tuple[Mapping[str, Any] | AcceptedMaintenanceRead, ...] = ()
    exploratory_reads: tuple[Mapping[str, Any] | AcceptedMaintenanceRead, ...] = ()
    work_item_id: str | None = None
    task_id: UUID | None = None
    frontier_key: str | None = None
    campaign_hypothesis: CampaignHypothesis | None = None

    @classmethod
    def direct(
        cls,
        *,
        family: str,
        strategy: str,
        seed_reads: Iterable[Mapping[str, Any] | AcceptedMaintenanceRead],
        support_reads: Iterable[Mapping[str, Any] | AcceptedMaintenanceRead] = (),
        exploratory_reads: Iterable[Mapping[str, Any] | AcceptedMaintenanceRead] = (),
        task_id: UUID | None = None,
        frontier_key: str | None = None,
        campaign_hypothesis: CampaignHypothesis | None = None,
    ) -> CurationFrontier:
        return cls(
            family=family,
            strategy=strategy,
            seed_reads=tuple(seed_reads),
            support_reads=tuple(support_reads),
            exploratory_reads=tuple(exploratory_reads),
            task_id=task_id,
            frontier_key=frontier_key,
            campaign_hypothesis=campaign_hypothesis,
        )

    @classmethod
    def claimed(
        cls,
        work_item: WorkItemRecordLike | str,
        *,
        family: str,
        strategy: str,
        seed_reads: Iterable[Mapping[str, Any] | AcceptedMaintenanceRead],
        support_reads: Iterable[Mapping[str, Any] | AcceptedMaintenanceRead] = (),
        exploratory_reads: Iterable[Mapping[str, Any] | AcceptedMaintenanceRead] = (),
        task_id: UUID | None = None,
        frontier_key: str | None = None,
        campaign_hypothesis: CampaignHypothesis | None = None,
    ) -> CurationFrontier:
        work_item_id = work_item.id if not isinstance(work_item, str) else work_item
        return cls(
            family=family,
            strategy=strategy,
            seed_reads=tuple(seed_reads),
            support_reads=tuple(support_reads),
            exploratory_reads=tuple(exploratory_reads),
            work_item_id=work_item_id,
            task_id=task_id,
            frontier_key=frontier_key,
            campaign_hypothesis=campaign_hypothesis,
        )


@dataclass(frozen=True, slots=True)
class CurationDryRunResult:
    """Typed dry-run evidence returned after the run and lifecycle write."""

    run: CurationRun
    result: CurationRunResult
    context: ImmutableCurationContextPacket
    validation: CurationValidationResult | None
    work_item: CurationWorkItemDecision
    planner_attempts: int
    specialist_work_items: tuple[WorkItemRecordLike, ...] = ()
    restore_result: dict[str, object] | None = None

    @property
    def outcome(self) -> CurationRunOutcome:
        return self.result.outcome


def _context_record_counts(
    context: ImmutableCurationContextPacket,
    *,
    source_seed_count: int,
    source_support_count: int,
    source_exploratory_count: int = 0,
) -> dict[str, int]:
    counts = {
        "included_seed_count": len(context.seeds),
        "included_support_count": len(context.support),
        "omitted_seed_count": max(source_seed_count - len(context.seeds), 0),
        "omitted_support_count": max(source_support_count - len(context.support), 0),
    }
    if source_exploratory_count or context.exploratory:
        counts.update(
            included_exploratory_count=len(context.exploratory),
            omitted_exploratory_count=max(
                source_exploratory_count - len(context.exploratory), 0
            ),
        )
    return counts


def _restore_result_payload(result: RestoreWaveResult) -> dict[str, object]:
    return {
        "status": result.status.value,
        "target_event_ids": [str(event_id) for event_id in result.target_event_ids],
        "restored_event_ids": [str(event_id) for event_id in result.restored_event_ids],
        "conflict_reason": result.conflict_reason,
        "results": [item.model_dump(mode="json") for item in result.results],
    }


def _materialize_read(
    value: Mapping[str, Any] | AcceptedMaintenanceRead,
) -> AcceptedMaintenanceRead:
    return value if isinstance(value, AcceptedMaintenanceRead) else accept_maintenance_read(value)


def _context_memory_ids(context: ImmutableCurationContextPacket) -> set[UUID]:
    return {
        UUID(memory_id)
        for memory_id in (*context.seed_memory_ids, *context.support_memory_ids)
    }


def _candidate_token(
    context: ImmutableCurationContextPacket,
    memory_id: UUID,
) -> str | None:
    record_token = context.record_tokens.get(str(memory_id))
    graph_token = context.graph_tokens.get(str(memory_id))
    if record_token is None or graph_token is None:
        return None
    return candidate_revision_token(record_token, graph_token)


def _action_memory_ids(action: Any) -> set[UUID]:
    ids: set[UUID] = set()
    for attribute in ("target_id", "source_id", "canonical_id"):
        value = getattr(action, attribute, None)
        if isinstance(value, UUID):
            ids.add(value)
    ids.update(value for value in getattr(action, "source_ids", ()) if isinstance(value, UUID))
    return ids


def _merge_candidate_outcomes(
    outcomes: dict[UUID, tuple[CandidateDisposition, str, bool]],
    memory_ids: Iterable[UUID],
    disposition: CandidateDisposition,
    reason: str,
    escalated: bool,
) -> None:
    priority = {
        CandidateDisposition.COOLDOWN: 0,
        CandidateDisposition.ACTIONED: 1,
        CandidateDisposition.ESCALATED: 2,
    }
    for memory_id in memory_ids:
        current = outcomes.get(memory_id)
        if current is None or priority[disposition] >= priority[current[0]]:
            outcomes[memory_id] = (disposition, reason, escalated)


def _assemble_context(
    *,
    family: str,
    strategy: str,
    seed_reads: tuple[AcceptedMaintenanceRead, ...],
    support_reads: tuple[AcceptedMaintenanceRead, ...],
    exploratory_reads: tuple[AcceptedMaintenanceRead, ...],
    provider: ProviderTrust,
    budget: CurationReadBudget,
    protections_by_memory: Mapping[UUID | str, set[ProtectionMode] | frozenset[ProtectionMode]] | None,
    sensitive_fields_by_memory: Mapping[UUID | str, set[str] | frozenset[str]] | None,
    require_authoritative_disclosure_context: bool,
    campaign_hypothesis: CampaignHypothesis | None = None,
) -> ImmutableCurationContextPacket:
    """Build the largest deterministic packet that fits the read budget.

    Support is advisory, so it is removed from the trailing edge first.  If
    the seeds still do not fit, trailing seeds are removed while preserving
    the highest-priority seed.  A single seed that cannot fit remains a typed
    budget failure instead of being silently dropped.
    """

    selected_seeds = list(seed_reads)
    selected_support = list(support_reads)
    selected_exploratory = list(exploratory_reads)
    while True:
        try:
            return build_context_packet(
                family=family,
                strategy=strategy,
                seed_reads=selected_seeds,
                support_reads=selected_support,
                exploratory_reads=selected_exploratory,
                provider=provider,
                budget=budget,
                protections_by_memory=cast(Any, protections_by_memory),
                sensitive_fields_by_memory=cast(Any, sensitive_fields_by_memory),
                require_authoritative_disclosure_context=require_authoritative_disclosure_context,
                campaign_hypothesis=campaign_hypothesis,
            )
        except CurationBudgetExhausted:
            if selected_exploratory:
                selected_exploratory.pop()
                continue
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
        quality_sampler: CurationQualitySampler | None = None,
        restore_action_store: Any | None = None,
        mutation_history: MutationHistoryStore | None = None,
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
        self._quality_sampler = quality_sampler
        self._restore_action_store = restore_action_store
        self._mutation_history = mutation_history
        self._execution_service = CurationExecutionService(
            curation_store=curation_store,
            executor=executor,
            verifier=verifier,
            memory_types=memory_types,
            contradictory_memory_ids=set(contradictory_memory_ids),
            protections_by_memory=protections_by_memory,
        )
        self._work_item_service = work_item_service or CurationWorkItemService(
            work_items,
            no_op_cooldown_seconds=self._config.no_op_cooldown_seconds,
        )

    async def run(self, frontier: CurationFrontier) -> CurationDryRunResult:
        seed_reads = tuple(_materialize_read(value) for value in frontier.seed_reads)
        support_reads = tuple(_materialize_read(value) for value in frontier.support_reads)
        exploratory_reads = tuple(
            _materialize_read(value) for value in frontier.exploratory_reads
        )
        context_failure: CurationBudgetExhausted | None = None
        try:
            context = _assemble_context(
                family=frontier.family,
                strategy=frontier.strategy,
                seed_reads=seed_reads,
                support_reads=support_reads,
                exploratory_reads=exploratory_reads,
                provider=self._config.provider,
                budget=self._config.read_budget,
                protections_by_memory=cast(Any, self._protections_by_memory),
                sensitive_fields_by_memory=cast(Any, self._sensitive_fields_by_memory),
                require_authoritative_disclosure_context=self._config.require_authoritative_disclosure_context,
                campaign_hypothesis=frontier.campaign_hypothesis,
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
                campaign_hypothesis=frontier.campaign_hypothesis,
            )
        context_record_counts = _context_record_counts(
            context,
            source_seed_count=len(seed_reads),
            source_support_count=len(support_reads),
            source_exploratory_count=len(exploratory_reads),
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
            budget_usage = project_budget_usage(context, [], None)
            retry_reason = context_failure.reason_code
            latest = None
            plan: CurationPlan | None = None
            validation: CurationValidationResult | None = None
            envelopes: list[PlannerExecutionEnvelope[Any]] = []
            failure: CurationPlannerError | BaseException | None = None
        else:
            request = CurationPlanningRequest(
                run_id=run_id,
                plan_id=plan_id,
                frontier_key=frontier_key,
                context_fingerprint=context.context_fingerprint,
                campaign_hypothesis=context.campaign_hypothesis,
                context=CurationContextPacket.from_visible_ids(
                    seed_memory_ids=context.seed_memory_ids,
                    support_memory_ids=context.support_memory_ids,
                    exploratory_memory_ids=context.exploratory_memory_ids,
                    context_fingerprint=context.context_fingerprint,
                    campaign_hypothesis=context.campaign_hypothesis,
                ),
            )
            plan, validation, envelopes, retry_reason, failure = await self._plan(
                request=request,
                context=context,
            )

            outcome, reason_code = self._classify_outcome(plan, validation, failure)
            rejection_codes = project_rejection_codes(validation)
            rejection_codes.extend(
                code for code in project_failure_rejection_codes(failure) if code not in rejection_codes
            )
            budget_usage = project_budget_usage(context, envelopes, validation)
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
            outcome, reason_code, terminal_state, receipts = self._execution_service.execute(
                run_id=run_id,
                executing=executing,
                validation=validation,
                context=context,
            )
            rejection_codes.extend(
                receipt.error_code
                for receipt in receipts
                if receipt.error_code is not None and receipt.error_code not in rejection_codes
            )
        else:
            receipts = ()
        self._persist_candidate_outcomes(
            plan=plan,
            validation=validation,
            context=context,
            run_id=run_id,
            frontier_key=frontier_key,
            mutation_family=frontier.family,
            strategy=frontier.strategy,
            reason_code=reason_code,
            receipts=receipts,
        )
        quality_evidence = (
            ()
            if self._quality_sampler is None
            else self._quality_sampler.evaluate(
                run=self._curation_store.get_run(run_id) or executing,
                receipts=receipts,
                campaign_hypothesis=frontier.campaign_hypothesis,
            )
        )
        override = self._quality_override(
            outcome=outcome,
            receipts=receipts,
            quality_evidence=quality_evidence,
        )
        if override is not None:
            override_confidence, override_reason, judge_evidence = override
            quality_evidence = tuple(
                evidence.model_copy(
                    update={
                        "override_confidence": override_confidence,
                        "override_reason": override_reason,
                        "override_judge_evidence": judge_evidence,
                        "override_outcome": "quality_override",
                        "wave_status": "accepted_override",
                        "productive_mutation_count": len(receipts),
                    }
                )
                for evidence in quality_evidence
            )
            outcome = CurationRunOutcome.QUALITY_OVERRIDE
            reason_code = "quality_override"
            restore_result = None
        else:
            restore_result = self._restore_rejected_wave(
                run_id=run_id,
                receipts=receipts,
                quality_evidence=quality_evidence,
            )
            if restore_result is not None:
                outcome = CurationRunOutcome.QUALITY_REJECTED
                reason_code = "quality_wave_rejected"
                if "quality_wave_rejected" not in rejection_codes:
                    rejection_codes.append("quality_wave_rejected")

        terminal = self._curation_store.terminalize_run(run_id, terminal_state, outcome)
        if terminal is None:
            terminal = self._curation_store.get_run(run_id)
        if terminal is None:
            raise RuntimeError(f"curation run {run_id} was not persisted")

        work_item = self._work_item_service.decide(
            outcome,
            reason_code,
            latest,
            quality_evidence=quality_evidence,
            campaign_hypothesis=frontier.campaign_hypothesis,
        )
        specialist_work_items: tuple[WorkItemRecordLike, ...] = ()
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
        run_result = build_run_result(
            run_id=run_id,
            outcome=outcome,
            plan_id=plan_id,
            receipts=receipts,
            rejection_codes=rejection_codes,
            retry_reason=retry_reason,
            budget_usage=budget_usage,
            context_record_counts=context_record_counts,
            failure=failure,
            envelopes=envelopes,
            quality_evidence=[
                evidence.model_dump(mode="json") for evidence in quality_evidence
            ],
            restore_result=restore_result,
            override_confidence=(
                None if override is None else override[0]
            ),
            override_reason=None if override is None else override[1],
            override_judge_evidence={} if override is None else override[2],
            override_outcome=(
                None if override is None else "quality_override"
            ),
        )
        return CurationDryRunResult(
            run=terminal,
            result=run_result,
            context=context,
            validation=validation,
            work_item=work_item,
            planner_attempts=len(envelopes),
            specialist_work_items=specialist_work_items,
            restore_result=restore_result,
        )

    def _quality_override(
        self,
        *,
        outcome: CurationRunOutcome,
        receipts: tuple[CurationActionReceipt, ...],
        quality_evidence: tuple[Any, ...],
    ) -> tuple[float, str, dict[str, object]] | None:
        confidence = getattr(self._planner, "override_confidence", None)
        reason = getattr(self._planner, "override_reason", None)
        eligible = getattr(self._planner, "override_eligible", True)
        if (
            not eligible
            or
            not isinstance(confidence, (int, float))
            or isinstance(confidence, bool)
            or confidence < 0.90
            or not isinstance(reason, str)
            or not reason.strip()
            or outcome not in {CurationRunOutcome.APPLIED}
            or not receipts
            or not quality_evidence
            or any(receipt.status is not CurationReceiptState.VERIFIED for receipt in receipts)
            or any(getattr(evidence, "acceptance_met", None) is False for evidence in quality_evidence)
            or any(getattr(evidence, "wave_status", None) != "rejected" for evidence in quality_evidence)
            or any(receipt.operation != "normalize_memory" for receipt in receipts)
        ):
            return None
        judge_evidence = cast(
            dict[str, object],
            {
            "quality_evidence": [
                evidence.model_dump(mode="json")
                if hasattr(evidence, "model_dump")
                else dict(evidence)
                for evidence in quality_evidence
            ],
            "original_outcome": str(outcome),
            "threshold": 0.90,
            },
        )
        return float(confidence), reason.strip(), judge_evidence

    def _restore_rejected_wave(
        self,
        *,
        run_id: UUID,
        receipts: tuple[CurationActionReceipt, ...],
        quality_evidence: tuple[Any, ...],
    ) -> dict[str, object] | None:
        rejected = [
            evidence
            for evidence in quality_evidence
            if getattr(evidence, "wave_status", None) == "rejected"
        ]
        if not rejected:
            return None
        if self._restore_action_store is None or self._mutation_history is None:
            return {"status": "unavailable", "reason": "restore_storage_unavailable"}

        wave_action_ids = {
            action_id
            for evidence in rejected
            for action_id in getattr(evidence, "wave_action_ids", ())
        }
        mutated_receipts = tuple(
            receipt
            for receipt in receipts
            if receipt.status is CurationReceiptState.VERIFIED
            and receipt.mutation_event_id is not None
        )
        if wave_action_ids != {receipt.action_id for receipt in mutated_receipts}:
            return {"status": "conflict", "reason": "quality_wave_not_fully_sampled"}
        if any(receipt.operation != "normalize_memory" for receipt in mutated_receipts):
            return {"status": "conflict", "reason": "unsupported_restore_wave_family"}

        event_ids = tuple(
            receipt.mutation_event_id
            for receipt in mutated_receipts
            if receipt.mutation_event_id is not None
        )
        expected_record_tokens: dict[UUID, str] = {}
        for event_id in event_ids:
            for revision in self._mutation_history.get_record_revisions(event_id):
                if revision.after_token is not None:
                    expected_record_tokens[revision.memory_id] = revision.after_token
        restore_run_id = uuid4()
        result = restore_wave(
            self._restore_action_store,
            self._mutation_history,
            event_ids,
            expected_record_tokens=expected_record_tokens,
            run_id=restore_run_id,
            idempotency_key=f"quality-rejection:{run_id}",
            curation_store=self._curation_store,
        )
        if self._curation_store.get_run(restore_run_id) is not None:
            restore_outcome = (
                CurationRunOutcome.APPLIED
                if result.status.value in {"applied", "already_applied"}
                else CurationRunOutcome.VERIFICATION_FAILED
            )
            self._curation_store.terminalize_run(
                restore_run_id,
                CurationRunState.EXECUTING,
                restore_outcome,
            )
        payload = _restore_result_payload(result)
        payload["run_id"] = str(restore_run_id)
        return payload

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
        result = await plan_and_validate(
            self._planner,
            CurationPlanningInput(
                request=request,
                context=context,
                mutation_budget=self._config.mutation_budget,
                memory_types=self._memory_types,
                contradictory_memory_ids=self._contradictory_memory_ids,
                protections_by_memory=self._protections_by_memory,
            ),
        )
        return result.plan, result.validation, result.envelopes, result.retry_reason, result.failure

    def _persist_candidate_outcomes(
        self,
        *,
        plan: CurationPlan | None,
        validation: CurationValidationResult | None,
        context: ImmutableCurationContextPacket,
        run_id: UUID,
        frontier_key: str,
        mutation_family: str,
        strategy: str,
        reason_code: str,
        receipts: tuple[CurationActionReceipt, ...],
    ) -> None:
        now = self._clock()
        if plan is None:
            for memory_id in _context_memory_ids(context):
                self._put_candidate_outcome(
                    memory_id=memory_id,
                    token=_candidate_token(context, memory_id),
                    disposition=CandidateDisposition.ESCALATED,
                    reason=reason_code,
                    strategy=strategy,
                    considered_strategy=strategy,
                    mutation_family=mutation_family,
                    frontier_key=frontier_key,
                    run_id=run_id,
                    now=now,
                )
            return

        outcomes: dict[UUID, tuple[CandidateDisposition, str, bool]] = {}

        for decision in plan.retained:
            outcomes[decision.memory_id] = (
                CandidateDisposition.COOLDOWN,
                str(decision.reason),
                False,
            )

        receipt_by_action_id = {receipt.action_id: receipt for receipt in receipts}
        if validation is not None:
            for item in validation.rejected_actions:
                _merge_candidate_outcomes(
                    outcomes,
                    _action_memory_ids(item.action),
                    CandidateDisposition.ESCALATED,
                    ",".join(str(code) for code in item.reason_codes),
                    True,
                )
            for route in validation.specialist_routes:
                _merge_candidate_outcomes(
                    outcomes,
                    _action_memory_ids(route.action),
                    CandidateDisposition.ESCALATED,
                    str(route.reason_code),
                    True,
                )
            for item in validation.accepted_actions:
                receipt = receipt_by_action_id.get(item.action.action_id)
                if receipt is not None and receipt.status is CurationReceiptState.VERIFIED:
                    disposition = CandidateDisposition.ACTIONED
                    reason = "verified_receipt"
                    escalated = False
                elif receipt is not None and receipt.status is CurationReceiptState.REJECTED:
                    disposition = CandidateDisposition.ESCALATED
                    reason = receipt.error_code or "rejected_receipt"
                    escalated = True
                else:
                    disposition = CandidateDisposition.ESCALATED
                    reason = (
                        "unverified_receipt"
                        if receipt is not None
                        else reason_code
                    )
                    escalated = True
                _merge_candidate_outcomes(
                    outcomes,
                    _action_memory_ids(item.action),
                    disposition,
                    reason,
                    escalated,
                )

        for memory_id in set(plan.seed_memory_ids) | set(outcomes):
            disposition, reason, escalated = outcomes.get(
                memory_id,
                (CandidateDisposition.ESCALATED, reason_code, True),
            )
            self._put_candidate_outcome(
                memory_id=memory_id,
                token=_candidate_token(context, memory_id),
                disposition=disposition,
                reason=reason,
                strategy=strategy if escalated else None,
                considered_strategy=strategy,
                mutation_family=mutation_family,
                frontier_key=frontier_key,
                run_id=run_id,
                now=now,
            )

    def _put_candidate_outcome(
        self,
        *,
        memory_id: UUID,
        token: str | None,
        disposition: CandidateDisposition,
        reason: str,
        strategy: str | None,
        considered_strategy: str,
        mutation_family: str,
        frontier_key: str,
        run_id: UUID,
        now: datetime,
    ) -> None:
        previous = self._curation_store.get_candidate_state(memory_id)
        if previous is not None and previous.last_run_id == run_id:
            return
        if (
            previous is not None
            and previous.last_observed_revision_token not in (None, token)
        ):
            return
        if (
            previous is not None
            and previous.last_frontier_key not in (None, frontier_key)
            and (
                previous.disposition is CandidateDisposition.ACTIONED
                or (
                    previous.disposition is CandidateDisposition.ESCALATED
                    and previous.last_escalated_strategy != strategy
                )
            )
        ):
            return
        no_op_count = 0
        escalation_count = 0
        if disposition is CandidateDisposition.COOLDOWN:
            no_op_count = (
                previous.consecutive_no_op_count + 1
                if previous is not None
                and previous.last_observed_revision_token == token
                else 1
            )
        if disposition is CandidateDisposition.ESCALATED:
            escalation_count = (
                previous.escalation_count + 1
                if previous is not None
                and previous.last_observed_revision_token == token
                else 1
            )
        self._curation_store.put_candidate_state(
            CurationCandidateState(
                memory_id=memory_id,
                last_observed_revision_token=token,
                disposition=disposition,
                consecutive_no_op_count=no_op_count,
                cooldown_until=(
                    now + timedelta(seconds=self._config.no_op_cooldown_seconds)
                    if disposition is CandidateDisposition.COOLDOWN
                    else None
                ),
                last_disposition_reason=reason,
                last_frontier_key=frontier_key,
                last_run_id=run_id,
                escalation_count=escalation_count,
                last_escalated_strategy=strategy,
                last_considered_at=now,
                last_considered_strategy=considered_strategy,
                last_mutation_family=(mutation_family if disposition is CandidateDisposition.ACTIONED else previous.last_mutation_family if previous else None),
                last_mutated_at=(now if disposition is CandidateDisposition.ACTIONED else previous.last_mutated_at if previous else None),
                coverage_evidence_json={
                    "reason": reason,
                    "disposition": str(disposition),
                    "strategy": considered_strategy,
                    "frontier_key": frontier_key,
                },
            )
        )

    def _classify_outcome(
        self,
        plan: CurationPlan | None,
        validation: CurationValidationResult | None,
        failure: CurationPlannerError | BaseException | None,
    ) -> tuple[CurationRunOutcome, str]:
        return project_classify_outcome(plan, validation, failure)

    def _execute_accepted_actions(
        self,
        *,
        run_id: UUID,
        executing: CurationRun,
        validation: CurationValidationResult,
        context: ImmutableCurationContextPacket,
    ) -> tuple[CurationRunOutcome, str, CurationRunState, tuple[CurationActionReceipt, ...]]:
        return self._execution_service.execute(
            run_id=run_id,
            executing=executing,
            validation=validation,
            context=context,
        )

def _uuid_or_none(value: str | None) -> UUID | None:
    return None if value is None else UUID(str(value))


def _rejection_codes(validation: CurationValidationResult | None) -> list[str]:
    return project_rejection_codes(validation)


def _budget_usage(
    context: ImmutableCurationContextPacket,
    envelopes: list[PlannerExecutionEnvelope[Any]],
    validation: CurationValidationResult | None,
) -> CurationBudgetUsage:
    return project_budget_usage(context, envelopes, validation)
