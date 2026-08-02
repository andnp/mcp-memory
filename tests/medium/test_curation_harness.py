from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import UUID, uuid4

import pytest

from mcp_memory.core.curation_context import AcceptedMaintenanceRead, CurationReadBudget, build_context_packet
from mcp_memory.core.curation_models import (
    ActionPreconditions,
    ArchiveMemoryAction,
    ClaimManifest,
    ClaimMapping,
    CreateLinkAction,
    CurationPlan,
    CurationRunOutcome,
    EvidenceRef,
    LinkAssertion,
    MergeMemoriesAction,
    NormalizeMemoryAction,
    RemoveLinkAction,
    RetentionDecision,
    RetentionReason,
    RewriteMemoryAction,
    SplitMemoryAction,
)
from mcp_memory.core.curation_identity import record_token
from mcp_memory.core.curation_harness import (
    CurationDryRunHarness,
    CurationFrontier,
    CurationHarnessConfig,
    WorkItemAction,
)
from mcp_memory.core.curation_planner import (
    CurationPlannerCancelledError,
    CurationPlannerProviderError,
    FakeCurationPlanner,
    FakePlannerScenario,
)
from mcp_memory.core.curation_disclosure import ProviderTrust, ProviderTrustClass
from mcp_memory.core.curation_routing import MaintenanceFamily
from mcp_memory.core.curation_validation import (
    AcceptedCurationAction,
    CurationSpecialistRoute,
    CurationValidationResult,
)
from mcp_memory.curation_store import CurationActionReceipt, CurationReceiptState, CurationRun, CurationRunState, SQLiteCurationStore
from mcp_memory.utils.db import DatabaseManager
from mcp_memory.work_item_store import SQLiteWorkItemRepository, WORK_ITEM_STATUS_DEFERRED


pytestmark = pytest.mark.medium


def _record(memory_id: UUID) -> dict[str, object]:
    return {
        "id": memory_id,
        "title": "Title",
        "content": "A useful memory.",
        "summary": "A useful summary.",
        "type": "observation",
        "status": "active",
        "tags": [],
        "workspace_ids": [],
    }


def _frontier(seed: UUID, *, work_item_id: str | None = None) -> CurationFrontier:
    return CurationFrontier(
        family="curator",
        strategy="focused",
        seed_reads=(AcceptedMaintenanceRead(_record(seed)),),
        work_item_id=work_item_id,
    )


def _plan(
    frontier: CurationFrontier,
    seed: UUID,
    *,
    seed_ids: tuple[UUID, ...] | None = None,
) -> CurationPlan:
    context = build_context_packet(
        family=frontier.family,
        strategy=frontier.strategy,
        seed_reads=frontier.seed_reads,
        provider=ProviderTrust(ProviderTrustClass.LOCAL),
    )
    return CurationPlan(
        plan_id=UUID("00000000-0000-0000-0000-000000000002"),
        run_id=UUID("00000000-0000-0000-0000-000000000001"),
        frontier_key=context.frontier_fingerprint,
        context_fingerprint=context.context_fingerprint,
        seed_memory_ids=list(seed_ids or (seed,)),
        retained=[
            RetentionDecision(memory_id=memory_id, reason=RetentionReason.ALREADY_FOCUSED, rationale="still focused")
            for memory_id in (seed_ids or (seed,))
        ],
        rationale="no mutation is needed",
    )


def _action_tokenized_record(memory_id: UUID) -> dict[str, object]:
    return _record(memory_id)


def _normalize_action(record: dict[str, object]) -> NormalizeMemoryAction:
    return NormalizeMemoryAction(
        action_id=uuid4(),
        target_id=cast(UUID, record["id"]),
        confidence=1.0,
        rationale="normalize",
        summary="Normalized summary.",
        preconditions=ActionPreconditions(record_tokens={cast(UUID, record["id"]): record_token(record)}),
    )


def _rewrite_action(record: dict[str, object]) -> RewriteMemoryAction:
    memory_id = cast(UUID, record["id"])
    return RewriteMemoryAction(
        action_id=uuid4(),
        target_id=memory_id,
        confidence=1.0,
        rationale="rewrite",
        content="Rewritten content.",
        summary="Rewritten summary.",
        evidence=[EvidenceRef(memory_id=memory_id)],
        claim_manifest=ClaimManifest(
            preserved_claims=["keep"],
            source_mapping=[ClaimMapping(output="Rewritten summary.", source_memory_ids=[memory_id])],
        ),
        preconditions=ActionPreconditions(record_tokens={memory_id: record_token(record)}),
    )


def _create_link_action(source: dict[str, object], target: dict[str, object]) -> CreateLinkAction:
    source_id = cast(UUID, source["id"])
    target_id = cast(UUID, target["id"])
    link = LinkAssertion(source_id=source_id, target_id=target_id, link_type="SUPPORTS", context="supports")
    return CreateLinkAction(
        action_id=uuid4(),
        source_id=source_id,
        target_id=target_id,
        link_type="SUPPORTS",
        context="supports",
        confidence=1.0,
        rationale="link",
        evidence=[EvidenceRef(link=link)],
        preconditions=ActionPreconditions(
            record_tokens={source_id: record_token(source), target_id: record_token(target)},
            absent_links=[link],
        ),
    )


def _remove_link_action(source: dict[str, object], target: dict[str, object]) -> RemoveLinkAction:
    source_id = cast(UUID, source["id"])
    target_id = cast(UUID, target["id"])
    return RemoveLinkAction(
        action_id=uuid4(),
        source_id=source_id,
        target_id=target_id,
        link_type="SUPPORTS",
        context="supports",
        confidence=1.0,
        rationale="remove link",
        evidence=[EvidenceRef(link=LinkAssertion(source_id=source_id, target_id=target_id, link_type="SUPPORTS"))],
        preconditions=ActionPreconditions(
            record_tokens={source_id: record_token(source), target_id: record_token(target)},
        ),
    )


def _merge_action(canonical: dict[str, object], source: dict[str, object]) -> MergeMemoriesAction:
    canonical_id = cast(UUID, canonical["id"])
    source_id = cast(UUID, source["id"])
    return MergeMemoriesAction(
        action_id=uuid4(),
        canonical_id=canonical_id,
        source_ids=[source_id],
        confidence=1.0,
        rationale="merge",
        content="Merged content.",
        evidence=[EvidenceRef(memory_id=canonical_id)],
        claim_manifest=ClaimManifest(
            preserved_claims=["keep"],
            source_mapping=[ClaimMapping(output="Merged content.", source_memory_ids=[canonical_id, source_id])],
            unresolved_tensions=["fact merge reviewed"],
        ),
        preconditions=ActionPreconditions(
            record_tokens={canonical_id: record_token(canonical), source_id: record_token(source)},
        ),
    )


def _split_action(record: dict[str, object]) -> SplitMemoryAction:
    memory_id = cast(UUID, record["id"])
    return SplitMemoryAction(
        action_id=uuid4(),
        target_id=memory_id,
        confidence=1.0,
        rationale="split",
        evidence=[EvidenceRef(memory_id=memory_id)],
        children=[ClaimMapping(output="Child summary.", source_memory_ids=[memory_id])],
        claim_manifest=ClaimManifest(
            preserved_claims=["keep"],
            source_mapping=[ClaimMapping(output="Child summary.", source_memory_ids=[memory_id])],
        ),
        preconditions=ActionPreconditions(record_tokens={memory_id: record_token(record)}),
    )


def _archive_action(record: dict[str, object]) -> ArchiveMemoryAction:
    memory_id = cast(UUID, record["id"])
    return ArchiveMemoryAction(
        action_id=uuid4(),
        target_id=memory_id,
        confidence=1.0,
        rationale="archive",
        evidence=[EvidenceRef(memory_id=memory_id)],
        claim_manifest=ClaimManifest(preserved_claims=["keep"]),
        preconditions=ActionPreconditions(record_tokens={memory_id: record_token(record)}),
    )


def _synthetic_validation(action) -> CurationValidationResult:
    plan = CurationPlan(
        plan_id=uuid4(),
        run_id=uuid4(),
        frontier_key="frontier",
        context_fingerprint="context",
        seed_memory_ids=[],
        actions=[action],
        retained=[],
        rationale="synthetic validation",
    )
    return CurationValidationResult(
        plan=plan,
        accepted_actions=(AcceptedCurationAction(action=action, family=MaintenanceFamily.CURATOR),),
    )


class _DispatchingExecutor:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def _receipt(self, action, affected_ids: list[UUID]) -> CurationActionReceipt:
        return CurationActionReceipt(
            run_id=uuid4(),
            action_id=action.action_id,
            operation=action.operation,
            affected_ids=affected_ids,
            status=CurationReceiptState.APPLIED_UNVERIFIED,
        )

    def execute_normalize(self, action, **kwargs):
        self.calls.append("execute_normalize")
        return self._receipt(action, [action.target_id])

    def execute_create_link(self, action, **kwargs):
        self.calls.append("execute_create_link")
        return self._receipt(action, [action.source_id, action.target_id])

    def execute_rewrite(self, action, **kwargs):
        self.calls.append("execute_rewrite")
        return self._receipt(action, [action.target_id])

    def execute_remove_link(self, action, **kwargs):
        self.calls.append("execute_remove_link")
        return self._receipt(action, [action.source_id, action.target_id])

    def execute_merge(self, action, **kwargs):
        self.calls.append("execute_merge")
        return self._receipt(action, [action.canonical_id, *action.source_ids])

    def execute_split(self, action, **kwargs):
        self.calls.append("execute_split")
        return self._receipt(action, [action.target_id])

    def execute_archive(self, action, **kwargs):
        self.calls.append("execute_archive")
        return self._receipt(action, [action.target_id])


class _DispatchingVerifier:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def verify(self, receipt: CurationActionReceipt, action) -> CurationActionReceipt:
        self.calls.append(action.operation)
        return receipt.model_copy(
            update={
                "status": CurationReceiptState.VERIFIED,
                "verified_at": datetime(2026, 1, 1, tzinfo=UTC),
            }
        )


class _FailingVerifier(_DispatchingVerifier):
    def verify(self, receipt: CurationActionReceipt, action) -> CurationActionReceipt:
        self.calls.append(action.operation)
        return receipt.model_copy(
            update={
                "status": CurationReceiptState.VERIFICATION_FAILED,
                "error_code": "after_token_mismatch",
            }
        )


class _TransitionOnlyStore:
    def __init__(self) -> None:
        self.transitions: list[tuple[UUID, CurationRunState]] = []

    def transition_run(self, run_id: UUID, state: CurationRunState, run: CurationRun) -> CurationRun:
        self.transitions.append((run_id, state))
        return run


class _AllActionsPlanner:
    def __init__(self, actions: list[Any], seed_ids: list[UUID]) -> None:
        self._actions = actions
        self._seed_ids = seed_ids
        self.calls = 0

    async def create_plan(self, request, tools):
        self.calls += 1
        plan = CurationPlan(
            plan_id=request.plan_id,
            run_id=request.run_id,
            frontier_key=request.frontier_key,
            context_fingerprint=request.context_fingerprint,
            seed_memory_ids=list(self._seed_ids),
            actions=list(self._actions),
            retained=[],
            rationale="exercise every supported action through the harness run path",
        )
        return await FakeCurationPlanner([FakePlannerScenario(plan=plan)]).create_plan(request, tools)


class _RequestBoundFakePlanner:
    """Bind static fake scenarios to the harness-created request identity."""

    def __init__(self, scenarios: list[FakePlannerScenario]) -> None:
        self._scenarios = scenarios
        self.calls = 0

    async def create_plan(self, request, tools):
        index = min(self.calls, len(self._scenarios) - 1)
        scenario = self._scenarios[index]
        self.calls += 1
        if scenario.plan is not None and isinstance(scenario.plan, CurationPlan):
            scenario = FakePlannerScenario(
                plan=scenario.plan.model_copy(
                    update={
                        "run_id": request.run_id,
                        "plan_id": request.plan_id,
                        "frontier_key": request.frontier_key,
                        "context_fingerprint": request.context_fingerprint,
                    }
                ),
                provider_key=scenario.provider_key,
                model_name=scenario.model_name,
            )
        return await FakeCurationPlanner([scenario]).create_plan(request, tools)


def _harness(
    db_manager: DatabaseManager,
    planner,
    *,
    config: CurationHarnessConfig | None = None,
) -> CurationDryRunHarness:
    return CurationDryRunHarness(
        curation_store=SQLiteCurationStore(db_manager),
        planner=planner,
        work_items=SQLiteWorkItemRepository(db_manager),
        config=config,
        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )


@pytest.mark.asyncio
async def test_valid_noop_completes_claimed_work_and_persists_disposition(db_manager: DatabaseManager) -> None:
    seed = uuid4()
    work_items = SQLiteWorkItemRepository(db_manager)
    item, _ = work_items.enqueue_unique(
        family_key="curator",
        execution_lane="agentic",
        payload={"memory_ids": [str(seed)]},
        idempotency_key="curation-harness-noop",
    )
    claimed = work_items.claim_batch(family_key="curator", execution_lane="agentic", lease_owner="worker", limit=1)[0]
    frontier = _frontier(seed, work_item_id=claimed.id)
    planner = _RequestBoundFakePlanner([FakePlannerScenario(plan=_plan(frontier, seed))])

    result = await _harness(db_manager, planner).run(frontier)

    assert result.outcome is CurationRunOutcome.NO_OP
    assert result.work_item.action is WorkItemAction.COMPLETE
    assert work_items.get_item(item.id).status == "completed"
    state = SQLiteCurationStore(db_manager).get_candidate_state(seed)
    assert state is not None
    assert state.disposition.value == "cooldown"
    assert state.last_run_id == result.run.run_id


@pytest.mark.asyncio
async def test_repeated_noop_advances_cooldown_metadata(db_manager: DatabaseManager) -> None:
    seed = uuid4()
    frontier = _frontier(seed)
    planner = _RequestBoundFakePlanner([
        FakePlannerScenario(plan=_plan(frontier, seed)),
        FakePlannerScenario(plan=_plan(frontier, seed)),
    ])
    harness = _harness(db_manager, planner)

    first = await harness.run(frontier)
    second = await harness.run(frontier)

    state = SQLiteCurationStore(db_manager).get_candidate_state(seed)
    assert state is not None
    assert state.disposition.value == "cooldown"
    assert state.consecutive_no_op_count == 2
    assert state.last_frontier_key == second.run.frontier_key
    assert state.last_run_id == second.run.run_id
    assert first.run.run_id != second.run.run_id


@pytest.mark.asyncio
async def test_valid_action_is_deferred_and_never_executes_mutation(db_manager: DatabaseManager) -> None:
    seed = uuid4()
    work_items = SQLiteWorkItemRepository(db_manager)
    item, _ = work_items.enqueue_unique(
        family_key="curator", execution_lane="agentic", idempotency_key="curation-harness-action"
    )
    claimed = work_items.claim_batch(family_key="curator", execution_lane="agentic", lease_owner="worker", limit=1)[0]
    frontier = _frontier(seed, work_item_id=claimed.id)
    plan = CurationPlan.model_validate(
        _plan(frontier, seed).model_dump(mode="python")
        | {
            "retained": [],
            "actions": [{
                "operation": "normalize_memory",
                "action_id": uuid4(),
                "target_id": seed,
                "confidence": 1,
                "rationale": "specific normalization",
                "summary": "A useful summary.",
            }],
        }
    )
    planner = _RequestBoundFakePlanner([FakePlannerScenario(plan=plan)])

    result = await _harness(db_manager, planner).run(frontier)

    assert result.outcome is CurationRunOutcome.DEFERRED
    assert result.work_item.action is WorkItemAction.DEFER
    assert work_items.get_item(item.id).status == WORK_ITEM_STATUS_DEFERRED
    assert db_manager.get_connection().execute("SELECT COUNT(*) FROM memory_mutation_events").fetchone()[0] == 0
    state = SQLiteCurationStore(db_manager).get_candidate_state(seed)
    assert state is not None
    assert state.disposition.value == "escalated"
    assert state.escalation_count == 1
    assert state.last_escalated_strategy == "focused"
    assert state.last_disposition_reason == "dry_run_requires_executor"


@pytest.mark.asyncio
async def test_schema_failure_retries_once_then_completes_noop(db_manager: DatabaseManager) -> None:
    seed = uuid4()
    frontier = _frontier(seed)
    valid = _plan(frontier, seed)
    planner = _RequestBoundFakePlanner([
        FakePlannerScenario(plan={"not": "a plan"}),
        FakePlannerScenario(plan=valid),
    ])

    result = await _harness(db_manager, planner).run(frontier)

    assert planner.calls == 2
    assert result.planner_attempts == 2
    assert result.outcome is CurationRunOutcome.NO_OP
    assert result.run.retry_reason == "schema_invalid"


@pytest.mark.asyncio
async def test_repeated_schema_failure_defers_claimed_work(db_manager: DatabaseManager) -> None:
    seed = uuid4()
    work_items = SQLiteWorkItemRepository(db_manager)
    item, _ = work_items.enqueue_unique(
        family_key="curator", execution_lane="agentic", idempotency_key="curation-harness-invalid"
    )
    claimed = work_items.claim_batch(family_key="curator", execution_lane="agentic", lease_owner="worker", limit=1)[0]
    frontier = _frontier(seed, work_item_id=claimed.id)
    planner = _RequestBoundFakePlanner([
        FakePlannerScenario(plan={"not": "a plan"}),
        FakePlannerScenario(plan={"not": "a plan"}),
    ])

    result = await _harness(db_manager, planner).run(frontier)

    assert planner.calls == 2
    assert result.outcome is CurationRunOutcome.INVALID_PLAN
    assert result.work_item.action is WorkItemAction.DEFER
    assert work_items.get_item(item.id).status == WORK_ITEM_STATUS_DEFERRED


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure, expected",
    [
        (CurationPlannerProviderError("provider unavailable"), CurationRunOutcome.PROVIDER_FAILED),
        (CurationPlannerCancelledError(), CurationRunOutcome.CANCELLED),
    ],
)
async def test_provider_and_cancellation_failures_defer_claimed_work(
    db_manager: DatabaseManager,
    failure: BaseException,
    expected: CurationRunOutcome,
) -> None:
    seed = uuid4()
    work_items = SQLiteWorkItemRepository(db_manager)
    item, _ = work_items.enqueue_unique(
        family_key="curator", execution_lane="agentic", idempotency_key=f"failure-{uuid4()}"
    )
    claimed = work_items.claim_batch(family_key="curator", execution_lane="agentic", lease_owner="worker", limit=1)[0]
    frontier = _frontier(seed, work_item_id=claimed.id)
    planner = _RequestBoundFakePlanner([FakePlannerScenario(failure=cast(Any, failure))])

    result = await _harness(db_manager, planner).run(frontier)

    assert result.outcome is expected
    assert result.work_item.action is WorkItemAction.DEFER
    assert work_items.get_item(item.id).status == WORK_ITEM_STATUS_DEFERRED


@pytest.mark.asyncio
async def test_context_budget_trims_support_before_seeds(db_manager: DatabaseManager) -> None:
    first, second, support = uuid4(), uuid4(), uuid4()
    retained_frontier = CurationFrontier(
        family="curator",
        strategy="focused",
        seed_reads=(AcceptedMaintenanceRead(_record(first)), AcceptedMaintenanceRead(_record(second))),
    )
    frontier = CurationFrontier(
        family="curator",
        strategy="focused",
        seed_reads=retained_frontier.seed_reads,
        support_reads=(AcceptedMaintenanceRead(_record(support)),),
    )
    planner = _RequestBoundFakePlanner([
        FakePlannerScenario(plan=_plan(retained_frontier, first, seed_ids=(first, second)))
    ])
    result = await _harness(
        db_manager,
        planner,
        config=CurationHarnessConfig(read_budget=CurationReadBudget(max_context_characters=450)),
    ).run(frontier)

    assert result.context.seed_memory_ids == (str(first), str(second))
    assert result.context.support_memory_ids == ()
    assert result.result.context_record_counts == {
        "included_seed_count": 2,
        "included_support_count": 0,
        "omitted_seed_count": 0,
        "omitted_support_count": 1,
    }


@pytest.mark.asyncio
async def test_context_budget_trims_lowest_priority_trailing_seed(db_manager: DatabaseManager) -> None:
    first, second = uuid4(), uuid4()
    retained_frontier = _frontier(first)
    frontier = CurationFrontier(
        family="curator",
        strategy="focused",
        seed_reads=(AcceptedMaintenanceRead(_record(first)), AcceptedMaintenanceRead(_record(second))),
    )
    planner = _RequestBoundFakePlanner([FakePlannerScenario(plan=_plan(retained_frontier, first))])
    result = await _harness(
        db_manager,
        planner,
        config=CurationHarnessConfig(read_budget=CurationReadBudget(max_context_characters=250)),
    ).run(frontier)

    assert result.context.seed_memory_ids == (str(first),)
    assert result.result.context_record_counts["omitted_seed_count"] == 1
    assert result.run.frontier_key == result.context.frontier_fingerprint
    assert result.validation is not None and result.validation.valid


@pytest.mark.asyncio
async def test_context_identity_is_deterministic_after_seed_trimming(db_manager: DatabaseManager) -> None:
    first, second = uuid4(), uuid4()
    frontier = CurationFrontier(
        family="curator",
        strategy="focused",
        seed_reads=(AcceptedMaintenanceRead(_record(first)), AcceptedMaintenanceRead(_record(second))),
        frontier_key="stale-frontier-key",
    )
    planner = _RequestBoundFakePlanner([FakePlannerScenario(plan=_plan(_frontier(first), first))])
    config = CurationHarnessConfig(read_budget=CurationReadBudget(max_context_characters=250))

    first_result = await _harness(db_manager, planner, config=config).run(frontier)
    second_result = await _harness(
        db_manager,
        _RequestBoundFakePlanner([FakePlannerScenario(plan=_plan(_frontier(first), first))]),
        config=config,
    ).run(frontier)

    assert first_result.context.context_fingerprint == second_result.context.context_fingerprint
    assert first_result.context.frontier_fingerprint == second_result.context.frontier_fingerprint
    assert first_result.run.frontier_key == first_result.context.frontier_fingerprint
    assert first_result.run.frontier_key != "stale-frontier-key"


@pytest.mark.parametrize(
    ("record_count", "build_action", "expected_executor_method"),
    [
        (1, lambda records: _normalize_action(records[0]), "execute_normalize"),
        (1, lambda records: _rewrite_action(records[0]), "execute_rewrite"),
        (2, lambda records: _create_link_action(records[0], records[1]), "execute_create_link"),
        (2, lambda records: _remove_link_action(records[0], records[1]), "execute_remove_link"),
        (2, lambda records: _merge_action(records[0], records[1]), "execute_merge"),
        (1, lambda records: _split_action(records[0]), "execute_split"),
        (1, lambda records: _archive_action(records[0]), "execute_archive"),
    ],
)
@pytest.mark.asyncio
async def test_accepted_actions_dispatch_through_executor_and_verifier(
    record_count: int,
    build_action: Any,
    expected_executor_method: str,
) -> None:
    records = [_action_tokenized_record(uuid4()) for _ in range(record_count)]
    action = build_action(records)
    validation = _synthetic_validation(action)
    frontier = CurationFrontier.direct(
        family="curator",
        strategy="focused",
        seed_reads=tuple(AcceptedMaintenanceRead(record) for record in records),
    )
    context = build_context_packet(
        family=frontier.family,
        strategy=frontier.strategy,
        seed_reads=frontier.seed_reads,
        provider=ProviderTrust(ProviderTrustClass.LOCAL),
    )
    executor = _DispatchingExecutor()
    verifier = _DispatchingVerifier()
    harness = CurationDryRunHarness(
        curation_store=cast(Any, _TransitionOnlyStore()),
        planner=cast(Any, object()),
        config=CurationHarnessConfig(provider=ProviderTrust(ProviderTrustClass.LOCAL)),
        executor=cast(Any, executor),
        verifier=cast(Any, verifier),
    )
    run = CurationRun(
        run_id=uuid4(),
        frontier_key=context.frontier_fingerprint,
        context_fingerprint=context.context_fingerprint,
        plan_id=uuid4(),
        state=CurationRunState.EXECUTING,
    )

    outcome, reason, terminal_state, receipts = harness._execute_accepted_actions(
        run_id=run.run_id,
        executing=run,
        validation=validation,
        context=context,
    )

    assert outcome is CurationRunOutcome.APPLIED
    assert reason == "verified_receipts"
    assert terminal_state is CurationRunState.VERIFYING
    assert executor.calls == [expected_executor_method]
    assert verifier.calls == [action.operation]
    assert len(receipts) == 1
    assert receipts[0].status is CurationReceiptState.VERIFIED


@pytest.mark.asyncio
async def test_execute_accepted_actions_runs_all_seven_operations_through_harness() -> None:
    db_path = Path(f".curation-harness-all-actions-{uuid4()}.sqlite")
    manager = DatabaseManager(db_path)
    records = [_action_tokenized_record(uuid4()) for _ in range(10)]
    normalize = records[0]
    rewrite = records[1]
    create_source = records[2]
    create_target = records[3]
    remove_source = records[4]
    remove_target = records[5]
    merge_canonical = records[6]
    merge_source = records[7]
    split_target = records[8]
    archive_target = records[9]
    create_action = _create_link_action(create_source, create_target)
    remove_action = _remove_link_action(remove_source, remove_target)
    merge_action = _merge_action(merge_canonical, merge_source)
    split_action = _split_action(split_target)
    archive_action = _archive_action(archive_target)
    action_sequence = [
        _normalize_action(normalize),
        _rewrite_action(rewrite),
        create_action,
        remove_action,
        merge_action,
        split_action,
        archive_action,
    ]
    planner = _AllActionsPlanner(action_sequence, [cast(UUID, record["id"]) for record in records])
    executor = _DispatchingExecutor()
    verifier = _DispatchingVerifier()
    try:
        harness = CurationDryRunHarness(
            curation_store=SQLiteCurationStore(manager),
            planner=cast(Any, planner),
            config=CurationHarnessConfig(execute_accepted_actions=True),
            memory_types={cast(UUID, record["id"]): "fact" for record in records},
            executor=cast(Any, executor),
            verifier=cast(Any, verifier),
        )
        result = await harness.run(
            CurationFrontier.direct(
                family="curator",
                strategy="focused",
                seed_reads=tuple(AcceptedMaintenanceRead(record) for record in records),
            )
        )

        assert planner.calls == 1
        assert result.outcome is CurationRunOutcome.APPLIED
        assert result.validation is not None
        assert [item.action.operation for item in result.validation.accepted_actions] == [
            "normalize_memory",
            "rewrite_memory",
            "create_link",
            "remove_link",
            "merge_memories",
            "split_memory",
            "archive_memory",
        ]
        assert not result.validation.specialist_routes
        assert executor.calls == [
            "execute_normalize",
            "execute_rewrite",
            "execute_create_link",
            "execute_remove_link",
            "execute_merge",
            "execute_split",
            "execute_archive",
        ]
        assert verifier.calls == [
            "normalize_memory",
            "rewrite_memory",
            "create_link",
            "remove_link",
            "merge_memories",
            "split_memory",
            "archive_memory",
        ]
        assert result.result.verified_action_count == 7
        assert len(result.result.receipts) == 7
        assert result.specialist_work_items == ()
        for record in records:
            state = SQLiteCurationStore(manager).get_candidate_state(cast(UUID, record["id"]))
            assert state is not None
            assert state.disposition.value == "actioned"
            assert state.last_observed_revision_token is not None
            assert state.last_frontier_key == result.run.frontier_key
            assert state.last_run_id == result.run.run_id
    finally:
        manager.close()
        for suffix in ("", "-wal", "-shm"):
            candidate = db_path.with_name(db_path.name + suffix)
            if candidate.exists():
                candidate.unlink()


@pytest.mark.asyncio
async def test_verification_failure_escalates_without_marking_candidate_actioned(
    db_manager: DatabaseManager,
) -> None:
    seed = uuid4()
    record = _action_tokenized_record(seed)
    action = _normalize_action(record)
    frontier = _frontier(seed)
    plan = CurationPlan(
        plan_id=uuid4(),
        run_id=uuid4(),
        frontier_key="frontier",
        context_fingerprint="context",
        seed_memory_ids=[seed],
        actions=[action],
        rationale="verification failure",
    )
    planner = _RequestBoundFakePlanner([
        FakePlannerScenario(plan=plan),
        FakePlannerScenario(plan=plan),
    ])
    executor = _DispatchingExecutor()
    verifier = _FailingVerifier()
    harness = CurationDryRunHarness(
        curation_store=SQLiteCurationStore(db_manager),
        planner=planner,
        config=CurationHarnessConfig(execute_accepted_actions=True),
        memory_types={seed: "fact"},
        executor=cast(Any, executor),
        verifier=cast(Any, verifier),
        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )

    first = await harness.run(frontier)
    second = await harness.run(frontier)

    assert first.outcome is CurationRunOutcome.VERIFICATION_FAILED
    assert second.outcome is CurationRunOutcome.VERIFICATION_FAILED
    state = SQLiteCurationStore(db_manager).get_candidate_state(seed)
    assert state is not None
    assert state.disposition.value == "escalated"
    assert state.escalation_count == 2
    assert state.last_run_id == second.run.run_id
    assert state.last_disposition_reason == "unverified_receipt"


def test_specialist_route_persists_escalated_candidate_state(db_manager: DatabaseManager) -> None:
    seed = uuid4()
    record = _action_tokenized_record(seed)
    frontier = _frontier(seed)
    context = build_context_packet(
        family=frontier.family,
        strategy=frontier.strategy,
        seed_reads=frontier.seed_reads,
        provider=ProviderTrust(ProviderTrustClass.LOCAL),
    )
    action = _normalize_action(record)
    plan = CurationPlan(
        plan_id=uuid4(),
        run_id=uuid4(),
        frontier_key=context.frontier_fingerprint,
        context_fingerprint=context.context_fingerprint,
        seed_memory_ids=[seed],
        actions=[action],
        rationale="specialist route",
    )
    validation = CurationValidationResult(
        plan=plan,
        specialist_routes=(
            CurationSpecialistRoute(action=action, family=MaintenanceFamily.CURATOR),
        ),
    )
    harness = CurationDryRunHarness(
        curation_store=SQLiteCurationStore(db_manager),
        planner=cast(Any, object()),
        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )
    run_id = uuid4()

    harness._persist_candidate_outcomes(
        plan=plan,
        validation=validation,
        context=context,
        run_id=run_id,
        frontier_key=context.frontier_fingerprint,
        strategy=frontier.strategy,
        reason_code="needs_different_specialist",
        receipts=(),
    )

    state = SQLiteCurationStore(db_manager).get_candidate_state(seed)
    assert state is not None
    assert state.disposition.value == "escalated"
    assert state.escalation_count == 1
    assert state.last_escalated_strategy == frontier.strategy
    assert state.last_frontier_key == context.frontier_fingerprint
    assert state.last_run_id == run_id


@pytest.mark.asyncio
async def test_impossible_seed_budget_persists_and_defers_without_planner_or_mutation(
    db_manager: DatabaseManager,
) -> None:
    seed = uuid4()
    work_items = SQLiteWorkItemRepository(db_manager)
    item, _ = work_items.enqueue_unique(
        family_key="curator",
        execution_lane="agentic",
        idempotency_key="curation-harness-impossible-budget",
    )
    claimed = work_items.claim_batch(family_key="curator", execution_lane="agentic", lease_owner="worker", limit=1)[0]

    class _NoCallPlanner:
        calls = 0

        async def create_plan(self, request, tools):
            self.calls += 1
            raise AssertionError("planner must not be called for an impossible context")

    planner = _NoCallPlanner()
    result = await _harness(
        db_manager,
        planner,
        config=CurationHarnessConfig(
            read_budget=CurationReadBudget(max_context_characters=100),
        ),
    ).run(_frontier(seed, work_item_id=claimed.id))

    assert planner.calls == 0
    assert result.outcome is CurationRunOutcome.BUDGET_EXHAUSTED
    assert result.run.retry_reason == "budget_exhausted"
    assert result.run.disclosure_audit["budget_failure"]["dimension"] == "context_characters"
    assert result.work_item.action is WorkItemAction.DEFER
    assert result.work_item.retry_delay_seconds == 3600.0
    assert work_items.get_item(item.id).status == WORK_ITEM_STATUS_DEFERRED
    assert db_manager.get_connection().execute("SELECT COUNT(*) FROM memory_mutation_events").fetchone()[0] == 0
