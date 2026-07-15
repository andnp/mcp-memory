from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast
from uuid import UUID, uuid4

import pytest

from mcp_memory.core.curation_context import AcceptedMaintenanceRead, build_context_packet
from mcp_memory.core.curation_harness import (
    CurationDryRunHarness,
    CurationFrontier,
    WorkItemAction,
)
from mcp_memory.core.curation_models import CurationPlan, CurationRunOutcome, RetentionDecision, RetentionReason
from mcp_memory.core.curation_planner import (
    CurationPlannerCancelledError,
    CurationPlannerProviderError,
    FakeCurationPlanner,
    FakePlannerScenario,
)
from mcp_memory.core.curation_disclosure import ProviderTrust, ProviderTrustClass
from mcp_memory.curation_store import SQLiteCurationStore
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


def _plan(frontier: CurationFrontier, seed: UUID) -> CurationPlan:
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
        seed_memory_ids=[seed],
        retained=[RetentionDecision(memory_id=seed, reason=RetentionReason.ALREADY_FOCUSED, rationale="still focused")],
        rationale="no mutation is needed",
    )


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
def _harness(db_manager: DatabaseManager, planner) -> CurationDryRunHarness:
    return CurationDryRunHarness(
        curation_store=SQLiteCurationStore(db_manager),
        planner=planner,
        work_items=SQLiteWorkItemRepository(db_manager),
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
