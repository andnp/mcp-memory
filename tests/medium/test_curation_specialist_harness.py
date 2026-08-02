from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from mcp_memory.core.curation_context import (
    AcceptedMaintenanceRead,
    build_context_packet,
)
from mcp_memory.core.curation_disclosure import ProviderTrust, ProviderTrustClass
from mcp_memory.core.curation_harness import CurationDryRunHarness, CurationFrontier
from mcp_memory.core.curation_models import (
    CreateLinkAction,
    CurationPlan,
    EvidenceRef,
    LinkAssertion,
)
from mcp_memory.core.curation_planner import FakeCurationPlanner, FakePlannerScenario
from mcp_memory.curation_store import SQLiteCurationStore
from mcp_memory.utils.db import DatabaseManager
from mcp_memory.work_item_store import (
    WORK_FAMILY_GRAPH_LINK_REVIEW,
    SQLiteWorkItemRepository,
)

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
        "revision_token": "record-revision-1",
    }


class _RequestBoundPlanner:
    async def create_plan(self, request, tools):
        source_id, target_id = request.context.seed_memory_ids
        plan = CurationPlan(
            plan_id=request.plan_id,
            run_id=request.run_id,
            frontier_key=request.frontier_key,
            context_fingerprint=request.context_fingerprint,
            seed_memory_ids=[source_id, target_id],
            rationale="route graph work without executing it",
            actions=[
                CreateLinkAction(
                    action_id=uuid4(),
                    source_id=source_id,
                    target_id=target_id,
                    link_type="DEPENDS_ON",
                    context="The source depends on the target.",
                    confidence=0.9,
                    rationale="the graph specialist owns this relationship",
                    evidence=[
                        EvidenceRef(
                            link=LinkAssertion(
                                source_id=source_id,
                                target_id=target_id,
                                link_type="DEPENDS_ON",
                                context="The source depends on the target.",
                            ),
                            revision_token="evidence-1",
                        )
                    ],
                )
            ],
        )
        return await FakeCurationPlanner([FakePlannerScenario(plan=plan)]).create_plan(request, tools)


def _harness(db_manager: DatabaseManager) -> CurationDryRunHarness:
    return CurationDryRunHarness(
        curation_store=SQLiteCurationStore(db_manager),
        planner=_RequestBoundPlanner(),
        work_items=SQLiteWorkItemRepository(db_manager),
        clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )


@pytest.mark.asyncio
async def test_harness_defers_accepted_link_without_executor(db_manager: DatabaseManager) -> None:
    source_id = uuid4()
    target_id = uuid4()
    context = build_context_packet(
        family="curator",
        strategy="focused",
        seed_reads=(AcceptedMaintenanceRead(_record(source_id)), AcceptedMaintenanceRead(_record(target_id))),
        provider=ProviderTrust(ProviderTrustClass.LOCAL),
    )
    frontier = CurationFrontier.direct(
        family="curator",
        strategy="focused",
        seed_reads=(AcceptedMaintenanceRead(_record(source_id)), AcceptedMaintenanceRead(_record(target_id))),
        frontier_key=context.frontier_fingerprint,
    )

    result = await _harness(db_manager).run(frontier)
    work_items = SQLiteWorkItemRepository(db_manager)

    assert not result.specialist_work_items
    assert result.outcome.value == "deferred"
    assert not work_items.list_items(family_key=WORK_FAMILY_GRAPH_LINK_REVIEW)
    assert db_manager.get_connection().execute("SELECT COUNT(*) FROM memory_mutation_events").fetchone()[0] == 0
