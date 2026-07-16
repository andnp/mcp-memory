from __future__ import annotations

from uuid import uuid4

import pytest

from mcp_memory.core.curation_models import ActionPreconditions, CreateLinkAction, EvidenceRef
from mcp_memory.core.curation_routing import MaintenanceFamily
from mcp_memory.core.curation_validation import CurationSpecialistRoute
from mcp_memory.core.task_handlers.maintenance_work_items import (
    enqueue_specialist_route,
    enqueue_specialist_routes,
)
from mcp_memory.work_item_store import (
    EXECUTION_LANE_AGENTIC,
    SQLiteWorkItemRepository,
    WORK_FAMILY_GRAPH_LINK_REVIEW,
    WORK_FAMILY_OPERATOR_REVIEW,
)


pytestmark = pytest.mark.small


def _route(
    *,
    revision_token: str = "revision-1",
    family: MaintenanceFamily = MaintenanceFamily.GRAPH_LINKER,
    source_id=None,
    target_id=None,
    action_id=None,
    link_type: str = "DEPENDS_ON",
    context: str | None = None,
):
    source_id = source_id or uuid4()
    target_id = target_id or uuid4()
    action = CreateLinkAction(
        action_id=action_id or uuid4(),
        source_id=source_id,
        target_id=target_id,
        link_type=link_type,
        context=context,
        confidence=0.9,
        rationale="route the relationship to its owning specialist",
        evidence=[EvidenceRef(memory_id=source_id, revision_token=revision_token)],
        preconditions=ActionPreconditions(record_tokens={source_id: revision_token}),
    )
    return CurationSpecialistRoute(action=action, family=family)


def test_specialist_route_is_idempotent_until_target_evidence_changes(db_manager) -> None:
    work_items = SQLiteWorkItemRepository(db_manager)
    source_id = uuid4()
    target_id = uuid4()
    action_id = uuid4()
    first, created_first = enqueue_specialist_route(
        work_items, _route(source_id=source_id, target_id=target_id, action_id=action_id)
    )
    repeated, created_repeated = enqueue_specialist_route(
        work_items, _route(source_id=source_id, target_id=target_id, action_id=action_id)
    )
    changed, created_changed = enqueue_specialist_route(
        work_items,
        _route(revision_token="revision-2", source_id=source_id, target_id=target_id),
    )

    assert created_first is True
    assert created_repeated is False
    assert repeated.id == first.id
    assert created_changed is True
    assert changed.id != first.id
    assert first.family_key == WORK_FAMILY_GRAPH_LINK_REVIEW
    assert first.execution_lane == EXECUTION_LANE_AGENTIC
    assert first.payload["target_revision_token"] != changed.payload["target_revision_token"]
    assert len(work_items.list_items(family_key=WORK_FAMILY_GRAPH_LINK_REVIEW)) == 2


def test_distinct_create_link_intents_do_not_collide(db_manager) -> None:
    work_items = SQLiteWorkItemRepository(db_manager)
    source_id = uuid4()
    target_id = uuid4()
    action_id = uuid4()

    first, created_first = enqueue_specialist_route(
        work_items,
        _route(source_id=source_id, target_id=target_id, action_id=action_id),
    )
    repeated, created_repeated = enqueue_specialist_route(
        work_items,
        _route(source_id=source_id, target_id=target_id, action_id=action_id),
    )
    distinct, created_distinct = enqueue_specialist_route(
        work_items,
        _route(
            source_id=source_id,
            target_id=target_id,
            action_id=uuid4(),
            link_type="BLOCKS",
            context="same endpoints, different relationship",
        ),
    )

    assert created_first is True
    assert created_repeated is False
    assert repeated.id == first.id
    assert created_distinct is True
    assert distinct.id != first.id
    assert distinct.payload["action"]["link_type"] == "BLOCKS"


def test_existing_conflicting_route_fails_closed(db_manager) -> None:
    work_items = SQLiteWorkItemRepository(db_manager)
    route = _route(action_id=uuid4())
    existing, _ = enqueue_specialist_route(work_items, route)
    existing.payload["action"]["link_type"] = "UNRELATED"

    class ConflictingWorkItems:
        def enqueue_unique(self, **kwargs):
            return existing, False

    with pytest.raises(RuntimeError, match="specialist_route_idempotency_conflict"):
        enqueue_specialist_route(ConflictingWorkItems(), route)


def test_unsupported_specialist_family_creates_operator_review_work(db_manager) -> None:
    work_items = SQLiteWorkItemRepository(db_manager)
    records = enqueue_specialist_routes(
        work_items,
        [_route(family=MaintenanceFamily.SUMMARIZER)],
    )

    assert len(records) == 1
    assert records[0].family_key == WORK_FAMILY_OPERATOR_REVIEW
    assert records[0].payload["primary_family"] == MaintenanceFamily.SUMMARIZER.value
    assert records[0].payload["operator_review_required"] is True
