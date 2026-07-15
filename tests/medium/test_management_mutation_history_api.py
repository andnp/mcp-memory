from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4

import pytest

from mcp_memory.context import ApplicationContext
from mcp_memory.curation_store import CurationActionReceipt, CurationReceiptState, CurationRun, SQLiteCurationStore
from mcp_memory.daemon_dispatch import dispatch_management_request
from mcp_memory.management.service import ManagementService
from mcp_memory.mutation_history import (
    MutationActorKind,
    MutationEvent,
    MutationEventStatus,
    RecordRevision,
    RevisionRole,
)
from mcp_memory.mutation_history_store import SQLiteMutationHistoryStore


pytestmark = pytest.mark.medium


def _service(db_manager, mutation_history, curation) -> ManagementService:
    return ManagementService(
        ApplicationContext(
            db_manager=db_manager,
            memory_path=db_manager.db_path.parent,
            mutation_history=mutation_history,
            curation=curation,
        ).management_view(),
        SimpleNamespace(has_runtime=True, client_count=1),
    )


def _seed_event(runtime, *, created_at: datetime, with_receipt: bool = False) -> MutationEvent:
    event_id = uuid4()
    run_id = uuid4() if with_receipt else None
    action_id = uuid4() if with_receipt else None
    event = runtime.mutation_history.append_event(
        MutationEvent(
            id=event_id,
            operation="rewrite_memory",
            actor_kind=MutationActorKind.MAINTENANCE,
            family="curator",
            curation_run_id=run_id,
            action_id=action_id,
            rationale="provider prose must not become audit history",
            status=MutationEventStatus.APPLIED,
            created_at=created_at,
        )
    )
    memory_id = uuid4()
    runtime.mutation_history.append_record_revisions(
        [
            RecordRevision(
                event_id=event.id,
                memory_id=memory_id,
                role=RevisionRole.TARGET,
                before_exists=True,
                before_snapshot={"title": "before", "content": "x" * 20_000},
                after_exists=True,
                after_snapshot={"title": "after"},
                before_token="before-token",
                after_token="after-token",
            )
        ]
    )
    if run_id is not None and action_id is not None:
        runtime.curation.create_run(
            CurationRun(
                run_id=run_id,
                frontier_key="frontier",
                context_fingerprint="fingerprint",
            )
        )
        runtime.curation.put_receipt(
            CurationActionReceipt(
                run_id=run_id,
                action_id=action_id,
                operation=event.operation,
                affected_ids=[memory_id],
                status=CurationReceiptState.APPLIED_UNVERIFIED,
                before_token="before-token",
                after_token="after-token",
                mutation_event_id=event.id,
            )
        )
    return event


def test_management_history_is_paginated_filtered_and_receipt_linked(db_manager) -> None:
    runtime = SimpleNamespace(
        db_manager=db_manager,
        mutation_history=SQLiteMutationHistoryStore(db_manager),
        curation=SQLiteCurationStore(db_manager),
    )
    now = datetime.now(UTC)
    newest = _seed_event(runtime, created_at=now, with_receipt=True)
    _seed_event(runtime, created_at=now - timedelta(seconds=1))
    _seed_event(runtime, created_at=now - timedelta(seconds=2))
    service = _service(db_manager, runtime.mutation_history, runtime.curation)

    page = dispatch_management_request(
        SimpleNamespace(service=service),
        SimpleNamespace(),
        "/api/mutation-history",
        {"limit": 1, "family": "curator"},
    )

    assert page["has_more"] is True
    assert page["next_offset"] == 1
    assert page["events"][0]["id"] == str(newest.id)
    assert page["events"][0]["receipt"]["mutation_event_id"] == str(newest.id)


def test_management_history_detail_and_diff_use_bounded_authoritative_revisions(db_manager) -> None:
    runtime = SimpleNamespace(
        db_manager=db_manager,
        mutation_history=SQLiteMutationHistoryStore(db_manager),
        curation=SQLiteCurationStore(db_manager),
    )
    event = _seed_event(runtime, created_at=datetime.now(UTC), with_receipt=True)
    service = _service(db_manager, runtime.mutation_history, runtime.curation)

    detail = dispatch_management_request(
        SimpleNamespace(service=service),
        SimpleNamespace(),
        f"/api/mutation-history/{event.id}",
        {},
    )
    diff = dispatch_management_request(
        SimpleNamespace(service=service),
        SimpleNamespace(),
        f"/api/mutation-history/{event.id}/diff",
        {},
    )

    assert "rationale" not in detail["event"]
    assert detail["receipt"]["action_id"]
    assert detail["records"][0]["before_snapshot"]["truncated"] is True
    assert diff["records"][0]["before"]["sha256"]
    assert diff["records"][0]["after"]["title"] == "after"
