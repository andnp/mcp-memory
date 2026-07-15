from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest

from mcp_memory.context import ApplicationContext
from mcp_memory.core.curation_executor import CurationExecutor
from mcp_memory.core.curation_identity import record_token
from mcp_memory.core.curation_models import ActionPreconditions, NormalizeMemoryAction
from mcp_memory.curation_action_store import SQLiteCurationActionStore
from mcp_memory.curation_store import CurationRun, CurationRunState, SQLiteCurationStore
from mcp_memory.daemon_dispatch import dispatch_management_request
from mcp_memory.management.service import ManagementService
from mcp_memory.mutation_history_store import SQLiteMutationHistoryStore
from mcp_memory.relational.repository import RelationalMemoryRepository


pytestmark = pytest.mark.medium


def _service(db_manager, repository, history, curation) -> ManagementService:
    return ManagementService(
        ApplicationContext(
            db_manager=db_manager,
            memory_path=db_manager.db_path.parent,
            repository=repository,
            mutation_history=history,
            curation=curation,
            curation_action_store=SQLiteCurationActionStore(db_manager),
        ).management_view(),
        SimpleNamespace(has_runtime=True, client_count=1),
    )


def _seed_normalization(db_manager):
    repository = RelationalMemoryRepository(db_manager)
    memory_id = uuid4()
    repository.create_memory(
        "Original title",
        "Stable content",
        ["workspace"],
        memory_id=str(memory_id),
        summary="Original summary",
        memory_type="observation",
    )
    before = repository.get_memory(str(memory_id))
    assert before is not None
    run = CurationRun(
        run_id=uuid4(),
        frontier_key="management-restore",
        context_fingerprint="management-context",
        state=CurationRunState.EXECUTING,
    )
    curation = SQLiteCurationStore(db_manager)
    curation.create_run(run)
    action = NormalizeMemoryAction(
        action_id=uuid4(),
        target_id=memory_id,
        confidence=1,
        rationale="specific title normalization",
        title="Normalized title",
        preconditions=ActionPreconditions(record_tokens={memory_id: record_token(before)}),
    )
    receipt = CurationExecutor(SQLiteCurationActionStore(db_manager)).execute_normalize(
        action,
        run_id=run.run_id,
        memory_type=before.type,
    )
    changed = repository.get_memory(str(memory_id))
    assert changed is not None and receipt.mutation_event_id is not None
    history = SQLiteMutationHistoryStore(db_manager)
    return repository, history, curation, memory_id, receipt.mutation_event_id, changed


def test_management_protection_api_crud_uses_typed_store_outcomes(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    memory_id = uuid4()
    repository.create_memory("Title", "Content", ["workspace"], memory_id=str(memory_id))
    history = SQLiteMutationHistoryStore(db_manager)
    service = _service(db_manager, repository, history, SQLiteCurationStore(db_manager))

    applied = dispatch_management_request(
        SimpleNamespace(service=service),
        SimpleNamespace(),
        "/api/admin/protections",
        {
            "memory_id": str(memory_id),
            "mode": "manual_review_required",
            "reason": "operator review",
            "actor_id": "operator",
        },
    )
    listed = dispatch_management_request(
        SimpleNamespace(service=service),
        SimpleNamespace(),
        f"/api/memories/{memory_id}/protections",
        {},
    )
    removed = dispatch_management_request(
        SimpleNamespace(service=service),
        SimpleNamespace(),
        "/api/admin/protections/delete",
        {"memory_id": str(memory_id), "mode": "manual_review_required"},
    )

    assert applied["status"] == "applied"
    assert listed["protections"][0]["mode"] == "manual_review_required"
    assert removed == {
        "status": "removed",
        "memory_id": str(memory_id),
        "mode": "manual_review_required",
        "protection": None,
    }


def test_management_restore_api_eligibility_and_request_are_idempotent(db_manager) -> None:
    repository, history, curation, memory_id, event_id, changed = _seed_normalization(db_manager)
    service = _service(db_manager, repository, history, curation)

    eligibility = dispatch_management_request(
        SimpleNamespace(service=service),
        SimpleNamespace(),
        f"/api/mutation-history/{event_id}/restore-eligibility",
        {},
    )
    request_payload = {
        "expected_record_tokens": {str(memory_id): record_token(changed)},
        "reason": "restore the accepted normalization",
        "idempotency_key": "management-restore-1",
    }
    restored = dispatch_management_request(
        SimpleNamespace(service=service),
        SimpleNamespace(),
        f"/api/mutation-history/{event_id}/restore",
        request_payload,
    )
    replay = dispatch_management_request(
        SimpleNamespace(service=service),
        SimpleNamespace(),
        f"/api/mutation-history/{event_id}/restore",
        request_payload,
    )

    assert eligibility["eligible"] is True
    assert eligibility["current_record_tokens"][str(memory_id)] == record_token(changed)
    assert restored["status"] == "applied"
    assert replay["status"] == "already_applied"
    assert repository.get_memory(str(memory_id)).title == "Original title"  # type: ignore[union-attr]


def test_management_restore_api_returns_current_token_conflict_and_requires_confirmation(db_manager) -> None:
    repository, history, curation, memory_id, event_id, changed = _seed_normalization(db_manager)
    repository.update_memory(str(memory_id), title="Later human edit")
    service = _service(db_manager, repository, history, curation)

    eligibility = service.get_restore_eligibility(str(event_id))
    conflict = service.request_restore(
        event_id=str(event_id),
        expected_record_tokens={str(memory_id): record_token(repository.get_memory(str(memory_id)))},  # type: ignore[arg-type]
        reason="do not overwrite later edit",
        idempotency_key="management-restore-conflict",
    )
    assert eligibility.conflict_code == "stale_state"
    assert conflict.status == "conflict"
    assert conflict.conflict_details["code"] == "stale_state"

    # A separate fresh mutation demonstrates that manual-review protection gates
    # the request, while explicit confirmation satisfies that policy gate.
    repository.update_memory(str(memory_id), title=changed.title)
    fresh = _service(db_manager, repository, history, curation)
    fresh.set_protection(memory_id=str(memory_id), mode="manual_review_required", reason="review first")
    eligibility = fresh.get_restore_eligibility(str(event_id))
    assert eligibility.requires_confirmation is True
    assert eligibility.conflict_code == "confirmation_required"
    current_token = record_token(repository.get_memory(str(memory_id)))  # type: ignore[arg-type]
    blocked = fresh.request_restore(
        event_id=str(event_id),
        expected_record_tokens={str(memory_id): current_token},
        reason="review before restoring",
        idempotency_key="management-restore-confirmation-required",
    )
    confirmed = fresh.request_restore(
        event_id=str(event_id),
        expected_record_tokens={str(memory_id): current_token},
        reason="operator confirmed restore",
        idempotency_key="management-restore-confirmed",
        confirmation=True,
    )
    assert blocked.status == "rejected"
    assert blocked.conflict_details["code"] == "confirmation_required"
    assert confirmed.status == "applied"
