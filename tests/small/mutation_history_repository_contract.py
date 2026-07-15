from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Protocol
from uuid import UUID, uuid4

import pytest

from mcp_memory.mutation_history import (
    LinkRevision,
    MutationActorKind,
    MutationEvent,
    MutationEventStatus,
    Protection,
    ProtectionMode,
    RecordRevision,
    RestoreRequest,
    RestoreResult,
    RestoreResultStatus,
    RevisionRole,
)
from mcp_memory.mutation_history_store import (
    MAX_HISTORY_READ_LIMIT,
    MutationHistoryIdentityConflictError,
    MutationHistoryTerminalizationError,
)


MEMORY_ID = UUID("00000000-0000-0000-0000-000000000001")
OTHER_MEMORY_ID = UUID("00000000-0000-0000-0000-000000000002")


class MutationHistoryRepositoryLike(Protocol):
    def append_event(self, event: MutationEvent) -> MutationEvent: ...

    def append_record_revisions(self, revisions: list[RecordRevision]) -> None: ...

    def append_link_revisions(self, revisions: list[LinkRevision]) -> None: ...

    def get_event(self, event_id: UUID) -> MutationEvent | None: ...

    def get_event_by_action(self, run_id: UUID, action_id: UUID) -> MutationEvent | None: ...

    def list_events(self, *, memory_id: UUID | None = None, limit: int = 100) -> list[MutationEvent]: ...

    def get_record_revisions(self, event_id: UUID) -> list[RecordRevision]: ...

    def get_link_revisions(self, event_id: UUID) -> list[LinkRevision]: ...

    def get_protections(self, memory_id: UUID) -> list[Protection]: ...

    def set_protection(self, protection: Protection) -> Protection: ...

    def remove_protection(self, memory_id: UUID, mode: ProtectionMode) -> None: ...

    def terminalize_event(
        self,
        event_id: UUID,
        status: MutationEventStatus,
        *,
        terminalized_at: datetime | None = None,
    ) -> MutationEvent: ...

    def request_restore(self, request: RestoreRequest) -> RestoreResult: ...

    def terminalize_restore_request(
        self,
        request_id: UUID,
        result: RestoreResult,
        *,
        terminalized_at: datetime | None = None,
    ) -> RestoreResult: ...


RepositoryFactory = Callable[[], MutationHistoryRepositoryLike]


def _event(
    *,
    created_at: datetime,
    run_id: UUID | None = None,
    action_id: UUID | None = None,
    idempotency_key: str | None = None,
) -> MutationEvent:
    return MutationEvent(
        id=uuid4(),
        operation="normalize_memory",
        actor_kind=MutationActorKind.MAINTENANCE,
        curation_run_id=run_id,
        action_id=action_id,
        idempotency_key=idempotency_key,
        status=MutationEventStatus.APPLIED,
        created_at=created_at,
    )


def assert_mutation_history_repository_contract(make_repository: RepositoryFactory) -> None:
    """Assert the backend-neutral mutation-history and protection contract."""
    repository = make_repository()
    run_id = uuid4()
    action_id = uuid4()
    event = _event(
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        run_id=run_id,
        action_id=action_id,
        idempotency_key="contract-action-1",
    )
    record_revision = RecordRevision(
        event_id=event.id,
        memory_id=MEMORY_ID,
        role=RevisionRole.TARGET,
        before_exists=True,
        before_snapshot={"content": "before"},
        after_exists=True,
        after_snapshot={"content": "after"},
        before_token="before-token",
        after_token="after-token",
    )
    link_revision = LinkRevision(
        event_id=event.id,
        source_id=MEMORY_ID,
        target_id=OTHER_MEMORY_ID,
        link_type="supports",
        context="contract context",
        before_exists=False,
        after_exists=True,
    )

    # The event and its complete before/after revision set are the durable
    # history unit consumed by later action transactions.
    persisted = repository.append_event(event)
    repository.append_record_revisions([record_revision])
    repository.append_link_revisions([link_revision])
    assert persisted == event
    assert repository.get_event(event.id) == event
    assert repository.get_event_by_action(run_id, action_id) == event
    assert repository.get_record_revisions(event.id) == [record_revision]
    assert repository.get_link_revisions(event.id) == [link_revision]

    # Replaying an action with a regenerated event UUID returns the original
    # event; changing the action payload is an identity conflict.
    assert repository.append_event(event.model_copy(update={"id": uuid4()})) == event
    with pytest.raises(MutationHistoryIdentityConflictError):
        repository.append_event(event.model_copy(update={"id": uuid4(), "operation": "create_link"}))

    # History reads are filtered by affected memory and always bounded.
    link_only_event = _event(created_at=datetime(2026, 1, 2, tzinfo=UTC))
    repository.append_event(link_only_event)
    repository.append_link_revisions(
        [
            LinkRevision(
                event_id=link_only_event.id,
                source_id=OTHER_MEMORY_ID,
                target_id=MEMORY_ID,
                link_type="references",
                before_exists=False,
                after_exists=True,
            )
        ]
    )
    assert repository.list_events(memory_id=MEMORY_ID, limit=10) == [link_only_event, event]

    for index in range(MAX_HISTORY_READ_LIMIT + 1):
        repository.append_event(
            _event(
                created_at=datetime(2026, 2, 1, tzinfo=UTC) + timedelta(seconds=index),
            )
        )
    assert len(repository.list_events(limit=MAX_HISTORY_READ_LIMIT + 1)) <= MAX_HISTORY_READ_LIMIT
    with pytest.raises(ValueError):
        repository.list_events(limit=0)

    # First terminalization wins, including for restore requests.
    terminal_event = repository.append_event(_event(created_at=datetime(2026, 1, 3, tzinfo=UTC)))
    terminalized_at = datetime(2026, 1, 3, 1, tzinfo=UTC)
    terminal = repository.terminalize_event(
        terminal_event.id,
        MutationEventStatus.FAILED,
        terminalized_at=terminalized_at,
    )
    assert terminal.status is MutationEventStatus.FAILED
    assert terminal.terminalized_at == terminalized_at
    assert repository.terminalize_event(
        terminal_event.id,
        MutationEventStatus.FAILED,
        terminalized_at=terminalized_at + timedelta(hours=1),
    ) == terminal
    with pytest.raises(MutationHistoryTerminalizationError):
        repository.terminalize_event(terminal_event.id, MutationEventStatus.STALE)

    protection = repository.set_protection(
        Protection(
            memory_id=MEMORY_ID,
            mode=ProtectionMode.PINNED_ACTIVE,
            reason="contract protection",
            actor_id="contract",
        )
    )
    updated_protection = repository.set_protection(
        protection.model_copy(update={"reason": "updated contract protection"})
    )
    assert repository.get_protections(MEMORY_ID) == [updated_protection]
    repository.remove_protection(MEMORY_ID, ProtectionMode.PINNED_ACTIVE)
    assert repository.get_protections(MEMORY_ID) == []

    restore_request = RestoreRequest(
        target_event_id=event.id,
        reason="contract restore",
        idempotency_key="contract-restore-1",
        expected_record_tokens={MEMORY_ID: "after-token"},
        expected_link_tokens={"contract-link": "after-token"},
        confirmation=True,
    )
    restore = repository.request_restore(restore_request)
    replay = repository.request_restore(restore_request)
    assert restore.status is RestoreResultStatus.APPLIED
    assert replay.status is RestoreResultStatus.ALREADY_APPLIED
    assert replay.request_id == restore.request_id
    with pytest.raises(MutationHistoryIdentityConflictError):
        repository.request_restore(restore_request.model_copy(update={"reason": "different restore"}))

    assert restore.request_id is not None
    restore_terminal = repository.terminalize_restore_request(
        restore.request_id,
        RestoreResult(
            status=RestoreResultStatus.CONFLICT,
            target_event_id=event.id,
            conflict_reason="changed",
        ),
    )
    assert restore_terminal.status is RestoreResultStatus.CONFLICT
    assert repository.terminalize_restore_request(
        restore.request_id,
        RestoreResult(
            status=RestoreResultStatus.CONFLICT,
            target_event_id=event.id,
            conflict_reason="changed",
        ),
    ) == restore_terminal
    with pytest.raises(MutationHistoryTerminalizationError):
        repository.terminalize_restore_request(
            restore.request_id,
            RestoreResult(status=RestoreResultStatus.APPLIED, target_event_id=event.id),
        )
