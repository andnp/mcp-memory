from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from mcp_memory.work_item_store import EXECUTION_LANE_DETERMINISTIC, WorkItemRecord


class WorkItemRepositoryLike(Protocol):
    def enqueue_unique(
        self,
        *,
        family_key: str,
        execution_lane: str,
        payload: dict[str, object] | None = None,
        workspace_id: str | None = None,
        priority: int = 100,
        available_at: float | None = None,
        idempotency_key: str | None = None,
    ) -> tuple[WorkItemRecord, bool]: ...

    def claim_batch(
        self,
        *,
        family_key: str,
        execution_lane: str,
        lease_owner: str,
        limit: int,
        workspace_id: str | None = None,
        now: float | None = None,
        lease_ttl_seconds: float = 1800.0,
    ) -> list[WorkItemRecord]: ...

    def heartbeat_item(
        self,
        item_id: str,
        *,
        lease_owner: str,
        lease_ttl_seconds: float = 1800.0,
        heartbeated_at: float | None = None,
    ) -> WorkItemRecord: ...

    def complete_item(
        self,
        item_id: str,
        *,
        completed_at: float | None = None,
    ) -> WorkItemRecord: ...

    def defer_item(
        self,
        item_id: str,
        *,
        error: str,
        retry_delay_seconds: float,
        deferred_at: float | None = None,
    ) -> WorkItemRecord: ...

    def release_item(
        self,
        item_id: str,
        *,
        released_at: float | None = None,
    ) -> WorkItemRecord: ...


RepositoryFactory = Callable[[], WorkItemRepositoryLike]


def assert_enqueue_unique_deduplicates_idempotency_keys(make_repository: RepositoryFactory) -> None:
    repository = make_repository()

    first, created = repository.enqueue_unique(
        family_key="memory_tagging",
        execution_lane=EXECUTION_LANE_DETERMINISTIC,
        payload={"memory_id": "memory-dedupe-original"},
        workspace_id="workspace-work-item-dedupe",
        priority=10,
        available_at=5.0,
        idempotency_key="work-item-dedupe-key",
    )
    duplicate, duplicate_created = repository.enqueue_unique(
        family_key="memory_tagging",
        execution_lane=EXECUTION_LANE_DETERMINISTIC,
        payload={"memory_id": "memory-dedupe-duplicate"},
        workspace_id="workspace-work-item-dedupe",
        priority=99,
        available_at=6.0,
        idempotency_key="work-item-dedupe-key",
    )

    assert created is True
    assert duplicate_created is False
    assert duplicate.id == first.id
    assert duplicate.payload == {"memory_id": "memory-dedupe-original"}
    assert duplicate.priority == 10
    assert duplicate.available_at == 5.0
    assert duplicate.status == "pending"
    assert duplicate.attempt_count == 0


def assert_claim_batch_orders_ready_items(make_repository: RepositoryFactory) -> None:
    repository = make_repository()

    later, _ = repository.enqueue_unique(
        family_key="memory_tagging",
        execution_lane=EXECUTION_LANE_DETERMINISTIC,
        payload={"memory_id": "memory-claim-later"},
        workspace_id="workspace-work-item-claim-order",
        priority=20,
        available_at=5.0,
        idempotency_key="work-item-order-later",
    )
    earlier, _ = repository.enqueue_unique(
        family_key="memory_tagging",
        execution_lane=EXECUTION_LANE_DETERMINISTIC,
        payload={"memory_id": "memory-claim-earlier"},
        workspace_id="workspace-work-item-claim-order",
        priority=10,
        available_at=5.0,
        idempotency_key="work-item-order-earlier",
    )
    repository.enqueue_unique(
        family_key="memory_tagging",
        execution_lane=EXECUTION_LANE_DETERMINISTIC,
        payload={"memory_id": "memory-claim-delayed"},
        workspace_id="workspace-work-item-claim-order",
        priority=1,
        available_at=50.0,
        idempotency_key="work-item-order-delayed",
    )

    claimed = repository.claim_batch(
        family_key="memory_tagging",
        execution_lane=EXECUTION_LANE_DETERMINISTIC,
        lease_owner="worker-order",
        limit=10,
        workspace_id="workspace-work-item-claim-order",
        now=10.0,
        lease_ttl_seconds=5.0,
    )

    assert [item.id for item in claimed] == [earlier.id, later.id]
    assert [item.status for item in claimed] == ["running", "running"]
    assert [item.attempt_count for item in claimed] == [1, 1]
    assert [item.lease_owner for item in claimed] == ["worker-order", "worker-order"]
    assert [item.claimed_at for item in claimed] == [10.0, 10.0]
    assert [item.lease_expires_at for item in claimed] == [15.0, 15.0]


def assert_heartbeat_extends_leases_and_allows_expired_reclaim(make_repository: RepositoryFactory) -> None:
    repository = make_repository()

    item, created = repository.enqueue_unique(
        family_key="memory_tagging",
        execution_lane=EXECUTION_LANE_DETERMINISTIC,
        payload={"memory_id": "memory-heartbeat"},
        workspace_id="workspace-work-item-heartbeat",
        priority=10,
        available_at=5.0,
        idempotency_key="work-item-heartbeat",
    )
    assert created is True

    claimed = repository.claim_batch(
        family_key="memory_tagging",
        execution_lane=EXECUTION_LANE_DETERMINISTIC,
        lease_owner="worker-heartbeat-a",
        limit=1,
        workspace_id="workspace-work-item-heartbeat",
        now=10.0,
        lease_ttl_seconds=10.0,
    )

    assert [record.id for record in claimed] == [item.id]
    first_expiry = claimed[0].lease_expires_at
    assert first_expiry is not None
    assert first_expiry == 20.0

    heartbeated = repository.heartbeat_item(
        item.id,
        lease_owner="worker-heartbeat-a",
        lease_ttl_seconds=25.0,
        heartbeated_at=12.0,
    )
    assert heartbeated.lease_expires_at is not None
    assert heartbeated.updated_at == 12.0
    assert heartbeated.lease_expires_at == 37.0
    assert heartbeated.lease_expires_at > first_expiry

    still_leased = repository.claim_batch(
        family_key="memory_tagging",
        execution_lane=EXECUTION_LANE_DETERMINISTIC,
        lease_owner="worker-heartbeat-b",
        limit=1,
        workspace_id="workspace-work-item-heartbeat",
        now=21.0,
    )
    reclaimed = repository.claim_batch(
        family_key="memory_tagging",
        execution_lane=EXECUTION_LANE_DETERMINISTIC,
        lease_owner="worker-heartbeat-b",
        limit=1,
        workspace_id="workspace-work-item-heartbeat",
        now=38.0,
        lease_ttl_seconds=7.0,
    )

    assert still_leased == []
    assert [record.id for record in reclaimed] == [item.id]
    assert reclaimed[0].attempt_count == 2
    assert reclaimed[0].lease_owner == "worker-heartbeat-b"
    assert reclaimed[0].claimed_at == 38.0
    assert reclaimed[0].lease_expires_at == 45.0


def assert_release_defer_and_complete_items(make_repository: RepositoryFactory) -> None:
    repository = make_repository()

    releasable, _ = repository.enqueue_unique(
        family_key="memory_tagging",
        execution_lane=EXECUTION_LANE_DETERMINISTIC,
        payload={"memory_id": "memory-release"},
        workspace_id="workspace-work-item-lifecycle",
        priority=10,
        available_at=5.0,
        idempotency_key="work-item-release",
    )
    deferrable, _ = repository.enqueue_unique(
        family_key="memory_tagging",
        execution_lane=EXECUTION_LANE_DETERMINISTIC,
        payload={"memory_id": "memory-defer"},
        workspace_id="workspace-work-item-lifecycle",
        priority=20,
        available_at=5.0,
        idempotency_key="work-item-defer",
    )
    completable, _ = repository.enqueue_unique(
        family_key="memory_tagging",
        execution_lane=EXECUTION_LANE_DETERMINISTIC,
        payload={"memory_id": "memory-complete"},
        workspace_id="workspace-work-item-lifecycle",
        priority=30,
        available_at=5.0,
        idempotency_key="work-item-complete",
    )

    claimed = repository.claim_batch(
        family_key="memory_tagging",
        execution_lane=EXECUTION_LANE_DETERMINISTIC,
        lease_owner="worker-lifecycle-a",
        limit=3,
        workspace_id="workspace-work-item-lifecycle",
        now=10.0,
    )

    assert [item.id for item in claimed] == [releasable.id, deferrable.id, completable.id]

    released = repository.release_item(releasable.id, released_at=11.0)
    deferred = repository.defer_item(
        deferrable.id,
        error="provider_rate_limited",
        retry_delay_seconds=30.0,
        deferred_at=12.0,
    )
    completed = repository.complete_item(completable.id, completed_at=13.0)

    assert released.status == "pending"
    assert released.lease_owner is None
    assert released.lease_expires_at is None
    assert released.completed_at is None

    assert deferred.status == "deferred"
    assert deferred.available_at == 42.0
    assert deferred.lease_owner is None
    assert deferred.lease_expires_at is None
    assert deferred.last_error == "provider_rate_limited"

    assert completed.status == "completed"
    assert completed.completed_at == 13.0
    assert completed.lease_owner is None
    assert completed.lease_expires_at is None
    assert completed.last_error is None

    reclaimed_pending = repository.claim_batch(
        family_key="memory_tagging",
        execution_lane=EXECUTION_LANE_DETERMINISTIC,
        lease_owner="worker-lifecycle-b",
        limit=10,
        workspace_id="workspace-work-item-lifecycle",
        now=20.0,
    )
    reclaimed_deferred = repository.claim_batch(
        family_key="memory_tagging",
        execution_lane=EXECUTION_LANE_DETERMINISTIC,
        lease_owner="worker-lifecycle-c",
        limit=10,
        workspace_id="workspace-work-item-lifecycle",
        now=43.0,
    )

    assert [item.id for item in reclaimed_pending] == [releasable.id]
    assert reclaimed_pending[0].attempt_count == 2
    assert reclaimed_pending[0].lease_owner == "worker-lifecycle-b"
    assert [item.id for item in reclaimed_deferred] == [deferrable.id]
    assert reclaimed_deferred[0].attempt_count == 2
    assert reclaimed_deferred[0].lease_owner == "worker-lifecycle-c"