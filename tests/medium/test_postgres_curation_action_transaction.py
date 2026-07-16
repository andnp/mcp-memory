from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
import threading
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest

from mcp_memory.core.curation_identity import graph_token, record_token
from mcp_memory.curation_action_store import (
    CurationActionFatalError,
    CurationActionInjectedFailure,
    CurationActionStaleError,
    MutationResult,
)
from mcp_memory.curation_store import CurationRun, CurationRunState
from mcp_memory.mutation_history import Protection, ProtectionMode
from mcp_memory.storage.postgres_curation_action_store import PostgresCurationActionStore
from mcp_memory.storage.postgres_connection import PostgresConnectionManager
from mcp_memory.storage.postgres_curation_store import PostgresCurationStore
from mcp_memory.storage.postgres import ensure_postgres_schema
from mcp_memory.storage.postgres_mutation_history_store import PostgresMutationHistoryStore
from mcp_memory.storage.postgres_repository import PostgresRelationalMemoryRepository
from mcp_memory.storage.session import CursorLike


pytestmark = pytest.mark.medium


def _seed(postgres_storage_config):
    ensure_postgres_schema(postgres_storage_config)
    manager = PostgresConnectionManager(postgres_storage_config)
    repository = PostgresRelationalMemoryRepository(manager)
    first_id = str(uuid4())
    second_id = str(uuid4())
    first = repository.create_memory("First title", "First content", ["workspace"], memory_id=first_id)
    second = repository.create_memory("Second title", "Second content", ["workspace"], memory_id=second_id)
    assert first is not None and second is not None
    run = CurationRun(
        run_id=uuid4(),
        frontier_key="test-frontier",
        context_fingerprint="test-context",
        state=CurationRunState.EXECUTING,
    )
    PostgresCurationStore(manager).create_run(run)
    return manager, repository, run, first, second


@pytest.mark.parametrize(
    "fault_stage",
    ["domain_mutation", "projection_update", "repair_intent", "history", "receipt_preparation"],
)
def test_postgres_action_faults_roll_back_every_authoritative_write(postgres_storage_config, fault_stage: str) -> None:
    manager, repository, run, first, _second = _seed(postgres_storage_config)
    action_id = uuid4()

    def apply(transaction):
        transaction.update_memory(str(first.id), content="rolled back")
        return MutationResult("rewrite_memory", [first.id])

    with pytest.raises(CurationActionInjectedFailure):
        PostgresCurationActionStore(manager, fault_stage=fault_stage).execute_action(
            run_id=run.run_id,
            action_id=action_id,
            target_ids=[str(first.id)],
            expected_tokens={str(first.id): record_token(first)},
            apply=apply,
            operation="rewrite_memory",
        )

    unchanged = repository.get_memory(str(first.id))
    assert unchanged is not None
    assert unchanged.content == first.content
    with manager.open_connection() as connection:
        with connection.cursor() as cursor:
            for table in (
                "embedding_repair_queue",
                "memory_mutation_events",
                "memory_record_revisions",
                "memory_link_revisions",
                "curation_action_receipts",
            ):
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                row = cursor.fetchone()
                assert row is not None and row[0] == 0
    manager.close()


def test_postgres_action_replay_is_idempotent(postgres_storage_config) -> None:
    manager, repository, run, first, _second = _seed(postgres_storage_config)
    action_id = uuid4()
    calls = 0

    def apply(transaction):
        nonlocal calls
        calls += 1
        transaction.update_memory(str(first.id), content="once")
        return MutationResult("rewrite_memory", [first.id])

    store = PostgresCurationActionStore(manager)
    arguments = {
        "run_id": run.run_id,
        "action_id": action_id,
        "target_ids": [str(first.id)],
        "expected_tokens": {str(first.id): record_token(first)},
        "apply": apply,
        "operation": "rewrite_memory",
    }
    first_receipt = store.execute_action(**arguments)
    replayed_receipt = store.execute_action(**arguments)

    with pytest.raises(CurationActionFatalError, match="operation is required"):
        store.execute_action(**{key: value for key, value in arguments.items() if key != "operation"})

    assert replayed_receipt == first_receipt
    assert calls == 1
    with manager.open_connection() as connection:
        with connection.cursor() as cursor:
            for table in ("curation_action_receipts", "memory_mutation_events", "embedding_repair_queue"):
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                row = cursor.fetchone()
                assert row is not None and row[0] == 1
    manager.close()


def test_postgres_legacy_receipt_without_intent_hash_fails_closed(postgres_storage_config) -> None:
    manager, repository, run, first, _second = _seed(postgres_storage_config)
    action_id = uuid4()
    calls = 0

    def apply(transaction):
        nonlocal calls
        calls += 1
        transaction.update_memory(str(first.id), content="once")
        return MutationResult("rewrite_memory", [first.id])

    store = PostgresCurationActionStore(manager)
    arguments = {
        "run_id": run.run_id,
        "action_id": action_id,
        "target_ids": [str(first.id)],
        "expected_tokens": {str(first.id): record_token(first)},
        "apply": apply,
        "operation": "rewrite_memory",
    }
    receipt = store.execute_action(**arguments)
    with manager.open_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "ALTER TABLE curation_action_receipts ALTER COLUMN intent_hash DROP NOT NULL"
            )
            cursor.execute(
                "UPDATE curation_action_receipts SET intent_hash = NULL WHERE run_id = %s AND action_id = %s",
                (str(run.run_id), str(action_id)),
            )
        connection.commit()

    with pytest.raises(CurationActionFatalError, match="legacy-unverifiable"):
        store.execute_action(**arguments)

    assert calls == 1
    with manager.open_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT intent_hash FROM curation_action_receipts WHERE run_id = %s AND action_id = %s",
                (str(run.run_id), str(action_id)),
            )
            row = cursor.fetchone()
            assert row is not None and row[0] is None
    assert receipt.intent_hash is not None
    manager.close()


def test_postgres_action_identity_collision_rejects_mismatched_intent(postgres_storage_config) -> None:
    manager, repository, run, first, second = _seed(postgres_storage_config)
    action_id = uuid4()
    preconditions = {"required_statuses": {str(first.id): "active"}}
    calls = 0

    def apply(transaction):
        nonlocal calls
        calls += 1
        transaction.update_memory(str(first.id), content="once")
        return MutationResult("rewrite_memory", [first.id])

    store = PostgresCurationActionStore(manager)
    arguments = {
        "run_id": run.run_id,
        "action_id": action_id,
        "target_ids": [str(first.id)],
        "expected_tokens": {str(first.id): record_token(first)},
        "preconditions": preconditions,
        "operation": "rewrite_memory",
        "payload": {"content": "once"},
        "apply": apply,
    }
    receipt = store.execute_action(**arguments)
    assert receipt.intent_hash is not None

    mismatches = [
        {"operation": "archive_memory"},
        {"target_ids": [str(second.id)], "expected_tokens": {str(second.id): record_token(second)}},
        {"preconditions": {"required_statuses": {str(first.id): "archived"}}},
        {"payload": {"content": "twice"}},
    ]
    for mismatch in mismatches:
        with pytest.raises(CurationActionFatalError, match="identity collision"):
            store.execute_action(**{**arguments, **mismatch})

    assert calls == 1
    with manager.open_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM memory_mutation_events")
            row = cursor.fetchone()
            assert row is not None and row[0] == 1
            cursor.execute("SELECT COUNT(*) FROM curation_action_receipts")
            row = cursor.fetchone()
            assert row is not None and row[0] == 1
    manager.close()


def test_postgres_action_stale_graph_token_writes_nothing(postgres_storage_config) -> None:
    manager, repository, run, first, second = _seed(postgres_storage_config)
    old_graph_token = graph_token(str(first.id), [])
    repository.add_link(str(first.id), str(second.id), "supports")
    called = False

    def apply(transaction):
        nonlocal called
        called = True
        transaction.update_memory(str(first.id), content="must not be written")
        return MutationResult("rewrite_memory", [first.id])

    with pytest.raises(CurationActionStaleError):
        PostgresCurationActionStore(manager).execute_action(
            run_id=run.run_id,
            action_id=uuid4(),
            target_ids=[str(first.id)],
            expected_tokens={"graph:" + str(first.id): old_graph_token},
            apply=apply,
            operation="rewrite_memory",
        )

    assert not called
    unchanged = repository.get_memory(str(first.id))
    assert unchanged is not None and unchanged.content == first.content
    with manager.open_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM curation_action_receipts")
            row = cursor.fetchone()
            assert row is not None and row[0] == 0
    manager.close()


@pytest.mark.parametrize("active", [False, True])
def test_postgres_action_store_uses_active_protections_only(postgres_storage_config, active: bool) -> None:
    manager, repository, run, first, _second = _seed(postgres_storage_config)
    PostgresMutationHistoryStore(manager).set_protection(
        Protection(
            memory_id=UUID(str(first.id)),
            mode=ProtectionMode.NO_AUTONOMOUS_MUTATION,
            reason="protection expiry test",
            expires_at=datetime.now(UTC) + (timedelta(days=1) if active else -timedelta(days=1)),
        )
    )
    called = False

    def apply(transaction):
        nonlocal called
        called = True
        transaction.update_memory(str(first.id), content="changed")
        return MutationResult("rewrite_memory", [first.id])

    store = PostgresCurationActionStore(manager)
    if active:
        with pytest.raises(CurationActionFatalError, match="autonomous mutation is protected"):
            store.execute_action(
                run_id=run.run_id,
                action_id=uuid4(),
                target_ids=[str(first.id)],
                expected_tokens={str(first.id): record_token(first)},
                apply=apply,
                operation="rewrite_memory",
            )
    else:
        store.execute_action(
            run_id=run.run_id,
            action_id=uuid4(),
            target_ids=[str(first.id)],
            expected_tokens={str(first.id): record_token(first)},
            apply=apply,
            operation="rewrite_memory",
        )

    assert called is not active
    refreshed = repository.get_memory(str(first.id))
    assert refreshed is not None and refreshed.content == ("changed" if not active else first.content)
    with manager.open_connection() as connection:
        with connection.cursor() as cursor:
            expected_writes = 0 if active else 1
            for table in ("memory_mutation_events", "memory_record_revisions", "embedding_repair_queue", "curation_action_receipts"):
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                row = cursor.fetchone()
                assert row is not None and row[0] == expected_writes
    manager.close()


def test_postgres_action_reversed_concurrent_targets_have_no_deadlock(postgres_storage_config) -> None:
    manager, _repository, run, first, second = _seed(postgres_storage_config)
    manager.close()
    expected_tokens = {str(first.id): record_token(first), str(second.id): record_token(second)}
    start_barrier = threading.Barrier(2)

    def execute(target_ids: list[str], action_id):
        worker_manager = PostgresConnectionManager(postgres_storage_config)
        try:
            start_barrier.wait(timeout=10)

            def apply(transaction):
                transaction.update_memory(target_ids[0], content=f"winner-{action_id}")
                return MutationResult("rewrite_memory", target_ids)

            return PostgresCurationActionStore(worker_manager).execute_action(
                run_id=run.run_id,
                action_id=action_id,
                target_ids=target_ids,
                expected_tokens=expected_tokens,
                apply=apply,
                operation="rewrite_memory",
            )
        finally:
            worker_manager.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(execute, [str(first.id), str(second.id)], uuid4()),
            executor.submit(execute, [str(second.id), str(first.id)], uuid4()),
        ]
        outcomes = []
        for future in futures:
            try:
                outcomes.append(future.result(timeout=20))
            except CurationActionStaleError:
                outcomes.append(None)

    assert sum(outcome is not None for outcome in outcomes) == 1
    manager = PostgresConnectionManager(postgres_storage_config)
    try:
        with manager.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM memory_mutation_events")
                row = cursor.fetchone()
                assert row is not None and row[0] == 1
    finally:
        manager.close()


def test_postgres_protection_writer_waits_for_action_commit(postgres_storage_config) -> None:
    manager, repository, run, first, _second = _seed(postgres_storage_config)
    manager.close()
    action_manager = PostgresConnectionManager(postgres_storage_config)
    protection_manager = PostgresConnectionManager(postgres_storage_config)
    action_callback_started = threading.Event()
    protection_lock_attempted = threading.Event()
    order: list[str] = []
    order_lock = threading.Lock()

    def record_order(value: str) -> None:
        with order_lock:
            order.append(value)

    class SignalingActionStore(PostgresCurationActionStore):
        def _invalidate_derivative_caches(self, memory_ids: Sequence[str]) -> None:
            record_order("action_commit")

    class SignalingProtectionStore(PostgresMutationHistoryStore):
        def _lock_targets(self, cursor: CursorLike, target_ids: Sequence[str]) -> None:
            protection_lock_attempted.set()
            super()._lock_targets(cursor, target_ids)

        def set_protection(self, protection: Protection) -> Protection:
            stored = super().set_protection(protection)
            record_order("protection_commit")
            return stored

    def apply(transaction):
        action_callback_started.set()
        assert protection_lock_attempted.wait(timeout=10)
        transaction.update_memory(str(first.id), content="action committed")
        return MutationResult("rewrite_memory", [first.id])

    try:
        action_store = SignalingActionStore(action_manager)
        protection_store = SignalingProtectionStore(protection_manager)
        with ThreadPoolExecutor(max_workers=2) as executor:
            action_future = executor.submit(
                action_store.execute_action,
                run_id=run.run_id,
                action_id=uuid4(),
                target_ids=[str(first.id)],
                expected_tokens={str(first.id): record_token(first)},
                apply=apply,
                operation="rewrite_memory",
            )
            assert action_callback_started.wait(timeout=10)
            protection_future = executor.submit(
                protection_store.set_protection,
                Protection(
                    memory_id=UUID(str(first.id)),
                    mode=ProtectionMode.NO_AUTONOMOUS_MUTATION,
                    reason="concurrency test",
                ),
            )
            action_future.result(timeout=20)
            protection_future.result(timeout=20)

        assert order == ["action_commit", "protection_commit"]
        changed = repository.get_memory(str(first.id))
        assert changed is not None and changed.content == "action committed"
        assert len(protection_store.get_protections(UUID(str(first.id)))) == 1
    finally:
        action_manager.close()
        protection_manager.close()
