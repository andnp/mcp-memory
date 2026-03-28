from __future__ import annotations

import asyncio

import pytest

from mcp_memory.context import ApplicationContext
from mcp_memory.core.task_handlers import CURATOR_TASK_NAME, RECURRING_TASK_INTERVAL_SECONDS
from mcp_memory.core.task_worker import RuntimeTaskWorker
from mcp_memory.storage.postgres_provider_usage_store import PostgresProviderUsageRepository
from mcp_memory.storage.postgres import ensure_postgres_schema
from mcp_memory.storage.postgres_connection import PostgresConnectionManager
from mcp_memory.storage.postgres_embedding_repair_store import PostgresEmbeddingRepairQueue
from mcp_memory.storage.postgres_journal import PostgresSystem1Journal
from mcp_memory.storage.postgres_migrations import POSTGRES_SCHEMA_VERSION
from mcp_memory.storage.postgres_task_execution_store import PostgresTaskExecutionAttemptRepository
from mcp_memory.storage.postgres_task_queue import PostgresTaskQueue
from mcp_memory.storage.postgres_vector_store import PostgresVectorStore
from mcp_memory.storage.postgres_work_item_store import PostgresWorkItemRepository
from mcp_memory.work_item_store import EXECUTION_LANE_DETERMINISTIC


pytestmark = pytest.mark.medium


def test_postgres_integration_bootstraps_schema_and_exercises_runtime_primitives(postgres_storage_config) -> None:
    state = ensure_postgres_schema(postgres_storage_config)

    assert state.schema_metadata_present is True
    assert state.schema_version == POSTGRES_SCHEMA_VERSION

    with PostgresConnectionManager(postgres_storage_config) as manager:
        journal = PostgresSystem1Journal(manager)
        work_items = PostgresWorkItemRepository(manager)
        repair_queue = PostgresEmbeddingRepairQueue(manager)
        vector_store = PostgresVectorStore(manager)

        entry = journal.record("hello from postgres", workspace_id="workspace-a")
        pending = journal.get_pending(workspace_id="workspace-a")

        work_item, created = work_items.enqueue_unique(
            family_key="memory_tagging",
            execution_lane=EXECUTION_LANE_DETERMINISTIC,
            payload={"memory_id": "memory-1"},
            workspace_id="workspace-a",
            priority=10,
            available_at=5.0,
            idempotency_key="memory-tagging:memory-1",
        )
        claimed_items = work_items.claim_batch(
            family_key="memory_tagging",
            execution_lane=EXECUTION_LANE_DETERMINISTIC,
            lease_owner="worker-a",
            limit=10,
            workspace_id="workspace-a",
            now=10.0,
        )
        completed_item = work_items.complete_item(work_item.id, completed_at=20.0)

        repair_item, repair_created = repair_queue.enqueue_unique(
            memory_id="memory-1",
            workspace_id="workspace-a",
            model_name="mini-embed",
            memory_updated_at="2026-03-28T00:00:00+00:00",
            available_at=5.0,
        )
        claimed_repairs = repair_queue.claim_batch(
            lease_owner="repair-worker",
            limit=10,
            workspace_id="workspace-a",
            now=10.0,
        )
        completed_repair = repair_queue.complete_item(repair_item.id, completed_at=30.0)

        vector_store.upsert(
            source_kind="memory",
            source_id="memory-1",
            workspace_id="workspace-a",
            model_name="mini-embed",
            embedding=[1.0, 0.0],
        )
        vector_store.upsert(
            source_kind="memory",
            source_id="memory-2",
            workspace_id="workspace-a",
            model_name="mini-embed",
            embedding=[0.0, 1.0],
        )
        ranked = vector_store.search(
            source_kind="memory",
            model_name="mini-embed",
            query_embedding=[0.9, 0.1],
            workspace_id="workspace-a",
            limit=2,
        )

    assert entry.content == "hello from postgres"
    assert [item.content for item in pending] == ["hello from postgres"]
    assert created is True
    assert [item.id for item in claimed_items] == [work_item.id]
    assert completed_item.status == "completed"
    assert repair_created is True
    assert [item.id for item in claimed_repairs] == [repair_item.id]
    assert completed_repair.status == "completed"
    assert [memory_id for memory_id, _score in ranked] == ["memory-1", "memory-2"]


def test_postgres_integration_task_queue_lifecycle_and_task_runs(postgres_storage_config) -> None:
    ensure_postgres_schema(postgres_storage_config)

    with PostgresConnectionManager(postgres_storage_config) as manager:
        queue = PostgresTaskQueue(manager)

        task, created = queue.enqueue_unique(
            "ingest-system1",
            data={"memory_id": "memory-1"},
            workspace_id="workspace-a",
            priority=10,
            max_retries=2,
            available_at=5.0,
        )
        duplicate, duplicate_created = queue.enqueue_unique(
            "ingest-system1",
            data={"memory_id": "memory-2"},
            workspace_id="workspace-a",
            priority=99,
            max_retries=2,
            available_at=6.0,
        )
        claimed = queue.claim_next(now=10.0, workspace_id="workspace-a")
        retried = queue.fail(task.id, "temporary failure", retry_delay_seconds=5.0, failed_at=12.0, execution_epoch=1)
        reclaimed = queue.claim_next(now=17.0, workspace_id="workspace-a")
        completed = queue.complete(task.id, completed_at=20.0, run_result={"lines_compressed": 3}, execution_epoch=2)
        task_runs = queue.list_task_runs(task_id=task.id)
        summaries = queue.summarize_task_runs(["ingest-system1"], workspace_id="workspace-a")

        assert created is True
        assert duplicate_created is False
        assert duplicate.id == task.id
        assert claimed is not None
        assert claimed.execution_epoch == 1
        assert retried.status == "pending"
        assert reclaimed is not None
        assert reclaimed.execution_epoch == 2
        assert completed.status == "completed"
        assert [run.status for run in task_runs] == ["completed", "retry"]
        assert summaries[0].completed_runs == 1
        assert summaries[0].retry_runs == 1
        assert summaries[0].total_lines_compressed == 3


@pytest.mark.asyncio
async def test_postgres_integration_runtime_worker_completes_real_queue_task(postgres_storage_config) -> None:
    ensure_postgres_schema(postgres_storage_config)

    with PostgresConnectionManager(postgres_storage_config) as manager:
        queue = PostgresTaskQueue(manager)
        provider_usage = PostgresProviderUsageRepository(manager, workspace_id=None)
        seen: list[dict[str, str]] = []

        def handle_ingest(_ctx: ApplicationContext, task) -> None:
            seen.append(task.data)

        task = queue.enqueue(
            "ingest-system1",
            data={"memory_id": "123"},
            workspace_id="workspace-a",
            available_at=0.0,
            task_id="postgres-worker-task",
        )
        worker = RuntimeTaskWorker(
            ApplicationContext(
                db_manager=manager,
                task_queue=queue,
                provider_usage=provider_usage,
                workspace_id="workspace-a",
            ),
            handlers={"ingest-system1": handle_ingest},
            poll_interval_seconds=0.01,
            abandoned_recovery_interval_seconds=0.05,
        )

        await worker.start()
        try:
            for _ in range(100):
                if queue.get_task(task.id).status == "completed":
                    break
                await asyncio.sleep(0.01)
        finally:
            await worker.stop(0.1)

        completed = queue.get_task(task.id)
        task_runs = queue.list_task_runs(task_id=task.id)

        assert completed.status == "completed"
        assert seen == [{"memory_id": "123"}]
        assert [run.status for run in task_runs] == ["completed"]


@pytest.mark.asyncio
async def test_postgres_integration_runtime_worker_reconciles_running_conversation_for_recovered_dead_subprocess_task(
    postgres_storage_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ensure_postgres_schema(postgres_storage_config)

    with PostgresConnectionManager(postgres_storage_config) as manager:
        queue = PostgresTaskQueue(manager)
        provider_usage = PostgresProviderUsageRepository(manager, workspace_id="workspace-a")
        task = queue.enqueue(
            "ingest-system1",
            workspace_id="workspace-a",
            available_at=0.0,
            task_id="postgres-orphaned-provider-task",
        )
        assert queue.claim_next(now=1.0, workspace_id="workspace-a") is not None
        queue.set_running_process(task.id, subprocess_pid=9999, request_id="req-orphaned", updated_at=2.0)
        provider_usage.record_conversation(
            request_id="req-orphaned",
            attempt=1,
            task_name="ingest-system1",
            task_id=task.id,
            provider_key="gemini-cli",
            provider_name="Gemini CLI",
            model_name="gemini-3-flash-preview",
            subprocess_pid=9999,
            prompt_text="ingest pending thoughts",
            response_text="",
            parsed=None,
            status="running",
            error_text=None,
            started_at=1.0,
            completed_at=1.0,
        )

        worker = RuntimeTaskWorker(
            ApplicationContext(
                db_manager=manager,
                task_queue=queue,
                provider_usage=provider_usage,
                workspace_id="workspace-a",
            ),
            handlers={"ingest-system1": lambda context, queued_task: None},
            poll_interval_seconds=0.01,
            abandoned_task_stale_after_seconds=300.0,
        )

        monkeypatch.setattr("mcp_memory.storage.postgres_task_queue._is_process_alive", lambda pid: False)

        await worker._run_reconciliation_pass(now=1000.0, reason="periodic")  # noqa: SLF001

        failed = queue.get_task(task.id)
        conversation = provider_usage.get_conversation("req-orphaned")[0]

        assert failed.status == "failed"
        assert failed.last_error == "Provider subprocess 9999 exited unexpectedly"
        assert conversation.status == "error"
        assert conversation.task_id == task.id
        assert conversation.error_text == "Provider subprocess 9999 exited unexpectedly"
        assert conversation.reason_category is None
        assert conversation.reason_code is None


@pytest.mark.asyncio
async def test_postgres_integration_runtime_worker_uses_attempt_heartbeat_to_keep_running_task_alive(
    postgres_storage_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ensure_postgres_schema(postgres_storage_config)

    with PostgresConnectionManager(postgres_storage_config) as manager:
        queue = PostgresTaskQueue(manager)
        attempt_repository = PostgresTaskExecutionAttemptRepository(manager, workspace_id="workspace-a")
        task = queue.enqueue(
            "heartbeat-task",
            workspace_id="workspace-a",
            available_at=0.0,
            task_id="postgres-attempt-heartbeat-task",
        )
        claimed = queue.claim_next(now=1.0, workspace_id="workspace-a")
        assert claimed is not None

        attempt_repository.start_attempt(
            task_id=task.id,
            execution_epoch=claimed.execution_epoch,
            task_name=task.task_name,
            request_id="req-attempt-heartbeat",
            subprocess_pid=9999,
            provider_key="gemini-cli",
            provider_name="Gemini CLI",
            model_name="gemini-3-flash-preview",
            started_at=1.0,
        )
        attempt_repository.heartbeat_attempt(
            task_id=task.id,
            execution_epoch=claimed.execution_epoch,
            heartbeat_at=950.0,
            subprocess_pid=9999,
        )

        worker = RuntimeTaskWorker(
            ApplicationContext(
                db_manager=manager,
                task_queue=queue,
                task_execution_attempts=attempt_repository,
                workspace_id="workspace-a",
            ),
            handlers={"heartbeat-task": lambda context, queued_task: None},
        )

        monkeypatch.setattr("mcp_memory.core.tasks._is_process_alive", lambda pid: False)

        await worker._run_reconciliation_pass(now=1000.0, reason="periodic")  # noqa: SLF001

        running = queue.get_task(task.id)
        attempt = attempt_repository.get_attempt(task_id=task.id, execution_epoch=claimed.execution_epoch)

        assert running.status == "running"
        assert attempt.status == "running"
        assert attempt.last_heartbeat_at == pytest.approx(950.0)


@pytest.mark.asyncio
async def test_postgres_integration_runtime_worker_reconciles_attempt_for_recovered_dead_process(
    postgres_storage_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ensure_postgres_schema(postgres_storage_config)

    with PostgresConnectionManager(postgres_storage_config) as manager:
        queue = PostgresTaskQueue(manager)
        attempt_repository = PostgresTaskExecutionAttemptRepository(manager, workspace_id="workspace-a")
        task = queue.enqueue(
            "orphaned-attempt-task",
            workspace_id="workspace-a",
            available_at=0.0,
            task_id="postgres-orphaned-attempt-task",
        )
        claimed = queue.claim_next(now=1.0, workspace_id="workspace-a")
        assert claimed is not None

        attempt_repository.start_attempt(
            task_id=task.id,
            execution_epoch=claimed.execution_epoch,
            task_name=task.task_name,
            request_id="req-orphaned-attempt",
            subprocess_pid=9999,
            provider_key="gemini-cli",
            provider_name="Gemini CLI",
            model_name="gemini-3-flash-preview",
            started_at=1.0,
        )

        worker = RuntimeTaskWorker(
            ApplicationContext(
                db_manager=manager,
                task_queue=queue,
                task_execution_attempts=attempt_repository,
                workspace_id="workspace-a",
            ),
            handlers={"orphaned-attempt-task": lambda context, queued_task: None},
        )

        monkeypatch.setattr("mcp_memory.core.tasks._is_process_alive", lambda pid: False)

        await worker._run_reconciliation_pass(now=1000.0, reason="periodic")  # noqa: SLF001

        failed = queue.get_task(task.id)
        attempt = attempt_repository.get_attempt(task_id=task.id, execution_epoch=claimed.execution_epoch)

        assert failed.status == "failed"
        assert failed.last_error == "Provider subprocess 9999 exited unexpectedly"
        assert attempt.status == "error"
        assert attempt.completed_at == pytest.approx(1000.0)
        assert attempt.termination_reason == "task_failed"


@pytest.mark.asyncio
async def test_postgres_integration_runtime_worker_releases_leaked_work_items_and_embedding_repairs_for_terminal_task(
    postgres_storage_config,
) -> None:
    ensure_postgres_schema(postgres_storage_config)

    with PostgresConnectionManager(postgres_storage_config) as manager:
        queue = PostgresTaskQueue(manager)
        work_items = PostgresWorkItemRepository(manager)
        repair_queue = PostgresEmbeddingRepairQueue(manager)
        task = queue.enqueue(
            "memory-curator",
            workspace_id="workspace-a",
            available_at=0.0,
            task_id="postgres-terminal-cleanup-task",
        )
        claimed = queue.claim_next(now=1.0, workspace_id="workspace-a")
        assert claimed is not None

        work_item, created = work_items.enqueue_unique(
            family_key="memory_tagging",
            execution_lane=EXECUTION_LANE_DETERMINISTIC,
            payload={"memory_id": "memory-1"},
            workspace_id="workspace-a",
            priority=10,
            available_at=5.0,
            idempotency_key="memory-tagging:memory-1",
        )
        claimed_work_items = work_items.claim_batch(
            family_key="memory_tagging",
            execution_lane=EXECUTION_LANE_DETERMINISTIC,
            lease_owner=task.id,
            limit=10,
            workspace_id="workspace-a",
            now=10.0,
        )

        repair_item, repair_created = repair_queue.enqueue_unique(
            memory_id="memory-1",
            workspace_id="workspace-a",
            model_name="mini-embed",
            memory_updated_at="2026-03-28T00:00:00+00:00",
            available_at=5.0,
        )
        claimed_repairs = repair_queue.claim_batch(
            lease_owner=task.id,
            limit=10,
            workspace_id="workspace-a",
            now=10.0,
        )

        completed_task = queue.complete(task.id, completed_at=20.0, execution_epoch=claimed.execution_epoch)
        worker = RuntimeTaskWorker(
            ApplicationContext(
                db_manager=manager,
                task_queue=queue,
                work_items=work_items,
                embedding_repair_queue=repair_queue,
                workspace_id="workspace-a",
            ),
            handlers={"memory-curator": lambda context, queued_task: None},
        )

        worker._reconcile_terminal_task_state(completed_task)  # noqa: SLF001

        released_work_item = work_items.get_item(work_item.id)
        released_repair = repair_queue.get_item(repair_item.id)

        assert created is True
        assert repair_created is True
        assert [item.id for item in claimed_work_items] == [work_item.id]
        assert [item.id for item in claimed_repairs] == [repair_item.id]
        assert released_work_item.status == "pending"
        assert released_work_item.lease_owner is None
        assert released_repair.status == "pending"
        assert released_repair.lease_owner is None


@pytest.mark.asyncio
async def test_postgres_integration_runtime_worker_periodic_reconciliation_releases_orphaned_work_items_and_embedding_repairs(
    postgres_storage_config,
) -> None:
    ensure_postgres_schema(postgres_storage_config)

    with PostgresConnectionManager(postgres_storage_config) as manager:
        queue = PostgresTaskQueue(manager)
        work_items = PostgresWorkItemRepository(manager)
        repair_queue = PostgresEmbeddingRepairQueue(manager)

        work_item, created = work_items.enqueue_unique(
            family_key="memory_tagging",
            execution_lane=EXECUTION_LANE_DETERMINISTIC,
            payload={"memory_id": "memory-orphaned"},
            workspace_id="workspace-a",
            priority=10,
            available_at=5.0,
            idempotency_key="memory-tagging:memory-orphaned",
        )
        claimed_work_items = work_items.claim_batch(
            family_key="memory_tagging",
            execution_lane=EXECUTION_LANE_DETERMINISTIC,
            lease_owner="missing-task",
            limit=10,
            workspace_id="workspace-a",
            now=10.0,
        )

        repair_item, repair_created = repair_queue.enqueue_unique(
            memory_id="memory-orphaned",
            workspace_id="workspace-a",
            model_name="mini-embed",
            memory_updated_at="2026-03-29T00:00:00+00:00",
            available_at=5.0,
        )
        claimed_repairs = repair_queue.claim_batch(
            lease_owner="missing-task",
            limit=10,
            workspace_id="workspace-a",
            now=10.0,
        )

        worker = RuntimeTaskWorker(
            ApplicationContext(
                db_manager=manager,
                task_queue=queue,
                work_items=work_items,
                embedding_repair_queue=repair_queue,
                workspace_id="workspace-a",
            ),
            handlers={},
            abandoned_task_stale_after_seconds=300.0,
        )

        await worker._run_reconciliation_pass(now=20.0, reason="periodic")  # noqa: SLF001

        released_work_item = work_items.get_item(work_item.id)
        released_repair = repair_queue.get_item(repair_item.id)

        assert created is True
        assert repair_created is True
        assert [item.id for item in claimed_work_items] == [work_item.id]
        assert [item.id for item in claimed_repairs] == [repair_item.id]
        assert queue.list_tasks(status="running", workspace_id=None, limit=10) == []
        assert released_work_item.status == "pending"
        assert released_work_item.lease_owner is None
        assert released_repair.status == "pending"
        assert released_repair.lease_owner is None


@pytest.mark.asyncio
async def test_postgres_integration_runtime_worker_recurring_follow_up_applies_jitter(
    postgres_storage_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ensure_postgres_schema(postgres_storage_config)

    with PostgresConnectionManager(postgres_storage_config) as manager:
        queue = PostgresTaskQueue(manager)
        task = queue.enqueue(
            CURATOR_TASK_NAME,
            workspace_id=None,
            data={"workspace_id": None, "trigger": "recurring_schedule", "interval_seconds": RECURRING_TASK_INTERVAL_SECONDS[CURATOR_TASK_NAME]},
            available_at=0.0,
            task_id="postgres-curator-follow-up-jitter",
        )
        claimed = queue.claim_next(now=100.0)
        assert claimed is not None

        worker = RuntimeTaskWorker(
            ApplicationContext(db_manager=manager, task_queue=queue),
            handlers={CURATOR_TASK_NAME: lambda context, queued_task: None},
            poll_interval_seconds=0.01,
        )
        monkeypatch.setattr("mcp_memory.core.task_worker.compute_recurring_jitter_seconds", lambda interval_seconds: 18.0)

        completed_task = queue.complete(task.id, completed_at=140.0, run_result={"mutations": 1}, execution_epoch=claimed.execution_epoch)

        await worker._schedule_follow_up(claimed, completed_task)  # noqa: SLF001

        follow_up = queue.find_open_task(CURATOR_TASK_NAME, None)

        assert follow_up is not None
        assert follow_up.id != task.id
        assert follow_up.available_at == pytest.approx(140.0 + RECURRING_TASK_INTERVAL_SECONDS[CURATOR_TASK_NAME] + 18.0)
        assert follow_up.data["trigger"] == "recurring_follow_up"
        assert follow_up.data["jitter_seconds"] == pytest.approx(18.0)


@pytest.mark.asyncio
async def test_postgres_integration_runtime_worker_requests_shutdown_cancellation_for_running_tasks(
    postgres_storage_config,
) -> None:
    ensure_postgres_schema(postgres_storage_config)

    with PostgresConnectionManager(postgres_storage_config) as manager:
        queue = PostgresTaskQueue(manager)
        task = queue.enqueue(
            "shutdown-me",
            workspace_id="workspace-a",
            available_at=0.0,
            task_id="postgres-shutdown-me",
        )

        started = asyncio.Event()
        released = asyncio.Event()
        cancelled = asyncio.Event()

        async def blocking_handler(_ctx: ApplicationContext, queued_task) -> None:
            started.set()
            try:
                await released.wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        worker = RuntimeTaskWorker(
            ApplicationContext(db_manager=manager, task_queue=queue, workspace_id="workspace-a"),
            handlers={"shutdown-me": blocking_handler},
            poll_interval_seconds=0.01,
            abandoned_recovery_interval_seconds=30.0,
        )

        await worker.start()
        try:
            await asyncio.wait_for(started.wait(), timeout=1.0)
            await worker.stop(0.05)
        finally:
            released.set()

        task_after_stop = queue.get_task(task.id)

        assert cancelled.is_set()
        assert task_after_stop.status == "cancelled"
        assert task_after_stop.cancellation_reason == "daemon_shutdown"
        assert task_after_stop.cancelled_by == "daemon"