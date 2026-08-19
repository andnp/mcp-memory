from __future__ import annotations

import asyncio
import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

from mcp_memory.context import ApplicationContext
from mcp_memory.core.agent_runtime import (
    FACT_CHECKER_TASK_NAME,
    PROJECT_MANAGER_TASK_NAME,
    SWEEPER_TASK_NAME,
    handle_fact_checker_task,
    handle_project_manager_task,
    handle_sweeper_task,
)
from mcp_memory.core.direct_mutation_evidence import DirectMutationEntityDelta, DirectMutationEvidence
from mcp_memory.core.ports.work_items import EXECUTION_LANE_DETERMINISTIC
from mcp_memory.core.task_handlers import (
    CURATOR_TASK_NAME,
    DEDUPLICATOR_TASK_NAME,
    RECURRING_TASK_INTERVAL_SECONDS,
)
from mcp_memory.core.task_worker import RuntimeTaskWorker
from mcp_memory.core.tasks import TaskRecord
from mcp_memory.daemon import create_daemon_app
from mcp_memory.daemon_ports import ManagementControllerPort
from mcp_memory.daemon_transport import request_daemon_json
from mcp_memory.hook_reminders import REMINDER_MESSAGE
from mcp_memory.management.service import ManagementService
from mcp_memory.mcp.handlers import call_memory_tool
from mcp_memory.mcp.runtime import create_runtime
from mcp_memory.storage.postgres import ensure_postgres_schema
from mcp_memory.storage.postgres_connection import PostgresConnectionManager
from mcp_memory.storage.postgres_direct_mutation_evidence_store import PostgresDirectMutationEvidenceStore
from mcp_memory.storage.postgres_embedding_repair_store import PostgresEmbeddingRepairQueue
from mcp_memory.storage.postgres_journal import PostgresSystem1Journal
from mcp_memory.storage.postgres_migrations import POSTGRES_SCHEMA_VERSION
from mcp_memory.storage.postgres_provider_usage_store import PostgresProviderUsageRepository
from mcp_memory.storage.postgres_repository import PostgresRelationalMemoryRepository
from mcp_memory.storage.postgres_task_execution_store import PostgresTaskExecutionAttemptRepository
from mcp_memory.storage.postgres_task_queue import PostgresTaskQueue
from mcp_memory.storage.postgres_vector_store import PostgresVectorStore
from mcp_memory.storage.postgres_work_item_store import PostgresWorkItemRepository
from tests.small.work_item_repository_contract import (
    assert_claim_batch_orders_ready_items,
    assert_enqueue_unique_deduplicates_idempotency_keys,
    assert_heartbeat_extends_leases_and_allows_expired_reclaim,
    assert_release_defer_and_complete_items,
)

pytestmark = pytest.mark.medium


def _write_postgres_test_config(tmp_path: Path, *, dsn: str) -> Path:
    home = tmp_path / "home"
    config_dir = home / ".config" / "mcp-memory"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.toml"
    config_path.write_text(
        "[ai]\n"
        'provider = "none"\n'
        '\n'
        "[storage]\n"
        'backend = "postgres"\n'
        '\n'
        "[storage.postgres]\n"
        f'dsn = "{dsn}"\n'
        "pool_min = 1\n"
        "pool_max = 4\n"
        "statement_timeout_ms = 30000\n"
        "lock_timeout_ms = 5000\n"
        'application_name = "mcp-memory-test"\n',
        encoding="utf-8",
    )
    return home


def _configure_postgres_runtime_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, dsn: str) -> Path:
    home = _write_postgres_test_config(tmp_path, dsn=dsn)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    return tmp_path / "workspace"


def _running_task(task_name: str, *, workspace_id: str | None, data: dict[str, object]) -> TaskRecord:
    return TaskRecord(
        id=f"{task_name}-test",
        task_name=task_name,
        data=data,
        workspace_id=workspace_id,
        status="running",
        priority=100,
        retries_count=0,
        max_retries=3,
        created_at=0.0,
        updated_at=0.0,
        available_at=0.0,
        claimed_at=0.0,
        started_at=0.0,
        completed_at=None,
        last_error=None,
    )


def test_postgres_integration_bootstraps_schema_and_exercises_runtime_primitives(postgres_storage_config) -> None:
    state = ensure_postgres_schema(postgres_storage_config)
    entry = None
    pending: list[Any] = []
    created = False
    work_item = None
    claimed_items: list[Any] = []
    completed_item = None
    repair_created = False
    repair_item = None
    claimed_repairs: list[Any] = []
    completed_repair = None
    ranked: list[tuple[str, float]] = []

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

    assert entry is not None
    assert work_item is not None
    assert completed_item is not None
    assert repair_item is not None
    assert completed_repair is not None
    assert entry.content == "hello from postgres"
    assert [item.content for item in pending] == ["hello from postgres"]
    assert created is True
    assert [item.id for item in claimed_items] == [work_item.id]
    assert completed_item.status == "completed"
    assert repair_created is True
    assert [item.id for item in claimed_repairs] == [repair_item.id]
    assert completed_repair.status == "completed"
    assert [memory_id for memory_id, _score in ranked] == ["memory-1", "memory-2"]


def test_postgres_integration_persists_direct_mutation_evidence(postgres_storage_config) -> None:
    """Persist boolean existence flags through the deployed integer schema."""
    ensure_postgres_schema(postgres_storage_config)
    evidence = DirectMutationEvidence.start(
        task_id="postgres-evidence-task", execution_epoch=1, session_id="session-1", call_id="call-1",
        sequence=1, tool_name="internal_update_memory_record", arguments={"memory_id": "memory-1"},
    ).finish(
        payload={"status": "ok", "record": {"id": "memory-1"}}, ledger_entry={"status": "success"},
        deltas=(
            DirectMutationEntityDelta("record", "memory-1", before_exists=False, after_exists=True),
            DirectMutationEntityDelta("link", "link-1", before_exists=True, after_exists=False),
        ),
    )

    with PostgresConnectionManager(postgres_storage_config) as manager:
        store = PostgresDirectMutationEvidenceStore(manager)
        assert store.save(evidence) == evidence
        assert store.get(evidence.evidence_id) == evidence
        assert store.append(evidence) == evidence

    assert [delta.before_exists for delta in evidence.deltas] == [False, True]
    assert [delta.after_exists for delta in evidence.deltas] == [True, False]


def test_postgres_integration_work_item_repository_matches_shared_contract(postgres_storage_config) -> None:
    ensure_postgres_schema(postgres_storage_config)

    with PostgresConnectionManager(postgres_storage_config) as manager:
        def make_repository() -> PostgresWorkItemRepository:
            return PostgresWorkItemRepository(manager)

        assert_enqueue_unique_deduplicates_idempotency_keys(make_repository)
        assert_claim_batch_orders_ready_items(make_repository)
        assert_heartbeat_extends_leases_and_allows_expired_reclaim(make_repository)
        assert_release_defer_and_complete_items(make_repository)


def test_postgres_search_projection_updates_with_memory_changes(postgres_storage_config) -> None:
    ensure_postgres_schema(postgres_storage_config)
    row: tuple[object, ...] | None = None

    with PostgresConnectionManager(postgres_storage_config) as manager:
        repository = PostgresRelationalMemoryRepository(manager)
        created = repository.create_memory(
            title="Projection alpha",
            content="Projection content for lexical search.",
            summary="Projection summary.",
            workspace_ids=["workspace-a"],
            memory_type="fact",
            tags=["alpha", "beta"],
            memory_id="projection-memory",
        )
        assert created is not None

        repository.update_memory(
            "projection-memory",
            title="Projection gamma",
            content="Updated content for gamma search.",
            summary="Updated gamma summary.",
            tags=["gamma"],
        )

        with manager.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT title, summary, content, tags_text,
                           search_document @@ plainto_tsquery('simple', %s) AS matches_gamma,
                           search_document @@ plainto_tsquery('simple', %s) AS matches_beta
                    FROM memory_search_documents
                    WHERE memory_id = %s
                    """,
                    ("gamma", "beta", "projection-memory"),
                )
                row = cursor.fetchone()

    assert row is not None
    assert row[0] == "Projection gamma"
    assert row[1] == "Updated gamma summary."
    assert row[2] == "Updated content for gamma search."
    assert row[3] == "gamma"
    assert row[4] is True
    assert row[5] is False


def test_postgres_schema_upgrade_backfills_search_projection(postgres_storage_config) -> None:
    ensure_postgres_schema(postgres_storage_config)
    row: tuple[object, ...] | None = None

    with PostgresConnectionManager(postgres_storage_config) as manager:
        with manager.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO memories (id, title, content, summary, type, status, created_at, updated_at, read_count, access_score, metadata) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)",
                    (
                        "backfill-memory",
                        "Backfill title",
                        "Backfill content",
                        "Backfill summary",
                        "fact",
                        "active",
                        "2026-03-28T00:00:00+00:00",
                        "2026-03-28T00:00:00+00:00",
                        0,
                        0.0,
                        "{}",
                    ),
                )
                cursor.execute(
                    "INSERT INTO memory_workspaces (memory_id, workspace_id) VALUES (%s, %s)",
                    ("backfill-memory", "workspace-a"),
                )
                cursor.execute("INSERT INTO tags (name) VALUES (%s)", ("backfill",))
                cursor.execute("SELECT id FROM tags WHERE name = %s", ("backfill",))
                tag_row = cursor.fetchone()
                assert tag_row is not None
                cursor.execute(
                    "INSERT INTO memory_tags (memory_id, tag_id) VALUES (%s, %s)",
                    ("backfill-memory", tag_row[0]),
                )
                cursor.execute("DROP TABLE IF EXISTS memory_search_documents")
                cursor.execute(
                    "UPDATE schema_metadata SET value = %s WHERE key = %s",
                    ("7", "schema_version"),
                )
            connection.commit()

    state = ensure_postgres_schema(postgres_storage_config)

    with PostgresConnectionManager(postgres_storage_config) as manager:
        with manager.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT tags_text, search_document @@ plainto_tsquery('simple', %s)
                    FROM memory_search_documents
                    WHERE memory_id = %s
                    """,
                    ("backfill", "backfill-memory"),
                )
                row = cursor.fetchone()

    assert state.schema_version == POSTGRES_SCHEMA_VERSION
    assert row is not None
    assert row[0] == "backfill"
    assert row[1] is True


def test_postgres_integration_create_runtime_uses_default_postgres_config(
    postgres_storage_config,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace = _configure_postgres_runtime_env(monkeypatch, tmp_path, dsn=postgres_storage_config.dsn)
    workspace.mkdir(parents=True)
    monkeypatch.setattr("mcp_memory.mcp.runtime.build_embedder", lambda _config: None)

    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    try:
        assert runtime.storage_backend == "postgres"
        assert isinstance(runtime.db_manager, PostgresConnectionManager)
        assert isinstance(runtime.task_queue, PostgresTaskQueue)
        assert isinstance(runtime.provider_usage, PostgresProviderUsageRepository)
        assert runtime.repository is not None

        created = runtime.repository.create_memory(
            title="Postgres runtime fact",
            content="The runtime booted through the default config path.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["postgres", "boot"],
        )

        assert created is not None
        assert runtime.repository.get_memory(created.id) is not None
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_postgres_integration_public_memory_tools_work_through_real_runtime(
    postgres_storage_config,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace = _configure_postgres_runtime_env(monkeypatch, tmp_path, dsn=postgres_storage_config.dsn)
    workspace.mkdir(parents=True)
    monkeypatch.setattr("mcp_memory.mcp.runtime.build_embedder", lambda _config: None)

    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    try:
        record_payload = json.loads(
            (
                await call_memory_tool(
                    runtime,
                    "record_thought",
                    {"content": "Postgres MCP smoke memory"},
                )
            )[0].text
        )

        assert runtime.repository is not None
        memory_record = runtime.repository.create_memory(
            title="Postgres searchable memory",
            content="This memory should be searchable through the public MCP tools.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["postgres", "mcp"],
        )
        assert memory_record is not None

        search_payload = json.loads(
            (
                await call_memory_tool(
                    runtime,
                    "search_memory_records",
                    {"query": "searchable public MCP tools", "limit": 5},
                )
            )[0].text
        )

        memory_id = memory_record.id
        assert memory_record.memory_ref is not None
        read_payload = json.loads(
            (
                await call_memory_tool(
                    runtime,
                    "read_memory_record",
                    {"memory_id": memory_id},
                )
            )[0].text
        )

        assert record_payload["status"] == "recorded"
        assert "status" not in search_payload
        assert any(
            result["memory_ref"] == f"mem-{memory_record.memory_ref}"
            for result in search_payload["results"]
        )
        assert "status" not in read_payload
        assert read_payload["record"]["memory_ref"] == f"mem-{memory_record.memory_ref}"
        assert read_payload["record"]["content"] == "This memory should be searchable through the public MCP tools."

        assert runtime.db_manager is not None
        with runtime.db_manager.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT event_kind, memory_id, query_text FROM memory_tool_events ORDER BY id ASC"
                )
                telemetry_rows = cursor.fetchall()

        assert [row[0] for row in telemetry_rows] == ["search", "read"]
        assert telemetry_rows[0][2] == "searchable public MCP tools"
        assert telemetry_rows[1][1] == memory_id
    finally:
        runtime.close()


def test_postgres_integration_daemon_app_health_overview_and_search_use_real_runtime_config(
    postgres_storage_config,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace = _configure_postgres_runtime_env(monkeypatch, tmp_path, dsn=postgres_storage_config.dsn)
    workspace.mkdir(parents=True)
    monkeypatch.setattr("mcp_memory.mcp.runtime.build_embedder", lambda _config: None)

    seed_runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    try:
        current_time = time.time()
        assert seed_runtime.repository is not None
        assert seed_runtime.runtime_logs is not None
        assert seed_runtime.provider_usage is not None
        assert seed_runtime.task_queue is not None
        seed_runtime.repository.create_memory(
            title="Postgres daemon search fact",
            content="Management search should find this through the daemon app.",
            workspace_ids=[seed_runtime.workspace_id or "global"],
            memory_type="fact",
            tags=["postgres", "daemon"],
        )
        seeded_task = seed_runtime.task_queue.enqueue(
            "memory-curator",
            workspace_id=seed_runtime.workspace_id,
            available_at=time.time() + 60.0,
            task_id="postgres-daemon-task",
        )
        seed_runtime.runtime_logs.write_log(
            source="daemon",
            logger_name="mcp_memory.server",
            level="INFO",
            message="postgres daemon smoke log",
            created_at=current_time,
            data={"backend": "postgres"},
        )
        seed_runtime.provider_usage.record_call(
            task_name="memory-curator",
            task_id=seeded_task.id,
            request_id="postgres-daemon-conversation",
            subprocess_pid=1234,
            provider_key="test-provider",
            provider_name="Test Provider",
            model_name="test-model",
            status="success",
            duration_seconds=1.0,
            created_at=current_time,
            error_text=None,
        )
        seed_runtime.provider_usage.record_conversation(
            request_id="postgres-daemon-conversation",
            attempt=1,
            task_name="memory-curator",
            task_id=seeded_task.id,
            provider_key="test-provider",
            provider_name="Test Provider",
            model_name="test-model",
            subprocess_pid=1234,
            prompt_text="prompt",
            response_text="response",
            parsed={"ok": True},
            status="success",
            error_text=None,
            started_at=current_time - 1.0,
            completed_at=current_time,
        )
    finally:
        seed_runtime.close()

    app = create_daemon_app()
    with TestClient(app) as client:
        health = client.get("/api/health")
        overview = client.get("/api/overview")
        search = client.post(
            "/api/memories/search",
            json={"query": "daemon search fact", "limit": 5},
        )
        tasks = client.get("/api/tasks")
        logs = client.post("/api/logs", json={"source": "daemon", "limit": 10})
        conversations = client.post("/api/ai-conversations", json={"limit": 10})

    assert health.status_code == 200
    assert health.json()["storage_backend"] == "postgres"
    assert health.json()["status"] == "ready"
    assert health.json()["search"]["semantic_enabled"] is False

    assert overview.status_code == 200
    assert overview.json()["memories"]["total"] == 1
    assert overview.json()["search"]["semantic_enabled"] is False
    assert overview.json()["recent_logs"][0]["message"] == "postgres daemon smoke log"
    assert overview.json()["provider_usage"][0]["provider_key"] == "test-provider"

    assert search.status_code == 200
    assert {item["title"] for item in search.json()["results"]} == {"Postgres daemon search fact"}
    assert tasks.status_code == 200
    assert "postgres-daemon-task" in {item["id"] for item in tasks.json()["tasks"]}
    assert logs.status_code == 200
    assert {item["message"] for item in logs.json()["logs"]} == {"postgres daemon smoke log"}
    assert conversations.status_code == 200
    assert {item["request_id"] for item in conversations.json()["conversations"]} == {"postgres-daemon-conversation"}


@pytest.mark.asyncio
async def test_postgres_integration_daemon_hooks_persist_conversation_state_and_reminders(
    postgres_storage_config,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace = _configure_postgres_runtime_env(monkeypatch, tmp_path, dsn=postgres_storage_config.dsn)
    workspace.mkdir(parents=True)
    monkeypatch.setattr("mcp_memory.mcp.runtime.build_embedder", lambda _config: None)

    app = create_daemon_app()
    row: tuple[object, ...] | None = None
    async with app.router.lifespan_context(app):
        metadata = app.state.metadata
        start = await asyncio.to_thread(
            request_daemon_json,
            metadata,
            "/api/hooks/session-start",
            {"sessionId": "conversation-1", "timestamp": 100.0},
        )
        noop = await asyncio.to_thread(
            request_daemon_json,
            metadata,
            "/api/hooks/post-tool-use",
            {"sessionId": "conversation-1", "tool_name": "read_file", "timestamp": 200.0},
        )
        reminder = await asyncio.to_thread(
            request_daemon_json,
            metadata,
            "/api/hooks/post-tool-use",
            {"sessionId": "conversation-1", "tool_name": "apply_patch", "timestamp": 401.0},
        )
        end = await asyncio.to_thread(
            request_daemon_json,
            metadata,
            "/api/hooks/session-end",
            {"sessionId": "conversation-1", "timestamp": 450.0},
        )

    assert start["status"] == "ok"
    assert noop == {}
    assert reminder["systemMessage"] == REMINDER_MESSAGE
    assert end["status"] == "ok"

    with PostgresConnectionManager(postgres_storage_config) as manager:
        with manager.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT conversation_id, workspace_id, last_tool_name, last_reminder_at, ended_at
                    FROM hook_conversations
                    WHERE conversation_id = %s
                    """,
                    ("conversation-1",),
                )
                row = cursor.fetchone()

    assert row is not None
    assert row[0] == "conversation-1"
    assert row[1] is None
    assert row[2] == "apply_patch"
    assert row[3] == pytest.approx(401.0)
    assert row[4] == pytest.approx(450.0)


@pytest.mark.asyncio
async def test_postgres_backup_loop_skips_sqlite_snapshot_work(
    postgres_storage_config,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The typed daemon bundle skips SQLite snapshots for Postgres storage."""

    from mcp_memory.config import Config
    from mcp_memory.daemon_app import _run_periodic_backup_loop

    called: list[bool] = []
    monkeypatch.setattr(
        "mcp_memory.daemon_app.create_and_prune_sqlite_backup",
        lambda *args, **kwargs: called.append(True),
    )

    config = Config()
    config.backups.enabled = True
    config.backups.create_startup_snapshot = True

    with PostgresConnectionManager(postgres_storage_config) as manager:
        daemon = SimpleNamespace(
            memory=SimpleNamespace(
                config=config,
                memory_path=tmp_path / "memories",
            ),
            resources=SimpleNamespace(
                storage=SimpleNamespace(
                    backend="postgres",
                    db_manager=manager,
                ),
            ),
        )

        await asyncio.wait_for(_run_periodic_backup_loop(daemon), timeout=1.0)

    assert called == []


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


def test_postgres_integration_task_queue_allows_only_one_background_cleanup_to_run(postgres_storage_config) -> None:
    ensure_postgres_schema(postgres_storage_config)

    with PostgresConnectionManager(postgres_storage_config) as manager:
        queue = PostgresTaskQueue(manager)
        first_cleanup = queue.enqueue(
            CURATOR_TASK_NAME,
            priority=1,
            available_at=0.0,
            task_id="postgres-cleanup-one",
        )
        second_cleanup = queue.enqueue(
            DEDUPLICATOR_TASK_NAME,
            priority=90,
            available_at=0.0,
            task_id="postgres-cleanup-two",
        )
        normal_task = queue.enqueue(
            "normal-background-task",
            priority=100,
            available_at=0.0,
            task_id="postgres-normal-task",
        )

        claimed_first = queue.claim_next(now=1.0)
        claimed_while_cleanup_runs = queue.claim_next(now=2.0)

        assert claimed_first is not None and claimed_first.id == first_cleanup.id
        assert claimed_while_cleanup_runs is not None and claimed_while_cleanup_runs.id == normal_task.id
        assert queue.get_task(second_cleanup.id).status == "pending"

        queue.complete(first_cleanup.id, completed_at=3.0, execution_epoch=claimed_first.execution_epoch)
        claimed_second = queue.claim_next(now=4.0)

        assert claimed_second is not None and claimed_second.id == second_cleanup.id


def test_postgres_integration_task_retry_clears_stale_cancellation_state(postgres_storage_config) -> None:
    ensure_postgres_schema(postgres_storage_config)

    with PostgresConnectionManager(postgres_storage_config) as manager:
        queue = PostgresTaskQueue(manager)

        task = queue.enqueue(
            "retry-after-cancel-race",
            workspace_id="workspace-a",
            available_at=0.0,
            task_id="postgres-retry-after-cancel-race",
        )
        claimed = queue.claim_next(now=1.0, workspace_id="workspace-a")

        assert claimed is not None

        queue.request_cancel(
            task.id,
            cancelled_by="cli",
            reason="operator_cancelled",
            requested_at=2.0,
        )

        retried = queue.fail(
            task.id,
            "temporary failure",
            retry_delay_seconds=5.0,
            failed_at=3.0,
            execution_epoch=claimed.execution_epoch,
        )

        assert retried.status == "pending"
        assert retried.last_error == "temporary failure"
        assert retried.cancellation_requested_at is None
        assert retried.cancelled_at is None
        assert retried.cancellation_reason is None
        assert retried.cancelled_by is None
        assert queue.is_cancellation_requested(task.id) is False

        reclaimed = queue.claim_next(now=8.0, workspace_id="workspace-a")

        assert reclaimed is not None
        assert reclaimed.id == task.id
        assert reclaimed.execution_epoch == claimed.execution_epoch + 1
        assert reclaimed.cancellation_requested_at is None


def test_postgres_integration_seeded_random_candidates_accept_status_filter(
    postgres_storage_config,
) -> None:
    ensure_postgres_schema(postgres_storage_config)

    with PostgresConnectionManager(postgres_storage_config) as manager:
        repository = PostgresRelationalMemoryRepository(manager)
        active = repository.create_memory(
            title="Active maintenance candidate",
            content="Eligible for seeded random maintenance selection.",
            workspace_ids=["workspace-a"],
            status="active",
        )
        stale = repository.create_memory(
            title="Stale maintenance candidate",
            content="Excluded by the status filter.",
            workspace_ids=["workspace-a"],
            status="stale",
        )

        assert active is not None
        assert stale is not None
        candidates = repository.query_seeded_random_candidates(
            "status-filter-regression",
            status="active",
            limit=10,
        )
        assert [record.id for record in candidates] == [active.id]


def test_postgres_integration_maintenance_housekeeping_handlers_update_state(
    postgres_storage_config,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace = _configure_postgres_runtime_env(monkeypatch, tmp_path, dsn=postgres_storage_config.dsn)
    workspace.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("mcp_memory.mcp.runtime.build_embedder", lambda _config: None)

    runtime = create_runtime(workspace_root_override=str(workspace), cwd=workspace)
    assert runtime.repository is not None
    assert runtime.db_manager is not None

    try:
        stale_timestamp = (datetime.now(UTC) - timedelta(days=90)).isoformat()
        stale_plan = runtime.repository.create_memory(
            title="Old plan",
            content="This plan is stale.",
            workspace_ids=[runtime.workspace_id or "global"],
            memory_type="plan",
            created_at=stale_timestamp,
            updated_at=stale_timestamp,
        )
        healthy_memory = runtime.repository.create_memory(
            title="Healthy ext link",
            content="Tracks a valid file.",
            workspace_ids=[runtime.workspace_id or "global"],
        )
        broken_memory = runtime.repository.create_memory(
            title="Broken ext link",
            content="Tracks a missing file.",
            workspace_ids=[runtime.workspace_id or "global"],
        )
        assert stale_plan is not None
        assert healthy_memory is not None
        assert broken_memory is not None

        valid_file = workspace / "README.md"
        valid_file.write_text("ok", encoding="utf-8")
        runtime.repository.add_link(healthy_memory.id, f"ext:{valid_file.name}", "REFERENCES")
        runtime.repository.add_link(broken_memory.id, "ext:missing.txt", "REFERENCES")

        cutoff_timestamp = (datetime.now(UTC) - timedelta(days=8)).timestamp()
        with runtime.db_manager.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO tasks (id, task_name, workspace_id, data, status, priority, retries_count, max_retries, created_at, updated_at, available_at, completed_at, last_error) VALUES (%s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    (
                        "old-completed-task",
                        "sweeper",
                        runtime.workspace_id,
                        "{}",
                        "completed",
                        100,
                        0,
                        3,
                        cutoff_timestamp,
                        cutoff_timestamp,
                        cutoff_timestamp,
                        cutoff_timestamp,
                        None,
                    ),
                )
                cursor.execute(
                    "INSERT INTO system1_journal (content, workspace_id, timestamp, status) VALUES (%s, %s, %s, %s)",
                    ("processed note", runtime.workspace_id, cutoff_timestamp, "processed"),
                )
            connection.commit()

        project_result = handle_project_manager_task(
            runtime,
            _running_task(
                PROJECT_MANAGER_TASK_NAME,
                workspace_id=runtime.workspace_id,
                data={"workspace_id": runtime.workspace_id},
            ),
        )
        fact_result = handle_fact_checker_task(
            runtime,
            _running_task(
                FACT_CHECKER_TASK_NAME,
                workspace_id=runtime.workspace_id,
                data={
                    "workspace_id": runtime.workspace_id,
                    "workspace_root": str(workspace),
                },
            ),
        )
        sweep_result = handle_sweeper_task(
            runtime,
            _running_task(
                SWEEPER_TASK_NAME,
                workspace_id=runtime.workspace_id,
                data={"workspace_id": runtime.workspace_id},
            ),
        )

        stale_plan_record = runtime.repository.get_memory(stale_plan.id)
        healthy_memory_record = runtime.repository.get_memory(healthy_memory.id)
        broken_memory_record = runtime.repository.get_memory(broken_memory.id)

        with runtime.db_manager.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM tasks WHERE id = %s", ("old-completed-task",))
                remaining_tasks = cursor.fetchone()
                cursor.execute(
                    "SELECT COUNT(*) FROM system1_journal WHERE status IN ('processed', 'archived')"
                )
                remaining_journal = cursor.fetchone()

        assert project_result == {"updated": 1}
        assert fact_result == {"degraded": 1, "restored": 0}
        assert sweep_result["deleted_tasks"] == 1
        assert sweep_result["deleted_journal_entries"] == 1
        assert sweep_result["gc_metadata_records"] == 0
        assert sweep_result["lineage_hotspots"]["examples"] == []
        assert sweep_result["dangling_links"]["deleted"] == 0
        assert sweep_result["memory_gc"]["mode"] == "report-only"
        assert sweep_result["memory_gc"]["deleted"] == 0
        assert stale_plan_record is not None
        assert healthy_memory_record is not None
        assert broken_memory_record is not None
        assert stale_plan_record.status == "stale"
        assert healthy_memory_record.status == "active"
        assert broken_memory_record.status == "degraded"
        assert remaining_tasks == (0,)
        assert remaining_journal == (0,)
    finally:
        runtime.close()


def test_postgres_integration_sweeper_preserves_unexpired_recoverable_entries_and_is_idempotent(
    postgres_storage_config,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace = _configure_postgres_runtime_env(monkeypatch, tmp_path, dsn=postgres_storage_config.dsn)
    workspace.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("mcp_memory.mcp.runtime.build_embedder", lambda _config: None)

    runtime = create_runtime(workspace_root_override=str(workspace), cwd=workspace)
    assert runtime.db_manager is not None

    class _VectorStoreSpy:
        def __init__(self) -> None:
            self.deleted: list[tuple[str, str, str | None]] = []

        def delete(self, *, source_kind: str, source_id: str, model_name: str | None) -> None:
            self.deleted.append((source_kind, source_id, model_name))

    runtime.vector_store = _VectorStoreSpy()
    runtime.embedder = None

    try:
        cutoff_timestamp = (datetime.now(UTC) - timedelta(days=8)).timestamp()
        future_recoverable_until = (datetime.now(UTC) + timedelta(days=1)).timestamp()
        with runtime.db_manager.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO tasks (id, task_name, workspace_id, data, status, priority, retries_count, max_retries, created_at, updated_at, available_at, completed_at, last_error) VALUES (%s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    (
                        "old-sweeper-completed-task",
                        "sweeper",
                        runtime.workspace_id,
                        "{}",
                        "completed",
                        100,
                        0,
                        3,
                        cutoff_timestamp,
                        cutoff_timestamp,
                        cutoff_timestamp,
                        cutoff_timestamp,
                        None,
                    ),
                )
                cursor.execute(
                    "INSERT INTO system1_journal (content, workspace_id, timestamp, status) VALUES (%s, %s, %s, %s)",
                    ("processed note", runtime.workspace_id, cutoff_timestamp, "processed"),
                )
                cursor.execute(
                    "INSERT INTO system1_journal (content, workspace_id, timestamp, status, recoverable_until) VALUES (%s, %s, %s, %s, %s) RETURNING id",
                    ("expired recoverable note", runtime.workspace_id, cutoff_timestamp, "recoverable", cutoff_timestamp),
                )
                expired_row = cursor.fetchone()
                cursor.execute(
                    "INSERT INTO system1_journal (content, workspace_id, timestamp, status, recoverable_until) VALUES (%s, %s, %s, %s, %s) RETURNING id",
                    (
                        "still recoverable note",
                        runtime.workspace_id,
                        cutoff_timestamp,
                        "recoverable",
                        future_recoverable_until,
                    ),
                )
                preserved_row = cursor.fetchone()
            connection.commit()

        assert expired_row is not None
        assert preserved_row is not None
        task = _running_task(
            SWEEPER_TASK_NAME,
            workspace_id=runtime.workspace_id,
            data={"workspace_id": runtime.workspace_id},
        )

        first = handle_sweeper_task(runtime, task)
        second = handle_sweeper_task(runtime, task)

        with runtime.db_manager.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT id FROM system1_journal WHERE status = 'recoverable' ORDER BY id ASC"
                )
                remaining_recoverable = cursor.fetchall()

        empty_lineage_hotspots = {
            "active_split_original_records": 0,
            "oversized_lineage_metadata_records": 0,
            "high_relationship_density_records": 0,
            "examples": [],
        }
        assert first["deleted_tasks"] == 1
        assert first["deleted_journal_entries"] == 2
        assert first["gc_metadata_records"] == 0
        assert first["lineage_hotspots"] == empty_lineage_hotspots
        assert first["dangling_links"]["deleted"] == 0
        assert first["memory_gc"]["mode"] == "report-only"
        assert first["memory_gc"]["deleted"] == 0
        assert second["deleted_tasks"] == 0
        assert second["deleted_journal_entries"] == 0
        assert second["gc_metadata_records"] == 0
        assert second["lineage_hotspots"] == empty_lineage_hotspots
        assert second["dangling_links"]["deleted"] == 0
        assert second["memory_gc"]["mode"] == "report-only"
        assert second["memory_gc"]["deleted"] == 0
        assert remaining_recoverable == [(preserved_row[0],)]
        assert runtime.vector_store.deleted == [("thought", str(expired_row[0]), None)]
    finally:
        runtime.close()


def test_postgres_integration_fact_checker_restores_degraded_links_when_file_reappears(
    postgres_storage_config,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace = _configure_postgres_runtime_env(monkeypatch, tmp_path, dsn=postgres_storage_config.dsn)
    workspace.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("mcp_memory.mcp.runtime.build_embedder", lambda _config: None)

    runtime = create_runtime(workspace_root_override=str(workspace), cwd=workspace)
    assert runtime.repository is not None

    try:
        memory = runtime.repository.create_memory(
            title="Repairable ext link",
            content="Tracks a file that will return.",
            workspace_ids=[runtime.workspace_id or "global"],
        )
        assert memory is not None
        runtime.repository.add_link(memory.id, "ext:docs/plan.md", "REFERENCES")

        task = _running_task(
            FACT_CHECKER_TASK_NAME,
            workspace_id=runtime.workspace_id,
            data={
                "workspace_id": runtime.workspace_id,
                "workspace_root": str(workspace),
            },
        )
        first = handle_fact_checker_task(runtime, task)

        repaired_file = workspace / "docs" / "plan.md"
        repaired_file.parent.mkdir(parents=True, exist_ok=True)
        repaired_file.write_text("restored", encoding="utf-8")

        second = handle_fact_checker_task(runtime, task)
        refreshed_memory = runtime.repository.get_memory(memory.id)

        assert first == {"degraded": 1, "restored": 0}
        assert second == {"degraded": 0, "restored": 1}
        assert refreshed_memory is not None
        assert refreshed_memory.status == "active"
    finally:
        runtime.close()


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
            retry_delay_seconds=45.0,
            abandoned_task_stale_after_seconds=300.0,
        )

        monkeypatch.setattr("mcp_memory.storage.postgres_task_queue._is_process_alive", lambda pid: False)

        await worker._run_reconciliation_pass(now=1000.0, reason="periodic")

        retried = queue.get_task(task.id)
        conversation = provider_usage.get_conversation("req-orphaned")[0]

        assert retried.status == "pending"
        assert retried.last_error == "Provider subprocess 9999 exited unexpectedly"
        assert retried.retries_count == 1
        assert retried.available_at == pytest.approx(retried.updated_at + 45.0)
        assert conversation.status == "error"
        assert conversation.task_id == task.id
        assert conversation.error_text == "Provider subprocess 9999 exited unexpectedly"
        assert conversation.reason_category == "recovery"
        assert conversation.reason_code == "provider_subprocess_exited_retry"


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

        await worker._run_reconciliation_pass(now=1000.0, reason="periodic")

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
            retry_delay_seconds=45.0,
        )

        monkeypatch.setattr("mcp_memory.core.tasks._is_process_alive", lambda pid: False)

        await worker._run_reconciliation_pass(now=1000.0, reason="periodic")

        retried = queue.get_task(task.id)
        attempt = attempt_repository.get_attempt(task_id=task.id, execution_epoch=claimed.execution_epoch)

        assert retried.status == "pending"
        assert retried.last_error == "Provider subprocess 9999 exited unexpectedly"
        assert retried.retries_count == 1
        assert retried.available_at == pytest.approx(retried.updated_at + 45.0)
        assert attempt.status == "error"
        assert attempt.completed_at == pytest.approx(1000.0)
        assert attempt.error_text == "Provider subprocess 9999 exited unexpectedly"
        assert attempt.termination_reason == "provider_subprocess_exited_retry"


@pytest.mark.asyncio
async def test_postgres_integration_runtime_worker_preserves_retry_recovery_conversation_after_late_provider_cancellation(
    postgres_storage_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ensure_postgres_schema(postgres_storage_config)

    with PostgresConnectionManager(postgres_storage_config) as manager:
        queue = PostgresTaskQueue(manager)
        attempt_repository = PostgresTaskExecutionAttemptRepository(manager, workspace_id="workspace-a")
        provider_usage = PostgresProviderUsageRepository(manager, workspace_id="workspace-a")
        task = queue.enqueue(
            "late-cancel-task",
            workspace_id="workspace-a",
            available_at=0.0,
            task_id="postgres-late-cancel-task",
        )
        ctx = ApplicationContext(
            db_manager=manager,
            task_queue=queue,
            task_execution_attempts=attempt_repository,
            provider_usage=provider_usage,
            workspace_id="workspace-a",
        )

        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def blocking_handler(_ctx: ApplicationContext, queued_task: TaskRecord) -> None:
            queue.set_running_process(
                queued_task.id,
                subprocess_pid=9999,
                request_id="req-late-cancel",
                updated_at=2.0,
                execution_epoch=queued_task.execution_epoch,
            )
            attempt_repository.start_attempt(
                task_id=queued_task.id,
                execution_epoch=queued_task.execution_epoch,
                task_name=queued_task.task_name,
                request_id="req-late-cancel",
                subprocess_pid=9999,
                provider_key="gemini-cli",
                provider_name="Gemini CLI",
                model_name="gemini-3-flash-preview",
                started_at=2.0,
            )
            provider_usage.record_conversation(
                request_id="req-late-cancel",
                attempt=1,
                task_name=queued_task.task_name,
                task_id=queued_task.id,
                provider_key="gemini-cli",
                provider_name="Gemini CLI",
                model_name="gemini-3-flash-preview",
                subprocess_pid=9999,
                prompt_text="keep running",
                response_text="",
                parsed=None,
                status="running",
                error_text=None,
                started_at=2.0,
                completed_at=2.0,
            )
            started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                cancelled.set()
                attempt_repository.finish_attempt(
                    task_id=queued_task.id,
                    execution_epoch=queued_task.execution_epoch,
                    status="cancelled",
                    completed_at=50.0,
                    request_id="req-late-cancel",
                    subprocess_pid=9999,
                    error_text="Command cancelled",
                    termination_reason="provider_cancelled",
                )
                provider_usage.finalize_running_conversation(
                    request_id="req-late-cancel",
                    status="cancelled",
                    error_text="Command cancelled",
                    reason_category="cancellation",
                    reason_code="provider_cancelled",
                    completed_at=50.0,
                )
                raise

        monkeypatch.setattr("mcp_memory.core.tasks._is_process_alive", lambda pid: False)
        worker = RuntimeTaskWorker(
            ctx,
            handlers={"late-cancel-task": blocking_handler},
            poll_interval_seconds=0.01,
            retry_delay_seconds=45.0,
            abandoned_recovery_interval_seconds=0.01,
            abandoned_task_stale_after_seconds=0.0,
        )

        await worker.start()
        try:
            await asyncio.wait_for(started.wait(), timeout=1.0)
            await asyncio.wait_for(cancelled.wait(), timeout=1.0)
            for _ in range(100):
                retried = queue.get_task(task.id)
                attempt = attempt_repository.get_attempt(task_id=task.id, execution_epoch=1)
                conversations = provider_usage.get_conversation("req-late-cancel")
                if (
                    retried.status == "pending"
                    and attempt.termination_reason == "provider_subprocess_exited_retry"
                    and conversations
                    and conversations[0].reason_code == "provider_subprocess_exited_retry"
                ):
                    break
                await asyncio.sleep(0.01)
        finally:
            await worker.stop(0.05)

        retried = queue.get_task(task.id)
        attempt = attempt_repository.get_attempt(task_id=task.id, execution_epoch=1)
        conversation = provider_usage.get_conversation("req-late-cancel")[0]

        assert retried.status == "pending"
        assert retried.last_error == "Provider subprocess 9999 exited unexpectedly"
        assert retried.available_at == pytest.approx(retried.updated_at + 45.0)
        assert attempt.status == "error"
        assert attempt.completed_at == pytest.approx(retried.updated_at)
        assert attempt.error_text == "Provider subprocess 9999 exited unexpectedly"
        assert attempt.termination_reason == "provider_subprocess_exited_retry"
        assert conversation.status == "error"
        assert conversation.error_text == "Provider subprocess 9999 exited unexpectedly"
        assert conversation.reason_category == "recovery"
        assert conversation.reason_code == "provider_subprocess_exited_retry"


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

        worker._reconcile_terminal_task_state(cast(Any, completed_task))

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

        await worker._run_reconciliation_pass(now=20.0, reason="periodic")

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

        await worker._schedule_follow_up(cast(Any, claimed), cast(Any, completed_task))

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
        task_runs = queue.list_task_runs(task_id=task.id)

        assert cancelled.is_set()
        assert task_after_stop.status == "cancelled"
        assert task_after_stop.cancellation_reason == "daemon_shutdown"
        assert task_after_stop.cancelled_by == "daemon"
        assert task_after_stop.last_error == "daemon_shutdown"
        assert [run.status for run in task_runs] == ["cancelled"]
        assert task_runs[0].result == {"cancelled_by": "daemon", "reason": "daemon_shutdown"}


def test_postgres_management_service_uses_backend_safe_noop_provider_usage_fallback_when_storage_backend_hint_is_missing(
    postgres_storage_config,
) -> None:
    ensure_postgres_schema(postgres_storage_config)

    with PostgresConnectionManager(postgres_storage_config) as manager:
        queue = PostgresTaskQueue(manager)
        service = ManagementService(
            ApplicationContext(
                db_manager=manager,
                task_queue=queue,
                workspace_id="workspace-a",
            ),
            cast(ManagementControllerPort, SimpleNamespace(has_runtime=True, client_count=1)),
        )

        conversations = service.list_ai_conversations(limit=10)

        assert conversations.conversations == []
