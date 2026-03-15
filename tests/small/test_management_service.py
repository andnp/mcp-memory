from types import SimpleNamespace
import logging
import time

import pytest

from mcp_memory.context import ApplicationContext
from mcp_memory.core.tasks import SQLiteTaskQueue
from mcp_memory.management.service import ManagementService
from mcp_memory.relational.repository import RelationalMemoryRepository
from mcp_memory.runtime_logging import SQLiteStructuredLogHandler


pytestmark = pytest.mark.small


def test_management_service_overview_and_memory_detail(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    task_queue = SQLiteTaskQueue(db_manager)
    handler = SQLiteStructuredLogHandler(
        db_manager=db_manager,
        workspace_id="workspace-a",
        source="daemon",
    )

    primary = repository.create_memory(
        title="Dashboard plan",
        content="Expose overview metrics and a memory detail view.",
        workspace_ids=["workspace-a"],
        memory_type="plan",
        tags=["dashboard", "api"],
        status="active",
    )
    secondary = repository.create_memory(
        title="Dashboard legacy plan",
        content="An older plan that has been replaced.",
        workspace_ids=["workspace-a"],
        memory_type="plan",
        tags=["dashboard"],
        status="stale",
    )
    assert primary is not None and secondary is not None

    repository.add_link(primary.id, secondary.id, "SUPERSEDES", "Replaced during epic 6")
    repository.record_access(primary.id, access_score=1.0, accessed_at="2026-03-15T10:00:00+00:00", increment_read_count=True)
    repository.record_access(primary.id, access_score=2.0, accessed_at="2026-03-15T10:05:00+00:00", increment_read_count=True)
    repository.record_access(secondary.id, access_score=1.0, accessed_at="2026-03-15T10:10:00+00:00", increment_read_count=True)

    task = task_queue.enqueue(
        "summarize-memory",
        task_id="task-1",
        workspace_id="workspace-a",
        available_at=0.0,
    )
    assert task_queue.claim_next(now=10.0) is not None
    task_queue.fail_permanently(task.id, "summary provider offline", failed_at=11.0)
    handler.emit(
        logging.LogRecord(
            name="mcp_memory.tests",
            level=logging.WARNING,
            pathname=__file__,
            lineno=42,
            msg="daemon log row",
            args=(),
            exc_info=None,
        )
    )
    db_manager.get_connection().execute(
        "INSERT INTO provider_usage (workspace_id, task_name, provider_key, provider_name, model_name, status, duration_seconds, created_at, error_text) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("workspace-a", "memory-curator", "gemini-cli", "Gemini CLI", "gemini-3-flash-preview", "success", 0.25, time.time(), None),
    )
    db_manager.get_connection().commit()

    ctx = ApplicationContext(
        workspace_id="workspace-a",
        memory_path=db_manager.db_path.parent,
        db_manager=db_manager,
        repository=repository,
        task_queue=task_queue,
    )
    controller = SimpleNamespace(has_runtime=True, client_count=1)
    service = ManagementService(ctx, controller)

    overview = service.get_overview()
    detail = service.get_memory_detail(primary.id)
    health = service.get_health()
    filtered_logs = service.list_logs(query="daemon", source="daemon")
    summary = service.summarize_logs(source="daemon")
    prune_result = service.prune_logs(max_runtime_logs=1, max_log_age_days=30)
    after_prune = service.list_logs(limit=10)

    assert overview.memories.total == 2
    assert overview.memories.by_status == {"active": 1, "stale": 1}
    assert overview.memory_metrics.total_memories == 2
    assert overview.memory_metrics.thought_buffer_entries == 0
    assert overview.agent_runs[0].task_name == "ingest-system1"
    assert overview.provider_usage[0].provider_key == "gemini-cli"
    assert overview.provider_usage[0].task_name == "memory-curator"
    assert overview.provider_usage[0].calls_last_hour == 1
    assert overview.tasks.failed_count == 1
    assert overview.failed_tasks[0]["last_error"] == "summary provider offline"
    assert overview.recent_logs[0].message == "daemon log row"
    assert overview.recent_logs[0].source == "daemon"
    assert [record.id for record in overview.top_read_memories] == [primary.id, secondary.id]
    assert [record.read_count for record in overview.top_read_memories] == [2, 1]
    assert filtered_logs.logs[0].logger_name == "mcp_memory.tests"
    assert summary.total == 1
    assert summary.by_level == {"WARNING": 1}
    assert summary.by_source == {"daemon": 1}
    assert prune_result.deleted == 0
    assert prune_result.max_runtime_logs == 1
    assert len(after_prune.logs) == 1
    assert detail.record["id"] == primary.id
    assert detail.relationships["outgoing"][0]["link_type"] == "SUPERSEDES"
    assert detail.superseded[0]["id"] == secondary.id
    assert overview.failed_tasks[0]["status"] == "failed"
    assert health.runtime_active is True
    assert health.workspace_id == "workspace-a"


def test_management_service_can_cancel_running_task_and_list_conversations(db_manager, monkeypatch) -> None:
    repository = RelationalMemoryRepository(db_manager)
    task_queue = SQLiteTaskQueue(db_manager)
    task = task_queue.enqueue(
        "memory-curator",
        task_id="task-cancel-1",
        workspace_id="workspace-a",
        available_at=0.0,
    )
    assert task_queue.claim_next(now=10.0) is not None
    task_queue.set_running_process(task.id, subprocess_pid=4321, request_id="req-123", updated_at=11.0)

    db_manager.get_connection().execute(
        "INSERT INTO ai_conversations (request_id, attempt, workspace_id, task_name, task_id, provider_key, provider_name, model_name, subprocess_pid, prompt_text, response_text, parsed_json, status, error_text, started_at, completed_at, duration_seconds) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "req-123",
            1,
            "workspace-a",
            "memory-curator",
            task.id,
            "gemini-cli",
            "Gemini CLI",
            "gemini-3-flash-preview",
            4321,
            "prompt text",
            '{"ok": true}',
            '{"ok": true}',
            "success",
            None,
            1.0,
            2.0,
            1.0,
        ),
    )
    db_manager.get_connection().commit()

    killed: list[int] = []
    monkeypatch.setattr("mcp_memory.management.service._terminate_process", lambda pid: killed.append(pid) or True)

    ctx = ApplicationContext(
        workspace_id="workspace-a",
        memory_path=db_manager.db_path.parent,
        db_manager=db_manager,
        repository=repository,
        task_queue=task_queue,
    )
    service = ManagementService(ctx, SimpleNamespace(has_runtime=True, client_count=1))

    cancel_payload = service.cancel_task(task.id, cancelled_by="cli", reason="manual_cancel")
    conversations = service.list_ai_conversations(task_name="memory-curator")

    assert cancel_payload["status"] == "cancellation_requested"
    assert cancel_payload["signal_sent"] is True
    assert killed == [4321]
    assert cancel_payload["task"]["cancellation_reason"] == "manual_cancel"
    assert conversations.conversations[0].request_id == "req-123"
    assert conversations.conversations[0].subprocess_pid == 4321


def test_terminate_process_escalates_to_sigkill_when_sigterm_does_not_exit(monkeypatch) -> None:
    sent_signals: list[int] = []
    alive_states = iter([True, True, False])

    monkeypatch.setattr("mcp_memory.management.service.os.kill", lambda pid, sig: sent_signals.append(sig))
    monkeypatch.setattr("mcp_memory.management.service.time.sleep", lambda _: None)
    monkeypatch.setattr("mcp_memory.management.service.time.monotonic", iter([0.0, 0.2, 0.4, 1.2]).__next__)
    monkeypatch.setattr("mcp_memory.management.service._is_process_alive", lambda pid: next(alive_states))

    result = __import__("mcp_memory.management.service", fromlist=["_terminate_process"])._terminate_process(1234)

    assert result is True
    assert sent_signals == [15, 9]
