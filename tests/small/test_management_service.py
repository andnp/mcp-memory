from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
import logging
import time

import pytest

from mcp_memory.context import ApplicationContext
from mcp_memory.core.journal import System1Journal
from mcp_memory.core.tasks import SQLiteTaskQueue
from mcp_memory.management.service import ManagementService
from mcp_memory.provider_usage_store import ProviderUsageRepository
from mcp_memory.relational.repository import RelationalMemoryRepository
from mcp_memory.runtime_logging import SQLiteStructuredLogHandler


pytestmark = pytest.mark.small


def _build_management_service(
    db_manager,
    *,
    workspace_id: str | None,
    repository: RelationalMemoryRepository | None = None,
    task_queue: SQLiteTaskQueue | None = None,
) -> ManagementService:
    return ManagementService(
        ApplicationContext(
            workspace_id=workspace_id,
            memory_path=db_manager.db_path.parent,
            db_manager=db_manager,
            repository=repository,
            task_queue=task_queue,
        ),
        SimpleNamespace(has_runtime=True, client_count=1),
    )


def test_management_service_reporting_handles_empty_store(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    task_queue = SQLiteTaskQueue(db_manager)
    service = _build_management_service(
        db_manager,
        workspace_id="workspace-a",
        repository=repository,
        task_queue=task_queue,
    )

    overview = service.get_overview()
    nerd_metrics = service.get_nerd_metrics(window_hours=24, bucket_minutes=60, now=100.0)

    assert overview.memories.total == 0
    assert overview.memories.by_type == {}
    assert overview.memories.by_status == {}
    assert overview.memory_metrics.total_memories == 0
    assert overview.memory_metrics.total_memory_lines == 0
    assert overview.memory_metrics.total_summary_lines == 0
    assert overview.memory_metrics.thought_buffer_entries == 0
    assert overview.queue_diagnostics == []
    assert overview.failed_tasks == []
    assert overview.provider_usage == []
    assert overview.recent_logs == []
    assert overview.top_read_memories == []
    assert overview.top_read_memories_active == []
    assert overview.tasks.by_status == {}
    assert overview.tasks.failed_count == 0
    assert nerd_metrics.graph_topology.total_memories == 0
    assert nerd_metrics.graph_topology.total_links == 0
    assert nerd_metrics.memory_lifecycle.by_status == {}
    assert nerd_metrics.composition.by_workspace == []
    assert nerd_metrics.composition.by_tag == []
    assert nerd_metrics.composition.by_type == []
    assert nerd_metrics.composition.by_status == []
    assert [bucket.count for bucket in nerd_metrics.distributions.created_age_buckets] == [0, 0, 0, 0, 0]
    assert [bucket.count for bucket in nerd_metrics.distributions.updated_age_buckets] == [0, 0, 0, 0, 0]
    assert [bucket.count for bucket in nerd_metrics.distributions.content_size_buckets] == [0, 0, 0, 0, 0]
    assert nerd_metrics.timelines.memory_activity == []
    assert nerd_metrics.maintenance.events == []
    assert nerd_metrics.agent_throughput == []
    assert nerd_metrics.provider_latency == []


def test_management_service_overview_respects_workspace_and_global_scopes(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    task_queue = SQLiteTaskQueue(db_manager)

    memory_a = repository.create_memory(
        title="Workspace A fact",
        content="Only workspace A should see this by default.",
        workspace_ids=["workspace-a"],
        memory_type="fact",
    )
    memory_b = repository.create_memory(
        title="Workspace B fact",
        content="Only global scope or workspace B should see this.",
        workspace_ids=["workspace-b"],
        memory_type="fact",
    )
    assert memory_a is not None and memory_b is not None

    db_manager.get_connection().executemany(
        "INSERT INTO provider_usage (workspace_id, task_name, provider_key, provider_name, model_name, status, duration_seconds, created_at, error_text) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("workspace-a", "memory-curator", "gemini-cli", "Gemini CLI", "gemini-3-flash-preview", "success", 0.25, 10.0, None),
            ("workspace-b", "memory-curator", "copilot-mini", "Copilot CLI", "gpt-5-mini", "success", 0.5, 20.0, None),
        ],
    )
    db_manager.get_connection().executemany(
        "INSERT INTO runtime_logs (workspace_id, source, logger_name, level, message, created_at, data_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            ("workspace-a", "daemon", "mcp_memory.tests", "INFO", "workspace-a log", 11.0, "{}"),
            ("workspace-b", "daemon", "mcp_memory.tests", "INFO", "workspace-b log", 21.0, "{}"),
        ],
    )
    task_a = task_queue.enqueue(
        "graph-linker",
        task_id="workspace-a-maintenance",
        workspace_id="workspace-a",
        available_at=0.0,
    )
    task_b = task_queue.enqueue(
        "graph-linker",
        task_id="workspace-b-maintenance",
        workspace_id="workspace-b",
        available_at=0.0,
    )
    assert task_queue.claim_next(now=30.0) is not None
    task_queue.complete(task_a.id, completed_at=31.0, run_result={"updated": 1})
    assert task_queue.claim_next(now=32.0) is not None
    task_queue.complete(task_b.id, completed_at=33.0, run_result={"updated": 2})
    db_manager.get_connection().commit()

    scoped_service = _build_management_service(
        db_manager,
        workspace_id="workspace-a",
        repository=repository,
        task_queue=task_queue,
    )
    global_service = _build_management_service(
        db_manager,
        workspace_id=None,
        repository=repository,
        task_queue=task_queue,
    )

    scoped_overview = scoped_service.get_overview()
    global_overview = global_service.get_overview()
    scoped_nerd = scoped_service.get_nerd_metrics(window_hours=24, bucket_minutes=60, now=100.0)
    global_nerd = global_service.get_nerd_metrics(window_hours=24, bucket_minutes=60, now=100.0)

    assert scoped_overview.memories.total == 1
    assert scoped_overview.memory_metrics.total_memories == 1
    assert [record.title for record in scoped_overview.recent_memories] == ["Workspace A fact"]
    assert {item.provider_key for item in scoped_overview.provider_usage} == {"gemini-cli"}
    assert [(item.key, item.count) for item in scoped_nerd.composition.by_workspace] == [("workspace-a", 1)]
    assert [item.task_id for item in scoped_nerd.maintenance.events] == [task_a.id]

    assert global_overview.memories.total == 2
    assert global_overview.memory_metrics.total_memories == 2
    assert {record.title for record in global_overview.recent_memories} == {"Workspace A fact", "Workspace B fact"}
    assert {item.provider_key for item in global_overview.provider_usage} == {"gemini-cli", "copilot-mini"}
    assert [(item.key, item.count) for item in global_nerd.composition.by_workspace] == [
        ("workspace-a", 1),
        ("workspace-b", 1),
    ]
    assert [item.task_id for item in global_nerd.maintenance.events] == [task_b.id, task_a.id]


def test_management_service_nerd_metrics_composition_distributions_and_timelines(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    task_queue = SQLiteTaskQueue(db_manager)
    now = datetime(2026, 3, 19, 12, 0, tzinfo=UTC)

    repository.create_memory(
        title="Fresh fact",
        content="a" * 120,
        workspace_ids=["workspace-a"],
        tags=["alpha", "shared"],
        memory_type="fact",
        status="active",
        created_at=(now - timedelta(hours=6)).isoformat(),
        updated_at=(now - timedelta(hours=2)).isoformat(),
    )
    repository.create_memory(
        title="Recent plan",
        content="b" * 2_000,
        workspace_ids=["workspace-a"],
        tags=["beta", "shared"],
        memory_type="plan",
        status="stale",
        created_at=(now - timedelta(days=2)).isoformat(),
        updated_at=(now - timedelta(hours=20)).isoformat(),
    )
    repository.create_memory(
        title="Older reflection",
        content="c" * 7_000,
        workspace_ids=["workspace-a"],
        tags=["alpha", "gamma"],
        memory_type="reflection",
        status="degraded",
        created_at=(now - timedelta(days=40)).isoformat(),
        updated_at=(now - timedelta(days=10)).isoformat(),
    )
    repository.create_memory(
        title="Other workspace fact",
        content="d" * 500,
        workspace_ids=["workspace-b"],
        tags=["beta", "excluded"],
        memory_type="fact",
        status="archived",
        created_at=(now - timedelta(days=5)).isoformat(),
        updated_at=(now - timedelta(days=1)).isoformat(),
    )

    service = _build_management_service(
        db_manager,
        workspace_id="workspace-a",
        repository=repository,
        task_queue=task_queue,
    )

    nerd_metrics = service.get_nerd_metrics(window_hours=24 * 45, bucket_minutes=24 * 60, now=now.timestamp())

    assert [(item.key, item.count) for item in nerd_metrics.composition.by_workspace] == [("workspace-a", 3)]
    assert [(item.key, item.count) for item in nerd_metrics.composition.by_type] == [
        ("fact", 1),
        ("plan", 1),
        ("reflection", 1),
    ]
    assert [(item.key, item.count) for item in nerd_metrics.composition.by_status] == [
        ("active", 1),
        ("degraded", 1),
        ("stale", 1),
    ]
    assert [(item.key, item.count) for item in nerd_metrics.composition.by_tag] == [
        ("alpha", 2),
        ("shared", 2),
        ("beta", 1),
        ("gamma", 1),
    ]

    assert [(bucket.key, bucket.count) for bucket in nerd_metrics.distributions.created_age_buckets] == [
        ("lt_1d", 1),
        ("1d_to_7d", 1),
        ("7d_to_30d", 0),
        ("30d_to_90d", 1),
        ("gte_90d", 0),
    ]
    assert [(bucket.key, bucket.count) for bucket in nerd_metrics.distributions.updated_age_buckets] == [
        ("lt_1d", 2),
        ("1d_to_7d", 0),
        ("7d_to_30d", 1),
        ("30d_to_90d", 0),
        ("gte_90d", 0),
    ]
    assert [(bucket.key, bucket.count) for bucket in nerd_metrics.distributions.content_size_buckets] == [
        ("0b_to_255b", 1),
        ("256b_to_1kb", 0),
        ("1kb_to_4kb", 1),
        ("4kb_to_16kb", 1),
        ("gte_16kb", 0),
    ]

    assert sum(bucket.created_count for bucket in nerd_metrics.timelines.memory_activity) == 3
    assert sum(bucket.updated_count for bucket in nerd_metrics.timelines.memory_activity) == 3
    assert nerd_metrics.timelines.memory_activity[-1].total_content_bytes == 9_120


def test_management_service_graph_topology_counts_cross_workspace_links(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    task_queue = SQLiteTaskQueue(db_manager)

    source = repository.create_memory(
        title="Workspace A dependency",
        content="Depends on a global/shared fact.",
        workspace_ids=["workspace-a"],
        memory_type="fact",
    )
    target = repository.create_memory(
        title="Workspace B dependency target",
        content="Target memory in another workspace.",
        workspace_ids=["workspace-b"],
        memory_type="fact",
    )
    assert source is not None and target is not None
    repository.add_link(source.id, target.id, "DEPENDS_ON", "Cross-workspace dependency")

    scoped_a = _build_management_service(
        db_manager,
        workspace_id="workspace-a",
        repository=repository,
        task_queue=task_queue,
    ).get_nerd_metrics(window_hours=24, bucket_minutes=60, now=100.0)
    scoped_b = _build_management_service(
        db_manager,
        workspace_id="workspace-b",
        repository=repository,
        task_queue=task_queue,
    ).get_nerd_metrics(window_hours=24, bucket_minutes=60, now=100.0)
    global_metrics = _build_management_service(
        db_manager,
        workspace_id=None,
        repository=repository,
        task_queue=task_queue,
    ).get_nerd_metrics(window_hours=24, bucket_minutes=60, now=100.0)

    assert scoped_a.graph_topology.total_memories == 1
    assert scoped_a.graph_topology.total_links == 1
    assert scoped_a.graph_topology.average_degree == 1.0
    assert scoped_a.graph_topology.orphan_count == 0
    assert scoped_a.graph_topology.link_type_counts == {"DEPENDS_ON": 1}

    assert scoped_b.graph_topology.total_memories == 1
    assert scoped_b.graph_topology.total_links == 1
    assert scoped_b.graph_topology.average_degree == 1.0
    assert scoped_b.graph_topology.orphan_count == 0
    assert scoped_b.graph_topology.link_type_counts == {"DEPENDS_ON": 1}

    assert global_metrics.graph_topology.total_memories == 2
    assert global_metrics.graph_topology.total_links == 1
    assert global_metrics.graph_topology.average_degree == 1.0
    assert global_metrics.graph_topology.graph_supported_count == 1
    assert global_metrics.graph_topology.link_type_counts == {"DEPENDS_ON": 1}


def test_management_service_analytics_handles_zero_duration_rows_and_zero_window(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    task_queue = SQLiteTaskQueue(db_manager)

    completed = task_queue.enqueue(
        "graph-linker",
        task_id="analytics-completed",
        workspace_id="workspace-a",
        available_at=0.0,
    )
    retried = task_queue.enqueue(
        "fact-checker",
        task_id="analytics-retried",
        workspace_id="workspace-a",
        available_at=0.0,
    )
    assert task_queue.claim_next(now=100.0) is not None
    task_queue.complete(
        completed.id,
        completed_at=100.0,
        run_result={
            "merged": 2,
            "requested_strategy": "semantic",
            "strategy_used": "semantic",
            "candidate_count": 5,
            "lines_compressed": 7,
        },
    )
    assert task_queue.claim_next(now=101.0) is not None
    task_queue.fail(retried.id, "retry me", retry_delay_seconds=0.0, failed_at=101.0)
    claimed_retry = task_queue.claim_next(now=102.0)
    assert claimed_retry is not None
    task_queue.fail_permanently(claimed_retry.id, "boom", failed_at=102.0)

    db_manager.get_connection().executemany(
        "INSERT INTO provider_usage (workspace_id, task_name, provider_key, provider_name, model_name, status, duration_seconds, created_at, error_text) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("workspace-a", "graph-linker", "gemini-cli", "Gemini CLI", "gemini-3-flash-preview", "success", 0.0, 100.0, None),
            ("workspace-a", "fact-checker", "gemini-cli", "Gemini CLI", "gemini-3-flash-preview", "error", 0.0, 101.0, "timeout"),
            ("workspace-b", "graph-linker", "copilot-mini", "Copilot CLI", "gpt-5-mini", "success", 0.9, 101.0, None),
        ],
    )
    db_manager.get_connection().commit()

    service = _build_management_service(
        db_manager,
        workspace_id="workspace-a",
        repository=repository,
        task_queue=task_queue,
    )

    nerd_metrics = service.get_nerd_metrics(window_hours=1, bucket_minutes=0, now=102.0)
    zero_window_metrics = service.get_nerd_metrics(window_hours=0, bucket_minutes=0, now=200.0)

    assert nerd_metrics.window_hours == 1
    assert nerd_metrics.bucket_minutes == 0
    assert len(nerd_metrics.agent_throughput) == 1
    assert nerd_metrics.agent_throughput[0].total_runs == 3
    assert nerd_metrics.agent_throughput[0].completed_runs == 1
    assert nerd_metrics.agent_throughput[0].failed_runs == 1
    assert nerd_metrics.agent_throughput[0].retry_runs == 1
    assert nerd_metrics.agent_throughput[0].avg_duration_seconds == 0.0
    assert len(nerd_metrics.provider_latency) == 1
    assert nerd_metrics.provider_latency[0].call_count == 2
    assert nerd_metrics.provider_latency[0].failure_count == 1
    assert nerd_metrics.provider_latency[0].avg_duration_seconds == 0.0
    assert nerd_metrics.provider_latency[0].p95_duration_seconds == 0.0
    assert [event.status for event in nerd_metrics.maintenance.events] == ["failed", "retry", "completed"]
    graph_linker_event = next(event for event in nerd_metrics.maintenance.events if event.task_name == "graph-linker")
    assert graph_linker_event.strategy_used == "semantic"
    assert graph_linker_event.candidate_count == 5
    assert graph_linker_event.merged_count == 2
    assert graph_linker_event.lines_compressed == 7
    assert graph_linker_event.impact_summary == "merged=2, lines=7"
    assert any(stat.key == "provider_failure_rate" and stat.value == 0.5 for stat in nerd_metrics.stats)
    assert any(alert.key == "provider_failure_rate" for alert in nerd_metrics.alerts)

    assert zero_window_metrics.agent_throughput == []
    assert zero_window_metrics.provider_latency == []
    assert zero_window_metrics.maintenance.events == []
    assert any(stat.key == "runs_last_window" and stat.value == 0.0 for stat in zero_window_metrics.stats)
    assert any(stat.key == "provider_calls_last_window" and stat.value == 0.0 for stat in zero_window_metrics.stats)


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
    completed = task_queue.enqueue(
        "deduplicator",
        task_id="task-2",
        workspace_id="workspace-a",
        available_at=0.0,
    )
    assert task_queue.claim_next(now=12.0) is not None
    task_queue.complete(
        completed.id,
        completed_at=14.0,
        run_result={
            "merged": 1,
            "requested_strategy": "semantic",
            "strategy_used": "semantic",
            "candidate_count": 8,
            "sampled_memory_ids": [primary.id],
        },
    )
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
    db_manager.get_connection().execute(
        "INSERT INTO provider_usage (workspace_id, task_name, provider_key, provider_name, model_name, status, duration_seconds, created_at, error_text) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("workspace-b", "memory-curator", "copilot-mini", "Copilot CLI", "gpt-5-mini", "success", 0.5, time.time(), None),
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
    nerd_metrics = service.get_nerd_metrics(window_hours=24, bucket_minutes=60, now=time.time())
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
    assert {item.provider_key for item in overview.provider_usage} == {"gemini-cli"}
    assert overview.tasks.failed_count == 1
    deduplicator = next(agent for agent in overview.agent_runs if agent.task_name == "deduplicator")
    assert deduplicator.last_result_metadata.strategy_used == "semantic"
    assert deduplicator.last_result_metadata.candidate_count == 8
    assert deduplicator.last_result_metadata.sampled_memory_ids == [primary.id]
    assert any(run.result_metadata.strategy_used == "semantic" for run in overview.recent_agent_runs)
    assert overview.failed_tasks[0]["last_error"] == "summary provider offline"
    assert overview.recent_logs[0].message == "daemon log row"
    assert overview.recent_logs[0].source == "daemon"
    assert [record.id for record in overview.top_read_memories] == [primary.id, secondary.id]
    assert [record.read_count for record in overview.top_read_memories] == [2, 1]
    assert [record.id for record in overview.top_read_memories_active] == [primary.id]
    assert [record.read_count for record in overview.top_read_memories_active] == [2]
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
    assert nerd_metrics.graph_topology.total_memories == 2
    assert nerd_metrics.graph_topology.total_links == 1
    assert nerd_metrics.graph_topology.link_type_counts == {"SUPERSEDES": 1}
    assert nerd_metrics.graph_topology.orphan_rate == 0.0
    assert nerd_metrics.memory_lifecycle.by_status == {"active": 1, "stale": 1}
    assert nerd_metrics.memory_lifecycle.by_type == {"plan": 2}
    assert nerd_metrics.memory_lifecycle.cold_memory_count == 0
    assert nerd_metrics.search_quality.semantic_enabled is False
    assert nerd_metrics.search_quality.graph_supported_rate == 0.0
    curator_route = next(item for item in nerd_metrics.route_audit if item.task_name == "memory-curator")
    summarize_route = next(item for item in nerd_metrics.route_audit if item.task_name == "summarize-memory")
    assert curator_route.task_class == "premium_agentic"
    assert curator_route.recent_provider_key == "gemini-cli"
    assert summarize_route.task_class == "deterministic"
    assert summarize_route.resolved_provider_key is None
    assert any(stat.key == "orphan_rate" for stat in nerd_metrics.stats)
    assert all(alert.key != "curator_route_fallback" for alert in nerd_metrics.alerts)


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


def test_management_service_lists_reconciled_terminal_conversation_status(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    task_queue = SQLiteTaskQueue(db_manager)
    provider_usage = ProviderUsageRepository(db_manager, workspace_id="workspace-a")

    provider_usage.record_conversation(
        request_id="req-terminal",
        attempt=1,
        task_name="ingest-system1",
        task_id="task-terminal",
        provider_key="gemini-cli",
        provider_name="Gemini CLI",
        model_name="gemini-3-flash-preview",
        subprocess_pid=2222,
        prompt_text="prompt text",
        response_text="",
        parsed=None,
        status="running",
        error_text=None,
        started_at=10.0,
        completed_at=10.0,
    )
    provider_usage.reconcile_running_task_conversations(
        task_id="task-terminal",
        status="error",
        error_text="Provider subprocess 2222 exited unexpectedly",
        completed_at=15.0,
    )

    ctx = ApplicationContext(
        workspace_id="workspace-a",
        memory_path=db_manager.db_path.parent,
        db_manager=db_manager,
        repository=repository,
        task_queue=task_queue,
    )
    service = ManagementService(ctx, SimpleNamespace(has_runtime=True, client_count=1))

    conversations = service.list_ai_conversations(task_name="ingest-system1")

    assert conversations.conversations[0].request_id == "req-terminal"
    assert conversations.conversations[0].status == "error"
    assert conversations.conversations[0].error_text == "Provider subprocess 2222 exited unexpectedly"


def test_management_service_ignores_scheduled_tasks_for_oldest_runnable_age(db_manager) -> None:
    task_queue = SQLiteTaskQueue(db_manager)
    ctx = ApplicationContext(
        workspace_id="workspace-a",
        memory_path=db_manager.db_path.parent,
        db_manager=db_manager,
        task_queue=task_queue,
    )
    service = ManagementService(ctx, SimpleNamespace(has_runtime=True, client_count=1))

    task_queue.enqueue(
        "ingest-system1",
        task_id="scheduled-only",
        workspace_id="workspace-a",
        available_at=time.time() + 500.0,
    )

    nerd_metrics = service.get_nerd_metrics(window_hours=24, bucket_minutes=60, now=time.time())

    assert nerd_metrics.queue_snapshot.runnable_count == 0
    assert nerd_metrics.queue_snapshot.scheduled_count == 1
    assert nerd_metrics.queue_snapshot.oldest_age_seconds == 0.0
    assert all(alert.key != "queue_oldest_age" for alert in nerd_metrics.alerts)


def test_management_service_can_record_thought_into_journal(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    journal = System1Journal(db_manager)
    task_queue = SQLiteTaskQueue(db_manager)

    ctx = ApplicationContext(
        workspace_id="workspace-a",
        memory_path=db_manager.db_path.parent,
        db_manager=db_manager,
        journal=journal,
        repository=repository,
        task_queue=task_queue,
    )
    service = ManagementService(ctx, SimpleNamespace(has_runtime=True, client_count=1))

    payload = service.record_thought("remember the command bar")

    assert payload["status"] == "recorded"
    pending = journal.get_pending(workspace_id="workspace-a")
    assert [entry.content for entry in pending] == ["remember the command bar"]


def test_terminate_process_escalates_to_sigkill_when_sigterm_does_not_exit(monkeypatch) -> None:
    sent_signals: list[tuple[str, str, bool]] = []
    wait_deadlines: list[float] = []
    alive_states = iter([True])

    monkeypatch.setattr("mcp_memory.management.service.time.monotonic", iter([0.0, 0.2, 0.4]).__next__)
    monkeypatch.setattr("mcp_memory.management.service._is_process_alive", lambda pid: next(alive_states))
    monkeypatch.setattr(
        "mcp_memory.management.service._send_process_signal",
        lambda pid, sig, *, scope, suppress_permission_errors=False: sent_signals.append(
            (sig.name, scope, suppress_permission_errors)
        ) or (None, True),
    )

    def _fake_wait(process_id: int, *, deadline: float, poll_interval_seconds: float, is_process_running) -> None:
        wait_deadlines.append(deadline)
        if len(wait_deadlines) == 1:
            raise RuntimeError("timeout")

    monkeypatch.setattr("mcp_memory.management.service._wait_for_process_exit", _fake_wait)

    result = __import__("mcp_memory.management.service", fromlist=["_terminate_process"])._terminate_process(1234)

    assert result is True
    assert sent_signals == [
        ("SIGTERM", "pid", True),
        ("SIGKILL", "pid", True),
    ]
    assert wait_deadlines == pytest.approx([1.0, 1.2])
