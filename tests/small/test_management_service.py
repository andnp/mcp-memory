from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
import logging
import time

import pytest

from mcp_memory.config import Config, CurationConfig, StorageCacheMode
from mcp_memory.context import ApplicationContext
from mcp_memory.core.journal import System1Journal
from mcp_memory.daemon_models import DaemonControllerView
from mcp_memory.management.analytics_reporting import is_provenance_process_tag
from mcp_memory.management.capabilities import ManagementCapabilities
from mcp_memory.management.health_reporting import build_embedding_status
from mcp_memory.management.models import ExecutionAttemptHealthPayload
from mcp_memory.management.models import NerdMetricsPayload
from mcp_memory.management.query_runner import SQLiteManagementQueryAdapter
from mcp_memory.management.reporting_rows import extract_copilot_premium_requests
from mcp_memory.core.tasks import SQLiteTaskQueue
from mcp_memory.embedding_repair_store import SQLiteEmbeddingRepairQueue
from mcp_memory.management.service import ManagementService
from mcp_memory.embeddings import EmbedderStatus, SQLiteVectorStore
from mcp_memory.provider_usage_store import ProviderUsageRepository
from mcp_memory.relational.repository import RelationalMemoryRepository
from mcp_memory.relational.search import RelationalMemorySearchService
from mcp_memory.runtime_logging import SQLiteStructuredLogHandler
from mcp_memory.storage.shared_read_cache import SharedReadCache
from mcp_memory.work_item_store import SQLiteWorkItemRepository


pytestmark = pytest.mark.small


@dataclass
class _ManagementContextStub:
    config: Config | None = None
    workspace_id: str | None = None
    workspace_root: Path | None = None
    memory_path: Path | None = None
    storage_backend: str | None = None
    db_manager: object = None
    journal: object = None
    repository: object = None
    relational_search: object = None
    read_cache: object = None
    task_queue: object = None
    ai_json_provider: object = None
    ai_agent_provider: object = None
    ai_provider_registry: dict[str, object] | None = None
    provider_usage: object = None
    runtime_logs: object = None
    retrieval_telemetry: object = None
    embedding_integrity_events: object = None
    embedder: object = None
    vector_store: object = None


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


def test_management_search_uses_retrieval_facade_and_preserves_surfacing(db_manager, monkeypatch) -> None:
    repository = RelationalMemoryRepository(db_manager)
    native_search = RelationalMemorySearchService(repository, Config())
    context = ApplicationContext(
        config=Config(),
        workspace_id="workspace-a",
        memory_path=db_manager.db_path.parent,
        db_manager=db_manager,
        repository=repository,
        relational_search=native_search,
    )
    service = ManagementService(
        context,
        SimpleNamespace(has_runtime=True, client_count=1),
    )
    record = repository.create_memory(
        title="Facade management result",
        content="Management retrieval facade routing.",
        workspace_ids=["workspace-a"],
        memory_type="fact",
    )
    assert record is not None

    def native_search_is_forbidden(*args, **kwargs):
        raise AssertionError("management search bypassed the retrieval facade")

    monkeypatch.setattr(native_search, "search_memories", native_search_is_forbidden)

    payload = service.search_memories(
        query="facade management",
        workspace_id="workspace-a",
        limit=5,
    )

    assert [result.title for result in payload.results] == ["Facade management result"]
    refreshed = repository.get_memory(record.id)
    assert refreshed is not None and refreshed.last_surfaced_at is not None


def test_management_service_accepts_structural_management_context(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    task_queue = SQLiteTaskQueue(db_manager)
    service = ManagementService(
        _ManagementContextStub(
            workspace_id="workspace-a",
            memory_path=db_manager.db_path.parent,
            db_manager=db_manager,
            repository=repository,
            task_queue=task_queue,
        ),
        SimpleNamespace(has_runtime=True, client_count=1),
    )

    health = service.get_health()

    assert health.storage_backend == "sqlite"
    assert service.workspace_id == "workspace-a"


def test_management_service_injects_storage_specific_nerd_metrics_adapter(db_manager, monkeypatch) -> None:
    service = _build_management_service(db_manager, workspace_id="workspace-a")
    seen: dict[str, object] = {}

    def capture_build_nerd_metrics(**kwargs):
        seen.update(kwargs)
        return NerdMetricsPayload(generated_at=0.0, window_hours=24, bucket_minutes=60)

    monkeypatch.setattr("mcp_memory.management.service.build_nerd_metrics", capture_build_nerd_metrics)

    service.get_nerd_metrics()

    assert isinstance(seen["query_adapter"], SQLiteManagementQueryAdapter)


def test_management_service_exposes_isolated_typed_capabilities(db_manager) -> None:
    service = _build_management_service(
        db_manager,
        workspace_id="workspace-a",
        repository=RelationalMemoryRepository(db_manager),
        task_queue=SQLiteTaskQueue(db_manager),
    )

    assert isinstance(service.capabilities, ManagementCapabilities)
    assert service.capabilities.reporting.get_health() == service.get_health()
    assert service.capabilities.maintenance.list_tasks() == service.list_tasks()
    assert service.capabilities.mutation.list_mutation_history().events == []
    assert service.capabilities.runtime.list_logs().logs == []
    assert service.capabilities.memory.list_memories().records == []


def test_management_service_omits_legacy_curation_mode_flags(db_manager) -> None:
    config = Config(curation=CurationConfig())
    service = ManagementService(
        ApplicationContext(
            config=config,
            memory_path=db_manager.db_path.parent,
            db_manager=db_manager,
            repository=RelationalMemoryRepository(db_manager),
            task_queue=SQLiteTaskQueue(db_manager),
        ),
        SimpleNamespace(has_runtime=True, client_count=1),
    )

    health = service.get_health()
    metrics = service.get_nerd_metrics(window_hours=24, bucket_minutes=60, now=100.0)

    assert not hasattr(health, "shadow_mode_enabled")
    assert not hasattr(health, "normalize_execution_enabled")
    assert not hasattr(health, "create_link_execution_enabled")
    assert not hasattr(metrics.curation, "shadow_mode_enabled")
    assert not hasattr(metrics.curation, "normalize_execution_enabled")
    assert not hasattr(metrics.curation, "create_link_execution_enabled")


def test_management_service_health_tolerates_hook_client_count_failures(db_manager, caplog: pytest.LogCaptureFixture) -> None:
    repository = RelationalMemoryRepository(db_manager)
    task_queue = SQLiteTaskQueue(db_manager)

    class _BrokenHookService:
        def get_active_client_count(self) -> int:
            raise RuntimeError("hook_store_unavailable")

    service = ManagementService(
        _ManagementContextStub(
            workspace_id="workspace-a",
            memory_path=db_manager.db_path.parent,
            db_manager=db_manager,
            repository=repository,
            task_queue=task_queue,
        ),
        DaemonControllerView(hook_service=_BrokenHookService()),
    )

    with caplog.at_level(logging.WARNING):
        health = service.get_health()

    assert health.client_count == 0
    assert any("Failed to read active daemon client count" in message for message in caplog.messages)


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
    assert overview.premium_usage.copilot_premium_requests_today == 0
    assert overview.premium_usage.copilot_premium_requests_last_day == 0
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
    assert nerd_metrics.composition.by_content_tag == []
    assert nerd_metrics.composition.by_provenance_tag == []
    assert nerd_metrics.composition.by_type == []
    assert nerd_metrics.composition.by_status == []
    assert [bucket.count for bucket in nerd_metrics.distributions.created_age_buckets] == [0, 0, 0, 0, 0]
    assert [bucket.count for bucket in nerd_metrics.distributions.updated_age_buckets] == [0, 0, 0, 0, 0]
    assert [bucket.count for bucket in nerd_metrics.distributions.content_size_buckets] == [0, 0, 0, 0, 0]
    assert nerd_metrics.timelines.memory_activity == []
    assert nerd_metrics.maintenance.events == []
    assert nerd_metrics.lifecycle_trends.status_events == []
    assert nerd_metrics.lifecycle_trends.never_surfaced_backlog
    assert nerd_metrics.lifecycle_trends.cold_tail
    assert nerd_metrics.lifecycle_trends.quality_signals == []
    assert nerd_metrics.quality_drilldown.signals == []
    assert nerd_metrics.quality_remediation.stats == []
    assert nerd_metrics.quality_remediation.activity == []
    assert [stat.value for stat in nerd_metrics.provider_policy.stats] == [0.0, 0.0, 0.0, 0.0]
    assert nerd_metrics.provider_policy.by_task == []
    assert nerd_metrics.provider_policy.by_provider == []
    assert all(bucket.count == 0 for bucket in nerd_metrics.lifecycle_trends.never_surfaced_backlog)
    assert all(bucket.count == 0 for bucket in nerd_metrics.lifecycle_trends.cold_tail)
    assert nerd_metrics.growth_dynamics.top_tag_trends == []
    assert nerd_metrics.growth_dynamics.workspace_contribution_share == []
    assert nerd_metrics.maintenance_summary.by_family == []
    assert nerd_metrics.maintenance_summary.by_agent == []
    assert nerd_metrics.maintenance_summary.family_delta_series == []
    assert nerd_metrics.agent_throughput == []
    assert nerd_metrics.provider_latency == []


def test_extract_copilot_premium_requests_preserves_fractional_values() -> None:
    assert extract_copilot_premium_requests(
        response_text='{"type":"result","usage":{"premiumRequests":0.33}}',
        parsed_json=None,
    ) == pytest.approx(0.33)

    assert extract_copilot_premium_requests(
        response_text='{"type":"message"}\n{"type":"result","usage":{"premiumRequests":1.67}}',
        parsed_json=None,
    ) == pytest.approx(1.67)


def test_management_service_overview_preserves_fractional_premium_usage(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    task_queue = SQLiteTaskQueue(db_manager)
    service = _build_management_service(
        db_manager,
        workspace_id="workspace-a",
        repository=repository,
        task_queue=task_queue,
    )

    now = time.time()
    db_manager.get_connection().execute(
        "INSERT INTO ai_conversations (request_id, attempt, workspace_id, task_name, task_id, provider_key, provider_name, model_name, subprocess_pid, prompt_text, response_text, parsed_json, status, error_text, started_at, completed_at, duration_seconds) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "req-fractional-premium",
            1,
            "workspace-a",
            "memory-curator",
            "task-fractional-premium",
            "copilot-cli",
            "Copilot CLI",
            "claude-haiku-4.5",
            4444,
            "prompt",
            '{"type":"result","usage":{"premiumRequests":0.33}}',
            None,
            "success",
            None,
            now - 30.0,
            now - 10.0,
            20.0,
        ),
    )
    db_manager.get_connection().commit()

    overview = service.get_overview()

    assert overview.premium_usage.copilot_premium_requests_today == pytest.approx(0.33)
    assert overview.premium_usage.copilot_premium_requests_last_day == pytest.approx(0.33)


def test_management_service_task_sampling_summary_aggregates_recent_run_utility(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    task_queue = SQLiteTaskQueue(db_manager)
    service = _build_management_service(
        db_manager,
        workspace_id="workspace-a",
        repository=repository,
        task_queue=task_queue,
    )

    task_one = task_queue.enqueue(
        "memory-curator",
        task_id="sampling-run-1",
        workspace_id="workspace-a",
        available_at=0.0,
    )
    task_two = task_queue.enqueue(
        "memory-curator",
        task_id="sampling-run-2",
        workspace_id="workspace-a",
        available_at=0.0,
    )
    task_three = task_queue.enqueue(
        "graph-linker",
        task_id="sampling-run-3",
        workspace_id="workspace-a",
        available_at=0.0,
    )

    assert task_queue.claim_next(now=10.0) is not None
    task_queue.complete(
        task_one.id,
        completed_at=12.0,
        run_result={
            "strategy_used": "semantic",
            "strategy_fallback_reason": "insufficient_candidates",
            "strategy_selection_mode": "utility_priors",
            "strategy_selection_reason": "utility prior tie-break",
            "candidate_count": 6,
            "tool_calls_executed": 4,
            "mutations": 2,
            "grouping_strategy_used": "compatibility",
        },
    )
    assert task_queue.claim_next(now=13.0) is not None
    task_queue.complete(
        task_two.id,
        completed_at=15.0,
        run_result={
            "strategy_used": "semantic",
            "strategy_selection_mode": "deterministic_scores",
            "strategy_selection_reason": "selected=semantic from deterministic scores",
            "candidate_count": 4,
            "tool_calls_executed": 1,
            "mutations": 0,
            "grouping_strategy_used": "compatibility",
            "grouping_fallback_reason": "single_group",
        },
    )
    assert task_queue.claim_next(now=16.0) is not None
    task_queue.complete(
        task_three.id,
        completed_at=18.0,
        run_result={
            "strategy_used": "lexical",
            "candidate_count": 5,
            "tool_calls_executed": 2,
            "mutations": 1,
        },
    )

    summary = service.get_task_sampling_summary(limit=10)

    assert [(row.name, row.runs, row.fallbacks) for row in summary.selection] == [
        ("semantic", 2, 1),
        ("lexical", 1, 0),
    ]
    assert [(row.name, row.runs, row.fallbacks) for row in summary.grouping] == [
        ("compatibility", 2, 1),
    ]
    assert [
        (
            row.task_name,
            row.strategy_used,
            row.runs,
            row.fallback_count,
            row.mutation_runs,
            row.total_mutations,
            row.total_tool_calls,
            row.average_candidate_count,
            row.mutation_rate,
            row.mutations_per_run,
            row.mutations_per_tool_call,
            row.no_op_runs,
            row.no_op_rate,
        )
        for row in summary.selection_utility
    ] == [
        ("graph-linker", "lexical", 1, 0, 1, 1, 2, 5.0, 1.0, 1.0, 0.5, 0, 0.0),
        ("memory-curator", "semantic", 2, 1, 1, 2, 5, 5.0, 0.5, 1.0, 0.4, 1, 0.5),
    ]
    assert [
        (
            row.task_name,
            row.strategy_selection_mode,
            row.strategy_used,
            row.reason_family,
            row.runs,
            row.fallback_count,
            row.total_mutations,
            row.total_tool_calls,
            row.average_candidate_count,
            row.no_op_runs,
            row.no_op_rate,
        )
        for row in summary.selector_behavior
    ] == [
        ("graph-linker", "unspecified", "lexical", "unknown", 1, 0, 1, 2, 5.0, 0, 0.0),
        ("memory-curator", "deterministic_scores", "semantic", "deterministic_signals", 1, 0, 0, 1, 4.0, 1, 1.0),
        ("memory-curator", "utility_priors", "semantic", "fallback", 1, 1, 2, 4, 6.0, 0, 0.0),
    ]


def test_management_service_health_reports_storage_backend(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    task_queue = SQLiteTaskQueue(db_manager)
    service = ManagementService(
        ApplicationContext(
            workspace_id="workspace-a",
            memory_path=db_manager.db_path.parent,
            db_manager=db_manager,
            repository=repository,
            task_queue=task_queue,
            storage_backend="sqlite",
        ),
        SimpleNamespace(has_runtime=True, client_count=1),
    )

    health = service.get_health()

    assert health.storage_backend == "sqlite"


def test_build_embedding_status_surfaces_blocked_fallback_persistence_policy_for_postgres() -> None:
    payload = build_embedding_status(
        SimpleNamespace(
            status=lambda: EmbedderStatus(
                model_name="hash:sentence-transformers/all-MiniLM-L6-v2",
                configured_model_name="sentence-transformers/all-MiniLM-L6-v2",
                backend="fallback",
                model_cached=False,
            )
        ),
        storage_backend="postgres",
        vector_store=SimpleNamespace(
            get_write_policy_state=lambda: SimpleNamespace(
                fallback_persistence_policy="blocked",
                blocked_fallback_write_count=2,
                last_blocked_fallback_model_name="hash:sentence-transformers/all-MiniLM-L6-v2",
            )
        ),
    )

    assert payload.model_name == "hash:sentence-transformers/all-MiniLM-L6-v2"
    assert payload.configured_model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert payload.backend == "fallback"
    assert payload.fallback_persistence_policy == "blocked"
    assert payload.blocked_fallback_write_count == 2
    assert payload.last_blocked_fallback_model_name == "hash:sentence-transformers/all-MiniLM-L6-v2"


def test_management_service_health_and_overview_surface_embedding_write_policy(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    task_queue = SQLiteTaskQueue(db_manager)
    fake_embedder = SimpleNamespace(
        status=lambda: EmbedderStatus(
            model_name="hash:sentence-transformers/all-MiniLM-L6-v2",
            configured_model_name="sentence-transformers/all-MiniLM-L6-v2",
            backend="fallback",
            model_cached=False,
        )
    )
    fake_vector_store = SimpleNamespace(
        get_write_policy_state=lambda: SimpleNamespace(
            fallback_persistence_policy="blocked",
            blocked_fallback_write_count=3,
            last_blocked_fallback_model_name="hash:sentence-transformers/all-MiniLM-L6-v2",
        )
    )
    service = ManagementService(
        ApplicationContext(
            workspace_id="workspace-a",
            memory_path=db_manager.db_path.parent,
            db_manager=db_manager,
            repository=repository,
            task_queue=task_queue,
            provider_usage=SimpleNamespace(
                summarize_usage=lambda workspace_id=None: [],
                list_conversations=lambda **kwargs: [],
            ),
            runtime_logs=SimpleNamespace(list_logs=lambda **kwargs: []),
            storage_backend="postgres",
            embedder=fake_embedder,
            vector_store=fake_vector_store,
        ),
        SimpleNamespace(has_runtime=True, client_count=1),
    )

    health = service.get_health()
    overview = service.get_overview()

    assert health.embeddings.configured_model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert health.embeddings.fallback_persistence_policy == "blocked"
    assert health.embeddings.blocked_fallback_write_count == 3
    assert health.embeddings.last_blocked_fallback_model_name == "hash:sentence-transformers/all-MiniLM-L6-v2"
    assert overview.embeddings.fallback_persistence_policy == "blocked"
    assert overview.embeddings.blocked_fallback_write_count == 3


def test_management_service_surfaces_persisted_embedding_integrity_event_summary(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    task_queue = SQLiteTaskQueue(db_manager)
    service = _build_management_service(
        db_manager,
        workspace_id="workspace-a",
        repository=repository,
        task_queue=task_queue,
    )

    db_manager.get_connection().executemany(
        """
        INSERT INTO embedding_integrity_events (
            workspace_id, event_kind, model_name, source_kind, source_id,
            scanned_row_count, invalid_row_count, mixed_dimension_group_count,
            details_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                "workspace-a",
                "integrity_scan_summary",
                "mini-embed",
                None,
                None,
                5,
                2,
                1,
                '{"expected_dimension":2}',
                100.0,
            ),
            (
                "workspace-a",
                "blocked_fallback_write",
                "hash:sentence-transformers/all-MiniLM-L6-v2",
                "memory",
                "memory-a",
                None,
                None,
                None,
                '{"reason":"fallback_embedding_persistence_blocked"}',
                101.0,
            ),
            (
                "workspace-b",
                "blocked_fallback_write",
                "hash:other-model",
                "memory",
                "memory-b",
                None,
                None,
                None,
                '{}',
                102.0,
            ),
        ],
    )
    db_manager.get_connection().commit()

    health = service.get_health()
    overview = service.get_overview()

    assert health.embeddings.integrity_events.total == 3
    assert health.embeddings.integrity_events.by_kind == {
        "blocked_fallback_write": 2,
        "integrity_scan_summary": 1,
    }
    assert health.embeddings.integrity_events.last_scan is not None
    assert health.embeddings.integrity_events.last_scan.model_name == "mini-embed"
    assert health.embeddings.integrity_events.last_scan.scanned_row_count == 5
    assert health.embeddings.integrity_events.last_scan.invalid_row_count == 2
    assert health.embeddings.integrity_events.last_scan.mixed_dimension_group_count == 1
    assert health.embeddings.integrity_events.last_blocked_fallback_write is not None
    assert health.embeddings.integrity_events.last_blocked_fallback_write.model_name == "hash:other-model"
    assert health.embeddings.integrity_events.last_blocked_fallback_write.source_id == "memory-b"

    assert overview.embeddings.integrity_events.total == 3
    assert overview.embeddings.integrity_events.by_kind == {
        "blocked_fallback_write": 2,
        "integrity_scan_summary": 1,
    }
    assert overview.embeddings.integrity_events.last_blocked_fallback_write is not None
    assert overview.embeddings.integrity_events.last_blocked_fallback_write.model_name == "hash:other-model"
    assert overview.embeddings.integrity_events.last_blocked_fallback_write.source_id == "memory-b"


def test_management_service_health_uses_global_embedding_integrity_summary_even_with_workspace_context(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    task_queue = SQLiteTaskQueue(db_manager)
    service = _build_management_service(
        db_manager,
        workspace_id="workspace-a",
        repository=repository,
        task_queue=task_queue,
    )

    db_manager.get_connection().executemany(
        """
        INSERT INTO embedding_integrity_events (
            workspace_id, event_kind, model_name, source_kind, source_id,
            scanned_row_count, invalid_row_count, mixed_dimension_group_count,
            details_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                None,
                "integrity_scan_summary",
                "mini-embed",
                None,
                None,
                5,
                2,
                1,
                '{"expected_dimension":2}',
                100.0,
            ),
            (
                None,
                "blocked_fallback_write",
                "hash:sentence-transformers/all-MiniLM-L6-v2",
                "memory",
                "memory-a",
                None,
                None,
                None,
                '{"reason":"fallback_embedding_persistence_blocked"}',
                101.0,
            ),
            (
                None,
                "blocked_fallback_write",
                "hash:other-model",
                "memory",
                "memory-b",
                None,
                None,
                None,
                '{}',
                102.0,
            ),
        ],
    )
    db_manager.get_connection().commit()

    health = service.get_health()
    overview = service.get_overview()

    assert health.embeddings.integrity_events.total == 3
    assert health.embeddings.integrity_events.by_kind == {
        "blocked_fallback_write": 2,
        "integrity_scan_summary": 1,
    }
    assert health.embeddings.integrity_events.last_blocked_fallback_write is not None
    assert health.embeddings.integrity_events.last_blocked_fallback_write.model_name == "hash:other-model"
    assert health.embeddings.integrity_events.last_blocked_fallback_write.source_id == "memory-b"
    assert overview.embeddings.integrity_events.total == 3
    assert overview.embeddings.integrity_events.by_kind == health.embeddings.integrity_events.by_kind
    assert overview.embeddings.integrity_events.last_blocked_fallback_write is not None
    assert overview.embeddings.integrity_events.last_blocked_fallback_write.model_name == "hash:other-model"


@pytest.mark.parametrize(
    ("backend", "cache_enabled", "cache_mode", "read_cache_path", "expected_state", "expected_path_suffix"),
    [
        ("sqlite", False, "readonly", None, "disabled", None),
        ("sqlite", True, "readonly", None, "unsupported_backend", None),
        ("postgres", True, "writeback", None, "inactive", "cache/shared_read_cache.sqlite3"),
        ("postgres", True, "writeback", "active-cache.sqlite3", "active", "active-cache.sqlite3"),
        ("postgres", True, "readonly", None, "inactive", "cache/shared_read_cache.sqlite3"),
        ("postgres", True, "readonly", "active-cache.sqlite3", "active", "active-cache.sqlite3"),
    ],
)
def test_management_service_health_reports_cache_state(
    db_manager,
    tmp_path,
    backend: str,
    cache_enabled: bool,
    cache_mode: StorageCacheMode,
    read_cache_path: str | None,
    expected_state: str,
    expected_path_suffix: str | None,
) -> None:
    repository = RelationalMemoryRepository(db_manager)
    task_queue = SQLiteTaskQueue(db_manager)
    config = Config()
    config.storage.cache.enabled = cache_enabled
    config.storage.cache.mode = cache_mode
    memory_path = tmp_path / "memories"
    read_cache = None if read_cache_path is None else SimpleNamespace(_db_path=tmp_path / read_cache_path)

    service = ManagementService(
        ApplicationContext(
            config=config,
            workspace_id="workspace-a",
            memory_path=memory_path,
            db_manager=db_manager,
            repository=repository,
            task_queue=task_queue,
            storage_backend=backend,
            read_cache=read_cache,
        ),
        SimpleNamespace(has_runtime=True, client_count=1),
    )

    health = service.get_health()

    assert health.cache.enabled is cache_enabled
    assert health.cache.mode == (cache_mode if cache_enabled else None)
    assert health.cache.state == expected_state
    if expected_path_suffix is None:
        assert health.cache.path is None
    else:
        assert health.cache.path is not None
        assert health.cache.path.endswith(expected_path_suffix)


def test_management_service_surfaces_cache_metrics_in_health_and_overview(db_manager, tmp_path) -> None:
    repository = RelationalMemoryRepository(db_manager)
    task_queue = SQLiteTaskQueue(db_manager)
    config = Config()
    config.storage.cache.enabled = True
    config.storage.cache.mode = "readonly"
    memory_path = tmp_path / "memories"
    read_cache = SharedReadCache(tmp_path / "cache" / "shared_read_cache.sqlite3")

    read_cache.increment_metric("search_requests", amount=5)
    read_cache.increment_metric("fresh_exact_search_hits", amount=2)
    read_cache.increment_metric("stale_exact_search_fallbacks", amount=1)
    read_cache.increment_metric("read_requests", amount=4)
    read_cache.increment_metric("validated_read_hits", amount=3)
    read_cache.increment_metric("warmed_projection_rows", amount=7)

    service = ManagementService(
        ApplicationContext(
            config=config,
            workspace_id="workspace-a",
            memory_path=memory_path,
            db_manager=db_manager,
            repository=repository,
            task_queue=task_queue,
            provider_usage=SimpleNamespace(
                summarize_usage=lambda workspace_id=None: [],
                list_conversations=lambda **kwargs: [],
            ),
            runtime_logs=SimpleNamespace(list_logs=lambda **kwargs: []),
            storage_backend="postgres",
            read_cache=read_cache,
        ),
        SimpleNamespace(has_runtime=True, client_count=1),
    )

    health = service.get_health()
    overview = service.get_overview()

    assert health.cache.state == "active"
    assert health.cache.metrics.search_requests == 5
    assert health.cache.metrics.fresh_exact_search_hits == 2
    assert health.cache.metrics.stale_exact_search_fallbacks == 1
    assert health.cache.metrics.read_requests == 4
    assert health.cache.metrics.validated_read_hits == 3
    assert health.cache.metrics.warmed_projection_rows == 7
    assert health.cache.metrics.fresh_exact_search_hit_rate == 0.4
    assert health.cache.metrics.validated_read_hit_rate == 0.75
    assert health.cache.metrics.recent.window_minutes == 15
    assert health.cache.metrics.recent.search_requests == 5
    assert health.cache.metrics.recent.stale_exact_search_fallbacks == 1
    assert health.cache.metrics.recent.read_requests == 4
    assert health.cache.metrics.recent.validated_read_hits == 3
    assert health.cache.metrics.recent.warmed_projection_rows == 7
    assert health.cache.metrics.recent.fresh_exact_search_hit_rate == 0.4
    assert health.cache.metrics.recent.validated_read_hit_rate == 0.75

    assert overview.cache.state == "active"
    assert overview.cache.metrics.search_requests == 5
    assert overview.cache.metrics.read_requests == 4
    assert overview.cache.metrics.warmed_projection_rows == 7
    assert overview.cache.metrics.recent.search_requests == 5
    assert overview.cache.metrics.recent.read_requests == 4
    assert overview.cache.metrics.recent.warmed_projection_rows == 7


def test_management_service_uses_postgres_runtime_log_repository_for_postgres_backend(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeRuntimeLogs:
        @property
        def retention_policy(self):
            return __import__("mcp_memory.config", fromlist=["LoggingConfig"]).LoggingConfig()

        def list_logs(self, **kwargs):
            captured["list_logs_kwargs"] = kwargs
            return []

        def summarize_logs(self, **kwargs):
            captured["summarize_logs_kwargs"] = kwargs
            return __import__("mcp_memory.runtime_log_store", fromlist=["RuntimeLogSummary"]).RuntimeLogSummary(total=0)

        def prune_logs(self, **kwargs):
            captured["prune_logs_kwargs"] = kwargs
            return 0

    monkeypatch.setattr(
        "mcp_memory.management.runtime_health_service.build_execution_attempt_health",
        lambda db_manager: ExecutionAttemptHealthPayload(),
    )

    db_manager = SimpleNamespace(db_path=None)
    service = ManagementService(
        ApplicationContext(
            workspace_id="workspace-a",
            db_manager=db_manager,
            storage_backend="postgres",
            runtime_logs=FakeRuntimeLogs(),
            provider_usage=SimpleNamespace(list_conversations=lambda **kwargs: []),
        ),
        SimpleNamespace(has_runtime=False, client_count=0),
    )

    assert service.get_health().storage_backend == "postgres"
    assert service.list_logs().logs == []
    assert service.summarize_logs().total == 0
    assert service.prune_logs().deleted == 0
    assert captured["list_logs_kwargs"] == {
        "workspace_id": "workspace-a",
        "level": None,
        "logger_name": None,
        "source": None,
        "query": None,
        "after": None,
        "before": None,
        "limit": 50,
    }


def test_operator_health_snapshot_includes_memory_tool_latency_summary(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    task_queue = SQLiteTaskQueue(db_manager)
    memory = repository.create_memory(
        title="Latency summary seed",
        content="Seed memory for read telemetry rows.",
        workspace_ids=["workspace-a"],
        memory_type="fact",
        tags=["telemetry"],
        memory_id="memory-1",
    )
    assert memory is not None
    service = _build_management_service(
        db_manager,
        workspace_id="workspace-a",
        repository=repository,
        task_queue=task_queue,
    )

    now = time.time()
    db_manager.get_connection().executemany(
        """
        INSERT INTO memory_tool_events (
            invocation_id, workspace_id, caller_kind, event_kind, memory_id, query_text,
            result_rank, result_count, duration_ms, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            ("search-1", "workspace-a", "external", "search", None, "alpha", None, 2, 120.0, now - 5),
            ("search-1", "workspace-a", "external", "search", "memory-1", "alpha", 1, 2, 120.0, now - 5),
            ("search-2", "workspace-a", "external", "search", None, "beta", None, 0, 2400.0, now - 4),
            ("read-1", "workspace-a", "external", "read", "memory-1", None, None, None, 35.0, now - 3),
        ],
    )
    db_manager.get_connection().commit()

    snapshot = service.get_operator_health_snapshot(log_window_minutes=15)

    metrics = {metric.event_kind: metric for metric in snapshot.tool_latency.by_event_kind}
    assert snapshot.tool_latency.window_minutes == 15
    assert metrics["search"].count == 2
    assert metrics["search"].slow_count == 1
    assert metrics["search"].max_duration_ms == 2400.0
    assert metrics["read"].count == 1
    assert metrics["read"].avg_duration_ms == 35.0


def test_management_service_overview_is_global_even_from_workspace_context(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    task_queue = SQLiteTaskQueue(db_manager)

    memory_a = repository.create_memory(
        title="Workspace A fact",
        content="Only workspace A should see this by default.",
        workspace_ids=["workspace-a"],
        memory_type="fact",
        created_at="1970-01-01T00:00:01+00:00",
        updated_at="1970-01-01T00:00:01+00:00",
    )
    memory_b = repository.create_memory(
        title="Workspace B fact",
        content="Only global scope or workspace B should see this.",
        workspace_ids=["workspace-b"],
        memory_type="fact",
        created_at="1970-01-01T00:00:02+00:00",
        updated_at="1970-01-01T00:00:02+00:00",
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
    now = time.time()
    db_manager.get_connection().executemany(
        "INSERT INTO ai_conversations (request_id, attempt, workspace_id, task_name, task_id, provider_key, provider_name, model_name, subprocess_pid, prompt_text, response_text, parsed_json, status, error_text, started_at, completed_at, duration_seconds) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                "req-a",
                1,
                "workspace-a",
                "memory-curator",
                "task-a",
                "copilot-cli",
                "Copilot CLI",
                "gpt-5",
                111,
                "prompt",
                '{"type":"result","usage":{"premiumRequests":3}}',
                None,
                "success",
                None,
                now - 120.0,
                now - 60.0,
                60.0,
            ),
            (
                "req-b",
                1,
                "workspace-b",
                "memory-curator",
                "task-b",
                "copilot-cli",
                "Copilot CLI",
                "gpt-5",
                222,
                "prompt",
                '{"type":"result","usage":{"premiumRequests":5}}',
                None,
                "success",
                None,
                now - 180.0,
                now - 90.0,
                90.0,
            ),
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

    assert scoped_overview.memories.total == 2
    assert scoped_overview.memory_metrics.total_memories == 2
    assert scoped_overview.premium_usage.copilot_premium_requests_today == 8
    assert scoped_overview.premium_usage.copilot_premium_requests_last_day == 8
    assert {record.title for record in scoped_overview.recent_memories} == {"Workspace A fact", "Workspace B fact"}
    assert {item.provider_key for item in scoped_overview.provider_usage} == {"gemini-cli", "copilot-mini"}
    assert [(item.key, item.count) for item in scoped_nerd.composition.by_workspace] == [("workspace-a", 1)]
    assert [item.task_id for item in scoped_nerd.maintenance.events] == [task_a.id]
    assert [item.key for item in scoped_nerd.growth_dynamics.workspace_contribution_share] == ["workspace-a"]
    assert scoped_nerd.growth_dynamics.workspace_contribution_share[0].buckets[-1].count == 1

    assert global_overview.memories.total == 2
    assert global_overview.memory_metrics.total_memories == 2
    assert global_overview.premium_usage.copilot_premium_requests_today == 8
    assert global_overview.premium_usage.copilot_premium_requests_last_day == 8
    assert {record.title for record in global_overview.recent_memories} == {"Workspace A fact", "Workspace B fact"}
    assert {item.provider_key for item in global_overview.provider_usage} == {"gemini-cli", "copilot-mini"}
    assert [(item.key, item.count) for item in global_nerd.composition.by_workspace] == [
        ("workspace-a", 1),
        ("workspace-b", 1),
    ]
    assert [item.task_id for item in global_nerd.maintenance.events] == [task_b.id, task_a.id]
    assert [item.key for item in global_nerd.growth_dynamics.workspace_contribution_share] == ["workspace-a", "workspace-b"]
    assert [item.buckets[-1].count for item in global_nerd.growth_dynamics.workspace_contribution_share] == [1, 1]


def test_management_service_nerd_metrics_accepts_global_and_workspace_overrides(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    task_queue = SQLiteTaskQueue(db_manager)

    repository.create_memory(
        title="Workspace A fact",
        content="Scoped to workspace A.",
        workspace_ids=["workspace-a"],
        memory_type="fact",
        created_at="1970-01-01T00:00:01+00:00",
        updated_at="1970-01-01T00:00:01+00:00",
    )
    repository.create_memory(
        title="Workspace B fact",
        content="Scoped to workspace B.",
        workspace_ids=["workspace-b"],
        memory_type="fact",
        created_at="1970-01-01T00:00:02+00:00",
        updated_at="1970-01-01T00:00:02+00:00",
    )

    db_manager.get_connection().executemany(
        "INSERT INTO provider_usage (workspace_id, task_name, provider_key, provider_name, model_name, status, duration_seconds, created_at, error_text) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("workspace-a", "memory-curator", "gemini-cli", "Gemini CLI", "gemini-3-flash-preview", "success", 0.25, 10.0, None),
            ("workspace-b", "memory-curator", "copilot-mini", "Copilot CLI", "gpt-5-mini", "success", 0.5, 20.0, None),
        ],
    )
    db_manager.get_connection().commit()

    service = _build_management_service(
        db_manager,
        workspace_id="workspace-a",
        repository=repository,
        task_queue=task_queue,
    )

    default_metrics = service.get_nerd_metrics(window_hours=24, bucket_minutes=60, now=100.0)
    global_metrics = service.get_nerd_metrics(scope="global", window_hours=24, bucket_minutes=60, now=100.0)
    workspace_override_metrics = service.get_nerd_metrics(
        scope="workspace",
        workspace_id="workspace-b",
        window_hours=24,
        bucket_minutes=60,
        now=100.0,
    )

    assert default_metrics.graph_topology.total_memories == 1
    assert [(item.key, item.count) for item in default_metrics.composition.by_workspace] == [("workspace-a", 1)]
    assert [item.provider_key for item in default_metrics.provider_latency] == ["gemini-cli"]

    assert global_metrics.graph_topology.total_memories == 2
    assert [(item.key, item.count) for item in global_metrics.composition.by_workspace] == [
        ("workspace-a", 1),
        ("workspace-b", 1),
    ]
    assert {item.provider_key for item in global_metrics.provider_latency} == {"gemini-cli", "copilot-mini"}

    assert workspace_override_metrics.graph_topology.total_memories == 1
    assert [(item.key, item.count) for item in workspace_override_metrics.composition.by_workspace] == [("workspace-b", 1)]
    assert [item.provider_key for item in workspace_override_metrics.provider_latency] == ["copilot-mini"]


def test_management_service_resolve_scoped_workspace_id_preserves_service_scoped_default_semantics(db_manager) -> None:
    service = _build_management_service(
        db_manager,
        workspace_id="workspace-a",
        repository=RelationalMemoryRepository(db_manager),
        task_queue=SQLiteTaskQueue(db_manager),
    )

    assert service._resolve_scoped_workspace_id(scope=None, workspace_id=None) == "workspace-a"
    assert service._resolve_scoped_workspace_id(scope="global", workspace_id=None) is None
    assert service._resolve_scoped_workspace_id(scope="workspace", workspace_id=None) == "workspace-a"
    assert service._resolve_scoped_workspace_id(scope="workspace", workspace_id="workspace-b") == "workspace-b"


def test_management_service_resolve_scoped_workspace_id_requires_workspace_context_for_workspace_scope(db_manager) -> None:
    service = _build_management_service(
        db_manager,
        workspace_id=None,
        repository=RelationalMemoryRepository(db_manager),
        task_queue=SQLiteTaskQueue(db_manager),
    )

    with pytest.raises(ValueError, match="workspace_id_required_for_workspace_scope"):
        service._resolve_scoped_workspace_id(scope="workspace", workspace_id=None)


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


def test_management_service_classifies_provenance_process_tags_and_splits_composition(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    task_queue = SQLiteTaskQueue(db_manager)

    repository.create_memory(
        title="Tagged fact",
        content="Tagged content.",
        workspace_ids=["workspace-a"],
        tags=[
            "alpha",
            "system1",
            "system1-appended",
            "auto-ingested",
            "auto-defragmented",
            "merged-by-deduplicator",
            "deduplicator-task-123",
            "planner-session",
            "code-session",
            "workflow",
            "testing-strategy",
            "user-preferences",
            "coding-standards",
            "documentation-style",
            "memory-operating-model",
            "startup",
        ],
        memory_type="fact",
    )
    repository.create_memory(
        title="Second tagged fact",
        content="More tagged content.",
        workspace_ids=["workspace-a"],
        tags=["alpha", "beta", "manual-review"],
        memory_type="fact",
    )

    service = _build_management_service(
        db_manager,
        workspace_id="workspace-a",
        repository=repository,
        task_queue=task_queue,
    )

    nerd_metrics = service.get_nerd_metrics(window_hours=24, bucket_minutes=60, now=100.0)

    assert is_provenance_process_tag("system1") is True
    assert is_provenance_process_tag("system1-appended") is True
    assert is_provenance_process_tag("auto-ingested") is True
    assert is_provenance_process_tag("auto-defragmented") is True
    assert is_provenance_process_tag("merged-by-deduplicator") is True
    assert is_provenance_process_tag("deduplicator-task-123") is True
    assert is_provenance_process_tag("planner-session") is True
    assert is_provenance_process_tag("code-session") is True
    assert is_provenance_process_tag("workflow") is True
    assert is_provenance_process_tag("testing-strategy") is True
    assert is_provenance_process_tag("user-preferences") is True
    assert is_provenance_process_tag("coding-standards") is True
    assert is_provenance_process_tag("documentation-style") is True
    assert is_provenance_process_tag("memory-operating-model") is True
    assert is_provenance_process_tag("startup") is True
    assert is_provenance_process_tag("alpha") is False

    assert [(item.key, item.count) for item in nerd_metrics.composition.by_content_tag] == [
        ("alpha", 2),
        ("beta", 1),
        ("manual-review", 1),
    ]
    assert [(item.key, item.count) for item in nerd_metrics.composition.by_provenance_tag] == [
        ("auto-defragmented", 1),
        ("auto-ingested", 1),
        ("code-session", 1),
        ("coding-standards", 1),
        ("deduplicator-task-123", 1),
        ("documentation-style", 1),
        ("memory-operating-model", 1),
        ("merged-by-deduplicator", 1),
        ("planner-session", 1),
        ("startup", 1),
        ("other", 5),
    ]
    assert [(item.key, item.count) for item in nerd_metrics.composition.by_tag] == [
        ("alpha", 2),
        ("auto-defragmented", 1),
        ("auto-ingested", 1),
        ("beta", 1),
        ("code-session", 1),
        ("coding-standards", 1),
        ("deduplicator-task-123", 1),
        ("documentation-style", 1),
        ("manual-review", 1),
        ("memory-operating-model", 1),
        ("other", 8),
    ]


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
    verification_family = next(item for item in nerd_metrics.maintenance_summary.by_family if item.key == "verification")
    graph_linker_yield = next(item for item in nerd_metrics.maintenance_summary.by_agent if item.key == "graph-linker")
    assert verification_family.total_runs == 3
    assert verification_family.completed_runs == 1
    assert verification_family.failed_runs == 1
    assert verification_family.retry_runs == 1
    assert verification_family.merged_count == 2
    assert verification_family.delta_total == 2
    assert graph_linker_yield.family_key == "verification"
    assert graph_linker_yield.delta_per_completed_run == 2.0
    assert graph_linker_yield.lines_per_completed_run == 7.0
    assert nerd_metrics.lifecycle_trends.status_events == []
    assert any(stat.key == "provider_failure_rate" and stat.value == 0.5 for stat in nerd_metrics.stats)
    assert any(alert.key == "provider_failure_rate" for alert in nerd_metrics.alerts)

    assert zero_window_metrics.agent_throughput == []
    assert zero_window_metrics.provider_latency == []
    assert zero_window_metrics.maintenance.events == []
    assert zero_window_metrics.lifecycle_trends.status_events == []
    assert zero_window_metrics.growth_dynamics.top_tag_trends == []
    assert zero_window_metrics.maintenance_summary.by_family == []
    assert any(stat.key == "runs_last_window" and stat.value == 0.0 for stat in zero_window_metrics.stats)
    assert any(stat.key == "provider_calls_last_window" and stat.value == 0.0 for stat in zero_window_metrics.stats)


def test_management_service_nerd_metrics_additive_trends_and_maintenance_summary(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    task_queue = SQLiteTaskQueue(db_manager)
    now = datetime(2026, 3, 19, 12, 0, tzinfo=UTC)

    alpha_memory = repository.create_memory(
        title="Alpha shared",
        content="alpha",
        workspace_ids=["workspace-a", "workspace-b"],
        tags=["alpha"],
        memory_type="fact",
        created_at=(now - timedelta(days=6)).isoformat(),
        updated_at=(now - timedelta(days=6)).isoformat(),
    )
    beta_memory = repository.create_memory(
        title="Beta",
        content="beta",
        workspace_ids=["workspace-a"],
        tags=["beta"],
        memory_type="fact",
        created_at=(now - timedelta(days=5)).isoformat(),
        updated_at=(now - timedelta(days=5)).isoformat(),
    )
    gamma_memory = repository.create_memory(
        title="Gamma",
        content="gamma",
        workspace_ids=["workspace-a"],
        tags=["gamma"],
        memory_type="fact",
        created_at=(now - timedelta(days=4)).isoformat(),
        updated_at=(now - timedelta(days=4)).isoformat(),
    )
    delta_memory = repository.create_memory(
        title="Delta",
        content="delta",
        workspace_ids=["workspace-a"],
        tags=["delta"],
        memory_type="fact",
        created_at=(now - timedelta(days=3)).isoformat(),
        updated_at=(now - timedelta(days=3)).isoformat(),
    )
    epsilon_memory = repository.create_memory(
        title="Epsilon",
        content="epsilon",
        workspace_ids=["workspace-a"],
        tags=["epsilon"],
        memory_type="fact",
        created_at=(now - timedelta(days=2)).isoformat(),
        updated_at=(now - timedelta(days=2)).isoformat(),
    )
    zeta_memory = repository.create_memory(
        title="Zeta",
        content="zeta",
        workspace_ids=["workspace-a"],
        tags=["zeta"],
        memory_type="fact",
        created_at=(now - timedelta(days=1)).isoformat(),
        updated_at=(now - timedelta(days=1)).isoformat(),
    )
    foreign_eta_memory = repository.create_memory(
        title="Foreign eta",
        content="eta",
        workspace_ids=["workspace-b"],
        tags=["eta"],
        memory_type="fact",
        created_at=(now - timedelta(hours=12)).isoformat(),
        updated_at=(now - timedelta(hours=12)).isoformat(),
    )
    assert alpha_memory is not None
    assert beta_memory is not None
    assert gamma_memory is not None
    assert delta_memory is not None
    assert epsilon_memory is not None
    assert zeta_memory is not None
    assert foreign_eta_memory is not None

    repository.record_access(
        alpha_memory.id,
        access_score=1.0,
        accessed_at=(now - timedelta(days=5, hours=20)).isoformat(),
        increment_read_count=False,
    )
    db_manager.get_connection().execute(
        "UPDATE memories SET last_surfaced_at = ? WHERE id = ?",
        ((now - timedelta(days=5, hours=18)).isoformat(), beta_memory.id),
    )

    task_project = task_queue.enqueue(
        "project-manager",
        task_id="project-manager-summary-1",
        workspace_id="workspace-a",
        available_at=0.0,
    )
    task_fact = task_queue.enqueue(
        "fact-checker",
        task_id="fact-checker-summary-1",
        workspace_id="workspace-a",
        available_at=0.0,
    )
    task_dedup = task_queue.enqueue(
        "deduplicator",
        task_id="deduplicator-summary-1",
        workspace_id="workspace-a",
        available_at=0.0,
    )
    assert task_queue.claim_next(now=now.timestamp() - 300.0) is not None
    task_queue.complete(
        task_project.id,
        completed_at=(now - timedelta(days=3, hours=12)).timestamp(),
        run_result={"stale": 2, "updated": 2, "meaningful_actions": 2},
    )
    assert task_queue.claim_next(now=now.timestamp() - 200.0) is not None
    task_queue.complete(
        task_fact.id,
        completed_at=(now - timedelta(days=2, hours=12)).timestamp(),
        run_result={"degraded": 1},
    )
    assert task_queue.claim_next(now=now.timestamp() - 100.0) is not None
    task_queue.complete(
        task_dedup.id,
        completed_at=(now - timedelta(days=1, hours=12)).timestamp(),
        run_result={"merged": 3, "archived": 1, "lines_compressed": 9, "meaningful_actions": 4},
    )
    db_manager.get_connection().commit()

    service = _build_management_service(
        db_manager,
        workspace_id="workspace-a",
        repository=repository,
        task_queue=task_queue,
    )

    nerd_metrics = service.get_nerd_metrics(window_hours=24 * 7, bucket_minutes=24 * 60, now=now.timestamp())

    assert [series.key for series in nerd_metrics.lifecycle_trends.status_events] == ["stale", "degraded", "archived"]
    assert [bucket.count for bucket in nerd_metrics.lifecycle_trends.never_surfaced_backlog][-1] == 5
    assert [bucket.count for bucket in nerd_metrics.lifecycle_trends.cold_tail][-1] == 5

    assert [series.key for series in nerd_metrics.growth_dynamics.top_tag_trends] == [
        "alpha",
        "beta",
        "delta",
        "epsilon",
        "gamma",
        "other",
    ]
    assert [series.buckets[-1].count for series in nerd_metrics.growth_dynamics.top_tag_trends] == [1, 1, 1, 1, 1, 1]
    assert [series.key for series in nerd_metrics.growth_dynamics.workspace_contribution_share] == [
        "workspace-a",
        "workspace-b",
    ]
    assert [series.buckets[-1].count for series in nerd_metrics.growth_dynamics.workspace_contribution_share] == [6, 1]
    assert [series.buckets[-1].share for series in nerd_metrics.growth_dynamics.workspace_contribution_share] == [0.8571, 0.1429]

    organization_family = next(item for item in nerd_metrics.maintenance_summary.by_family if item.key == "organization")
    verification_family = next(item for item in nerd_metrics.maintenance_summary.by_family if item.key == "verification")
    compaction_family = next(item for item in nerd_metrics.maintenance_summary.by_family if item.key == "compaction")
    assert organization_family.task_names == ["project-manager"]
    assert organization_family.updated_count == 2
    assert organization_family.delta_total == 2
    assert verification_family.degraded_count == 1
    assert verification_family.delta_total == 1
    assert compaction_family.merged_count == 3
    assert compaction_family.archived_count == 1
    assert compaction_family.lines_compressed == 9
    assert compaction_family.delta_total == 4

    deduplicator_yield = next(item for item in nerd_metrics.maintenance_summary.by_agent if item.key == "deduplicator")
    assert deduplicator_yield.family_key == "compaction"
    assert deduplicator_yield.meaningful_actions == 4
    assert deduplicator_yield.delta_per_completed_run == 4.0
    assert deduplicator_yield.lines_per_completed_run == 9.0

    compaction_series = next(item for item in nerd_metrics.maintenance_summary.family_delta_series if item.key == "compaction")
    assert compaction_series.task_names == ["deduplicator"]
    assert sum(bucket.merged_count for bucket in compaction_series.buckets) == 3
    assert sum(bucket.archived_count for bucket in compaction_series.buckets) == 1


def test_management_service_nerd_metrics_surface_memory_quality_signals(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    task_queue = SQLiteTaskQueue(db_manager)
    now = time.time() + 60.0

    repository.create_memory(
        title="task_complete: deduplicator-1",
        content="Operational closeout should not live here.",
        workspace_ids=["workspace-a"],
        memory_type="fact",
        summary="Concrete summary.",
    )
    repository.create_memory(
        title="Broad observation",
        content="Short but durable note.",
        workspace_ids=["workspace-a"],
        memory_type="observation",
        summary="Covers several related findings.",
    )
    repository.create_memory(
        title="Oversized durable memory",
        content="x" * 4_200,
        workspace_ids=["workspace-a"],
        memory_type="fact",
        summary="Detailed but concrete summary.",
        tags=["durable"],
        metadata={"split_from_memory_id": "source-1", "split_group_id": "split-1"},
    )

    service = _build_management_service(
        db_manager,
        workspace_id="workspace-a",
        repository=repository,
        task_queue=task_queue,
    )

    nerd_metrics = service.get_nerd_metrics(window_hours=24, bucket_minutes=60, now=now)
    stats = {stat.key: stat.value for stat in nerd_metrics.stats}
    alerts = {alert.key: alert for alert in nerd_metrics.alerts}

    assert stats["trace_like_memory_count"] == 1.0
    assert stats["generic_summary_count"] == 1.0
    assert stats["untagged_observation_count"] == 1.0
    assert stats["untagged_observation_rate"] == 1.0
    assert stats["oversized_memory_count"] == 1.0

    quality_signal_keys = [series.key for series in nerd_metrics.lifecycle_trends.quality_signals]
    assert quality_signal_keys == [
        "trace_like_memory_count",
        "generic_summary_count",
        "untagged_observation_count",
        "oversized_memory_count",
    ]
    assert [series.buckets[-1].count for series in nerd_metrics.lifecycle_trends.quality_signals] == [1, 1, 1, 1]

    drilldown_counts = {signal.key: signal.count for signal in nerd_metrics.quality_drilldown.signals}
    assert drilldown_counts == {
        "trace_like_memory_count": 1,
        "generic_summary_count": 1,
        "untagged_observation_count": 1,
        "oversized_memory_count": 1,
    }
    trace_like_records = next(signal.records for signal in nerd_metrics.quality_drilldown.signals if signal.key == "trace_like_memory_count")
    assert trace_like_records[0].title == "task_complete: deduplicator-1"
    oversized_records = next(signal.records for signal in nerd_metrics.quality_drilldown.signals if signal.key == "oversized_memory_count")
    assert oversized_records[0].tags == ["durable"]

    remediation_stats = {stat.key: stat.value for stat in nerd_metrics.quality_remediation.stats}
    assert remediation_stats == {
        "tagged_observation_count": 0.0,
        "concrete_summary_count": 2.0,
        "split_lineage_count": 1.0,
    }
    remediation_activity = {
        series.key: sum(bucket.count for bucket in series.buckets)
        for series in nerd_metrics.quality_remediation.activity
    }
    assert remediation_activity == {
        "concrete_summary_updates": 2,
        "split_lineage_updates": 1,
    }

    assert alerts["trace_like_memory_count"].severity == "warning"
    assert alerts["untagged_observation_rate"].severity == "warning"
    assert alerts["generic_summary_count"].severity == "info"
    assert alerts["oversized_memory_count"].severity == "info"


def test_management_service_lists_prioritized_quality_cleanup_candidates(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    task_queue = SQLiteTaskQueue(db_manager)
    now = time.time() + 120.0

    trace_like = repository.create_memory(
        title="task_complete: cleanup-run-1",
        content="Operational closeout should be curated.",
        workspace_ids=["workspace-a"],
        memory_type="fact",
        summary="Covers several cleanup follow-ups.",
        created_at=datetime.fromtimestamp(now - 300.0, tz=UTC).isoformat(),
        updated_at=datetime.fromtimestamp(now - 60.0, tz=UTC).isoformat(),
    )
    low_conversion = repository.create_memory(
        title="Low conversion memory",
        content="Useful content that still fails to convert when surfaced.",
        workspace_ids=["workspace-a"],
        memory_type="fact",
        summary="Concrete summary.",
        tags=["quality"],
        created_at=datetime.fromtimestamp(now - 400.0, tz=UTC).isoformat(),
        updated_at=datetime.fromtimestamp(now - 120.0, tz=UTC).isoformat(),
    )
    oversized = repository.create_memory(
        title="Oversized cleanup candidate",
        content="x" * 4_500,
        workspace_ids=["workspace-a"],
        memory_type="fact",
        summary="Detailed but concrete summary.",
        tags=["durable"],
        created_at=datetime.fromtimestamp(now - 500.0, tz=UTC).isoformat(),
        updated_at=datetime.fromtimestamp(now - 180.0, tz=UTC).isoformat(),
    )
    assert trace_like is not None and low_conversion is not None and oversized is not None

    db_manager.get_connection().executemany(
        """
        INSERT INTO memory_tool_events (
            invocation_id, workspace_id, caller_kind, event_kind, memory_id, query_text,
            result_rank, result_count, duration_ms, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            ("search-1", "workspace-a", "operator", "search", None, "cleanup", None, 1, 50.0, now - 30.0),
            ("search-1", "workspace-a", "operator", "search", low_conversion.id, "cleanup", 1, 1, 50.0, now - 30.0),
            ("search-2", "workspace-a", "operator", "search", None, "cleanup", None, 1, 55.0, now - 20.0),
            ("search-2", "workspace-a", "operator", "search", low_conversion.id, "cleanup", 1, 1, 55.0, now - 20.0),
            ("search-3", "workspace-a", "operator", "search", None, "cleanup", None, 1, 60.0, now - 10.0),
            ("search-3", "workspace-a", "operator", "search", low_conversion.id, "cleanup", 1, 1, 60.0, now - 10.0),
        ],
    )
    db_manager.get_connection().commit()

    service = _build_management_service(
        db_manager,
        workspace_id="workspace-a",
        repository=repository,
        task_queue=task_queue,
    )

    candidates = service.list_quality_cleanup_candidates(window_hours=24, bucket_minutes=60, now=now, limit=10)

    assert candidates.total_candidates == 3
    assert [candidate.memory_id for candidate in candidates.candidates] == [
        trace_like.id,
        low_conversion.id,
        oversized.id,
    ]

    trace_like_candidate = candidates.candidates[0]
    assert trace_like_candidate.priority_score == 95
    assert [criterion.key for criterion in trace_like_candidate.criteria] == [
        "trace_like_memory_count",
        "generic_summary_count",
    ]
    assert [recommendation.key for recommendation in trace_like_candidate.recommendations] == [
        "trace_like_memory_count",
        "generic_summary_count",
    ]

    low_conversion_candidate = candidates.candidates[1]
    assert low_conversion_candidate.priority_score == 50
    assert [criterion.key for criterion in low_conversion_candidate.criteria] == ["low_conversion"]
    assert [recommendation.key for recommendation in low_conversion_candidate.recommendations] == ["low_conversion"]
    assert low_conversion_candidate.search_count == 3
    assert low_conversion_candidate.read_count == 0
    assert low_conversion_candidate.converted_search_count == 0
    assert low_conversion_candidate.conversion_rate == 0.0

    oversized_candidate = candidates.candidates[2]
    assert oversized_candidate.priority_score == 20
    assert [criterion.key for criterion in oversized_candidate.criteria] == ["oversized_memory_count"]


@pytest.mark.parametrize(
    "task_name",
    [
        "project-manager",
        "fact-checker",
        "graph-linker",
        "conflict-detector",
        "defragmenter",
        "deduplicator",
        "taxonomist",
    ],
)
def test_management_service_redirects_manual_cleanup_aliases_to_memory_curator(db_manager, task_name: str) -> None:
    repository = RelationalMemoryRepository(db_manager)
    task_queue = SQLiteTaskQueue(db_manager)
    service = _build_management_service(
        db_manager,
        workspace_id="workspace-a",
        repository=repository,
        task_queue=task_queue,
    )

    payload = service.enqueue_background_task(task_name, force=False)

    assert payload["status"] == "enqueued"
    assert payload["task"]["task_name"] == "memory-curator"
    assert payload["redirected_from_task_name"] == task_name


def test_management_service_groups_manual_cleanup_aliases_in_run_all(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    task_queue = SQLiteTaskQueue(db_manager)
    service = _build_management_service(
        db_manager,
        workspace_id="workspace-a",
        repository=repository,
        task_queue=task_queue,
    )

    results = service.enqueue_all_background_tasks(force=False)

    task_names = {result["task"]["task_name"] for result in results}
    assert task_names == {
        "ingest-system1",
        "memory-curator",
        "sweeper",
    }
    curator_result = next(result for result in results if result["task"]["task_name"] == "memory-curator")
    assert curator_result["redirected_from_task_names"] == [
        "project-manager",
        "fact-checker",
        "graph-linker",
        "conflict-detector",
        "defragmenter",
        "deduplicator",
        "taxonomist",
    ]


def test_management_service_merges_quality_and_low_conversion_criteria_per_memory(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    task_queue = SQLiteTaskQueue(db_manager)
    now = time.time() + 60.0

    candidate = repository.create_memory(
        title="Cleanup follow-up",
        content="Short durable note.",
        workspace_ids=["workspace-a"],
        memory_type="observation",
        summary="Covers related retrieval cleanup findings.",
        created_at=datetime.fromtimestamp(now - 300.0, tz=UTC).isoformat(),
        updated_at=datetime.fromtimestamp(now - 90.0, tz=UTC).isoformat(),
    )
    below_threshold = repository.create_memory(
        title="Below threshold memory",
        content="Another durable note.",
        workspace_ids=["workspace-a"],
        memory_type="fact",
        summary="Concrete summary.",
        tags=["quality"],
        created_at=datetime.fromtimestamp(now - 200.0, tz=UTC).isoformat(),
        updated_at=datetime.fromtimestamp(now - 80.0, tz=UTC).isoformat(),
    )
    assert candidate is not None and below_threshold is not None

    db_manager.get_connection().executemany(
        """
        INSERT INTO memory_tool_events (
            invocation_id, workspace_id, caller_kind, event_kind, memory_id, query_text,
            result_rank, result_count, duration_ms, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            ("merge-search-1", "workspace-a", "operator", "search", None, "follow up", None, 1, 40.0, now - 30.0),
            ("merge-search-1", "workspace-a", "operator", "search", candidate.id, "follow up", 1, 1, 40.0, now - 30.0),
            ("merge-search-2", "workspace-a", "operator", "search", None, "follow up", None, 1, 42.0, now - 20.0),
            ("merge-search-2", "workspace-a", "operator", "search", candidate.id, "follow up", 1, 1, 42.0, now - 20.0),
            ("merge-search-3", "workspace-a", "operator", "search", None, "follow up", None, 1, 45.0, now - 10.0),
            ("merge-search-3", "workspace-a", "operator", "search", candidate.id, "follow up", 1, 1, 45.0, now - 10.0),
            ("low-threshold-1", "workspace-a", "operator", "search", None, "threshold", None, 1, 35.0, now - 25.0),
            ("low-threshold-1", "workspace-a", "operator", "search", below_threshold.id, "threshold", 1, 1, 35.0, now - 25.0),
            ("low-threshold-2", "workspace-a", "operator", "search", None, "threshold", None, 1, 36.0, now - 15.0),
            ("low-threshold-2", "workspace-a", "operator", "search", below_threshold.id, "threshold", 1, 1, 36.0, now - 15.0),
        ],
    )
    db_manager.get_connection().commit()

    service = _build_management_service(
        db_manager,
        workspace_id="workspace-a",
        repository=repository,
        task_queue=task_queue,
    )

    candidates = service.list_quality_cleanup_candidates(window_hours=24, bucket_minutes=60, now=now, limit=10)

    assert [candidate_result.memory_id for candidate_result in candidates.candidates] == [candidate.id]
    merged_candidate = candidates.candidates[0]
    assert merged_candidate.priority_score == 115
    assert [criterion.key for criterion in merged_candidate.criteria] == [
        "low_conversion",
        "generic_summary_count",
        "untagged_observation_count",
    ]
    assert [recommendation.key for recommendation in merged_candidate.recommendations] == [
        "low_conversion",
        "generic_summary_count",
        "untagged_observation_count",
    ]
    assert merged_candidate.search_count == 3
    assert merged_candidate.conversion_rate == 0.0


def test_management_service_overview_and_memory_detail(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    task_queue = SQLiteTaskQueue(db_manager)
    work_items = SQLiteWorkItemRepository(db_manager)
    embedding_repair_queue = SQLiteEmbeddingRepairQueue(db_manager)
    handler = SQLiteStructuredLogHandler(
        db_manager=db_manager,
        workspace_id="workspace-a",
        source="daemon",
    )
    relational_search = RelationalMemorySearchService(
        repository,
        config=__import__("mcp_memory.config", fromlist=["Config"]).Config(),
        embedder=None,
        vector_store=SQLiteVectorStore(db_manager),
        db_manager=db_manager,
        task_queue=task_queue,
        work_items=work_items,
        embedding_repair_queue=embedding_repair_queue,
        background_repair_wait_seconds=5.0,
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
    db_manager.get_connection().executemany(
        "INSERT INTO runtime_logs (workspace_id, source, logger_name, level, message, created_at, data_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (
                "workspace-a",
                "daemon",
                "mcp_memory.core.provider_policy",
                "WARNING",
                "Provider routing exhausted all configured routes",
                time.time(),
                '{"extra":{"task_name":"memory-curator","candidate_routes":["gemini-cheap","copilot-mini"]}}',
            ),
            (
                "workspace-a",
                "daemon",
                "mcp_memory.core.provider_policy",
                "WARNING",
                "Legacy fallback provider is unavailable due to admission control",
                time.time(),
                '{"extra":{"task_name":"memory-curator","reason":"provider_quota_exhausted"}}',
            ),
        ],
    )
    db_manager.get_connection().executemany(
        "INSERT INTO provider_policy_events (workspace_id, task_name, task_id, event_kind, warning_kind, provider_key, provider_name, model_name, route_key, candidate_routes_json, reason_category, reason_code, retry_delay_seconds, warning_suppressed, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                "workspace-a",
                "memory-curator",
                None,
                "route_exhausted",
                "provider_routing_exhausted",
                None,
                None,
                None,
                None,
                '["gemini-cheap","copilot-mini"]',
                None,
                None,
                None,
                1,
                time.time(),
            ),
            (
                "workspace-a",
                "memory-curator",
                None,
                "legacy_fallback_denied",
                "legacy_fallback_denied",
                None,
                None,
                None,
                None,
                '[]',
                "upstream",
                "provider_quota_exhausted",
                90.0,
                0,
                time.time(),
            ),
            (
                "workspace-a",
                "memory-curator",
                None,
                "route_skipped",
                None,
                "gemini-cli",
                "Gemini CLI",
                "gemini-3-flash-preview",
                "gemini-cheap",
                '[]',
                "upstream",
                "provider_quota_exhausted",
                90.0,
                0,
                time.time(),
            ),
        ],
    )
    db_manager.get_connection().execute(
        "INSERT INTO provider_usage (workspace_id, task_name, provider_key, provider_name, model_name, status, duration_seconds, created_at, error_text) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("workspace-a", "memory-curator", "gemini-cli", "Gemini CLI", "gemini-3-flash-preview", "success", 0.25, time.time(), None),
    )
    db_manager.get_connection().execute(
        "INSERT INTO ai_conversations (request_id, attempt, workspace_id, task_name, task_id, provider_key, provider_name, model_name, subprocess_pid, prompt_text, response_text, parsed_json, status, error_text, started_at, completed_at, duration_seconds) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "req-premium",
            1,
            "workspace-a",
            "memory-curator",
            "task-premium",
            "copilot-cli",
            "Copilot CLI",
            "gpt-5",
            3333,
            "prompt text",
            '{"type":"message"}\n{"type":"result","usage":{"premiumRequests":13}}',
            None,
            "success",
            None,
            time.time() - 30.0,
            time.time() - 10.0,
            20.0,
        ),
    )
    db_manager.get_connection().execute(
        "INSERT INTO provider_usage (workspace_id, task_name, provider_key, provider_name, model_name, status, duration_seconds, created_at, error_text, reason_category, reason_code, retry_delay_seconds) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "workspace-a",
            "memory-curator",
            "gemini-cli",
            "Gemini CLI",
            "gemini-3-flash-preview",
            "skipped",
            0.0,
            time.time(),
            "quota reset pending",
            "upstream",
            "provider_quota_exhausted",
            90.0,
        ),
    )
    db_manager.get_connection().execute(
        "INSERT OR REPLACE INTO provider_admission_state (provider_key, model_name, reason_category, reason_code, error_text, retry_delay_seconds, active_until, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "gemini-cli",
            "gemini-3-flash-preview",
            "upstream",
            "provider_quota_exhausted",
            "quota reset pending",
            90.0,
            time.time() + 90.0,
            time.time(),
        ),
    )
    db_manager.get_connection().execute(
        "INSERT INTO provider_usage (workspace_id, task_name, provider_key, provider_name, model_name, status, duration_seconds, created_at, error_text) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("workspace-b", "memory-curator", "copilot-mini", "Copilot CLI", "gpt-5-mini", "success", 0.5, time.time(), None),
    )
    embedding_repair_queue.enqueue_unique(
        memory_id=primary.id,
        workspace_id=None,
        model_name="demo-model",
        memory_updated_at=primary.updated_at,
    )
    db_manager.get_connection().commit()

    ctx = ApplicationContext(
        workspace_id="workspace-a",
        memory_path=db_manager.db_path.parent,
        db_manager=db_manager,
        repository=repository,
        task_queue=task_queue,
        work_items=work_items,
        embedding_repair_queue=embedding_repair_queue,
        relational_search=relational_search,
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
    assert overview.premium_usage.copilot_premium_requests_today == 13
    assert overview.premium_usage.copilot_premium_requests_last_day == 13
    assert overview.agent_runs[0].task_name == "ingest-system1"
    assert overview.provider_usage[0].provider_key == "gemini-cli"
    assert overview.provider_usage[0].task_name == "memory-curator"
    assert overview.provider_usage[0].calls_last_hour == 1
    assert overview.provider_usage[0].skips_last_day == 1
    assert overview.provider_usage[0].top_skip_reason_last_day == "provider_quota_exhausted"
    assert overview.provider_usage[0].active_admission_reason == "provider_quota_exhausted"
    assert {item.provider_key for item in overview.provider_usage} == {"gemini-cli", "copilot-mini"}
    assert overview.tasks.failed_count == 1
    deduplicator = next(agent for agent in overview.agent_runs if agent.task_name == "deduplicator")
    assert deduplicator.last_result_metadata.strategy_used == "semantic"
    assert deduplicator.last_result_metadata.candidate_count == 8
    assert deduplicator.last_result_metadata.sampled_memory_ids == [primary.id]
    assert any(run.result_metadata.strategy_used == "semantic" for run in overview.recent_agent_runs)
    assert overview.failed_tasks[0]["last_error"] == "summary provider offline"
    assert any(record.message == "daemon log row" for record in overview.recent_logs)
    assert overview.recent_logs[0].source == "daemon"
    assert [record.id for record in overview.top_read_memories] == [primary.id, secondary.id]
    assert [record.read_count for record in overview.top_read_memories] == [2, 1]
    assert [record.id for record in overview.top_read_memories_active] == [primary.id]
    assert [record.read_count for record in overview.top_read_memories_active] == [2]
    assert any(log.logger_name == "mcp_memory.tests" for log in filtered_logs.logs)
    assert summary.total == 3
    assert summary.by_level == {"WARNING": 3}
    assert summary.by_source == {"daemon": 3}
    assert prune_result.deleted == 2
    assert prune_result.max_runtime_logs == 1
    assert len(after_prune.logs) == 1
    assert detail.record["id"] == primary.id
    assert detail.relationships["outgoing"][0]["link_type"] == "SUPERSEDES"
    assert detail.superseded[0]["id"] == secondary.id
    assert overview.failed_tasks[0]["status"] == "failed"
    assert health.runtime_active is True
    assert health.search.background_repair_enabled is True
    assert health.search.background_repair_wait_seconds == 5.0
    assert health.search.queued_repair_backlog_count == 0
    assert health.search.running_repair_count == 0
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
    assert curator_route.recent_provider_key == "copilot-cli"
    assert summarize_route.task_class == "deterministic"
    assert summarize_route.resolved_provider_key is None
    assert nerd_metrics.provider_policy.stats[0].key == "provider_policy_route_exhaustion_count"
    assert nerd_metrics.provider_policy.stats[0].value == 1.0
    assert nerd_metrics.provider_policy.stats[1].value == 1.0
    assert nerd_metrics.provider_policy.stats[2].value == 1.0
    assert nerd_metrics.provider_policy.stats[3].key == "provider_policy_warning_suppressed_count"
    assert nerd_metrics.provider_policy.stats[3].value == 1.0
    provider_policy_task = next(item for item in nerd_metrics.provider_policy.by_task if item.task_name == "memory-curator")
    assert provider_policy_task.route_exhaustion_count == 1
    assert provider_policy_task.legacy_fallback_denied_count == 1
    assert provider_policy_task.admission_skip_count == 1
    assert provider_policy_task.top_skip_provider_key == "gemini-cli"
    assert provider_policy_task.top_skip_reason_code == "provider_quota_exhausted"
    provider_policy_provider = next(item for item in nerd_metrics.provider_policy.by_provider if item.provider_key == "gemini-cli")
    assert provider_policy_provider.admission_skip_count == 1
    assert provider_policy_provider.top_task_name == "memory-curator"
    assert provider_policy_provider.top_reason_code == "provider_quota_exhausted"
    assert provider_policy_provider.active_admission_reason == "provider_quota_exhausted"
    assert any(stat.key == "orphan_rate" for stat in nerd_metrics.stats)
    assert any(stat.key == "copilot_premium_requests_today" and stat.value == 13.0 for stat in nerd_metrics.stats)
    assert any(stat.key == "copilot_premium_requests_last_day" and stat.value == 13.0 for stat in nerd_metrics.stats)
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


def test_management_service_operator_health_snapshot_aggregates_recent_signals(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    task_queue = SQLiteTaskQueue(db_manager)
    provider_usage = ProviderUsageRepository(db_manager, workspace_id=None)

    recent_iso = datetime.now(UTC).isoformat()
    repository.create_memory(
        title="Recent operator edit",
        content="Fresh memory update for snapshot reporting.",
        workspace_ids=["workspace-a"],
        memory_type="fact",
        updated_at=recent_iso,
    )

    db_manager.get_connection().execute(
        "INSERT INTO runtime_logs (workspace_id, source, logger_name, level, message, created_at, data_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            "workspace-a",
            "daemon",
            "mcp_memory.tests",
            "ERROR",
            "recent operator error",
            time.time(),
            "{}",
        ),
    )
    db_manager.get_connection().execute(
        "INSERT INTO runtime_logs (workspace_id, source, logger_name, level, message, created_at, data_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            "workspace-a",
            "daemon",
            "mcp_memory.core.provider_policy",
            "WARNING",
            "Provider routing exhausted all configured routes",
            time.time(),
            '{"extra":{"task_name":"graph-linker"}}',
        ),
    )

    task = task_queue.enqueue(
        "graph-linker",
        task_id="snapshot-run-1",
        workspace_id="workspace-a",
        available_at=0.0,
    )
    assert task_queue.claim_next(now=10.0) is not None
    task_queue.complete(task.id, completed_at=12.0, run_result={"updated": 1})

    retry_task = task_queue.enqueue(
        "graph-linker",
        task_id="snapshot-run-2",
        workspace_id="workspace-a",
        available_at=0.0,
    )
    assert task_queue.claim_next(now=13.0) is not None
    task_queue.fail(retry_task.id, "retry me", retry_delay_seconds=30.0, failed_at=14.0)

    db_manager.get_connection().execute(
        "INSERT INTO provider_policy_events (workspace_id, task_name, task_id, event_kind, warning_kind, provider_key, provider_name, model_name, route_key, candidate_routes_json, reason_category, reason_code, retry_delay_seconds, warning_suppressed, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "workspace-a",
            "graph-linker",
            task.id,
            "route_exhausted",
            "provider_routing_exhausted",
            None,
            None,
            None,
            None,
            '["gemini-cheap","copilot-mini"]',
            None,
            None,
            None,
            0,
            time.time(),
        ),
    )
    db_manager.get_connection().execute(
        "INSERT INTO provider_usage (workspace_id, task_name, provider_key, provider_name, model_name, status, duration_seconds, created_at, error_text, reason_category, reason_code, retry_delay_seconds) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "workspace-a",
            "graph-linker",
            "gemini-cli",
            "Gemini CLI",
            "gemini-3-flash-preview",
            "skipped",
            0.0,
            time.time(),
            "quota reset pending",
            "upstream",
            "provider_quota_exhausted",
            90.0,
        ),
    )

    provider_usage.record_conversation(
        request_id="snapshot-conversation",
        attempt=1,
        task_name="graph-linker",
        task_id=task.id,
        provider_key="gemini-cli",
        provider_name="Gemini CLI",
        model_name="gemini-3-flash-preview",
        subprocess_pid=1111,
        prompt_text="prompt",
        response_text="response",
        parsed=None,
        status="error",
        error_text="provider timeout",
        started_at=time.time() - 30.0,
        completed_at=time.time() - 20.0,
    )
    db_manager.get_connection().commit()

    service = _build_management_service(
        db_manager,
        workspace_id=None,
        repository=repository,
        task_queue=task_queue,
    )

    snapshot = service.get_operator_health_snapshot(
        recent_error_limit=5,
        recent_run_limit=5,
        conversation_limit=5,
        recent_memory_limit=5,
    )

    assert snapshot.status == "warn"
    assert "recent_error_logs" in snapshot.alerts
    assert "recent_task_retries" in snapshot.alerts
    assert "recent_ai_errors" in snapshot.alerts
    assert "provider_policy_churn" in snapshot.alerts
    assert snapshot.logs.by_level["ERROR"] == 1
    assert snapshot.logs.recent_errors[0].message == "recent operator error"
    assert snapshot.warnings.total == 1
    assert snapshot.warnings.recent[0].logger_name == "mcp_memory.core.provider_policy"
    assert snapshot.tasks.recent_status_counts == {"completed": 1, "retry": 1}
    assert [run.task_id for run in snapshot.tasks.recent] == [retry_task.id, task.id]
    assert snapshot.tasks.recent_failure_count == 0
    assert snapshot.tasks.recent_retry_count == 1
    assert snapshot.tasks.recent_retries[0].task_id == retry_task.id
    assert snapshot.conversations.by_status == {"error": 1}
    assert snapshot.conversations.recent[0].request_id == "snapshot-conversation"
    assert snapshot.memory_activity.updated_last_15_minutes == 1
    assert snapshot.memory_activity.recent[0].title == "Recent operator edit"
    provider_policy_stats = {stat.key: stat.value for stat in snapshot.provider_policy.stats}
    assert provider_policy_stats["provider_policy_route_exhaustion_count"] == 1.0
    assert provider_policy_stats["provider_policy_admission_skip_count"] == 1.0
    assert snapshot.provider_policy.by_task[0].task_name == "graph-linker"


def test_management_service_operator_health_snapshot_applies_conversation_and_memory_windows(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    task_queue = SQLiteTaskQueue(db_manager)
    provider_usage = ProviderUsageRepository(db_manager, workspace_id="workspace-a")
    now = time.time()

    recent_memory = repository.create_memory(
        title="Recent memory",
        content="Updated within 15 minutes.",
        workspace_ids=["workspace-a"],
        memory_type="fact",
        updated_at=(datetime.now(UTC) - timedelta(minutes=5)).isoformat(),
    )
    hour_memory = repository.create_memory(
        title="Hour memory",
        content="Updated within an hour.",
        workspace_ids=["workspace-a"],
        memory_type="fact",
        updated_at=(datetime.now(UTC) - timedelta(minutes=45)).isoformat(),
    )
    day_memory = repository.create_memory(
        title="Day memory",
        content="Updated within a day.",
        workspace_ids=["workspace-a"],
        memory_type="fact",
        updated_at=(datetime.now(UTC) - timedelta(hours=23)).isoformat(),
    )
    stale_memory = repository.create_memory(
        title="Stale memory",
        content="Outside the one-day window.",
        workspace_ids=["workspace-a"],
        memory_type="fact",
        updated_at=(datetime.now(UTC) - timedelta(days=2)).isoformat(),
    )
    assert recent_memory is not None
    assert hour_memory is not None
    assert day_memory is not None
    assert stale_memory is not None

    provider_usage.record_conversation(
        request_id="recent-success",
        attempt=1,
        task_name="graph-linker",
        task_id="task-success",
        provider_key="gemini-cli",
        provider_name="Gemini CLI",
        model_name="gemini-3-flash-preview",
        subprocess_pid=111,
        prompt_text="prompt",
        response_text="response",
        parsed=None,
        status="success",
        error_text=None,
        started_at=now - 600.0,
        completed_at=now - 540.0,
    )
    provider_usage.record_conversation(
        request_id="recent-error",
        attempt=1,
        task_name="graph-linker",
        task_id="task-error",
        provider_key="gemini-cli",
        provider_name="Gemini CLI",
        model_name="gemini-3-flash-preview",
        subprocess_pid=222,
        prompt_text="prompt",
        response_text="response",
        parsed=None,
        status="error",
        error_text="timeout",
        started_at=now - 900.0,
        completed_at=now - 840.0,
    )
    provider_usage.record_conversation(
        request_id="old-error",
        attempt=1,
        task_name="graph-linker",
        task_id="task-old-error",
        provider_key="gemini-cli",
        provider_name="Gemini CLI",
        model_name="gemini-3-flash-preview",
        subprocess_pid=333,
        prompt_text="prompt",
        response_text="response",
        parsed=None,
        status="error",
        error_text="stale timeout",
        started_at=now - (26 * 3600),
        completed_at=now - (25 * 3600),
    )
    db_manager.get_connection().commit()

    service = _build_management_service(
        db_manager,
        workspace_id="workspace-a",
        repository=repository,
        task_queue=task_queue,
    )

    snapshot = service.get_operator_health_snapshot(conversation_window_hours=24)

    assert snapshot.conversations.by_status == {"error": 1, "success": 1}
    assert snapshot.conversations.total == 2
    assert snapshot.memory_activity.updated_last_15_minutes == 1
    assert snapshot.memory_activity.updated_last_hour == 2
    assert snapshot.memory_activity.updated_last_day == 3


def test_management_service_operator_health_snapshot_ignores_service_workspace_scope(db_manager) -> None:
    repository = RelationalMemoryRepository(db_manager)
    task_queue = SQLiteTaskQueue(db_manager)
    provider_usage = ProviderUsageRepository(db_manager, workspace_id="workspace-a")
    now = time.time()

    repository.create_memory(
        title="Workspace A recent memory",
        content="Updated recently in workspace A.",
        workspace_ids=["workspace-a"],
        memory_type="fact",
        updated_at=datetime.now(UTC).isoformat(),
    )
    repository.create_memory(
        title="Workspace B recent memory",
        content="Updated recently in workspace B.",
        workspace_ids=["workspace-b"],
        memory_type="fact",
        updated_at=datetime.now(UTC).isoformat(),
    )

    db_manager.get_connection().executemany(
        "INSERT INTO runtime_logs (workspace_id, source, logger_name, level, message, created_at, data_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            ("workspace-a", "daemon", "mcp_memory.tests", "ERROR", "workspace-a error", now - 5.0, "{}"),
            ("workspace-b", "daemon", "mcp_memory.tests", "ERROR", "workspace-b error", now - 4.0, "{}"),
        ],
    )

    task_a = task_queue.enqueue(
        "graph-linker",
        task_id="snapshot-workspace-a",
        workspace_id="workspace-a",
        available_at=0.0,
    )
    task_b = task_queue.enqueue(
        "graph-linker",
        task_id="snapshot-workspace-b",
        workspace_id="workspace-b",
        available_at=0.0,
    )
    assert task_queue.claim_next(now=10.0) is not None
    task_queue.complete(task_a.id, completed_at=11.0, run_result={"updated": 1})
    assert task_queue.claim_next(now=12.0) is not None
    task_queue.complete(task_b.id, completed_at=13.0, run_result={"updated": 1})

    provider_usage.record_conversation(
        request_id="workspace-a-conversation",
        attempt=1,
        task_name="graph-linker",
        task_id=task_a.id,
        provider_key="gemini-cli",
        provider_name="Gemini CLI",
        model_name="gemini-3-flash-preview",
        subprocess_pid=111,
        prompt_text="prompt",
        response_text="response",
        parsed=None,
        status="success",
        error_text=None,
        started_at=now - 60.0,
        completed_at=now - 50.0,
    )
    provider_usage.record_conversation(
        request_id="workspace-b-conversation",
        attempt=1,
        task_name="graph-linker",
        task_id=task_b.id,
        provider_key="gemini-cli",
        provider_name="Gemini CLI",
        model_name="gemini-3-flash-preview",
        subprocess_pid=222,
        prompt_text="prompt",
        response_text="response",
        parsed=None,
        status="error",
        error_text="timeout",
        started_at=now - 40.0,
        completed_at=now - 30.0,
    )
    db_manager.get_connection().commit()

    service = _build_management_service(
        db_manager,
        workspace_id="workspace-a",
        repository=repository,
        task_queue=task_queue,
    )

    snapshot = service.get_operator_health_snapshot(recent_run_limit=10, conversation_limit=10, recent_memory_limit=10)

    assert snapshot.logs.by_level == {"ERROR": 2}
    assert {run.task_id for run in snapshot.tasks.recent} == {task_a.id, task_b.id}
    assert snapshot.conversations.by_status == {"error": 1, "success": 1}
    assert {record.title for record in snapshot.memory_activity.recent} == {
        "Workspace A recent memory",
        "Workspace B recent memory",
    }


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


def test_management_service_record_thought_uses_effective_service_workspace_not_boot_runtime_info(db_manager, monkeypatch) -> None:
    repository = RelationalMemoryRepository(db_manager)
    journal = System1Journal(db_manager)
    task_queue = SQLiteTaskQueue(db_manager)

    ctx = ApplicationContext(
        workspace_id="workspace-request",
        memory_path=db_manager.db_path.parent,
        db_manager=db_manager,
        journal=journal,
        repository=repository,
        task_queue=task_queue,
    )
    service = ManagementService(ctx, SimpleNamespace(has_runtime=True, client_count=1))
    monkeypatch.setattr(service, "_runtime_info", replace(service._runtime_info, workspace_id="workspace-boot"))

    payload = service.record_thought("remember the request workspace")
    entry = payload["entry"]

    assert payload["status"] == "recorded"
    assert isinstance(entry, dict)
    assert entry["workspace_id"] == "workspace-request"
    assert [entry.content for entry in journal.get_pending(workspace_id="workspace-request")] == [
        "remember the request workspace"
    ]
    assert journal.get_pending(workspace_id="workspace-boot") == []


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
