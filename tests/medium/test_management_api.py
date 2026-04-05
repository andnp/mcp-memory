from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
from mcp_memory.provider_usage_store import ProviderUsageRepository
from datetime import UTC, datetime
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace


from typing import Any, cast

from fastapi.testclient import TestClient
import pytest
import mcp_memory.daemon_app as daemon_app_module

from mcp_memory.daemon import create_daemon_app
from mcp_memory.daemon_models import DaemonControllerView
from mcp_memory.daemon_transport import DaemonZmqServer, request_daemon_json
from mcp_memory.config import Config
from mcp_memory.context import ApplicationContext
from mcp_memory.hook_reminders import HookReminderService
from mcp_memory.management.service import ManagementService
from mcp_memory.mcp.services import read_memory_record_service, search_memory_records_service
from mcp_memory.mcp.runtime import create_runtime
from mcp_memory.runtime_logging import SQLiteStructuredLogHandler
from mcp_memory.storage.shared_read_cache import SharedReadCache


pytestmark = pytest.mark.medium


async def _request_json(
    metadata,
    path: str,
    payload: dict | None = None,
    *,
    timeout_seconds: float = 5,
) -> dict:
    return await asyncio.to_thread(
        request_daemon_json,
        metadata,
        path,
        payload,
        timeout_seconds=timeout_seconds,
    )


@pytest.mark.asyncio
async def test_management_api_exposes_dashboard_and_json_views(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    base_time = time.time() - 300.0
    primary_created_at = datetime.fromtimestamp(base_time - 120.0, tz=UTC).isoformat()
    primary_updated_at = datetime.fromtimestamp(base_time - 60.0, tz=UTC).isoformat()
    superseded_created_at = datetime.fromtimestamp(base_time - 90.0, tz=UTC).isoformat()
    superseded_updated_at = datetime.fromtimestamp(base_time - 30.0, tz=UTC).isoformat()

    seed_runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    try:
        assert seed_runtime.repository is not None
        assert seed_runtime.task_queue is not None
        assert seed_runtime.workspace_id is not None
        assert seed_runtime.db_manager is not None
        seed_runtime.db_manager.get_connection().execute(
            "INSERT INTO runtime_logs (workspace_id, source, logger_name, level, message, created_at, data_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                seed_runtime.workspace_id,
                "daemon",
                "mcp_memory.server",
                "INFO",
                "seeded daemon log",
                time.time(),
                "{}",
            ),
        )
        seed_runtime.db_manager.get_connection().execute(
            "INSERT INTO provider_usage (workspace_id, task_name, provider_key, provider_name, model_name, status, duration_seconds, created_at, error_text) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                seed_runtime.workspace_id,
                "graph-linker",
                "gemini-cli",
                "Gemini CLI",
                "gemini-3-flash-preview",
                "success",
                0.42,
                    base_time + 30.0,
                None,
            ),
        )
        seed_runtime.db_manager.get_connection().commit()
        primary = seed_runtime.repository.create_memory(
            title="Management API plan",
            content="Expose runtime health and recent memories.",
            summary="Operations dashboard plan.",
            workspace_ids=[seed_runtime.workspace_id],
            memory_type="plan",
            tags=["dashboard", "api"],
            created_at=primary_created_at,
            updated_at=primary_updated_at,
        )
        superseded = seed_runtime.repository.create_memory(
            title="Management API legacy plan",
            content="An older plan for the same area.",
            workspace_ids=[seed_runtime.workspace_id],
            memory_type="plan",
            status="stale",
            tags=["dashboard"],
            created_at=superseded_created_at,
            updated_at=superseded_updated_at,
        )
        assert primary is not None and superseded is not None
        seed_runtime.repository.add_link(primary.id, superseded.id, "SUPERSEDES", "Superseded during epic 6")
        search_memory_records_service(seed_runtime, {"query": "runtime health", "limit": 5, "status": "active"})
        search_memory_records_service(seed_runtime, {"query": "runtime health", "limit": 5, "status": "archived"})
        read_memory_record_service(seed_runtime, {"memory_id": primary.id})
        read_memory_record_service(seed_runtime, {"memory_id": primary.id})
        foreign_memory = seed_runtime.repository.create_memory(
            title="Other workspace memory",
            content="This should stay out of scoped nerd metrics.",
            workspace_ids=["workspace-b"],
            memory_type="fact",
            tags=["foreign-tag"],
        )
        assert foreign_memory is not None

        task = seed_runtime.task_queue.enqueue(
            "fact-checker",
            task_id="fact-checker-1",
            workspace_id=seed_runtime.workspace_id,
            available_at=0.0,
        )
        strategy_task = seed_runtime.task_queue.enqueue(
            "deduplicator",
            task_id="dedup-observability-1",
            workspace_id=seed_runtime.workspace_id,
            available_at=0.0,
        )
        premium_yield_task = seed_runtime.task_queue.enqueue(
            "taxonomist",
            task_id="taxonomist-premium-yield-1",
            workspace_id=seed_runtime.workspace_id,
            available_at=0.0,
        )
        seed_runtime.task_queue.enqueue(
            "summarize-memory",
            task_id="summarize-pending-1",
            workspace_id=seed_runtime.workspace_id,
            available_at=time.time() + 60.0,
            data={"trigger": "summary_follow_up"},
        )
        assert seed_runtime.task_queue.claim_next(now=base_time + 10.0) is not None
        seed_runtime.task_queue.fail_permanently(task.id, "missing ext link", failed_at=base_time + 11.0)
        assert seed_runtime.task_queue.claim_next(now=base_time + 12.0) is not None
        seed_runtime.task_queue.complete(
            strategy_task.id,
            completed_at=base_time + 13.0,
            run_result={
                "merged": 1,
                "lines_compressed": 12,
                "requested_strategy": "semantic",
                "strategy_used": "semantic",
                "candidate_count": 6,
                "sampled_memory_ids": [primary.id],
            },
        )
        assert seed_runtime.task_queue.claim_next(now=base_time + 14.0) is not None
        seed_runtime.task_queue.complete(
            premium_yield_task.id,
            completed_at=base_time + 15.0,
            run_result={
                "updated": 3,
                "campaign_key": "lightweight_review",
                "campaign_origin_family": "memory_tagging",
                "campaign_family_keys": ["memory_tagging", "graph_link_review", "conflict_review"],
                "campaign_continuation_supported": True,
                "provider_calls_used": 1,
                "claimed_work_item_count": 3,
                "tool_calls_executed": 6,
                "mutations": 4,
                "compatible_batch_calls": 1,
            },
        )
    finally:
        seed_runtime.close()

    app = create_daemon_app(workspace_root_override=None, cwd=workspace)
    async with app.router.lifespan_context(app):
        metadata = app.state.metadata

        health = await _request_json(metadata, "/api/health")
        overview = await _request_json(metadata, "/api/overview")
        logs = await _request_json(metadata, "/api/logs", {"source": "daemon", "q": "seeded"})
        log_summary = await _request_json(metadata, "/api/logs/summary", {"source": "daemon"})
        nerd_metrics = await _request_json(metadata, "/api/metrics/nerd", {"window_hours": 24, "bucket_minutes": 60, "now": base_time + 60.0})
        global_nerd_metrics = await _request_json(
            metadata,
            "/api/metrics/nerd",
            {"scope": "global", "window_hours": 24, "bucket_minutes": 60, "now": base_time + 60.0},
        )
        workspace_override_nerd_metrics = await _request_json(
            metadata,
            "/api/metrics/nerd?scope=workspace&workspace_id=workspace-b",
            {"window_hours": 24, "bucket_minutes": 60, "now": base_time + 60.0},
        )
        prune_logs = await _request_json(
            metadata,
            "/api/admin/logs/prune",
            {"max_runtime_logs": 1, "max_log_age_days": 30},
        )
        repair_search = await _request_json(metadata, "/api/admin/search/repair", {}, timeout_seconds=20)
        tasks = await _request_json(metadata, "/api/tasks", {"status": "failed"})
        record_thought = await _request_json(
            metadata,
            "/api/record-thought",
            {"content": "ship the command center incrementally"},
        )
        overview_after_thought = await _request_json(metadata, "/api/overview")
        memories = await _request_json(metadata, "/api/memories", {"workspace_id": seed_runtime.workspace_id})
        detail = await _request_json(metadata, f"/api/memories/{primary.id}")
        missing_detail = await _request_json(metadata, "/api/memories/missing-memory-id")
        invalid_limit = await _request_json(metadata, "/api/tasks", {"limit": 0})
        internal_tools = await _request_json(metadata, "/internal/maintenance/tools")
        hook_start = await _request_json(
            metadata,
            "/api/hooks/session-start",
            {"sessionId": "conversation-1", "timestamp": 100.0},
        )
        hook_noop = await _request_json(
            metadata,
            "/api/hooks/post-tool-use",
            {"sessionId": "conversation-1", "tool_name": "read_file", "timestamp": 200.0},
        )
        hook_reminder = await _request_json(
            metadata,
            "/api/hooks/post-tool-use",
            {"sessionId": "conversation-1", "tool_name": "apply_patch", "timestamp": 401.0},
        )
        hook_end = await _request_json(
            metadata,
            "/api/hooks/session-end",
            {"sessionId": "conversation-1", "timestamp": 500.0},
        )
        run_agent = await _request_json(
            metadata,
            "/api/admin/agents/run",
            {"task_name": "graph-linker", "force": True},
        )
        run_all = await _request_json(
            metadata,
            "/api/admin/agents/run-all",
            {"force": False},
        )
        running_task = await _request_json(
            metadata,
            "/api/admin/agents/run",
            {"task_name": "memory-curator", "force": True},
        )
        cancel_task = await _request_json(
            metadata,
            f"/api/admin/tasks/{running_task['task']['id']}/cancel",
            {"cancelled_by": "api-test", "reason": "operator_cancelled"},
        )
        created_link = await _request_json(
            metadata,
            "/api/admin/links",
            {
                "source_id": primary.id,
                "target_id": "ext:README.md",
                "link_type": "REFERENCES",
                "context": "Referenced from the dashboard admin flow.",
            },
        )
        deleted_link = await _request_json(
            metadata,
            "/api/admin/links/delete",
            {
                "source_id": primary.id,
                "target_id": "ext:README.md",
                "link_type": "REFERENCES",
            },
        )
        app.state.routes.ctx.db_manager.get_connection().execute(
            "INSERT INTO ai_conversations (request_id, attempt, workspace_id, task_name, task_id, provider_key, provider_name, model_name, subprocess_pid, prompt_text, response_text, parsed_json, status, error_text, started_at, completed_at, duration_seconds) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "req-api-1",
                1,
                seed_runtime.workspace_id,
                "graph-linker",
                run_agent["task"]["id"],
                "gemini-cli",
                "Gemini CLI",
                "gemini-3-flash-preview",
                6543,
                "prompt",
                "response",
                '{"ok": true}',
                "success",
                None,
                1.0,
                2.0,
                1.0,
            ),
        )
        app.state.routes.ctx.db_manager.get_connection().commit()
        conversations = await _request_json(metadata, "/api/ai-conversations", {"task_name": "graph-linker"})
        dashboard = app.state.routes.service.load_dashboard_html()

        assert health["status"] == "ready"
        assert health["workspace_root"] == str(workspace)
        assert "embeddings" in health
        assert "backend" in health["embeddings"]
        assert health["search"]["semantic_enabled"] is True
        assert "transport_diagnostics" in health
        assert health["transport_diagnostics"]["max_concurrent_requests"] == 8
        assert health["transport_diagnostics"]["recent_queue_wait_max_ms"] >= 0.0
        assert "execution_attempts" in health
        assert health["execution_attempts"]["running_task_count"] >= 0
        assert overview["memories"]["total"] == 3
        assert overview["embeddings"]["model_name"] is not None
        assert overview["search"]["semantic_enabled"] is True
        assert "execution_attempts" in overview
        assert overview["execution_attempts"]["missing_attempt_count"] >= 0
        assert overview["memory_metrics"]["total_memories"] == 3
        assert overview["queue_diagnostics"]
        assert all(item["pending_state"] in {"scheduled", "runnable"} for item in overview["queue_diagnostics"])
        assert any(item["task_name"] in {"project-manager", "summarize-memory"} for item in overview["queue_diagnostics"])
        assert "agent_runs" in overview
        assert overview["provider_usage"][0]["provider_key"] == "gemini-cli"
        assert overview["provider_usage"][0]["task_name"] == "graph-linker"
        assert overview["provider_usage"][0]["calls_last_day"] == 1
        assert overview["provider_usage"][0]["skips_last_day"] == 0
        assert overview["provider_usage"][0]["top_failure_reason_last_day"] is None
        assert overview["recent_logs"][0]["message"] == "seeded daemon log"
        assert [record["id"] for record in overview["top_read_memories"][:1]] == [primary.id]
        assert [record["read_count"] for record in overview["top_read_memories"][:1]] == [2]
        assert [record["id"] for record in overview["top_read_memories_active"][:1]] == [primary.id]
        assert overview["tasks"]["failed_count"] == 1
        assert record_thought["status"] == "recorded"
        assert overview_after_thought["journal"]["pending_count"] >= overview["journal"]["pending_count"] + 1
        assert overview["failed_tasks"][0]["last_error"] == "missing ext link"
        assert logs["logs"][0]["source"] == "daemon"
        assert log_summary["total"] == 1
        assert log_summary["by_level"] == {"INFO": 1}
        assert nerd_metrics["stats"]
        assert "composition" in nerd_metrics
        assert "distributions" in nerd_metrics
        assert "timelines" in nerd_metrics
        assert "maintenance" in nerd_metrics
        assert "lifecycle_trends" in nerd_metrics
        assert "quality_signals" in nerd_metrics["lifecycle_trends"]
        assert "growth_dynamics" in nerd_metrics
        assert "quality_drilldown" in nerd_metrics
        assert "quality_remediation" in nerd_metrics
        assert "retrieval" in nerd_metrics
        assert "maintenance_summary" in nerd_metrics
        assert nerd_metrics["agent_throughput"]
        assert nerd_metrics["provider_latency"]
        assert any(stat["key"] == "provider_p95_latency" for stat in nerd_metrics["stats"])
        assert any(stat["key"] == "running_attempt_count" for stat in nerd_metrics["stats"])
        assert any(stat["key"] == "stale_attempt_count" for stat in nerd_metrics["stats"])
        assert any(stat["key"] == "premium_execution_count" and stat["value"] == 1.0 for stat in nerd_metrics["stats"])
        assert any(stat["key"] == "compatible_batch_calls" and stat["value"] == 1.0 for stat in nerd_metrics["stats"])
        assert any(stat["key"] == "work_items_per_premium_execution" and stat["value"] == 3.0 for stat in nerd_metrics["stats"])
        assert any(stat["key"] == "mutations_per_premium_execution" and stat["value"] == 4.0 for stat in nerd_metrics["stats"])
        assert any(stat["key"] == "tool_calls_per_premium_execution" and stat["value"] == 6.0 for stat in nerd_metrics["stats"])
        assert nerd_metrics["graph_topology"]["total_memories"] == 3
        assert nerd_metrics["graph_topology"]["total_links"] == 1
        assert nerd_metrics["graph_topology"]["link_type_counts"] == {"SUPERSEDES": 1}
        assert global_nerd_metrics["graph_topology"]["total_memories"] == 3
        assert global_nerd_metrics["composition"]["by_workspace"] == [
            {"key": seed_runtime.workspace_id, "label": seed_runtime.workspace_id, "count": 2},
            {"key": "workspace-b", "label": "workspace-b", "count": 1},
        ]
        assert workspace_override_nerd_metrics["graph_topology"]["total_memories"] == 1
        assert workspace_override_nerd_metrics["composition"]["by_workspace"] == [
            {"key": "workspace-b", "label": "workspace-b", "count": 1}
        ]
        assert nerd_metrics["memory_lifecycle"]["by_status"]["active"] == 2
        assert nerd_metrics["memory_lifecycle"]["by_status"]["stale"] == 1
        assert nerd_metrics["memory_lifecycle"]["by_type"] == {"fact": 1, "plan": 2}
        assert nerd_metrics["memory_lifecycle"]["cold_memory_count"] == 2
        assert nerd_metrics["composition"]["by_workspace"] == [
            {"key": seed_runtime.workspace_id, "label": seed_runtime.workspace_id, "count": 2},
            {"key": "workspace-b", "label": "workspace-b", "count": 1},
        ]
        assert {item["key"] for item in nerd_metrics["composition"]["by_tag"]} == {"api", "dashboard", "foreign-tag"}
        assert {item["key"] for item in nerd_metrics["composition"]["by_content_tag"]} == {"api", "dashboard", "foreign-tag"}
        assert nerd_metrics["composition"]["by_provenance_tag"] == []
        assert nerd_metrics["composition"]["by_status"] == [
            {"key": "active", "label": "active", "count": 2},
            {"key": "stale", "label": "stale", "count": 1},
        ]
        assert [bucket["key"] for bucket in nerd_metrics["distributions"]["created_age_buckets"]] == [
            "lt_1d",
            "1d_to_7d",
            "7d_to_30d",
            "30d_to_90d",
            "gte_90d",
        ]
        assert sum(bucket["created_count"] for bucket in nerd_metrics["timelines"]["memory_activity"]) == 2
        assert nerd_metrics["timelines"]["memory_activity"][-1]["total_content_bytes"] > 0
        deduplicator_event = next(item for item in nerd_metrics["maintenance"]["events"] if item["task_name"] == "deduplicator")
        assert deduplicator_event["strategy_used"] == "semantic"
        assert deduplicator_event["merged_count"] == 1
        assert deduplicator_event["candidate_count"] == 6
        assert deduplicator_event["lines_compressed"] == 12
        assert deduplicator_event["impact_summary"] == "merged=1, lines=12"
        assert nerd_metrics["lifecycle_trends"]["never_surfaced_backlog"]
        assert nerd_metrics["lifecycle_trends"]["cold_tail"]
        assert nerd_metrics["growth_dynamics"]["top_tag_trends"]
        assert nerd_metrics["growth_dynamics"]["workspace_contribution_share"]
        assert nerd_metrics["retrieval"]["summary"]["search_invocations"] == 2
        assert nerd_metrics["retrieval"]["summary"]["zero_result_searches"] == 1
        assert nerd_metrics["retrieval"]["summary"]["read_events"] == 2
        assert nerd_metrics["retrieval"]["top_read_memories"][0]["memory_id"] == primary.id
        assert nerd_metrics["retrieval"]["top_read_memories"][0]["read_count"] == 2
        assert nerd_metrics["retrieval"]["top_search_memories"][0]["memory_id"] == primary.id
        compaction_family = next(item for item in nerd_metrics["maintenance_summary"]["by_family"] if item["key"] == "compaction")
        deduplicator_yield = next(item for item in nerd_metrics["maintenance_summary"]["by_agent"] if item["key"] == "deduplicator")
        assert compaction_family["merged_count"] == 1
        assert compaction_family["lines_compressed"] == 12
        assert deduplicator_yield["family_key"] == "compaction"
        assert deduplicator_yield["delta_per_completed_run"] == 1.0
        assert any(item["key"] == "compaction" for item in nerd_metrics["maintenance_summary"]["family_delta_series"])
        assert nerd_metrics["search_quality"]["semantic_enabled"] is True
        assert any(item["task_name"] == "memory-curator" for item in nerd_metrics["route_audit"])
        assert nerd_metrics["provider_policy"]["stats"][0]["key"] == "provider_policy_route_exhaustion_count"
        assert "by_task" in nerd_metrics["provider_policy"]
        assert "by_provider" in nerd_metrics["provider_policy"]
        summarize_route = next(item for item in nerd_metrics["route_audit"] if item["task_name"] == "summarize-memory")
        assert summarize_route["task_class"] == "deterministic"
        assert summarize_route["resolved_provider_key"] is None
        assert any(alert["key"] == "cold_memory_rate" for alert in nerd_metrics["alerts"])
        assert prune_logs["deleted"] == 0
        assert repair_search["rebuilt"] is True
        fact_checker = next(agent for agent in overview["agent_runs"] if agent["task_name"] == "fact-checker")
        deduplicator = next(agent for agent in overview["agent_runs"] if agent["task_name"] == "deduplicator")
        taxonomist = next(agent for agent in overview["agent_runs"] if agent["task_name"] == "taxonomist")
        assert fact_checker["failed_runs"] == 1
        assert deduplicator["last_result_metadata"]["strategy_used"] == "semantic"
        assert deduplicator["last_result_metadata"]["candidate_count"] == 6
        assert deduplicator["last_result_metadata"]["campaign_key"] is None
        assert taxonomist["last_result_metadata"]["campaign_key"] == "lightweight_review"
        assert taxonomist["last_result_metadata"]["premium_execution_count"] == 1
        assert taxonomist["last_result_metadata"]["compatible_batch_calls"] == 1
        assert taxonomist["last_result_metadata"]["work_items_per_premium_execution"] == 3.0
        assert taxonomist["last_result_metadata"]["mutations_per_premium_execution"] == 4.0
        assert taxonomist["last_result_metadata"]["tool_calls_per_premium_execution"] == 6.0
        assert overview["recent_agent_runs"]
        assert any(run["result_metadata"]["strategy_used"] == "semantic" for run in overview["recent_agent_runs"])
        assert any(run["result_metadata"]["campaign_key"] == "lightweight_review" for run in overview["recent_agent_runs"])
        assert tasks["tasks"][0]["last_error"] == "missing ext link"
        assert {record["id"] for record in memories["records"]} == {primary.id, superseded.id}
        assert detail["record"]["id"] == primary.id
        assert missing_detail == {"status": "error", "error": "memory_not_found"}
        assert invalid_limit == {"status": "error", "error": "limit_out_of_range"}
        assert any(tool["name"] == "internal_merge_memory_into_canonical" for tool in internal_tools["tools"])
        assert detail["superseded"][0]["id"] == superseded.id
        assert hook_start["status"] == "ok"
        assert hook_start["active_client_count"] == 1
        assert hook_start["shutdown_scheduled"] is False
        assert hook_noop == {}
        assert "record_thought" in hook_reminder["systemMessage"]
        assert hook_end["status"] == "ok"
        assert hook_end["active_client_count"] == 0
        assert hook_end["shutdown_scheduled"] is False
        assert "running_count" in fact_checker
        assert "next_available_at" in fact_checker
        assert "last_result_summary" in fact_checker
        assert run_agent["status"] == "enqueued"
        assert len(run_all["results"]) >= 1
        assert cancel_task["status"] in {"cancellation_requested", "cancelled"}
        assert cancel_task["task"]["cancellation_reason"] == "operator_cancelled"
        assert conversations["conversations"][0]["request_id"] == "req-api-1"
        assert conversations["conversations"][0]["subprocess_pid"] == 6543
        assert created_link["status"] == "created"
        assert deleted_link["status"] == "deleted"
        assert "MCP Memory Dashboard" in dashboard or "MCP Memory Command Center" in dashboard
        assert "Command Bar" in dashboard or '<div id="root"></div>' in dashboard
        if "MCP Memory Dashboard" in dashboard:
            assert "Background Agents" in dashboard
            assert "Memory Metrics" in dashboard
            assert "Embedding Backend" in dashboard
            assert "Recent Agent Runs" in dashboard
            assert "Recent Logs" in dashboard
            assert "AI Provider Usage" in dashboard
            assert "Refresh Logs" in dashboard
            assert "Agent Controls" in dashboard


@pytest.mark.asyncio
async def test_management_api_health_and_overview_include_cache_metrics(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)

    app = create_daemon_app(workspace_root_override=None, cwd=workspace)
    async with app.router.lifespan_context(app):
        service = app.state.routes.service
        config = Config()
        config.storage.cache.enabled = True
        config.storage.cache.mode = "readonly"
        cache = SharedReadCache(tmp_path / "shared_read_cache.sqlite3")
        cache.increment_metric("search_requests", amount=6)
        cache.increment_metric("fresh_exact_search_hits", amount=3)
        cache.increment_metric("projection_fallbacks", amount=1)
        cache.increment_metric("read_requests", amount=4)
        cache.increment_metric("validated_read_hits", amount=2)
        cache.increment_metric("warmed_projection_rows", amount=9)

        service._storage_backend = "postgres"
        service._config = config
        service._read_cache = cache

        health = await _request_json(app.state.metadata, "/api/health")
        overview = await _request_json(app.state.metadata, "/api/overview")

    assert health["cache"]["state"] == "active"
    assert health["cache"]["metrics"]["search_requests"] == 6
    assert health["cache"]["metrics"]["fresh_exact_search_hits"] == 3
    assert health["cache"]["metrics"]["projection_fallbacks"] == 1
    assert health["cache"]["metrics"]["fresh_exact_search_hit_rate"] == 0.5
    assert health["cache"]["metrics"]["warmed_projection_rows"] == 9
    assert health["cache"]["metrics"]["recent"]["window_minutes"] == 15
    assert health["cache"]["metrics"]["recent"]["search_requests"] == 6
    assert health["cache"]["metrics"]["recent"]["projection_fallbacks"] == 1
    assert health["cache"]["metrics"]["recent"]["fresh_exact_search_hit_rate"] == 0.5
    assert health["cache"]["metrics"]["recent"]["warmed_projection_rows"] == 9
    assert overview["cache"]["state"] == "active"
    assert overview["cache"]["metrics"]["read_requests"] == 4
    assert overview["cache"]["metrics"]["validated_read_hits"] == 2
    assert overview["cache"]["metrics"]["recent"]["read_requests"] == 4
    assert overview["cache"]["metrics"]["recent"]["validated_read_hits"] == 2


@pytest.mark.asyncio
async def test_daemon_lifespan_attempts_embedding_model_cache(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    cache_calls: list[str] = []

    class FakeEmbedder:
        model_name = "fake-local-model"

        def cache_model(self) -> bool:
            cache_calls.append("called")
            return True

    runtime.embedder = FakeEmbedder()
    monkeypatch.setattr("mcp_memory.daemon_app.create_runtime_from_spec", lambda spec: runtime)

    app = create_daemon_app(workspace_root_override=None, cwd=workspace)

    try:
        async with app.router.lifespan_context(app):
            await asyncio.sleep(0.05)
            assert cache_calls == ["called"]
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_daemon_lifespan_waits_for_embedding_model_cache_before_ready(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    cache_started = threading.Event()
    release_cache = threading.Event()

    class FakeEmbedder:
        model_name = "fake-local-model"

        def cache_model(self) -> bool:
            cache_started.set()
            assert release_cache.wait(timeout=1.0)
            return True

    runtime.embedder = FakeEmbedder()
    monkeypatch.setattr("mcp_memory.daemon_app.create_runtime_from_spec", lambda spec: runtime)

    app = create_daemon_app(workspace_root_override=None, cwd=workspace)
    lifespan = app.router.lifespan_context(app)
    enter_task: asyncio.Task[object] | None = None

    try:
        enter_task = asyncio.create_task(lifespan.__aenter__())
        await asyncio.to_thread(cache_started.wait, 1.0)
        assert cache_started.is_set() is True
        assert enter_task is not None
        assert enter_task.done() is False

        release_cache.set()
        assert enter_task is not None
        await asyncio.wait_for(enter_task, timeout=1.0)
        assert hasattr(app.state, "metadata")
    finally:
        if enter_task is not None and not enter_task.done():
            enter_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await enter_task
        elif enter_task is not None:
            await lifespan.__aexit__(None, None, None)
        runtime.close()


@pytest.mark.asyncio
async def test_daemon_idle_shutdown_waits_for_last_client_and_cancels_on_reconnect(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("mcp_memory.daemon_app._IDLE_SHUTDOWN_DELAY_SECONDS", 0.01)
    shutdown_calls: list[tuple[int, int]] = []

    monkeypatch.setattr("mcp_memory.daemon_app._count_running_background_tasks", lambda _app: 0)

    monkeypatch.setattr("mcp_memory.daemon_app.os.getpid", lambda: 4321)
    monkeypatch.setattr("mcp_memory.daemon_app.os.kill", lambda pid, sig: shutdown_calls.append((pid, sig)))

    idle_app = SimpleNamespace(
        state=SimpleNamespace(
            idle_shutdown_task=None,
            routes=SimpleNamespace(
                hook_service=SimpleNamespace(get_active_client_count=lambda: 0),
            ),
        ),
    )

    idle_task = asyncio.create_task(daemon_app_module._shutdown_daemon_when_idle(cast(Any, idle_app)))
    idle_app.state.idle_shutdown_task = idle_task
    await asyncio.wait_for(idle_task, timeout=0.2)

    assert shutdown_calls == [(4321, 15)]
    assert idle_app.state.idle_shutdown_task is None

    shutdown_calls.clear()

    reconnect_app = SimpleNamespace(
        state=SimpleNamespace(
            idle_shutdown_task=None,
            routes=SimpleNamespace(
                hook_service=SimpleNamespace(get_active_client_count=lambda: 1),
            ),
        ),
    )

    reconnect_task = asyncio.create_task(daemon_app_module._shutdown_daemon_when_idle(cast(Any, reconnect_app)))
    reconnect_app.state.idle_shutdown_task = reconnect_task
    await asyncio.wait_for(reconnect_task, timeout=0.2)

    assert shutdown_calls == []
    assert reconnect_app.state.idle_shutdown_task is None


@pytest.mark.asyncio
async def test_daemon_idle_shutdown_defers_while_background_tasks_are_running(monkeypatch) -> None:
    monkeypatch.setattr("mcp_memory.daemon_app._IDLE_SHUTDOWN_DELAY_SECONDS", 0.01)

    shutdown_calls: list[tuple[int, int]] = []
    running_counts_seen: list[int] = []
    deferred_check = asyncio.Event()
    running_counts = iter([1, 1, 0])

    def _fake_count_running_background_tasks(_app) -> int:
        count = next(running_counts, 0)
        running_counts_seen.append(count)
        if count > 0:
            deferred_check.set()
        return count

    monkeypatch.setattr(
        "mcp_memory.daemon_app._count_running_background_tasks",
        _fake_count_running_background_tasks,
    )
    monkeypatch.setattr("mcp_memory.daemon_app.os.getpid", lambda: 2468)
    monkeypatch.setattr("mcp_memory.daemon_app.os.kill", lambda pid, sig: shutdown_calls.append((pid, sig)))

    app = SimpleNamespace(
        state=SimpleNamespace(
            idle_shutdown_task=None,
            routes=SimpleNamespace(
                hook_service=SimpleNamespace(get_active_client_count=lambda: 0),
            ),
        ),
    )

    shutdown_task = asyncio.create_task(daemon_app_module._shutdown_daemon_when_idle(cast(Any, app)))
    app.state.idle_shutdown_task = shutdown_task

    await asyncio.wait_for(deferred_check.wait(), timeout=0.2)
    assert shutdown_calls == []

    await asyncio.wait_for(shutdown_task, timeout=0.2)
    assert shutdown_calls == [(2468, 15)]
    assert running_counts_seen[:2] == [1, 1]
    assert 0 in running_counts_seen
    assert app.state.idle_shutdown_task is None


@pytest.mark.asyncio
async def test_daemon_idle_shutdown_defers_while_recent_http_activity_exists(monkeypatch) -> None:
    monkeypatch.setattr("mcp_memory.daemon_app._IDLE_SHUTDOWN_DELAY_SECONDS", 0.01)
    monkeypatch.setattr("mcp_memory.daemon_app._HTTP_ACTIVITY_GRACE_SECONDS", 1.0)

    shutdown_calls: list[tuple[int, int]] = []

    monkeypatch.setattr("mcp_memory.daemon_app._count_running_background_tasks", lambda _app: 0)
    monkeypatch.setattr("mcp_memory.daemon_app.os.getpid", lambda: 9753)
    monkeypatch.setattr("mcp_memory.daemon_app.os.kill", lambda pid, sig: shutdown_calls.append((pid, sig)))

    app = SimpleNamespace(
        state=SimpleNamespace(
            idle_shutdown_task=None,
            last_http_activity_at=time.monotonic(),
            routes=SimpleNamespace(
                hook_service=SimpleNamespace(get_active_client_count=lambda: 0),
            ),
        ),
    )

    shutdown_task = asyncio.create_task(daemon_app_module._shutdown_daemon_when_idle(cast(Any, app)))
    app.state.idle_shutdown_task = shutdown_task

    await asyncio.sleep(0.05)
    assert shutdown_calls == []

    app.state.last_http_activity_at = 0.0

    await asyncio.wait_for(shutdown_task, timeout=0.2)

    assert shutdown_calls == [(9753, 15)]
    assert app.state.idle_shutdown_task is None


@pytest.mark.asyncio
async def test_record_http_activity_cancels_idle_shutdown(monkeypatch) -> None:
    monkeypatch.setattr("mcp_memory.daemon_app.time.monotonic", lambda: 12.5)

    cancelled: list[str] = []

    async def _fake_cancel(_app) -> None:
        cancelled.append("cancelled")

    monkeypatch.setattr("mcp_memory.daemon_app._cancel_idle_shutdown_task", _fake_cancel)

    app = SimpleNamespace(
        state=SimpleNamespace(
            last_http_activity_at=0.0,
        ),
    )

    await daemon_app_module._record_http_activity(cast(Any, app))

    assert app.state.last_http_activity_at == 12.5
    assert cancelled == ["cancelled"]


@pytest.mark.asyncio
async def test_management_api_overview_includes_top_read_memories(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)

    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    try:
        assert runtime.repository is not None
        assert runtime.workspace_id is not None
        alpha = runtime.repository.create_memory(
            title="Alpha read memory",
            content="Alpha details.",
            workspace_ids=[runtime.workspace_id],
            memory_type="fact",
        )
        beta = runtime.repository.create_memory(
            title="Beta read memory",
            content="Beta details.",
            workspace_ids=[runtime.workspace_id],
            memory_type="fact",
            status="archived",
        )
        assert alpha is not None and beta is not None
        runtime.repository.record_access(alpha.id, 1.0, "2026-03-15T10:00:00+00:00", increment_read_count=True)
        runtime.repository.record_access(alpha.id, 2.0, "2026-03-15T10:05:00+00:00", increment_read_count=True)
        runtime.repository.record_access(beta.id, 1.0, "2026-03-15T10:10:00+00:00", increment_read_count=True)
    finally:
        runtime.close()

    app = create_daemon_app(workspace_root_override=None, cwd=workspace)
    async with app.router.lifespan_context(app):
        overview = await _request_json(app.state.metadata, "/api/overview")

        assert [record["title"] for record in overview["top_read_memories"]] == [
            "Alpha read memory",
            "Beta read memory",
        ]
        assert [record["read_count"] for record in overview["top_read_memories"]] == [2, 1]
        assert [record["title"] for record in overview["top_read_memories_active"]] == [
            "Alpha read memory",
        ]
        assert [record["status"] for record in overview["top_read_memories_active"]] == ["active"]


def test_daemon_http_dashboard_and_api_routes(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)

    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    try:
        assert runtime.repository is not None
        assert runtime.workspace_id is not None
        runtime.repository.create_memory(
            title="HTTP dashboard fact",
            content="Serve the command center over HTTP.",
            workspace_ids=[runtime.workspace_id],
            memory_type="fact",
        )
    finally:
        runtime.close()

    app = create_daemon_app(workspace_root_override=None, cwd=workspace)
    with TestClient(app) as client:
        dashboard = client.get("/dashboard")
        search_dashboard = client.get("/dashboard/search")
        overview = client.get("/api/overview")
        search = client.post("/api/memories/search", json={"query": "command center", "limit": 5})
        record_thought = client.post("/api/record-thought", json={"content": "dogfood the react shell"})
        overview_after = client.get("/api/overview")
        missing_asset = client.get("/assets/missing.js")

    assert dashboard.status_code == 200
    assert search_dashboard.status_code == 200
    assert "MCP Memory Dashboard" in dashboard.text or "Memory Command Center" in dashboard.text
    assert overview.status_code == 200
    assert overview.json()["memories"]["total"] >= 1
    assert overview.json()["premium_usage"]["copilot_premium_requests_today"] == 0
    assert search.status_code == 200
    assert any(result["title"] == "HTTP dashboard fact" for result in search.json()["results"])
    assert record_thought.status_code == 200
    assert record_thought.json()["status"] == "recorded"
    assert overview_after.json()["journal"]["pending_count"] >= overview.json()["journal"]["pending_count"] + 1
    assert missing_asset.status_code == 404


def test_http_overview_defaults_to_global_scope_for_dashboard_calls(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace_a = tmp_path / "workspace-a"
    workspace_b = tmp_path / "workspace-b"
    workspace_a.mkdir(parents=True)
    workspace_b.mkdir(parents=True)

    runtime_a = create_runtime(workspace_root_override=None, cwd=workspace_a)
    runtime_b = create_runtime(workspace_root_override=None, cwd=workspace_b)
    try:
        assert runtime_a.repository is not None
        assert runtime_b.repository is not None
        assert runtime_a.workspace_id is not None
        assert runtime_b.workspace_id is not None
        runtime_a.repository.create_memory(
            title="Workspace A fact",
            content="A scoped fact.",
            workspace_ids=[runtime_a.workspace_id],
            memory_type="fact",
        )
        runtime_b.repository.create_memory(
            title="Workspace B fact",
            content="Another scoped fact.",
            workspace_ids=[runtime_b.workspace_id],
            memory_type="fact",
        )
        runtime_a.repository.create_memory(
            title="Shared fact",
            content="Belongs to both workspaces.",
            workspace_ids=[runtime_a.workspace_id, runtime_b.workspace_id],
            memory_type="fact",
        )
    finally:
        runtime_a.close()
        runtime_b.close()

    app = create_daemon_app(workspace_root_override=None, cwd=workspace_a)
    with TestClient(app) as client:
        global_overview = client.get("/api/overview")
        ignored_scoped_overview = client.get("/api/overview", params={"scope": "workspace", "workspace_id": runtime_b.workspace_id})

    assert global_overview.status_code == 200
    assert ignored_scoped_overview.status_code == 200
    assert global_overview.json()["memories"]["total"] == 3
    assert global_overview.json()["memory_metrics"]["total_memories"] == 3
    assert ignored_scoped_overview.json()["memories"]["total"] == 3


def test_http_operator_lists_default_to_global_scope_and_search_uses_workspace_ranking_context(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace_a = tmp_path / "workspace-a"
    workspace_b = tmp_path / "workspace-b"
    workspace_a.mkdir(parents=True)
    workspace_b.mkdir(parents=True)

    runtime_a = create_runtime(workspace_root_override=None, cwd=workspace_a)
    runtime_b = create_runtime(workspace_root_override=None, cwd=workspace_b)
    try:
        assert runtime_a.repository is not None
        assert runtime_b.repository is not None
        assert runtime_a.task_queue is not None
        assert runtime_a.db_manager is not None
        assert runtime_a.workspace_id is not None
        assert runtime_b.workspace_id is not None

        runtime_a.repository.create_memory(
            title="Workspace A API fact",
            content="Scoped to workspace A.",
            workspace_ids=[runtime_a.workspace_id],
            memory_type="fact",
        )
        runtime_b.repository.create_memory(
            title="Workspace B API fact",
            content="Scoped to workspace B.",
            workspace_ids=[runtime_b.workspace_id],
            memory_type="fact",
        )
        runtime_a.task_queue.enqueue("memory-curator", task_id="api-task-a", workspace_id=runtime_a.workspace_id)
        runtime_a.task_queue.enqueue("deduplicator", task_id="api-task-b", workspace_id=runtime_b.workspace_id)
        runtime_a.db_manager.get_connection().executemany(
            "INSERT INTO runtime_logs (workspace_id, source, logger_name, level, message, created_at, data_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (runtime_a.workspace_id, "daemon", "mcp_memory.server", "INFO", "workspace-a api log", 10.0, "{}"),
                (runtime_b.workspace_id, "daemon", "mcp_memory.server", "INFO", "workspace-b api log", 20.0, "{}"),
            ],
        )
        runtime_a.db_manager.get_connection().commit()
        ProviderUsageRepository(runtime_a.db_manager, workspace_id=runtime_a.workspace_id).record_conversation(
            request_id="api-req-a",
            attempt=1,
            task_name="memory-curator",
            task_id="api-task-a",
            provider_key="gemini-cli",
            provider_name="Gemini CLI",
            model_name="gemini-3-flash-preview",
            subprocess_pid=1111,
            prompt_text="prompt a",
            response_text="response a",
            parsed=None,
            status="completed",
            error_text=None,
            started_at=30.0,
            completed_at=32.0,
        )
        ProviderUsageRepository(runtime_b.db_manager, workspace_id=runtime_b.workspace_id).record_conversation(
            request_id="api-req-b",
            attempt=1,
            task_name="deduplicator",
            task_id="api-task-b",
            provider_key="copilot-mini",
            provider_name="Copilot CLI",
            model_name="gpt-5-mini",
            subprocess_pid=2222,
            prompt_text="prompt b",
            response_text="response b",
            parsed=None,
            status="completed",
            error_text=None,
            started_at=40.0,
            completed_at=44.0,
        )
    finally:
        runtime_a.close()
        runtime_b.close()

    app = create_daemon_app(workspace_root_override=None, cwd=workspace_a)
    with TestClient(app) as client:
        global_tasks = client.get("/api/tasks")
        workspace_tasks = client.get("/api/tasks", params={"scope": "workspace"})
        global_memories = client.get("/api/memories")
        workspace_memories = client.get("/api/memories", params={"scope": "workspace"})
        global_search = client.post("/api/memories/search", json={"query": "API fact", "limit": 10})
        workspace_search = client.post("/api/memories/search?scope=workspace", json={"query": "API fact", "limit": 10})
        global_logs = client.post("/api/logs", json={"source": "daemon"})
        workspace_logs = client.post("/api/logs?scope=workspace", json={"source": "daemon"})
        global_conversations = client.post("/api/ai-conversations", json={"limit": 10})
        workspace_conversations = client.post("/api/ai-conversations?scope=workspace", json={"limit": 10})

    assert global_tasks.status_code == 200
    global_task_ids = {task["id"] for task in global_tasks.json()["tasks"]}
    workspace_task_ids = {task["id"] for task in workspace_tasks.json()["tasks"]}
    assert {"api-task-a", "api-task-b"} <= global_task_ids
    assert "api-task-a" in workspace_task_ids
    assert "api-task-b" not in workspace_task_ids

    assert global_memories.status_code == 200
    global_memory_titles = {record["title"] for record in global_memories.json()["records"]}
    workspace_memory_titles = {record["title"] for record in workspace_memories.json()["records"]}
    assert {"Workspace A API fact", "Workspace B API fact"} <= global_memory_titles
    assert "Workspace A API fact" in workspace_memory_titles
    assert "Workspace B API fact" not in workspace_memory_titles

    assert global_search.status_code == 200
    assert {record["title"] for record in global_search.json()["results"]} == {"Workspace A API fact", "Workspace B API fact"}
    workspace_results = workspace_search.json()["results"]
    # Search remains global; workspace scope only supplies ranking context for the active workspace.
    assert {record["title"] for record in workspace_results} == {"Workspace A API fact", "Workspace B API fact"}
    assert workspace_results[0]["title"] == "Workspace A API fact"
    workspace_a_result = next(record for record in workspace_results if record["title"] == "Workspace A API fact")
    workspace_b_result = next(record for record in workspace_results if record["title"] == "Workspace B API fact")
    assert workspace_a_result["score"] >= workspace_b_result["score"]

    assert global_logs.status_code == 200
    global_log_messages = {record["message"] for record in global_logs.json()["logs"]}
    workspace_log_messages = {record["message"] for record in workspace_logs.json()["logs"]}
    assert {"workspace-a api log", "workspace-b api log"} <= global_log_messages
    assert "workspace-a api log" in workspace_log_messages
    assert "workspace-b api log" not in workspace_log_messages

    assert global_conversations.status_code == 200
    global_request_ids = {record["request_id"] for record in global_conversations.json()["conversations"]}
    workspace_request_ids = {record["request_id"] for record in workspace_conversations.json()["conversations"]}
    assert {"api-req-a", "api-req-b"} <= global_request_ids
    assert "api-req-a" in workspace_request_ids
    assert "api-req-b" not in workspace_request_ids


def test_daemon_lifespan_ensures_dashboard_frontend_is_built(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)

    ensured_static_roots: list[Path] = []
    monkeypatch.setattr(
        "mcp_memory.daemon_app._ensure_dashboard_frontend_ready",
        lambda static_root: ensured_static_roots.append(static_root),
    )

    app = create_daemon_app(workspace_root_override=None, cwd=workspace)

    with TestClient(app):
        pass

    assert ensured_static_roots
    assert ensured_static_roots[0].name == "static"


def test_management_api_accepts_30_day_nerd_metrics_window(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)

    app = create_daemon_app(workspace_root_override=None, cwd=workspace)
    with TestClient(app) as client:
        response = client.get(
            "/api/metrics/nerd",
            params={"window_hours": 24 * 30, "bucket_minutes": 60},
        )

    assert response.status_code == 200
    assert response.json()["window_hours"] == 24 * 30


@pytest.mark.asyncio
async def test_daemon_zmq_dispatch_rejects_invalid_transport_payload_shapes(tmp_path: Path) -> None:
    socket_path = tmp_path / "daemon.sock"
    server = DaemonZmqServer(
        context_factory=lambda _arguments: None,
        hook_handlers={},
        routes_provider=lambda: None,
        socket_path=socket_path,
        metadata_provider=lambda: None,
    )

    assert await server._dispatch(b"not-json") == {"status": "error", "error": "invalid_transport_payload"}
    assert await server._dispatch(json.dumps(["not", "an", "object"]).encode("utf-8")) == {
        "status": "error",
        "error": "invalid_transport_payload",
    }
    assert await server._dispatch(json.dumps({"payload": {}}).encode("utf-8")) == {
        "status": "error",
        "error": "transport_path_required",
    }
    assert await server._dispatch(json.dumps({"path": "/api/overview", "payload": []}).encode("utf-8")) == {
        "status": "error",
        "error": "transport_payload_must_be_object",
    }


@pytest.mark.asyncio
async def test_daemon_zmq_dispatch_processes_tool_requests_concurrently(monkeypatch, tmp_path: Path) -> None:
    socket_path = tmp_path / "daemon.sock"

    def slow_service(_ctx: ApplicationContext, arguments: dict) -> dict:
        time.sleep(0.25)
        return {"status": "ok", "query": arguments.get("query")}

    monkeypatch.setattr(
        "mcp_memory.mcp.transport.tool_services",
        lambda: {"search_memory_records": slow_service},
    )

    server = DaemonZmqServer(
        context_factory=lambda _arguments: ApplicationContext(),
        hook_handlers={},
        routes_provider=lambda: None,
        socket_path=socket_path,
        metadata_provider=lambda: None,
        max_concurrent_requests=4,
    )
    await server.start()
    await asyncio.sleep(0.01)

    try:
        metadata = SimpleNamespace(socket_path=str(socket_path), transport="zmq")
        async def _worker(index: int) -> tuple[int, str, float]:
            started_at = time.monotonic()
            payload = await asyncio.to_thread(
                request_daemon_json,
                metadata,
                "/internal/tools/search_memory_records",
                {"query": f"query-{index}"},
                timeout_seconds=2.0,
            )
            elapsed = time.monotonic() - started_at
            return index, payload["contents"][0]["text"], elapsed

        started_at = time.monotonic()
        results = await asyncio.gather(*(_worker(index) for index in range(4)))
        total_elapsed = time.monotonic() - started_at
    finally:
        await server.stop()

    assert len(results) == 4
    assert total_elapsed < 0.7
    for index, payload_text, elapsed in results:
        payload = json.loads(payload_text)
        assert payload == {"status": "ok", "query": f"query-{index}"}
        assert elapsed < 0.7


@dataclass(frozen=True)
class _HealthMetadata:
    status: str
    transport: str
    socket_path: str


@pytest.mark.asyncio
async def test_daemon_zmq_record_thought_tool_fast_path_ignores_saturated_request_pool(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    blocking_request_started = asyncio.Event()
    release_blocking_request = asyncio.Event()

    async def _blocking_post_tool_use(_ctx, _arguments: dict[str, object]) -> dict[str, object]:
        blocking_request_started.set()
        await release_blocking_request.wait()
        return {"status": "ok"}

    monkeypatch.setattr(daemon_app_module, "_handle_post_tool_use", _blocking_post_tool_use)

    app = create_daemon_app(workspace_root_override=None, cwd=workspace)
    async with app.router.lifespan_context(app):
        metadata = app.state.metadata
        blocking_request = asyncio.create_task(
            asyncio.to_thread(
                request_daemon_json,
                metadata,
                "/api/hooks/post-tool-use",
                {"sessionId": "conversation-1", "tool_name": "apply_patch", "timestamp": 1.0},
                timeout_seconds=1.0,
            )
        )
        await asyncio.wait_for(blocking_request_started.wait(), timeout=0.5)

        request_started_at = time.monotonic()
        response = await asyncio.wait_for(
            asyncio.to_thread(
                request_daemon_json,
                metadata,
                "/internal/tools/record_thought",
                {"content": "fast-path thought", "__workspace_root": str(workspace)},
                timeout_seconds=1.0,
            ),
            timeout=1.5,
        )
        request_elapsed = time.monotonic() - request_started_at

        decoded = json.loads(response["contents"][0]["text"])
        assert blocking_request.done() is False
        assert decoded["status"] == "recorded"
        assert decoded["entry"]["workspace_id"] == app.state.routes.ctx.workspace_id
        assert request_elapsed < 1.0

        pending_entries = app.state.routes.ctx.journal.get_pending(limit=10)
        assert len(pending_entries) == 1
        assert pending_entries[0].workspace_id == app.state.routes.ctx.workspace_id

        release_blocking_request.set()
        assert await asyncio.wait_for(blocking_request, timeout=0.5) == {"status": "ok"}


@pytest.mark.asyncio
async def test_daemon_zmq_api_record_thought_fast_path_ignores_saturated_request_pool(tmp_path: Path) -> None:
    socket_path = tmp_path / "daemon.sock"
    blocking_request_started = asyncio.Event()
    release_blocking_request = asyncio.Event()

    async def _blocking_handler(_payload: dict[str, object]) -> dict[str, object]:
        blocking_request_started.set()
        await release_blocking_request.wait()
        return {"status": "ok"}

    routes = SimpleNamespace(
        service=SimpleNamespace(
            record_thought=lambda content: {"status": "recorded", "content": content},
        )
    )
    server = DaemonZmqServer(
        context_factory=lambda _arguments: None,
        hook_handlers={"/api/hooks/block": _blocking_handler},
        routes_provider=lambda: routes,
        socket_path=socket_path,
        metadata_provider=lambda: _HealthMetadata(
            status="ready",
            transport="zmq",
            socket_path=str(socket_path),
        ),
        max_concurrent_requests=1,
    )
    await server.start()
    await asyncio.sleep(0.01)

    try:
        metadata = SimpleNamespace(socket_path=str(socket_path), transport="zmq")
        blocking_request = asyncio.create_task(
            asyncio.to_thread(
                request_daemon_json,
                metadata,
                "/api/hooks/block",
                {},
                timeout_seconds=1.0,
            )
        )
        await asyncio.wait_for(blocking_request_started.wait(), timeout=0.5)

        request_started_at = time.monotonic()
        response = await asyncio.wait_for(
            asyncio.to_thread(
                request_daemon_json,
                metadata,
                "/api/record-thought",
                {"content": "fast api thought"},
                timeout_seconds=0.2,
            ),
            timeout=0.5,
        )
        request_elapsed = time.monotonic() - request_started_at

        assert blocking_request.done() is False
        assert response == {"status": "recorded", "content": "fast api thought"}
        assert request_elapsed < 0.2

        release_blocking_request.set()
        assert await asyncio.wait_for(blocking_request, timeout=0.5) == {"status": "ok"}
    finally:
        release_blocking_request.set()
        await server.stop()


@pytest.mark.asyncio
async def test_daemon_zmq_health_fast_path_ignores_saturated_request_pool(tmp_path: Path) -> None:
    socket_path = tmp_path / "daemon.sock"
    blocking_request_started = asyncio.Event()
    release_blocking_request = asyncio.Event()

    async def _blocking_handler(_payload: dict[str, object]) -> dict:
        blocking_request_started.set()
        await release_blocking_request.wait()
        return {"status": "ok"}

    server = DaemonZmqServer(
        context_factory=lambda _arguments: ApplicationContext(),
        hook_handlers={"/api/hooks/block": _blocking_handler},
        routes_provider=lambda: None,
        socket_path=socket_path,
        metadata_provider=lambda: _HealthMetadata(
            status="ready",
            transport="zmq",
            socket_path=str(socket_path),
        ),
        max_concurrent_requests=1,
    )
    await server.start()
    await asyncio.sleep(0.01)

    try:
        metadata = SimpleNamespace(socket_path=str(socket_path), transport="zmq")
        blocking_request = asyncio.create_task(
            asyncio.to_thread(
                request_daemon_json,
                metadata,
                "/api/hooks/block",
                {},
                timeout_seconds=1.0,
            )
        )
        await asyncio.wait_for(blocking_request_started.wait(), timeout=0.5)

        health_started_at = time.monotonic()
        health = await asyncio.wait_for(
            asyncio.to_thread(
                request_daemon_json,
                metadata,
                "/internal/health",
                None,
                timeout_seconds=0.2,
            ),
            timeout=0.5,
        )
        health_elapsed = time.monotonic() - health_started_at

        assert blocking_request.done() is False
        assert health["status"] == "ready"
        assert health["transport"] == "zmq"
        assert health["socket_path"] == str(socket_path)
        assert health["transport_diagnostics"]["max_concurrent_requests"] == 1
        assert health["transport_diagnostics"]["queued_waiter_count"] == 0
        assert isinstance(health["transport_diagnostics"]["active_requests"], list)
        assert isinstance(health["transport_diagnostics"]["recent_requests"], list)
        assert health_elapsed < 0.2

        release_blocking_request.set()
        assert await asyncio.wait_for(blocking_request, timeout=0.5) == {"status": "ok"}
    finally:
        release_blocking_request.set()
        await server.stop()


@pytest.mark.asyncio
async def test_management_health_reports_transport_queue_wait_and_pressure(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    socket_path = tmp_path / "daemon.sock"
    first_request_started = asyncio.Event()
    release_first_request = asyncio.Event()

    async def _blocking_handler(payload: dict[str, object]) -> dict:
        if payload.get("request") == "first":
            first_request_started.set()
            await release_first_request.wait()
        return {"status": "ok", "request": payload.get("request")}

    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    hook_service = HookReminderService(runtime.db_manager, runtime.workspace_id)
    server = DaemonZmqServer(
        context_factory=lambda _arguments: ApplicationContext(),
        hook_handlers={"/api/hooks/block": _blocking_handler},
        routes_provider=lambda: None,
        socket_path=socket_path,
        metadata_provider=lambda: _HealthMetadata(
            status="ready",
            transport="zmq",
            socket_path=str(socket_path),
        ),
        max_concurrent_requests=1,
    )
    service = ManagementService(
        runtime,
        controller=DaemonControllerView(hook_service=hook_service, transport_server=server),
    )

    await server.start()
    await asyncio.sleep(0.01)

    try:
        metadata = SimpleNamespace(socket_path=str(socket_path), transport="zmq")
        first_request = asyncio.create_task(
            asyncio.to_thread(
                request_daemon_json,
                metadata,
                "/api/hooks/block",
                {"request": "first"},
                timeout_seconds=1.0,
            )
        )
        await asyncio.wait_for(first_request_started.wait(), timeout=0.5)

        queued_request = asyncio.create_task(
            asyncio.to_thread(
                request_daemon_json,
                metadata,
                "/api/hooks/block",
                {"request": "second"},
                timeout_seconds=1.0,
            )
        )

        for _ in range(50):
            transport_diagnostics = service.get_health().transport_diagnostics
            if transport_diagnostics.queued_waiter_count == 1:
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("expected one queued transport waiter during saturation")

        assert transport_diagnostics.current_in_flight_count == 1
        assert transport_diagnostics.current_constrained_in_flight_count == 1
        assert transport_diagnostics.request_slots_available == 0
        assert transport_diagnostics.max_concurrent_requests == 1
        assert [request.path for request in transport_diagnostics.active_requests] == [
            "/api/hooks/block",
            "/api/hooks/block",
        ]
        assert {request.phase for request in transport_diagnostics.active_requests} == {"dispatching", "queued"}
        assert all(request.age_ms >= 0.0 for request in transport_diagnostics.active_requests)

        release_first_request.set()
        assert await asyncio.wait_for(first_request, timeout=0.5) == {"status": "ok", "request": "first"}
        assert await asyncio.wait_for(queued_request, timeout=0.5) == {"status": "ok", "request": "second"}

        completed_diagnostics = service.get_health().transport_diagnostics
        assert completed_diagnostics.current_in_flight_count == 0
        assert completed_diagnostics.queued_waiter_count == 0
        assert completed_diagnostics.recent_completed_request_count >= 2
        assert completed_diagnostics.recent_queue_wait_avg_ms > 0.0
        assert completed_diagnostics.recent_queue_wait_max_ms > 0.0
        assert completed_diagnostics.recent_execution_max_ms > 0.0
        completed_requests = [
            request
            for request in completed_diagnostics.recent_requests
            if request.path == "/api/hooks/block"
        ]
        assert len(completed_requests) >= 2
        assert {request.phase for request in completed_requests[:2]} == {"reply_sent"}
        assert {request.response_status for request in completed_requests[:2]} == {"ok"}
        assert any(request.queue_wait_ms > 0.0 for request in completed_requests[:2])
        assert all(request.total_ms >= request.execution_ms for request in completed_requests[:2])
    finally:
        release_first_request.set()
        await server.stop()
        runtime.close()


@pytest.mark.asyncio
async def test_transport_slow_path_warning_is_persisted_to_runtime_logs(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    socket_path = tmp_path / "daemon.sock"

    async def _slow_handler(_payload: dict[str, object]) -> dict[str, object]:
        await asyncio.sleep(1.05)
        return {"status": "ok"}

    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    assert runtime.db_manager is not None
    assert runtime.workspace_id is not None

    transport_logger = logging.getLogger("mcp_memory.daemon_transport")
    handler = SQLiteStructuredLogHandler(
        db_manager=runtime.db_manager,
        workspace_id=runtime.workspace_id,
        source="daemon",
    )
    previous_level = transport_logger.level
    previous_propagate = transport_logger.propagate
    transport_logger.addHandler(handler)
    transport_logger.setLevel(logging.WARNING)
    transport_logger.propagate = False

    server = DaemonZmqServer(
        context_factory=lambda _arguments: ApplicationContext(),
        hook_handlers={"/api/hooks/slow": _slow_handler},
        routes_provider=lambda: None,
        socket_path=socket_path,
        metadata_provider=lambda: _HealthMetadata(
            status="ready",
            transport="zmq",
            socket_path=str(socket_path),
        ),
    )
    service = ManagementService(
        runtime,
        controller=SimpleNamespace(has_runtime=True, client_count=0),
    )

    await server.start()
    try:
        metadata = SimpleNamespace(socket_path=str(socket_path), transport="zmq")
        response = await _request_json(metadata, "/api/hooks/slow", {}, timeout_seconds=2.5)

        assert response == {"status": "ok"}

        logs = service.list_logs(
            source="daemon",
            logger_name="mcp_memory.daemon_transport",
            query="Slow daemon transport request",
            limit=10,
        ).logs

        assert logs
        extra = cast(dict[str, object], logs[0].data["extra"])
        assert logs[0].message == "Slow daemon transport request"
        assert extra["path"] == "/api/hooks/slow"
        assert extra["phase"] == "reply_sent"
        assert cast(float, extra["execution_ms"]) >= 1000.0
        assert extra["response_status"] == "ok"
    finally:
        await server.stop()
        transport_logger.removeHandler(handler)
        transport_logger.setLevel(previous_level)
        transport_logger.propagate = previous_propagate
        handler.close()
        runtime.close()


def test_http_and_zmq_management_dispatch_parity_on_edge_routes(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)

    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    try:
        assert runtime.repository is not None
        assert runtime.task_queue is not None
        assert runtime.workspace_id is not None
        memory = runtime.repository.create_memory(
            title="Parity memory",
            content="Keep HTTP and ZMQ dispatch aligned.",
            workspace_ids=[runtime.workspace_id],
            memory_type="fact",
        )
        assert memory is not None
    finally:
        runtime.close()

    app = create_daemon_app(workspace_root_override=None, cwd=workspace)
    with TestClient(app) as client:
        metadata = app.state.metadata

        http_detail = client.get(f"/api/memories/{memory.id}")
        zmq_detail = request_daemon_json(metadata, f"/api/memories/{memory.id}", None, timeout_seconds=5)

        assert http_detail.status_code == 200
        assert http_detail.json() == zmq_detail

        http_invalid_limit = client.get("/api/tasks", params={"limit": "0"})
        zmq_invalid_limit = request_daemon_json(metadata, "/api/tasks?limit=0", None, timeout_seconds=5)

        assert http_invalid_limit.status_code == 400
        assert http_invalid_limit.json() == zmq_invalid_limit

        http_unknown = client.get("/api/not-a-route")
        zmq_unknown = request_daemon_json(metadata, "/api/not-a-route", None, timeout_seconds=5)

        assert http_unknown.status_code == 404
        assert http_unknown.json() == {"detail": "unknown_transport_path"}
        assert zmq_unknown == {
            "status": "error",
            "error": "unknown_transport_path",
            "path": "/api/not-a-route",
        }

        queue = app.state.routes.ctx.task_queue
        assert queue is not None
        http_task = queue.enqueue(
            "memory-curator",
            task_id="http-cancel-task",
            workspace_id=app.state.routes.ctx.workspace_id,
            available_at=time.time() + 3600.0,
        )
        zmq_task = queue.enqueue(
            "memory-curator",
            task_id="zmq-cancel-task",
            workspace_id=app.state.routes.ctx.workspace_id,
            available_at=time.time() + 3600.0,
        )

        http_cancel = client.post(
            f"/api/admin/tasks/{http_task.id}/cancel",
            json={"cancelled_by": "http-test", "reason": "http_cancelled"},
        )
        zmq_cancel = request_daemon_json(
            metadata,
            f"/api/admin/tasks/{zmq_task.id}/cancel",
            {"cancelled_by": "zmq-test", "reason": "zmq_cancelled"},
            timeout_seconds=5,
        )

        assert http_cancel.status_code == 200
        assert http_cancel.json()["status"] == "cancelled"
        assert http_cancel.json()["task"]["id"] == http_task.id
        assert http_cancel.json()["task"]["cancellation_reason"] == "http_cancelled"
        assert zmq_cancel["status"] == "cancelled"
        assert zmq_cancel["task"]["id"] == zmq_task.id
        assert zmq_cancel["task"]["cancellation_reason"] == "zmq_cancelled"


def test_http_selector_stats_endpoint_reports_fresh_seeded_and_unknown(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)

    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    try:
        assert runtime.task_queue is not None
        assert runtime.workspace_id is not None
        fresh_task = runtime.task_queue.enqueue(
            "memory-curator",
            task_id="selector-fresh-1",
            workspace_id=runtime.workspace_id,
            available_at=0.0,
        )
        seeded_task = runtime.task_queue.enqueue(
            "memory-curator",
            task_id="selector-seeded-1",
            workspace_id=runtime.workspace_id,
            available_at=0.0,
        )
        unknown_task = runtime.task_queue.enqueue(
            "graph-linker",
            task_id="selector-unknown-1",
            workspace_id=runtime.workspace_id,
            available_at=0.0,
        )
        assert runtime.task_queue.claim_next(now=100.0) is not None
        runtime.task_queue.complete(
            fresh_task.id,
            completed_at=101.0,
            run_result={
                "requested_strategy": "semantic",
                "strategy_used": "semantic",
                "strategy_selection_mode": "deterministic_scores",
                "strategy_selection_reason": "selected=semantic from deterministic scores",
                "strategy_selection_scores": {"semantic": 0.95, "lexical": 0.4},
                "selector_feature_snapshot": {
                    "strategy_signals": {"semantic_overlap_share": 0.82},
                    "candidate_population": {
                        "count": 6,
                        "metrics": {
                            "content_chars": {"count": 6, "min": 10.0, "p50": 20.0, "p90": 40.0, "max": 45.0, "mean": 23.0},
                        },
                        "shares": {"never_surfaced_share": 0.5},
                    },
                    "selected_population": {
                        "count": 3,
                        "metrics": {
                            "content_chars": {"count": 3, "min": 12.0, "p50": 24.0, "p90": 36.0, "max": 36.0, "mean": 24.0},
                        },
                        "shares": {"never_surfaced_share": 0.3333},
                    },
                },
                "candidate_count": 6,
                "claimed_work_item_count": 0,
                "mutations": 2,
                "tool_calls_executed": 3,
            },
        )
        assert runtime.task_queue.claim_next(now=102.0) is not None
        runtime.task_queue.complete(
            seeded_task.id,
            completed_at=103.0,
            run_result={
                "requested_strategy": "semantic",
                "strategy_used": "semantic",
                "strategy_selection_mode": "utility_priors",
                "strategy_selection_reason": "utility prior tie-break",
                "strategy_fallback_reason": "insufficient_candidates",
                "candidate_count": 5,
                "claimed_work_item_count": 2,
                "mutations": 0,
                "tool_calls_executed": 1,
            },
        )
        assert runtime.task_queue.claim_next(now=104.0) is not None
        runtime.task_queue.complete(
            unknown_task.id,
            completed_at=105.0,
            run_result={
                "strategy_used": "lexical",
                "candidate_count": 4,
                "mutations": 1,
                "tool_calls_executed": 1,
            },
        )
        runtime.db_manager.get_connection().commit()
    finally:
        runtime.close()

    app = create_daemon_app(workspace_root_override=None, cwd=workspace)
    with TestClient(app) as client:
        response = client.get("/api/selector-stats", params={"window_hours": 24, "now": 110.0, "limit": 3})

    assert response.status_code == 200
    payload = response.json()
    assert payload["window_hours"] == 24
    assert payload["summary"]["total_runs"] == 3
    assert payload["summary"]["fresh_selector_runs"] == 1
    assert payload["summary"]["seeded_claimed_runs"] == 1
    assert payload["summary"]["unknown_runs"] == 1
    assert payload["summary"]["fallback_runs"] == 1
    assert payload["classification_breakdown"] == [
        {"key": "fresh_selector", "label": "Fresh selector", "runs": 1},
        {"key": "seeded_claimed", "label": "Seeded/claimed", "runs": 1},
        {"key": "unknown", "label": "Unknown", "runs": 1},
    ]
    assert len(payload["outcome_rows"]) == 3
    assert payload["feature_rollup_rows"] == [
        {
            "task_name": "memory-curator",
            "run_classification": "fresh_selector",
            "runs": 1,
            "snapshot_runs": 1,
            "candidate_metric_means": {"content_chars": 23.0},
            "selected_metric_means": {"content_chars": 24.0},
            "candidate_share_means": {"never_surfaced_share": 0.5},
            "selected_share_means": {"never_surfaced_share": 0.3333},
            "strategy_signal_means": {"semantic_overlap_share": 0.82},
        }
    ]
    assert payload["recent_runs"][0]["task_id"] == "selector-unknown-1"
    assert payload["recent_runs"][0]["run_classification"] == "unknown"
    fresh_run = next(row for row in payload["recent_runs"] if row["task_id"] == "selector-fresh-1")
    assert fresh_run["run_classification"] == "fresh_selector"
    assert fresh_run["strategy_selection_scores"] == {"semantic": 0.95, "lexical": 0.4}
    assert fresh_run["selector_feature_snapshot"]["strategy_signals"] == {"semantic_overlap_share": 0.82}
    assert fresh_run["selector_feature_snapshot"]["candidate_population"]["count"] == 6
    seeded_run = next(row for row in payload["recent_runs"] if row["task_id"] == "selector-seeded-1")
    assert seeded_run["run_classification"] == "seeded_claimed"
    assert seeded_run["classification_reason"] == "claimed_work_item_count=2"


def test_management_api_rejects_invalid_json_and_non_object_json_body(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)

    app = create_daemon_app(workspace_root_override=None, cwd=workspace)
    with TestClient(app) as client:
        invalid_json = client.post(
            "/api/memories/search",
            content=b"{",
            headers={"content-type": "application/json"},
        )
        non_object_json = client.post(
            "/api/memories/search",
            json=["not", "an", "object"],
        )

    assert invalid_json.status_code == 400
    assert invalid_json.json() == {"detail": "invalid_json_body"}
    assert non_object_json.status_code == 400
    assert non_object_json.json() == {"detail": "json_body_must_be_object"}
