from __future__ import annotations

import asyncio
import time
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from mcp_memory.daemon import create_daemon_app
from mcp_memory.daemon_transport import request_daemon_json
from mcp_memory.mcp.runtime import create_runtime


pytestmark = pytest.mark.medium


async def _request_json(metadata, path: str, payload: dict | None = None) -> dict:
    return await asyncio.to_thread(
        request_daemon_json,
        metadata,
        path,
        payload,
        timeout_seconds=5,
    )


@pytest.mark.asyncio
async def test_management_api_exposes_dashboard_and_json_views(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)

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
                time.time(),
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
        )
        superseded = seed_runtime.repository.create_memory(
            title="Management API legacy plan",
            content="An older plan for the same area.",
            workspace_ids=[seed_runtime.workspace_id],
            memory_type="plan",
            tags=["dashboard"],
        )
        assert primary is not None and superseded is not None
        seed_runtime.repository.add_link(primary.id, superseded.id, "SUPERSEDES", "Superseded during epic 6")

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
        seed_runtime.task_queue.enqueue(
            "summarize-memory",
            task_id="summarize-pending-1",
            workspace_id=seed_runtime.workspace_id,
            available_at=time.time() + 60.0,
            data={"trigger": "summary_follow_up"},
        )
        assert seed_runtime.task_queue.claim_next(now=10.0) is not None
        seed_runtime.task_queue.fail_permanently(task.id, "missing ext link", failed_at=11.0)
        assert seed_runtime.task_queue.claim_next(now=12.0) is not None
        seed_runtime.task_queue.complete(
            strategy_task.id,
            completed_at=13.0,
            run_result={
                "merged": 1,
                "requested_strategy": "semantic",
                "strategy_used": "semantic",
                "candidate_count": 6,
                "sampled_memory_ids": [primary.id],
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
        prune_logs = await _request_json(
            metadata,
            "/api/admin/logs/prune",
            {"max_runtime_logs": 1, "max_log_age_days": 30},
        )
        repair_search = await _request_json(metadata, "/api/admin/search/repair", {})
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
        assert health["workspace_id"] == seed_runtime.workspace_id
        assert health["workspace_root"] == str(workspace)
        assert "embeddings" in health
        assert "backend" in health["embeddings"]
        assert health["search"]["semantic_enabled"] is True
        assert overview["memories"]["total"] == 2
        assert overview["embeddings"]["model_name"] is not None
        assert overview["search"]["semantic_enabled"] is True
        assert overview["memory_metrics"]["total_memories"] == 2
        assert overview["queue_diagnostics"]
        assert all(item["pending_state"] in {"scheduled", "runnable"} for item in overview["queue_diagnostics"])
        assert any(item["task_name"] in {"project-manager", "summarize-memory"} for item in overview["queue_diagnostics"])
        assert "agent_runs" in overview
        assert overview["provider_usage"][0]["provider_key"] == "gemini-cli"
        assert overview["provider_usage"][0]["task_name"] == "graph-linker"
        assert overview["provider_usage"][0]["calls_last_day"] == 1
        assert overview["recent_logs"][0]["message"] == "seeded daemon log"
        assert overview["top_read_memories"] == []
        assert overview["tasks"]["failed_count"] == 1
        assert record_thought["status"] == "recorded"
        assert overview_after_thought["journal"]["pending_count"] >= overview["journal"]["pending_count"] + 1
        assert overview["failed_tasks"][0]["last_error"] == "missing ext link"
        assert logs["logs"][0]["source"] == "daemon"
        assert log_summary["total"] == 1
        assert log_summary["by_level"] == {"INFO": 1}
        assert prune_logs["deleted"] == 0
        assert repair_search["rebuilt"] is True
        fact_checker = next(agent for agent in overview["agent_runs"] if agent["task_name"] == "fact-checker")
        deduplicator = next(agent for agent in overview["agent_runs"] if agent["task_name"] == "deduplicator")
        assert fact_checker["failed_runs"] == 1
        assert deduplicator["last_result_metadata"]["strategy_used"] == "semantic"
        assert deduplicator["last_result_metadata"]["candidate_count"] == 6
        assert overview["recent_agent_runs"]
        assert any(run["result_metadata"]["strategy_used"] == "semantic" for run in overview["recent_agent_runs"])
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
        assert "MCP Memory Dashboard" in dashboard
        assert "Command Bar" in dashboard
        assert "Background Agents" in dashboard
        assert "Memory Metrics" in dashboard
        assert "Embedding Backend" in dashboard
        assert "Recent Agent Runs" in dashboard
        assert "Recent Logs" in dashboard
        assert "AI Provider Usage" in dashboard
        assert "Refresh Logs" in dashboard
        assert "Agent Controls" in dashboard


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
async def test_daemon_idle_shutdown_waits_for_last_client_and_cancels_on_reconnect(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    shutdown_calls: list[tuple[int, int]] = []

    monkeypatch.setattr("mcp_memory.daemon_app.os.kill", lambda pid, sig: shutdown_calls.append((pid, sig)))

    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    runtime.embedder = None
    monkeypatch.setattr("mcp_memory.daemon_app.create_runtime_from_spec", lambda spec: runtime)
    app = create_daemon_app(workspace_root_override=None, cwd=workspace, enable_idle_shutdown=True)

    try:
        async with app.router.lifespan_context(app):
            metadata = app.state.metadata
            start = await _request_json(metadata, "/api/hooks/session-start", {"sessionId": "conv-1", "timestamp": 100.0})
            health = await _request_json(metadata, "/api/health")
            end = await _request_json(metadata, "/api/hooks/session-end", {"sessionId": "conv-1", "timestamp": 101.0})

            assert start["active_client_count"] == 1
            assert health["client_count"] == 1
            assert end["active_client_count"] == 0
            assert end["shutdown_scheduled"] is True

            await asyncio.sleep(0.35)
            assert len(shutdown_calls) == 1
    finally:
        runtime.close()

    shutdown_calls.clear()

    runtime = create_runtime(workspace_root_override=None, cwd=workspace)
    runtime.embedder = None
    monkeypatch.setattr("mcp_memory.daemon_app.create_runtime_from_spec", lambda spec: runtime)
    app = create_daemon_app(workspace_root_override=None, cwd=workspace, enable_idle_shutdown=True)

    try:
        async with app.router.lifespan_context(app):
            metadata = app.state.metadata
            await _request_json(metadata, "/api/hooks/session-start", {"sessionId": "conv-2", "timestamp": 200.0})
            end = await _request_json(metadata, "/api/hooks/session-end", {"sessionId": "conv-2", "timestamp": 201.0})
            reconnect = await _request_json(metadata, "/api/hooks/session-start", {"sessionId": "conv-3", "timestamp": 202.0})

            assert end["shutdown_scheduled"] is True
            assert reconnect["active_client_count"] == 1

            await asyncio.sleep(0.35)
            assert shutdown_calls == []
            assert (await _request_json(metadata, "/api/health"))["client_count"] == 1
    finally:
        runtime.close()


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
        overview = client.get("/api/overview")
        record_thought = client.post("/api/record-thought", json={"content": "dogfood the react shell"})
        overview_after = client.get("/api/overview")
        missing_asset = client.get("/assets/missing.js")

    assert dashboard.status_code == 200
    assert "MCP Memory Dashboard" in dashboard.text or "Memory Command Center" in dashboard.text
    assert overview.status_code == 200
    assert overview.json()["memories"]["total"] >= 1
    assert record_thought.status_code == 200
    assert record_thought.json()["status"] == "recorded"
    assert overview_after.json()["journal"]["pending_count"] >= overview.json()["journal"]["pending_count"] + 1
    assert missing_asset.status_code == 404
