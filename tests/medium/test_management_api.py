from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from pathlib import Path
from urllib.request import Request, urlopen

import pytest
import uvicorn

from mcp_memory.daemon import create_daemon_app
from mcp_memory.mcp.runtime import create_runtime


pytestmark = pytest.mark.medium


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _fetch_json(url: str) -> dict:
    with urlopen(url, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def _fetch_text(url: str) -> str:
    with urlopen(url, timeout=5) as response:
        return response.read().decode("utf-8")


def _post_json(url: str, payload: dict) -> dict:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def _start_server(app, port: int):
    config = uvicorn.Config(app=app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            _fetch_json(f"http://127.0.0.1:{port}/api/health")
            return server, thread
        except Exception:
            time.sleep(0.05)
    raise RuntimeError("daemon app failed to start")


def _stop_server(server: uvicorn.Server, thread: threading.Thread) -> None:
    server.should_exit = True
    thread.join(timeout=10)


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
        seed_runtime.db_manager.get_connection().execute(
            "INSERT INTO runtime_logs (workspace_id, source, logger_name, level, message, created_at, data_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                seed_runtime.workspace_id,
                "daemon",
                "mcp_memory.server",
                "INFO",
                "seeded daemon log",
                12.0,
                "{}",
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
        assert seed_runtime.task_queue.claim_next(now=10.0) is not None
        seed_runtime.task_queue.fail_permanently(task.id, "missing ext link", failed_at=11.0)
    finally:
        seed_runtime.close()

    port = _free_port()
    app = create_daemon_app(workspace_root_override=None, cwd=workspace, host="127.0.0.1", port=port)
    server, thread = _start_server(app, port)
    try:
        health = _fetch_json(f"http://127.0.0.1:{port}/api/health")
        overview = _fetch_json(f"http://127.0.0.1:{port}/api/overview")
        logs = _fetch_json(f"http://127.0.0.1:{port}/api/logs?source=daemon&q=seeded")
        log_summary = _fetch_json(f"http://127.0.0.1:{port}/api/logs/summary?source=daemon")
        tasks = _fetch_json(f"http://127.0.0.1:{port}/api/tasks?status=failed")
        memories = _fetch_json(f"http://127.0.0.1:{port}/api/memories?workspace_id={seed_runtime.workspace_id}")
        detail = _fetch_json(f"http://127.0.0.1:{port}/api/memories/{primary.id}")
        internal_tools = _fetch_json(f"http://127.0.0.1:{port}/internal/maintenance/tools")
        hook_start = _post_json(
            f"http://127.0.0.1:{port}/api/hooks/session-start",
            {"sessionId": "conversation-1", "timestamp": 100.0},
        )
        hook_noop = _post_json(
            f"http://127.0.0.1:{port}/api/hooks/post-tool-use",
            {"sessionId": "conversation-1", "tool_name": "read_file", "timestamp": 200.0},
        )
        hook_reminder = _post_json(
            f"http://127.0.0.1:{port}/api/hooks/post-tool-use",
            {"sessionId": "conversation-1", "tool_name": "apply_patch", "timestamp": 401.0},
        )
        hook_end = _post_json(
            f"http://127.0.0.1:{port}/api/hooks/session-end",
            {"sessionId": "conversation-1", "timestamp": 500.0},
        )
        run_agent = _post_json(
            f"http://127.0.0.1:{port}/api/admin/agents/run",
            {"task_name": "graph-linker", "force": True},
        )
        run_all = _post_json(
            f"http://127.0.0.1:{port}/api/admin/agents/run-all",
            {"force": False},
        )
        created_link = _post_json(
            f"http://127.0.0.1:{port}/api/admin/links",
            {
                "source_id": primary.id,
                "target_id": "ext:README.md",
                "link_type": "REFERENCES",
                "context": "Referenced from the dashboard admin flow.",
            },
        )
        deleted_link = _post_json(
            f"http://127.0.0.1:{port}/api/admin/links/delete",
            {
                "source_id": primary.id,
                "target_id": "ext:README.md",
                "link_type": "REFERENCES",
            },
        )
        dashboard = _fetch_text(f"http://127.0.0.1:{port}/")

        assert health["status"] == "ready"
        assert health["workspace_id"] == seed_runtime.workspace_id
        assert health["workspace_root"] == str(workspace)
        assert "embeddings" in health
        assert "backend" in health["embeddings"]
        assert overview["memories"]["total"] == 2
        assert overview["embeddings"]["model_name"] is not None
        assert overview["memory_metrics"]["total_memories"] == 2
        assert "agent_runs" in overview
        assert overview["recent_logs"][0]["message"] == "seeded daemon log"
        assert overview["tasks"]["failed_count"] == 1
        assert overview["failed_tasks"][0]["last_error"] == "missing ext link"
        assert logs["logs"][0]["source"] == "daemon"
        assert log_summary["total"] == 1
        assert log_summary["by_level"] == {"INFO": 1}
        fact_checker = next(agent for agent in overview["agent_runs"] if agent["task_name"] == "fact-checker")
        assert fact_checker["failed_runs"] == 1
        assert overview["recent_agent_runs"]
        assert tasks["tasks"][0]["last_error"] == "missing ext link"
        assert {record["id"] for record in memories["records"]} == {primary.id, superseded.id}
        assert detail["record"]["id"] == primary.id
        assert any(tool["name"] == "internal_merge_memory_into_canonical" for tool in internal_tools["tools"])
        assert detail["superseded"][0]["id"] == superseded.id
        assert hook_start["status"] == "ok"
        assert hook_noop == {}
        assert "record_thought" in hook_reminder["systemMessage"]
        assert hook_end["status"] == "ok"
        assert "running_count" in fact_checker
        assert "next_available_at" in fact_checker
        assert "last_result_summary" in fact_checker
        assert run_agent["status"] == "enqueued"
        assert len(run_all["results"]) >= 1
        assert created_link["status"] == "created"
        assert deleted_link["status"] == "deleted"
        assert "MCP Memory Dashboard" in dashboard
        assert "Background Agents" in dashboard
        assert "Memory Metrics" in dashboard
        assert "Embedding Backend" in dashboard
        assert "Recent Agent Runs" in dashboard
        assert "Recent Logs" in dashboard
        assert "Refresh Logs" in dashboard
        assert "Agent Controls" in dashboard
    finally:
        _stop_server(server, thread)


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
