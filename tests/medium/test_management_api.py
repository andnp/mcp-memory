from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path
from urllib.request import urlopen

import pytest
import uvicorn

from mcp_memory.daemon import DaemonLifecycleController
from mcp_memory.management.api import create_management_app
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
    raise RuntimeError("management API failed to start")


def _stop_server(server: uvicorn.Server, thread: threading.Thread) -> None:
    server.should_exit = True
    thread.join(timeout=10)


@pytest.mark.asyncio
async def test_management_api_exposes_dashboard_and_json_views(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)

    seed_runtime = create_runtime(project_override=None, cwd=workspace)
    try:
        assert seed_runtime.repository is not None
        assert seed_runtime.task_queue is not None
        primary = seed_runtime.repository.create_memory(
            title="Management API plan",
            content="Expose runtime health and recent memories.",
            summary="Operations dashboard plan.",
            workspace_ids=["workspace-a"],
            memory_type="plan",
            tags=["dashboard", "api"],
        )
        superseded = seed_runtime.repository.create_memory(
            title="Management API legacy plan",
            content="An older plan for the same area.",
            workspace_ids=["workspace-a"],
            memory_type="plan",
            tags=["dashboard"],
        )
        assert primary is not None and superseded is not None
        seed_runtime.repository.add_link(primary.id, superseded.id, "SUPERSEDES", "Superseded during epic 6")

        task = seed_runtime.task_queue.enqueue(
            "fact-checker",
            task_id="fact-checker-1",
            workspace_id="workspace-a",
            available_at=0.0,
        )
        assert seed_runtime.task_queue.claim_next(now=10.0) is not None
        seed_runtime.task_queue.fail_permanently(task.id, "missing ext link", failed_at=11.0)
    finally:
        seed_runtime.close()

    controller = DaemonLifecycleController(shutdown_grace_seconds=0.05)
    app = create_management_app(project_override=None, cwd=workspace, controller=controller)
    port = _free_port()
    server, thread = _start_server(app, port)
    try:
        health = _fetch_json(f"http://127.0.0.1:{port}/api/health")
        overview = _fetch_json(f"http://127.0.0.1:{port}/api/overview")
        tasks = _fetch_json(f"http://127.0.0.1:{port}/api/tasks?status=failed")
        memories = _fetch_json(f"http://127.0.0.1:{port}/api/memories?workspace_id=workspace-a")
        detail = _fetch_json(f"http://127.0.0.1:{port}/api/memories/{primary.id}")
        dashboard = _fetch_text(f"http://127.0.0.1:{port}/")

        assert health["status"] == "ok"
        assert health["runtime_active"] is True
        assert overview["memories"]["total"] == 2
        assert overview["tasks"]["failed_count"] == 1
        assert tasks["tasks"][0]["last_error"] == "missing ext link"
        assert memories["records"][0]["title"] == "Management API legacy plan" or memories["records"][0]["title"] == "Management API plan"
        assert detail["record"]["id"] == primary.id
        assert detail["superseded"][0]["id"] == superseded.id
        assert "MCP Memory Dashboard" in dashboard
    finally:
        _stop_server(server, thread)
        await controller.shutdown_now()
