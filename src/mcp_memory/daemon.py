from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import fcntl
import json
import logging
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from typing import IO, Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

from mcp_memory.config import (
    resolve_daemon_lock_path,
    resolve_daemon_metadata_path,
    resolve_workspace_id,
)
from mcp_memory.core.agent_runtime import bootstrap_background_tasks, build_runtime_task_worker
from mcp_memory.management.service import ManagementService
from mcp_memory.mcp.handlers import call_memory_tool
from mcp_memory.mcp.runtime import create_runtime_from_spec, resolve_runtime_spec
from mcp_memory.mcp.tools import get_memory_tools


logger = logging.getLogger(__name__)


class DaemonLockTimeoutError(TimeoutError):
    pass


@dataclass
class FilesystemLock:
    lock_path: Path
    _handle: IO[str] | None = None

    def acquire(self, timeout_seconds: float | None = 10.0) -> None:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.lock_path.open("a+", encoding="utf-8")
        deadline = None if timeout_seconds is None else time.monotonic() + timeout_seconds
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._handle = handle
                return
            except BlockingIOError as exc:
                if deadline is not None and time.monotonic() >= deadline:
                    handle.close()
                    raise DaemonLockTimeoutError(f"Timed out acquiring daemon lock: {self.lock_path}") from exc
                time.sleep(0.05)
            except Exception:
                handle.close()
                raise

    def release(self) -> None:
        handle = self._handle
        self._handle = None
        if handle is None:
            return
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


@dataclass
class DaemonMetadata:
    workspace_id: str
    workspace_root: str
    host: str
    port: int
    pid: int
    started_at: float
    status: str

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass
class _DaemonRoutes:
    ctx: Any
    service: ManagementService
    metadata_path: Path


def create_daemon_app(
    workspace_root_override: str | None = None,
    cwd: Path | None = None,
    host: str | None = None,
    port: int | None = None,
):
    spec = resolve_runtime_spec(workspace_root_override, cwd)
    daemon_host = host or spec.config.daemon.host
    daemon_port = port if port is not None else _find_free_port()
    metadata_path = resolve_daemon_metadata_path(spec.workspace_id)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime = create_runtime_from_spec(spec)
        bootstrap_background_tasks(runtime)
        worker = build_runtime_task_worker(runtime)
        if worker is not None:
            await worker.start()

        routes = _DaemonRoutes(
            ctx=runtime,
            service=ManagementService(runtime, controller=_DaemonControllerView()),
            metadata_path=metadata_path,
        )
        app.state.routes = routes
        app.state.metadata = DaemonMetadata(
            workspace_id=spec.workspace_id,
            workspace_root=str(spec.workspace_root),
            host=daemon_host,
            port=daemon_port,
            pid=os.getpid(),
            started_at=time.time(),
            status="ready",
        )
        _write_metadata(metadata_path, app.state.metadata)
        try:
            yield
        finally:
            if worker is not None:
                await worker.stop(spec.config.daemon.shutdown_grace_seconds)
            runtime.close()
            _remove_metadata(metadata_path)

    app = FastAPI(title="mcp-memory daemon", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    async def dashboard():
        return HTMLResponse(app.state.routes.service.load_dashboard_html())

    @app.get("/api/health")
    async def health():
        payload = app.state.routes.service.get_health().model_dump()
        payload["pid"] = app.state.metadata.pid
        payload["status"] = app.state.metadata.status
        return payload

    @app.get("/api/overview")
    async def overview():
        return app.state.routes.service.get_overview().model_dump()

    @app.get("/api/tasks")
    async def tasks(
        status: str | None = Query(default=None),
        workspace_id: str | None = Query(default=None),
        limit: int = Query(default=20, ge=1, le=200),
    ):
        return app.state.routes.service.list_tasks(
            status=status,
            workspace_id=workspace_id,
            limit=limit,
        ).model_dump()

    @app.get("/api/memories")
    async def memories(
        workspace_id: str | None = Query(default=None),
        memory_type: str | None = Query(default=None),
        status: str | None = Query(default=None),
        limit: int = Query(default=20, ge=1, le=200),
    ):
        records = app.state.routes.service.list_memories(
            workspace_id=workspace_id,
            memory_type=memory_type,
            status=status,
            limit=limit,
        )
        return {"records": [record.model_dump() for record in records]}

    @app.get("/api/memories/{memory_id}")
    async def memory_detail(memory_id: str):
        try:
            return app.state.routes.service.get_memory_detail(memory_id).model_dump()
        except ValueError as exc:
            if str(exc) == "memory_not_found":
                raise HTTPException(status_code=404, detail="memory_not_found") from exc
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/internal/health")
    async def internal_health():
        return asdict(app.state.metadata)

    @app.get("/internal/tools")
    async def list_tools():
        return {
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": tool.inputSchema,
                }
                for tool in get_memory_tools()
            ]
        }

    @app.post("/internal/tools/{name}")
    async def call_tool(name: str, arguments: dict[str, Any]):
        response = await call_memory_tool(app.state.routes.ctx, name, arguments)
        return {
            "contents": [
                {
                    "type": content.type,
                    "text": content.text,
                }
                for content in response
            ]
        }

    return app


class _DaemonControllerView:
    @property
    def has_runtime(self) -> bool:
        return True

    @property
    def client_count(self) -> int:
        return 1


def ensure_daemon_started(
    workspace_root_override: str | None = None,
    cwd: Path | None = None,
) -> DaemonMetadata:
    spec = resolve_runtime_spec(workspace_root_override, cwd)
    metadata_path = resolve_daemon_metadata_path(spec.workspace_id)
    lock = FilesystemLock(resolve_daemon_lock_path(spec.workspace_id))
    timeout_seconds = spec.config.daemon.auto_start_timeout_seconds
    lock.acquire(timeout_seconds=timeout_seconds)
    try:
        existing = read_daemon_metadata(metadata_path)
        if existing is not None and _is_daemon_healthy(existing):
            return existing

        daemon_port = _find_free_port()
        _spawn_daemon_process(spec.workspace_root, spec.config.daemon.host, daemon_port)
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            current = read_daemon_metadata(metadata_path)
            if current is not None and _is_daemon_healthy(current):
                return current
            time.sleep(spec.config.daemon.healthcheck_interval_seconds)
        raise RuntimeError(f"Timed out waiting for daemon startup for {spec.workspace_id}")
    finally:
        lock.release()


def read_daemon_metadata(metadata_path: Path) -> DaemonMetadata | None:
    if not metadata_path.exists():
        return None
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    try:
        return DaemonMetadata(**payload)
    except TypeError:
        return None


def daemon_url(workspace_root_override: str | None = None, cwd: Path | None = None) -> str:
    return ensure_daemon_started(workspace_root_override, cwd).base_url


def _spawn_daemon_process(workspace_root: Path, host: str, port: int) -> None:
    command = [
        sys.executable,
        "-m",
        "mcp_memory.cli",
        "daemon",
        "--workspace-root",
        str(workspace_root),
        "--host",
        host,
        "--port",
        str(port),
    ]
    subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _write_metadata(metadata_path: Path, metadata: DaemonMetadata) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(asdict(metadata), sort_keys=True), encoding="utf-8")


def _remove_metadata(metadata_path: Path) -> None:
    try:
        metadata_path.unlink()
    except FileNotFoundError:
        return


def _is_daemon_healthy(metadata: DaemonMetadata) -> bool:
    try:
        request = Request(f"{metadata.base_url}/internal/health", method="GET")
        with urlopen(request, timeout=1) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return payload.get("workspace_id") == metadata.workspace_id and payload.get("status") == "ready"
    except (URLError, OSError, TimeoutError, json.JSONDecodeError):
        return False
