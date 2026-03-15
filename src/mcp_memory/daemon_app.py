from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

from mcp_memory.config import resolve_daemon_metadata_path
from mcp_memory.core.agent_runtime import bootstrap_background_tasks, build_runtime_task_worker
from mcp_memory.daemon_models import DaemonControllerView, DaemonMetadata, DaemonRoutes
from mcp_memory.daemon_process import find_free_port, remove_metadata, write_metadata
from mcp_memory.management.service import ManagementService
from mcp_memory.mcp.handlers import call_memory_tool
from mcp_memory.mcp.runtime import create_runtime_from_spec, resolve_runtime_spec
from mcp_memory.mcp.tools import get_memory_tools


def create_daemon_app(
    workspace_root_override: str | None = None,
    cwd: Path | None = None,
    host: str | None = None,
    port: int | None = None,
):
    spec = resolve_runtime_spec(workspace_root_override, cwd)
    daemon_host = host or spec.config.daemon.host
    daemon_port = port if port is not None else find_free_port()
    metadata_path = resolve_daemon_metadata_path(spec.workspace_id)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime = create_runtime_from_spec(spec)
        bootstrap_background_tasks(runtime)
        worker = build_runtime_task_worker(runtime)
        if worker is not None:
            await worker.start()

        routes = DaemonRoutes(
            ctx=runtime,
            service=ManagementService(runtime, controller=DaemonControllerView()),
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
        write_metadata(metadata_path, app.state.metadata)
        try:
            yield
        finally:
            if worker is not None:
                await worker.stop(spec.config.daemon.shutdown_grace_seconds)
            runtime.close()
            remove_metadata(metadata_path)

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
        return app.state.routes.service.list_memories(
            workspace_id=workspace_id,
            memory_type=memory_type,
            status=status,
            limit=limit,
        ).model_dump()

    @app.get("/api/memories/{memory_id}")
    async def memory_detail(memory_id: str):
        try:
            return app.state.routes.service.get_memory_detail(memory_id).model_dump()
        except ValueError as exc:
            if str(exc) == "memory_not_found":
                raise HTTPException(status_code=404, detail="memory_not_found") from exc
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/admin/links")
    async def create_link(arguments: dict[str, Any]):
        try:
            return app.state.routes.service.create_memory_link(
                source_id=str(arguments.get("source_id", "")).strip(),
                target_id=str(arguments.get("target_id", "")).strip(),
                link_type=str(arguments.get("link_type", "")).strip(),
                context=str(arguments.get("context", "")).strip(),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/admin/links/delete")
    async def delete_link(arguments: dict[str, Any]):
        try:
            return app.state.routes.service.delete_memory_link(
                source_id=str(arguments.get("source_id", "")).strip(),
                target_id=str(arguments.get("target_id", "")).strip(),
                link_type=str(arguments.get("link_type", "")).strip(),
            )
        except ValueError as exc:
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