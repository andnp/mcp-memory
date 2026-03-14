from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

from mcp_memory.daemon import get_daemon_lifecycle_controller
from mcp_memory.management.service import ManagementService


def create_management_app(
    project_override: str | None = None,
    cwd: Path | None = None,
    controller=None,
):
    active_controller = controller or get_daemon_lifecycle_controller()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime = await active_controller.acquire_runtime(project_override, cwd)
        app.state.ctx = runtime
        app.state.controller = active_controller
        app.state.management_service = ManagementService(runtime, active_controller)
        try:
            yield
        finally:
            shutdown_task = await active_controller.release_runtime(runtime)
            if shutdown_task is not None:
                await asyncio.shield(shutdown_task)

    app = FastAPI(title="mcp-memory management", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    async def dashboard():
        service = app.state.management_service
        return HTMLResponse(service.load_dashboard_html())

    @app.get("/api/health")
    async def health():
        return app.state.management_service.get_health().model_dump()

    @app.get("/api/overview")
    async def overview():
        return app.state.management_service.get_overview().model_dump()

    @app.get("/api/tasks")
    async def tasks(
        status: str | None = Query(default=None),
        workspace_id: str | None = Query(default=None),
        limit: int = Query(default=20, ge=1, le=200),
    ):
        return app.state.management_service.list_tasks(
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
        records = app.state.management_service.list_memories(
            workspace_id=workspace_id,
            memory_type=memory_type,
            status=status,
            limit=limit,
        )
        return {"records": [record.model_dump() for record in records]}

    @app.get("/api/memories/{memory_id}")
    async def memory_detail(memory_id: str):
        try:
            payload = app.state.management_service.get_memory_detail(memory_id)
        except ValueError as exc:
            if str(exc) == "memory_not_found":
                raise HTTPException(status_code=404, detail="memory_not_found") from exc
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return payload.model_dump()

    return app