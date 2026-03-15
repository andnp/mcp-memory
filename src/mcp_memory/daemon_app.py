from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import suppress
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
from mcp_memory.hook_reminders import HookReminderService
from mcp_memory.management.service import ManagementService
from mcp_memory.mcp.handlers import call_internal_memory_tool, call_memory_tool
from mcp_memory.mcp.runtime import create_runtime_from_spec, resolve_runtime_spec
from mcp_memory.mcp.internal_tools import get_internal_maintenance_tools
from mcp_memory.mcp.tools import get_memory_tools


logger = logging.getLogger(__name__)


async def _warm_embedding_model(embedder: Any) -> bool:
    if embedder is None:
        return False
    cache_model = getattr(embedder, "cache_model", None)
    if not callable(cache_model):
        return False
    return bool(await asyncio.to_thread(cache_model))


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
        assert runtime.db_manager is not None
        bootstrap_background_tasks(runtime)
        worker = build_runtime_task_worker(runtime)
        warmup_task = asyncio.create_task(_warm_embedding_model(runtime.embedder))
        if worker is not None:
            await worker.start()

        routes = DaemonRoutes(
            ctx=runtime,
            service=ManagementService(runtime, controller=DaemonControllerView()),
            hook_service=HookReminderService(runtime.db_manager, runtime.workspace_id),
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
            if not warmup_task.done():
                warmup_task.cancel()
                with suppress(asyncio.CancelledError):
                    await warmup_task
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

    @app.post("/api/admin/agents/run")
    async def run_agent(arguments: dict[str, Any]):
        try:
            return app.state.routes.service.enqueue_background_task(
                str(arguments.get("task_name", "")).strip(),
                force=bool(arguments.get("force", False)),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/admin/agents/run-all")
    async def run_all_agents(arguments: dict[str, Any]):
        return {
            "results": app.state.routes.service.enqueue_all_background_tasks(
                force=bool(arguments.get("force", False))
            )
        }

    @app.post("/api/hooks/session-start")
    async def hook_session_start(arguments: dict[str, Any]):
        try:
            return app.state.routes.hook_service.record_session_start(
                str(arguments.get("conversation_id") or arguments.get("sessionId") or arguments.get("session_id") or ""),
                arguments,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/hooks/post-tool-use")
    async def hook_post_tool_use(arguments: dict[str, Any]):
        try:
            return app.state.routes.hook_service.record_post_tool_use(arguments)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/hooks/session-end")
    async def hook_session_end(arguments: dict[str, Any]):
        try:
            return app.state.routes.hook_service.record_session_end(
                str(arguments.get("conversation_id") or arguments.get("sessionId") or arguments.get("session_id") or ""),
                arguments,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

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

    @app.get("/api/logs")
    async def logs(
        level: str | None = Query(default=None),
        logger_name: str | None = Query(default=None),
        source: str | None = Query(default=None),
        q: str | None = Query(default=None),
        after: float | None = Query(default=None),
        before: float | None = Query(default=None),
        limit: int = Query(default=50, ge=1, le=200),
    ):
        return app.state.routes.service.list_logs(
            level=level,
            logger_name=logger_name,
            source=source,
            query=q,
            after=after,
            before=before,
            limit=limit,
        ).model_dump()

    @app.get("/api/logs/summary")
    async def logs_summary(
        level: str | None = Query(default=None),
        logger_name: str | None = Query(default=None),
        source: str | None = Query(default=None),
        q: str | None = Query(default=None),
        after: float | None = Query(default=None),
        before: float | None = Query(default=None),
    ):
        return app.state.routes.service.summarize_logs(
            level=level,
            logger_name=logger_name,
            source=source,
            query=q,
            after=after,
            before=before,
        ).model_dump()

    @app.post("/api/admin/logs/prune")
    async def prune_logs(arguments: dict[str, Any]):
        return app.state.routes.service.prune_logs(
            max_runtime_logs=int(arguments["max_runtime_logs"]) if arguments.get("max_runtime_logs") is not None else None,
            max_log_age_days=int(arguments["max_log_age_days"]) if arguments.get("max_log_age_days") is not None else None,
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

    @app.get("/internal/maintenance/tools")
    async def list_internal_tools():
        return {
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": tool.inputSchema,
                }
                for tool in get_internal_maintenance_tools()
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

    @app.post("/internal/maintenance/tools/{name}")
    async def call_internal_tool(name: str, arguments: dict[str, Any]):
        response = await call_internal_memory_tool(app.state.routes.ctx, name, arguments)
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