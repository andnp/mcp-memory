from __future__ import annotations

import asyncio
import importlib.metadata
import logging
import os
import signal
import sys
import time
from contextlib import suppress
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

from mcp_memory.config import GLOBAL_DAEMON_IDENTITY, resolve_daemon_metadata_path
from mcp_memory.core.agent_runtime import bootstrap_background_tasks, build_runtime_task_worker
from mcp_memory.daemon_lifecycle import DaemonLockTimeoutError, FilesystemLock
from mcp_memory.daemon_models import DaemonControllerView, DaemonMetadata, DaemonRoutes
from mcp_memory.daemon_process import find_free_port, remove_metadata, write_metadata
from mcp_memory.hook_reminders import HookReminderService
from mcp_memory.management.service import ManagementService
from mcp_memory.mcp.handlers import call_internal_memory_tool, call_memory_tool
from mcp_memory.mcp.runtime import create_runtime_from_spec, resolve_runtime_spec
from mcp_memory.mcp.internal_tools import get_internal_maintenance_tools
from mcp_memory.mcp.tools import get_memory_tools


logger = logging.getLogger(__name__)
_IDLE_SHUTDOWN_DELAY_SECONDS = 0.25


async def _warm_embedding_model(embedder: Any) -> bool:
    if embedder is None:
        return False
    cache_model = getattr(embedder, "cache_model", None)
    if not callable(cache_model):
        return False
    return bool(await asyncio.to_thread(cache_model))


async def _cancel_idle_shutdown_task(app: FastAPI) -> None:
    shutdown_task = getattr(app.state, "idle_shutdown_task", None)
    if shutdown_task is None:
        return
    if shutdown_task.done():
        app.state.idle_shutdown_task = None
        return
    shutdown_task.cancel()
    with suppress(asyncio.CancelledError):
        await shutdown_task
    app.state.idle_shutdown_task = None


async def _shutdown_daemon_when_idle(app: FastAPI) -> None:
    try:
        await asyncio.sleep(_IDLE_SHUTDOWN_DELAY_SECONDS)
        active_clients = app.state.routes.hook_service.get_active_client_count()
        if active_clients > 0:
            logger.debug("Skipping idle daemon shutdown; %s client(s) remain active", active_clients)
            return
        logger.info("Stopping daemon after last MCP client exited")
        os.kill(os.getpid(), signal.SIGTERM)
    finally:
        app.state.idle_shutdown_task = None


def _schedule_idle_shutdown_if_needed(app: FastAPI) -> None:
    shutdown_task = getattr(app.state, "idle_shutdown_task", None)
    if shutdown_task is not None and not shutdown_task.done():
        return
    app.state.idle_shutdown_task = asyncio.create_task(_shutdown_daemon_when_idle(app))


def create_daemon_app(
    workspace_root_override: str | None = None,
    cwd: Path | None = None,
    host: str | None = None,
    port: int | None = None,
    *,
    enable_idle_shutdown: bool = False,
):
    spec = resolve_runtime_spec(workspace_root_override, cwd)
    daemon_host = host or spec.config.daemon.host
    daemon_port = port if port is not None else find_free_port()
    metadata_path = resolve_daemon_metadata_path(GLOBAL_DAEMON_IDENTITY)
    runtime_lock = FilesystemLock(spec.lock_path.with_suffix(".runtime.lock"))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            runtime_lock.acquire(timeout_seconds=0.1)
        except DaemonLockTimeoutError as exc:
            raise RuntimeError("daemon_runtime_lock_unavailable:global") from exc

        runtime = create_runtime_from_spec(spec)
        assert runtime.db_manager is not None
        bootstrap_background_tasks(runtime)
        worker = build_runtime_task_worker(runtime)
        warmup_task = asyncio.create_task(_warm_embedding_model(runtime.embedder))
        if worker is not None:
            await worker.start()

        hook_service = HookReminderService(runtime.db_manager, runtime.workspace_id)

        routes = DaemonRoutes(
            ctx=runtime,
            service=ManagementService(runtime, controller=DaemonControllerView(hook_service=hook_service)),
            hook_service=hook_service,
            metadata_path=metadata_path,
        )
        app.state.routes = routes
        app.state.idle_shutdown_task = None
        app.state.enable_idle_shutdown = enable_idle_shutdown
        app.state.metadata = DaemonMetadata(
            host=daemon_host,
            port=daemon_port,
            pid=os.getpid(),
            started_at=time.time(),
            status="ready",
            daemon_scope=GLOBAL_DAEMON_IDENTITY,
            binary_path=sys.executable,
            version=_resolve_runtime_version(),
            transport="http",
        )
        write_metadata(metadata_path, app.state.metadata)
        try:
            yield
        finally:
            await _cancel_idle_shutdown_task(app)
            if not warmup_task.done():
                warmup_task.cancel()
                with suppress(asyncio.CancelledError):
                    await warmup_task
            if worker is not None:
                await worker.stop(spec.config.daemon.shutdown_grace_seconds)
            runtime.close()
            remove_metadata(metadata_path)
            runtime_lock.release()

    app = FastAPI(title="mcp-memory daemon", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    async def dashboard():
        return HTMLResponse(app.state.routes.service.load_dashboard_html())

    @app.get("/api/health")
    async def health():
        payload = app.state.routes.service.get_health().model_dump()
        payload["pid"] = app.state.metadata.pid
        payload["status"] = app.state.metadata.status
        payload["daemon_scope"] = app.state.metadata.daemon_scope
        payload["binary_path"] = app.state.metadata.binary_path
        payload["version"] = app.state.metadata.version
        payload["transport"] = app.state.metadata.transport
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

    @app.post("/api/admin/tasks/{task_id}/cancel")
    async def cancel_task(task_id: str, arguments: dict[str, Any]):
        try:
            return app.state.routes.service.cancel_task(
                task_id,
                cancelled_by=str(arguments.get("cancelled_by", "cli") or "cli"),
                reason=str(arguments.get("reason", "cancelled_by_user") or "cancelled_by_user"),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/hooks/session-start")
    async def hook_session_start(arguments: dict[str, Any]):
        try:
            await _cancel_idle_shutdown_task(app)
            response = app.state.routes.hook_service.record_session_start(
                str(arguments.get("conversation_id") or arguments.get("sessionId") or arguments.get("session_id") or ""),
                arguments,
            )
            response["active_client_count"] = app.state.routes.hook_service.get_active_client_count()
            response["shutdown_scheduled"] = False
            return response
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
            response = app.state.routes.hook_service.record_session_end(
                str(arguments.get("conversation_id") or arguments.get("sessionId") or arguments.get("session_id") or ""),
                arguments,
            )
            active_client_count = app.state.routes.hook_service.get_active_client_count()
            should_schedule_shutdown = bool(app.state.enable_idle_shutdown) and active_client_count == 0
            if should_schedule_shutdown:
                _schedule_idle_shutdown_if_needed(app)
            else:
                await _cancel_idle_shutdown_task(app)
            response["active_client_count"] = active_client_count
            response["shutdown_scheduled"] = should_schedule_shutdown
            return response
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

    @app.get("/api/ai-conversations")
    async def ai_conversations(
        request_id: str | None = Query(default=None),
        task_name: str | None = Query(default=None),
        status: str | None = Query(default=None),
        limit: int = Query(default=50, ge=1, le=200),
    ):
        return app.state.routes.service.list_ai_conversations(
            request_id=request_id,
            task_name=task_name,
            status=status,
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


def _resolve_runtime_version() -> str | None:
    try:
        return importlib.metadata.version("mcp-memory")
    except importlib.metadata.PackageNotFoundError:
        return None