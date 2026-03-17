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
from dataclasses import replace
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from mcp_memory.config import GLOBAL_DAEMON_IDENTITY, resolve_daemon_metadata_path, resolve_daemon_socket_path, resolve_workspace_id, resolve_workspace_root
from mcp_memory.core.agent_runtime import bootstrap_background_tasks, build_runtime_task_worker
from mcp_memory.daemon_lifecycle import DaemonLockTimeoutError, FilesystemLock
from mcp_memory.daemon_models import DaemonControllerView, DaemonMetadata, DaemonRoutes
from mcp_memory.daemon_process import find_free_port, remove_metadata, write_metadata
from mcp_memory.daemon_transport import DaemonZmqServer, dispatch_management_request
from mcp_memory.hook_reminders import HookReminderService
from mcp_memory.management.service import ManagementService
from mcp_memory.mcp.runtime import create_runtime_from_spec, resolve_runtime_spec


logger = logging.getLogger(__name__)
_IDLE_SHUTDOWN_DELAY_SECONDS = 0.25
_REQUEST_WORKSPACE_ROOT_KEY = "__workspace_root"


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
    if port in {None, 0}:
        daemon_port = find_free_port()
    else:
        assert port is not None
        daemon_port = int(port)
    metadata_path = resolve_daemon_metadata_path(GLOBAL_DAEMON_IDENTITY)
    socket_path = resolve_daemon_socket_path()
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

        zmq_server = DaemonZmqServer(
            context_factory=lambda arguments: _context_for_request(runtime, arguments),
            hook_handlers={
                "/api/hooks/session-start": lambda arguments: _handle_session_start(app, runtime, hook_service, arguments),
                "/api/hooks/post-tool-use": lambda arguments: _handle_post_tool_use(runtime, arguments),
                "/api/hooks/session-end": lambda arguments: _handle_session_end(app, runtime, hook_service, arguments),
            },
            routes_provider=lambda: app.state.routes,
            socket_path=socket_path,
            metadata_provider=lambda: app.state.metadata,
        )
        await zmq_server.start()

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
            transport="zmq",
            socket_path=str(socket_path),
        )
        write_metadata(metadata_path, app.state.metadata)
        try:
            yield
        finally:
            await _cancel_idle_shutdown_task(app)
            await zmq_server.stop()
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
    @app.get("/dashboard", response_class=HTMLResponse)
    @app.get("/dashboard/{dashboard_path:path}", response_class=HTMLResponse)
    async def dashboard_html(dashboard_path: str = "") -> HTMLResponse:
        return HTMLResponse(app.state.routes.service.load_dashboard_html())

    @app.get("/assets/{asset_path:path}")
    async def dashboard_asset(asset_path: str):
        resolved = app.state.routes.service.resolve_dashboard_asset_path(asset_path)
        if resolved is None:
            raise HTTPException(status_code=404, detail="dashboard_asset_not_found")
        return FileResponse(resolved)

    @app.api_route("/api/{api_path:path}", methods=["GET", "POST"])
    async def dashboard_api(api_path: str, request: Request):
        payload = await _payload_from_http_request(request)
        result = dispatch_management_request(
            app.state.routes,
            app.state.metadata,
            f"/api/{api_path}",
            payload,
        )
        if result.get("status") == "error" and result.get("error") == "unknown_transport_path":
            raise HTTPException(status_code=404, detail=str(result.get("error")))
        if result.get("status") == "error":
            return JSONResponse(result, status_code=400)
        return JSONResponse(result)

    return app


async def _payload_from_http_request(request: Request) -> dict[str, object]:
    payload: dict[str, object] = dict(request.query_params)
    if request.method != "POST":
        return payload
    body = await request.body()
    if not body:
        return payload
    try:
        decoded = await request.json()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid_json_body") from exc
    if not isinstance(decoded, dict):
        raise HTTPException(status_code=400, detail="json_body_must_be_object")
    payload.update(decoded)
    return payload


def _resolve_runtime_version() -> str | None:
    try:
        return importlib.metadata.version("mcp-memory")
    except importlib.metadata.PackageNotFoundError:
        return None


def _context_for_request(ctx, arguments: dict[str, Any] | None):
    if arguments is None:
        return ctx
    workspace_root_value = arguments.get(_REQUEST_WORKSPACE_ROOT_KEY) or arguments.get("workspace_root")
    if not isinstance(workspace_root_value, str) or not workspace_root_value.strip():
        return ctx
    workspace_root = resolve_workspace_root(workspace_root=workspace_root_value)
    workspace_id = resolve_workspace_id(workspace_root=workspace_root_value)
    return replace(ctx, workspace_root=workspace_root, workspace_id=workspace_id)


async def _handle_session_start(app: FastAPI, ctx, hook_service: HookReminderService, arguments: dict[str, Any]) -> dict[str, Any]:
    request_ctx = _context_for_request(ctx, arguments)
    assert request_ctx.db_manager is not None
    await _cancel_idle_shutdown_task(app)
    response: dict[str, Any] = dict(
        HookReminderService(request_ctx.db_manager, request_ctx.workspace_id).record_session_start(
            str(arguments.get("conversation_id") or arguments.get("sessionId") or arguments.get("session_id") or ""),
            arguments,
        )
    )
    response["active_client_count"] = hook_service.get_active_client_count()
    response["shutdown_scheduled"] = False
    return response


async def _handle_post_tool_use(ctx, arguments: dict[str, Any]) -> dict[str, Any]:
    request_ctx = _context_for_request(ctx, arguments)
    assert request_ctx.db_manager is not None
    return HookReminderService(request_ctx.db_manager, request_ctx.workspace_id).record_post_tool_use(arguments)


async def _handle_session_end(app: FastAPI, ctx, hook_service: HookReminderService, arguments: dict[str, Any]) -> dict[str, Any]:
    request_ctx = _context_for_request(ctx, arguments)
    assert request_ctx.db_manager is not None
    response: dict[str, Any] = dict(
        HookReminderService(request_ctx.db_manager, request_ctx.workspace_id).record_session_end(
            str(arguments.get("conversation_id") or arguments.get("sessionId") or arguments.get("session_id") or ""),
            arguments,
        )
    )
    active_client_count = hook_service.get_active_client_count()
    should_schedule_shutdown = bool(app.state.enable_idle_shutdown) and active_client_count == 0
    if should_schedule_shutdown:
        _schedule_idle_shutdown_if_needed(app)
    else:
        await _cancel_idle_shutdown_task(app)
    response["active_client_count"] = active_client_count
    response["shutdown_scheduled"] = should_schedule_shutdown
    return response