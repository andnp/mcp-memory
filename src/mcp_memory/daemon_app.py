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

from mcp_memory.config import GLOBAL_DAEMON_IDENTITY, resolve_backup_dir, resolve_daemon_metadata_path, resolve_daemon_socket_path, resolve_workspace_id, resolve_workspace_root
from mcp_memory.core.agent_runtime import bootstrap_background_tasks, build_runtime_task_worker
from mcp_memory.daemon_dispatch import dispatch_management_request, error_payload
from mcp_memory.daemon_lifecycle import DaemonLockTimeoutError, FilesystemLock
from mcp_memory.daemon_models import DaemonControllerView, DaemonMetadata, DaemonRoutes
from mcp_memory.daemon_process import find_free_port, remove_metadata, write_metadata
from mcp_memory.daemon_transport import DaemonZmqServer
from mcp_memory.hook_reminders import HookReminderService
from mcp_memory.management.frontend_build import ensure_dashboard_frontend_built
from mcp_memory.management.service import ManagementService
from mcp_memory.mcp.runtime import create_runtime_from_spec, resolve_runtime_spec
from mcp_memory.sqlite_backup import create_and_prune_sqlite_backup, log_shared_storage_risks


logger = logging.getLogger(__name__)
_IDLE_SHUTDOWN_DELAY_SECONDS = 0.25
_HTTP_ACTIVITY_GRACE_SECONDS = 60.0
_REQUEST_WORKSPACE_ROOT_KEY = "__workspace_root"
_REQUEST_SESSION_ID_KEY = "__session_id"
_GLOBAL_DEFAULT_API_PATHS = {
    "overview",
    "metrics/nerd",
    "tasks",
    "memories",
    "memories/search",
    "logs",
    "logs/summary",
    "ai-conversations",
    "admin/logs/prune",
}


def _ensure_dashboard_frontend_ready(static_root: Path) -> None:
    result = ensure_dashboard_frontend_built(static_root=static_root)
    if result.status == "up_to_date":
        logger.debug("Dashboard frontend bundle is up to date")
        return
    if result.status == "built":
        logger.info("Dashboard frontend rebuilt at startup from %s", result.frontend_root)
        return
    logger.warning(
        "Dashboard frontend build step did not complete cleanly: status=%s returncode=%s message=%s",
        result.status,
        result.returncode,
        result.message,
    )


async def _warm_embedding_model(embedder: Any) -> bool:
    if embedder is None:
        return False
    cache_model = getattr(embedder, "cache_model", None)
    if not callable(cache_model):
        return False
    return bool(await asyncio.to_thread(cache_model))


def _consume_embedding_warmup_result(task: asyncio.Task[bool]) -> None:
    with suppress(asyncio.CancelledError):
        task.result()


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


async def _record_http_activity(app: FastAPI) -> None:
    app.state.last_http_activity_at = time.monotonic()
    await _cancel_idle_shutdown_task(app)


def _count_running_background_tasks(app: FastAPI) -> int:
    routes = getattr(app.state, "routes", None)
    if routes is None:
        return 0
    ctx = getattr(routes, "ctx", None)
    task_queue = getattr(ctx, "task_queue", None)
    if task_queue is None:
        return 0
    return int(task_queue.count_by_status().get("running", 0))


async def _shutdown_daemon_when_idle(app: FastAPI) -> None:
    try:
        while True:
            await asyncio.sleep(_IDLE_SHUTDOWN_DELAY_SECONDS)
            active_clients = app.state.routes.hook_service.get_active_client_count()
            if active_clients > 0:
                logger.debug("Skipping idle daemon shutdown; %s client(s) remain active", active_clients)
                return
            running_tasks = _count_running_background_tasks(app)
            if running_tasks > 0:
                logger.debug("Deferring idle daemon shutdown; %s background task(s) still running", running_tasks)
                continue
            last_http_activity_at = getattr(app.state, "last_http_activity_at", None)
            if isinstance(last_http_activity_at, (int, float)):
                recent_http_age_seconds = max(time.monotonic() - float(last_http_activity_at), 0.0)
                if recent_http_age_seconds < _HTTP_ACTIVITY_GRACE_SECONDS:
                    logger.debug(
                        "Deferring idle daemon shutdown; recent HTTP activity %.2fs ago",
                        recent_http_age_seconds,
                    )
                    continue
            logger.info("Stopping daemon after last MCP client exited")
            os.kill(os.getpid(), signal.SIGTERM)
            return
    finally:
        app.state.idle_shutdown_task = None


def _schedule_idle_shutdown_if_needed(app: FastAPI) -> None:
    shutdown_task = getattr(app.state, "idle_shutdown_task", None)
    if shutdown_task is not None and not shutdown_task.done():
        return
    app.state.idle_shutdown_task = asyncio.create_task(_shutdown_daemon_when_idle(app))


async def _cancel_background_task(app: FastAPI, task_name: str) -> None:
    task = getattr(app.state, task_name, None)
    if task is None:
        return
    if task.done():
        setattr(app.state, task_name, None)
        return
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task
    setattr(app.state, task_name, None)


async def _run_periodic_backup_loop(runtime) -> None:
    config = getattr(runtime, "config", None)
    db_manager = getattr(runtime, "db_manager", None)
    memory_path = getattr(runtime, "memory_path", None)
    storage_backend = getattr(runtime, "storage_backend", None) or "sqlite"
    if config is None or db_manager is None or memory_path is None:
        return

    backup_config = config.backups
    app_data_dir = Path(memory_path).parent
    if backup_config.warn_on_shared_storage:
        log_shared_storage_risks(app_data_dir)

    if storage_backend != "sqlite":
        logger.info("Skipping periodic SQLite backup loop for storage backend %s", storage_backend)
        return

    if not backup_config.enabled:
        return

    backup_dir = resolve_backup_dir()
    should_run_immediately = backup_config.create_startup_snapshot
    while True:
        if not should_run_immediately:
            await asyncio.sleep(backup_config.interval_seconds)
        should_run_immediately = False
        try:
            result = await asyncio.to_thread(
                create_and_prune_sqlite_backup,
                Path(db_manager.db_path),
                backup_dir,
                max_snapshots=backup_config.max_snapshots,
            )
            logger.info(
                "SQLite backup snapshot created: %s pruned=%s",
                result.backup_path,
                len(result.pruned_paths),
            )
        except Exception as exc:
            logger.warning("SQLite backup snapshot failed: %s", exc)


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
    if port is None:
        daemon_port = spec.config.daemon.port
    elif port == 0:
        daemon_port = find_free_port()
    else:
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
        warmup_task.add_done_callback(_consume_embedding_warmup_result)
        backup_task = asyncio.create_task(_run_periodic_backup_loop(runtime))
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
            service=ManagementService(
                runtime,
                controller=DaemonControllerView(hook_service=hook_service, transport_server=zmq_server),
            ),
            hook_service=hook_service,
            metadata_path=metadata_path,
        )
        await asyncio.to_thread(_ensure_dashboard_frontend_ready, routes.service.dashboard_static_root)
        app.state.routes = routes
        app.state.idle_shutdown_task = None
        app.state.backup_task = backup_task
        app.state.enable_idle_shutdown = enable_idle_shutdown
        app.state.last_http_activity_at = time.monotonic()
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
            await _cancel_background_task(app, "backup_task")
            await zmq_server.stop()
            if not warmup_task.done():
                warmup_task.cancel()
                with suppress(asyncio.CancelledError):
                    await warmup_task
            if worker is not None:
                await worker.stop(spec.config.daemon.shutdown_grace_seconds)
            runtime.close()
            remove_metadata(metadata_path, expected_pid=os.getpid())
            runtime_lock.release()

    app = FastAPI(title="mcp-memory daemon", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    @app.get("/dashboard", response_class=HTMLResponse)
    @app.get("/dashboard/{dashboard_path:path}", response_class=HTMLResponse)
    async def dashboard_html(dashboard_path: str = "") -> HTMLResponse:
        await _record_http_activity(app)
        return HTMLResponse(app.state.routes.service.load_dashboard_html())

    @app.get("/assets/{asset_path:path}")
    async def dashboard_asset(asset_path: str):
        await _record_http_activity(app)
        resolved = app.state.routes.service.resolve_dashboard_asset_path(asset_path)
        if resolved is None:
            raise HTTPException(status_code=404, detail="dashboard_asset_not_found")
        return FileResponse(resolved)

    @app.api_route("/api/{api_path:path}", methods=["GET", "POST"])
    async def dashboard_api(api_path: str, request: Request):
        await _record_http_activity(app)
        payload = await _payload_from_http_request(request)
        if api_path in _GLOBAL_DEFAULT_API_PATHS and "scope" not in payload and "workspace_id" not in payload:
            payload["scope"] = "global"
        try:
            result = await asyncio.to_thread(
                dispatch_management_request,
                app.state.routes,
                app.state.metadata,
                f"/api/{api_path}",
                payload,
            )
        except ValueError as exc:
            result = error_payload(exc)
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
    request_ctx = ctx
    session_id_value = arguments.get(_REQUEST_SESSION_ID_KEY) or arguments.get("session_id")
    if isinstance(session_id_value, str) and session_id_value.strip():
        request_ctx = replace(request_ctx, session_id=session_id_value.strip())
    workspace_root_value = arguments.get(_REQUEST_WORKSPACE_ROOT_KEY) or arguments.get("workspace_root")
    if not isinstance(workspace_root_value, str) or not workspace_root_value.strip():
        return request_ctx
    workspace_root = resolve_workspace_root(workspace_root=workspace_root_value)
    workspace_id = resolve_workspace_id(workspace_root=workspace_root_value)
    return replace(request_ctx, workspace_root=workspace_root, workspace_id=workspace_id)


async def _handle_session_start(app: FastAPI, ctx, hook_service: HookReminderService, arguments: dict[str, Any]) -> dict[str, Any]:
    request_ctx = _context_for_request(ctx, arguments)
    assert request_ctx.db_manager is not None
    await _cancel_idle_shutdown_task(app)
    request_timestamp = _hook_payload_timestamp(arguments)
    response: dict[str, Any] = dict(
        HookReminderService(request_ctx.db_manager, request_ctx.workspace_id).record_session_start(
            str(arguments.get("conversation_id") or arguments.get("sessionId") or arguments.get("session_id") or ""),
            arguments,
        )
    )
    response["active_client_count"] = hook_service.get_active_client_count(now=request_timestamp)
    response["shutdown_scheduled"] = False
    return response


async def _handle_post_tool_use(ctx, arguments: dict[str, Any]) -> dict[str, Any]:
    request_ctx = _context_for_request(ctx, arguments)
    assert request_ctx.db_manager is not None
    return HookReminderService(request_ctx.db_manager, request_ctx.workspace_id).record_post_tool_use(arguments)


async def _handle_session_end(app: FastAPI, ctx, hook_service: HookReminderService, arguments: dict[str, Any]) -> dict[str, Any]:
    request_ctx = _context_for_request(ctx, arguments)
    assert request_ctx.db_manager is not None
    request_timestamp = _hook_payload_timestamp(arguments)
    response: dict[str, Any] = dict(
        HookReminderService(request_ctx.db_manager, request_ctx.workspace_id).record_session_end(
            str(arguments.get("conversation_id") or arguments.get("sessionId") or arguments.get("session_id") or ""),
            arguments,
        )
    )
    active_client_count = hook_service.get_active_client_count(now=request_timestamp)
    should_schedule_shutdown = bool(app.state.enable_idle_shutdown) and active_client_count == 0
    if should_schedule_shutdown:
        _schedule_idle_shutdown_if_needed(app)
    else:
        await _cancel_idle_shutdown_task(app)
    response["active_client_count"] = active_client_count
    response["shutdown_scheduled"] = should_schedule_shutdown
    return response


def _hook_payload_timestamp(arguments: dict[str, Any]) -> float | None:
    raw = arguments.get("timestamp")
    if isinstance(raw, bool):
        return float(int(raw))
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str) and raw.strip():
        try:
            return float(raw.strip())
        except ValueError:
            return None
    return None
