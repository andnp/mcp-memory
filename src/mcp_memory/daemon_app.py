from __future__ import annotations

import asyncio
import importlib.metadata
import logging
import os
import signal
import time
from contextlib import asynccontextmanager, suppress
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse

from mcp_memory.config import resolve_workspace_id, resolve_workspace_root
from mcp_memory.daemon_dispatch import (
    dispatch_federation_request,
    dispatch_management_request,
    error_payload,
)
from mcp_memory.daemon_background import (
    RECORD_THOUGHT_WRITEBACK_FLUSH_INTERVAL_SECONDS,
    flush_record_thought_writeback_once as _background_flush_record_thought_writeback_once,
)
from mcp_memory.daemon_process import find_free_port
from mcp_memory.daemon_runtime import DaemonRuntimeSession
from mcp_memory.core.journal_operations import flush_record_thought_writeback_outbox
from mcp_memory.sqlite_backup import create_and_prune_sqlite_backup
from mcp_memory.daemon_background import (
    consume_embedding_warmup_result as _background_consume_embedding_warmup_result,
    ensure_dashboard_frontend_ready as _background_ensure_dashboard_frontend_ready,
    warm_embedding_model as _background_warm_embedding_model,
)
from mcp_memory.hook_reminders import HookReminderService
from mcp_memory.mcp.runtime import create_runtime_from_spec, resolve_global_daemon_bootstrap_spec


logger = logging.getLogger(__name__)
# This is the sleep at the top of every _shutdown_daemon_when_idle poll
# iteration, so it also doubles as the grace window between the last client
# disappearing and the daemon actually killing itself. It used to be 0.25s,
# which gave a reconnecting client (e.g. a client restart across `/clear`, an
# IDE reload, or a brief network drop) no realistic chance to send a fresh
# session-start and cancel the shutdown via _cancel_idle_shutdown_task before
# SIGTERM fired -- causing avoidable cold-start churn (~15-20s Postgres +
# embedding-model init) for every reconnect that raced the shutdown.
_IDLE_SHUTDOWN_DELAY_SECONDS = 15.0
_HTTP_ACTIVITY_GRACE_SECONDS = 60.0
_RECORD_THOUGHT_WRITEBACK_FLUSH_INTERVAL_SECONDS = RECORD_THOUGHT_WRITEBACK_FLUSH_INTERVAL_SECONDS
_REQUEST_WORKSPACE_ROOT_KEY = "__workspace_root"
_REQUEST_SESSION_ID_KEY = "__session_id"


@dataclass(frozen=True)
class _RequestScope:
    session_id: str | None = None
    workspace_root: Path | None = None


def _ensure_dashboard_frontend_ready(static_root: Path) -> None:
    _background_ensure_dashboard_frontend_ready(static_root)


async def _warm_embedding_model(embedder: Any) -> bool:
    return await _background_warm_embedding_model(embedder)


def _consume_embedding_warmup_result(task: asyncio.Task[bool]) -> None:
    _background_consume_embedding_warmup_result(task)


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
            try:
                active_clients = app.state.routes.hook_service.get_active_client_count()
            except Exception as exc:
                logger.warning("Unable to read active daemon client count during idle shutdown check", exc_info=exc)
                continue
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


async def _run_periodic_idle_shutdown_scheduler(app: FastAPI) -> None:
    try:
        while True:
            interval = 5.0 if _IDLE_SHUTDOWN_DELAY_SECONDS >= 0.1 else 0.05
            await asyncio.sleep(interval)
            if bool(getattr(app.state, "enable_idle_shutdown", False)):
                _schedule_idle_shutdown_if_needed(app)
    except asyncio.CancelledError:
        pass


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
    from mcp_memory.daemon_background import run_periodic_backup_loop

    await run_periodic_backup_loop(runtime, backup_fn=create_and_prune_sqlite_backup)


def _flush_record_thought_writeback_once(runtime) -> int:
    return _background_flush_record_thought_writeback_once(
        runtime,
        flush_fn=flush_record_thought_writeback_outbox,
    )


async def _run_record_thought_writeback_flush_loop(
    runtime,
    *,
    executor: ThreadPoolExecutor,
    poll_seconds: float = _RECORD_THOUGHT_WRITEBACK_FLUSH_INTERVAL_SECONDS,
) -> None:
    loop = asyncio.get_running_loop()
    while True:
        try:
            await loop.run_in_executor(executor, _flush_record_thought_writeback_once, runtime)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Record-thought writeback flush loop failed", exc_info=True)
        await asyncio.sleep(poll_seconds)


def create_daemon_app(
    host: str | None = None,
    port: int | None = None,
    *,
    enable_idle_shutdown: bool = False,
):
    spec = resolve_global_daemon_bootstrap_spec()
    daemon_host = host or spec.config.daemon.host
    if port is None:
        daemon_port = spec.config.daemon.port
    elif port == 0:
        daemon_port = find_free_port()
    else:
        daemon_port = int(port)
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        session: DaemonRuntimeSession | None = None

        def runtime():
            assert session is not None
            return session.runtime

        def hook_service():
            assert session is not None
            assert session.hook_service is not None
            return session.hook_service

        session = DaemonRuntimeSession(
            app=app,
            spec=spec,
            host=daemon_host,
            port=daemon_port,
            enable_idle_shutdown=enable_idle_shutdown,
            request_scope_context_factory=lambda arguments: _apply_request_scope_to_context(
                runtime(),
                _request_scope_for_arguments(arguments),
            ),
            session_start_handler=lambda arguments: _handle_session_start(
                app,
                runtime(),
                hook_service(),
                arguments,
            ),
            post_tool_use_handler=lambda arguments: _handle_post_tool_use(runtime(), arguments),
            session_end_handler=lambda arguments: _handle_session_end(
                app,
                runtime(),
                hook_service(),
                arguments,
            ),
            runtime_version=_resolve_runtime_version(),
            runtime_factory=lambda runtime_spec: create_runtime_from_spec(runtime_spec),
            embedding_warmup=_warm_embedding_model,
            embedding_warmup_done=_consume_embedding_warmup_result,
            dashboard_builder=_ensure_dashboard_frontend_ready,
        )
        await session.start()
        app.state.idle_shutdown_task = None
        app.state.idle_shutdown_scheduler_task = None
        app.state.backup_task = session.backup_task
        app.state.record_thought_writeback_flush_task = session.writeback_task
        app.state.record_thought_writeback_flush_executor = session.writeback_executor
        app.state.enable_idle_shutdown = enable_idle_shutdown
        app.state.last_http_activity_at = time.monotonic()
        if enable_idle_shutdown:
            _schedule_idle_shutdown_if_needed(app)
            app.state.idle_shutdown_scheduler_task = asyncio.create_task(
                _run_periodic_idle_shutdown_scheduler(app)
            )
        try:
            yield
        finally:
            await _cancel_background_task(app, "idle_shutdown_scheduler_task")
            await _cancel_idle_shutdown_task(app)
            await session.stop()
            app.state.backup_task = None
            app.state.record_thought_writeback_flush_task = None
            app.state.record_thought_writeback_flush_executor = None

    app = FastAPI(title="mcp-memory daemon", lifespan=lifespan)

    @app.get("/")
    async def dashboard_root() -> RedirectResponse:
        await _record_http_activity(app)
        return RedirectResponse(url="/dashboard", status_code=307)

    @app.get("/dashboard", response_class=HTMLResponse)
    @app.get("/dashboard/{dashboard_path:path}", response_class=HTMLResponse)
    async def dashboard_html(dashboard_path: str = "") -> HTMLResponse:
        await _record_http_activity(app)
        return HTMLResponse(app.state.routes.service.load_dashboard_html())

    @app.get("/assets/{asset_path:path}")
    @app.get("/dashboard/assets/{asset_path:path}")
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

    @app.api_route("/v1/{v1_path:path}", methods=["GET", "POST"])
    async def federation_api(v1_path: str, request: Request):
        await _record_http_activity(app)
        payload = await _payload_from_http_request(request)
        try:
            result = await dispatch_federation_request(
                app.state.routes,
                f"/v1/{v1_path}",
                payload,
            )
        except (TypeError, ValueError) as exc:
            return JSONResponse({"status": "error", "error": str(exc)}, status_code=400)
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


def _request_scope_for_arguments(arguments: dict[str, Any] | None) -> _RequestScope:
    if arguments is None:
        return _RequestScope()
    session_id = _optional_request_str(arguments, _REQUEST_SESSION_ID_KEY, "session_id")
    workspace_root_value = _optional_request_str(arguments, _REQUEST_WORKSPACE_ROOT_KEY, "workspace_root")
    if workspace_root_value is None:
        return _RequestScope(session_id=session_id)
    workspace_root = resolve_workspace_root(workspace_root=workspace_root_value)
    return _RequestScope(
        session_id=session_id,
        workspace_root=workspace_root,
    )


def _apply_request_scope_to_context(ctx, request_scope: _RequestScope):
    if request_scope.session_id is None and request_scope.workspace_root is None:
        return ctx
    request_ctx = ctx
    if request_scope.session_id is not None:
        request_ctx = replace(request_ctx, session_id=request_scope.session_id)
    if request_scope.workspace_root is None:
        return request_ctx
    return replace(
        request_ctx,
        workspace_root=request_scope.workspace_root,
        workspace_id=_workspace_id_for_request_scope(request_scope),
    )


def _workspace_id_for_request_scope(request_scope: _RequestScope) -> str | None:
    if request_scope.workspace_root is None:
        return None
    return resolve_workspace_id(workspace_root=str(request_scope.workspace_root))


def _context_for_request(ctx, arguments: dict[str, Any] | None):
    return _apply_request_scope_to_context(ctx, _request_scope_for_arguments(arguments))


def _optional_request_str(arguments: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


async def _handle_session_start(app: FastAPI, ctx, hook_service: HookReminderService, arguments: dict[str, Any]) -> dict[str, Any]:
    request_scope = _request_scope_for_arguments(arguments)
    assert ctx.db_manager is not None
    await _cancel_idle_shutdown_task(app)
    request_timestamp = _hook_payload_timestamp(arguments)
    response: dict[str, Any] = dict(
        HookReminderService(ctx.db_manager, _workspace_id_for_request_scope(request_scope)).record_session_start(
            str(arguments.get("conversation_id") or arguments.get("sessionId") or arguments.get("session_id") or ""),
            arguments,
        )
    )
    response["active_client_count"] = hook_service.get_active_client_count(now=request_timestamp)
    response["shutdown_scheduled"] = False
    return response


async def _handle_post_tool_use(ctx, arguments: dict[str, Any]) -> dict[str, Any]:
    request_scope = _request_scope_for_arguments(arguments)
    assert ctx.db_manager is not None
    return HookReminderService(ctx.db_manager, _workspace_id_for_request_scope(request_scope)).record_post_tool_use(arguments)


async def _handle_session_end(app: FastAPI, ctx, hook_service: HookReminderService, arguments: dict[str, Any]) -> dict[str, Any]:
    request_scope = _request_scope_for_arguments(arguments)
    assert ctx.db_manager is not None
    request_timestamp = _hook_payload_timestamp(arguments)
    response: dict[str, Any] = dict(
        HookReminderService(ctx.db_manager, _workspace_id_for_request_scope(request_scope)).record_session_end(
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
