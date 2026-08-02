from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, cast

from fastapi import FastAPI

from mcp_memory.config import GLOBAL_DAEMON_IDENTITY, resolve_daemon_metadata_path, resolve_daemon_socket_path
from mcp_memory.core.agent_runtime import bootstrap_background_tasks, build_runtime_task_worker
from mcp_memory.daemon_background import (
    consume_embedding_warmup_result,
    ensure_dashboard_frontend_ready,
    resolve_record_thought_writeback_flush_context,
    run_periodic_backup_loop,
    run_record_thought_writeback_flush_loop,
    warm_embedding_model,
)
from mcp_memory.daemon_lifecycle import DaemonLockTimeoutError, FilesystemLock
from mcp_memory.daemon_models import DaemonControllerView, DaemonMetadata, DaemonRoutes
from mcp_memory.daemon_process import remove_metadata, write_metadata
from mcp_memory.daemon_transport import DaemonZmqServer
from mcp_memory.hook_reminders import HookReminderService
from mcp_memory.management.service import ManagementService
from mcp_memory.context import ApplicationContext
from mcp_memory.mcp.runtime import (
    RuntimeBootstrapSpec,
    RuntimeCapabilityBundles,
    RuntimeComposition,
    create_runtime_composition,
)


class DaemonRuntimeSession:
    """Own the global daemon's resource, worker, and transport lifecycle."""

    def __init__(
        self,
        *,
        app: FastAPI,
        spec,
        host: str,
        port: int,
        enable_idle_shutdown: bool,
        request_scope_context_factory: Callable[[dict[str, object]], object],
        session_start_handler: Callable[[dict[str, object]], Any],
        post_tool_use_handler: Callable[[dict[str, object]], Any],
        session_end_handler: Callable[[dict[str, object]], Any],
        runtime_version: str | None,
        runtime_factory: Callable[[RuntimeBootstrapSpec], RuntimeComposition | ApplicationContext] = create_runtime_composition,
        embedding_warmup: Callable[[Any], Any] = warm_embedding_model,
        embedding_warmup_done: Callable[[asyncio.Task[bool]], None] = consume_embedding_warmup_result,
        dashboard_builder: Callable[[Path], None] = ensure_dashboard_frontend_ready,
    ) -> None:
        self.app = app
        self.spec = spec
        self.host = host
        self.port = port
        self.enable_idle_shutdown = enable_idle_shutdown
        self.request_scope_context_factory = request_scope_context_factory
        self.session_start_handler = session_start_handler
        self.post_tool_use_handler = post_tool_use_handler
        self.session_end_handler = session_end_handler
        self.runtime_version = runtime_version
        self.runtime_factory = runtime_factory
        self.embedding_warmup = embedding_warmup
        self.embedding_warmup_done = embedding_warmup_done
        self.dashboard_builder = dashboard_builder
        self.metadata_path = resolve_daemon_metadata_path(GLOBAL_DAEMON_IDENTITY)
        self.socket_path = resolve_daemon_socket_path()
        self.runtime_lock = FilesystemLock(spec.lock_path.with_suffix(".runtime.lock"))
        self.composition: RuntimeComposition | None = None
        self.worker = None
        self.hook_service: HookReminderService | None = None
        self.transport: DaemonZmqServer | None = None
        self.warmup_task: asyncio.Task[bool] | None = None
        self.backup_task: asyncio.Task[None] | None = None
        self.writeback_task: asyncio.Task[None] | None = None
        self.writeback_executor: ThreadPoolExecutor | None = None

    @property
    def runtime(self):
        if self.composition is None:
            raise RuntimeError("daemon_runtime_not_started")
        return self.composition.context

    async def start(self) -> None:
        try:
            self.runtime_lock.acquire(timeout_seconds=0.1)
        except DaemonLockTimeoutError as exc:
            raise RuntimeError("daemon_runtime_lock_unavailable:global") from exc

        try:
            composition = self.runtime_factory(self.spec)
            if isinstance(composition, RuntimeComposition):
                self.composition = composition
            else:
                context = cast(ApplicationContext, composition)
                self.composition = RuntimeComposition(
                    context=context,
                    capabilities=RuntimeCapabilityBundles(
                        memory=context.memory_capabilities(),
                        mutation=context.mutation_capabilities(),
                        provider=context.provider_capabilities(),
                        background=context.background_task_capabilities(),
                        task=context.task_runtime_capabilities(),
                        management=context.management_capabilities(),
                    ),
                )
            runtime = self.runtime
            assert runtime.db_manager is not None
            bootstrap_background_tasks(self.composition.capabilities.background)
            self.worker = build_runtime_task_worker(self.composition.capabilities.task)
            self.warmup_task = asyncio.create_task(self.embedding_warmup(runtime.embedder))
            self.warmup_task.add_done_callback(self.embedding_warmup_done)
            self.backup_task = asyncio.create_task(run_periodic_backup_loop(runtime))
            if self.worker is not None:
                await self.worker.start()

            self.hook_service = HookReminderService(runtime.db_manager, None)
            self.transport = DaemonZmqServer(
                context_factory=self.request_scope_context_factory,
                hook_handlers={
                    "/api/hooks/session-start": self.session_start_handler,
                    "/api/hooks/post-tool-use": self.post_tool_use_handler,
                    "/api/hooks/session-end": self.session_end_handler,
                },
                routes_provider=lambda: self.app.state.routes,
                socket_path=self.socket_path,
                metadata_provider=lambda: self.app.state.metadata,
            )
            await self.transport.start()

            service = ManagementService(
                runtime.management_capabilities(),
                controller=DaemonControllerView(
                    hook_service=self.hook_service,
                    transport_server=self.transport,
                ),
            )
            self.app.state.routes = DaemonRoutes(
                ctx=runtime,
                service=service,
                management=service.capabilities,
                hook_service=self.hook_service,
                metadata_path=self.metadata_path,
            )
            self.app.state.metadata = DaemonMetadata(
                host=self.host,
                port=self.port,
                pid=os.getpid(),
                started_at=time.time(),
                status="ready",
                daemon_scope=GLOBAL_DAEMON_IDENTITY,
                binary_path=sys.executable,
                version=self.runtime_version,
                transport="zmq",
                socket_path=str(self.socket_path),
            )
            write_metadata(self.metadata_path, self.app.state.metadata)
            await asyncio.to_thread(
                self.dashboard_builder,
                self.app.state.routes.service.dashboard_static_root,
            )
            if resolve_record_thought_writeback_flush_context(runtime) is not None:
                self.writeback_executor = ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix="record-thought-writeback-flush",
                )
                self.writeback_task = asyncio.create_task(
                    run_record_thought_writeback_flush_loop(
                        runtime,
                        executor=self.writeback_executor,
                    )
                )
            self.app.state.enable_idle_shutdown = self.enable_idle_shutdown
        except BaseException:
            await self.stop()
            raise

    async def stop(self) -> None:
        for task_name in ("backup_task", "writeback_task"):
            task = getattr(self, task_name)
            if task is None:
                continue
            if not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
            setattr(self, task_name, None)
        if self.writeback_executor is not None:
            await asyncio.to_thread(self.writeback_executor.shutdown, True)
            self.writeback_executor = None
        if self.transport is not None:
            await self.transport.stop()
            self.transport = None
        if self.warmup_task is not None and not self.warmup_task.done():
            self.warmup_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.warmup_task
        self.warmup_task = None
        if self.worker is not None:
            await self.worker.stop(self.spec.config.daemon.shutdown_grace_seconds)
            self.worker = None
        if self.composition is not None:
            self.composition.close()
            self.composition = None
        remove_metadata(self.metadata_path, expected_pid=os.getpid())
        self.runtime_lock.release()
