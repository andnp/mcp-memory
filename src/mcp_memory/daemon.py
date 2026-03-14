from __future__ import annotations

import asyncio
import fcntl
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

from mcp_memory.mcp.runtime import (
    RuntimeSpec,
    create_runtime_from_spec,
    resolve_runtime_spec,
)


logger = logging.getLogger(__name__)

DEFAULT_LOCK_TIMEOUT_SECONDS = 10.0
DEFAULT_SHUTDOWN_GRACE_SECONDS = 5.0
LOCK_POLL_INTERVAL_SECONDS = 0.05


class DaemonLockTimeoutError(TimeoutError):
    pass


@dataclass
class FilesystemLock:
    lock_path: Path
    _handle: object | None = None

    def acquire(self, timeout_seconds: float | None = DEFAULT_LOCK_TIMEOUT_SECONDS):
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.lock_path.open("a+", encoding="utf-8")
        deadline = None
        if timeout_seconds is not None:
            deadline = time.monotonic() + timeout_seconds

        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._write_metadata(handle)
                self._handle = handle
                return
            except BlockingIOError as exc:
                if deadline is not None and time.monotonic() >= deadline:
                    handle.close()
                    raise DaemonLockTimeoutError(
                        f"Timed out acquiring daemon lock: {self.lock_path}"
                    ) from exc
                time.sleep(LOCK_POLL_INTERVAL_SECONDS)
            except Exception:
                handle.close()
                raise

    def release(self):
        handle = self._handle
        self._handle = None
        if handle is None:
            return

        try:
            handle.seek(0)
            handle.truncate()
            handle.flush()
            os.fsync(handle.fileno())
        except OSError:
            logger.debug("Failed clearing daemon lock metadata for %s", self.lock_path)

        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()

    def _write_metadata(self, handle):
        payload = {
            "pid": os.getpid(),
            "acquired_at": time.time(),
            "lock_path": str(self.lock_path),
        }
        handle.seek(0)
        handle.truncate()
        json.dump(payload, handle, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())


class DaemonLifecycleController:
    def __init__(
        self,
        shutdown_grace_seconds: float = DEFAULT_SHUTDOWN_GRACE_SECONDS,
        lock_timeout_seconds: float | None = DEFAULT_LOCK_TIMEOUT_SECONDS,
        runtime_spec_resolver=resolve_runtime_spec,
        runtime_factory=create_runtime_from_spec,
    ):
        self._shutdown_grace_seconds = shutdown_grace_seconds
        self._lock_timeout_seconds = lock_timeout_seconds
        self._runtime_spec_resolver = runtime_spec_resolver
        self._runtime_factory = runtime_factory
        self._state_lock = asyncio.Lock()
        self._runtime = None
        self._runtime_spec: RuntimeSpec | None = None
        self._runtime_lock: FilesystemLock | None = None
        self._client_count = 0
        self._shutdown_task: asyncio.Task[None] | None = None

    @property
    def client_count(self) -> int:
        return self._client_count

    @property
    def has_runtime(self) -> bool:
        return self._runtime is not None

    async def acquire_runtime(
        self,
        project_override: str | None = None,
        cwd: Path | None = None,
    ):
        spec = await asyncio.to_thread(
            self._runtime_spec_resolver,
            project_override,
            cwd,
        )

        async with self._state_lock:
            self._cancel_shutdown_locked()

            if self._runtime is not None:
                if self._runtime_spec != spec:
                    raise RuntimeError(
                        "daemon lifecycle controller already owns a different runtime"
                    )
                self._client_count += 1
                return self._runtime

            runtime_lock = FilesystemLock(spec.memory_path / ".daemon.lock")

            try:
                await asyncio.to_thread(
                    runtime_lock.acquire,
                    self._lock_timeout_seconds,
                )
                runtime = await asyncio.to_thread(self._runtime_factory, spec)
            except Exception:
                await asyncio.to_thread(runtime_lock.release)
                raise

            self._runtime = runtime
            self._runtime_spec = spec
            self._runtime_lock = runtime_lock
            self._client_count = 1
            return runtime

    async def release_runtime(self, runtime=None):
        async with self._state_lock:
            if self._runtime is None or self._client_count == 0:
                raise RuntimeError("release_runtime called without an acquired runtime")

            if runtime is not None and runtime is not self._runtime:
                raise RuntimeError("release_runtime called for a non-owned runtime")

            self._client_count -= 1
            if self._client_count == 0:
                self._schedule_shutdown_locked()
                return self._shutdown_task

            return None

    async def shutdown_now(self):
        shutdown_task: asyncio.Task[None] | None = None
        async with self._state_lock:
            shutdown_task = self._shutdown_task
            self._shutdown_task = None

        if shutdown_task is not None and not shutdown_task.done():
            shutdown_task.cancel()
            await asyncio.gather(shutdown_task, return_exceptions=True)

        await self._shutdown_if_idle(force=True)

    def _cancel_shutdown_locked(self):
        if self._shutdown_task is None:
            return
        if not self._shutdown_task.done():
            self._shutdown_task.cancel()
        self._shutdown_task = None

    def _schedule_shutdown_locked(self):
        if self._shutdown_task is not None and not self._shutdown_task.done():
            return
        self._shutdown_task = asyncio.create_task(self._shutdown_after_grace())

    async def _shutdown_after_grace(self):
        try:
            await asyncio.sleep(self._shutdown_grace_seconds)
        except asyncio.CancelledError:
            return

        await self._shutdown_if_idle(force=False)

    async def _shutdown_if_idle(self, force: bool):
        runtime = None
        runtime_lock = None

        async with self._state_lock:
            if self._runtime is None:
                self._shutdown_task = None
                return

            if not force and self._client_count != 0:
                self._shutdown_task = None
                return

            runtime = self._runtime
            runtime_lock = self._runtime_lock
            self._runtime = None
            self._runtime_spec = None
            self._runtime_lock = None
            self._client_count = 0
            self._shutdown_task = None

        try:
            await asyncio.to_thread(runtime.close)
        finally:
            if runtime_lock is not None:
                await asyncio.to_thread(runtime_lock.release)


_DEFAULT_CONTROLLER: DaemonLifecycleController | None = None


def get_daemon_lifecycle_controller():
    global _DEFAULT_CONTROLLER
    if _DEFAULT_CONTROLLER is None:
        _DEFAULT_CONTROLLER = DaemonLifecycleController()
    return _DEFAULT_CONTROLLER


async def reset_daemon_lifecycle_controller():
    global _DEFAULT_CONTROLLER
    if _DEFAULT_CONTROLLER is not None:
        await _DEFAULT_CONTROLLER.shutdown_now()
        _DEFAULT_CONTROLLER = None