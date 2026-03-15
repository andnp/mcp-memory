from __future__ import annotations

from pathlib import Path
import time

from mcp_memory.config import (
    resolve_daemon_lock_path,
    resolve_daemon_metadata_path,
)
from mcp_memory.daemon_app import create_daemon_app
from mcp_memory.daemon_lifecycle import DaemonLockTimeoutError, FilesystemLock
from mcp_memory.daemon_models import DaemonMetadata
from mcp_memory.daemon_process import (
    find_free_port as _find_free_port,
    is_daemon_healthy as _is_daemon_healthy,
    read_daemon_metadata as _read_daemon_metadata,
    spawn_daemon_process as _spawn_daemon_process,
)
from mcp_memory.mcp.runtime import resolve_runtime_spec


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
        existing = _read_daemon_metadata(metadata_path)
        if existing is not None and _is_daemon_healthy(existing):
            return existing

        daemon_port = _find_free_port()
        _spawn_daemon_process(spec.workspace_root, spec.config.daemon.host, daemon_port)
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            current = _read_daemon_metadata(metadata_path)
            if current is not None and _is_daemon_healthy(current):
                return current
            time.sleep(spec.config.daemon.healthcheck_interval_seconds)
        raise RuntimeError(f"Timed out waiting for daemon startup for {spec.workspace_id}")
    finally:
        lock.release()


def read_daemon_metadata(metadata_path: Path) -> DaemonMetadata | None:
    return _read_daemon_metadata(metadata_path)


def daemon_url(workspace_root_override: str | None = None, cwd: Path | None = None) -> str:
    return ensure_daemon_started(workspace_root_override, cwd).base_url


__all__ = [
    "DaemonLockTimeoutError",
    "DaemonMetadata",
    "create_daemon_app",
    "daemon_url",
    "ensure_daemon_started",
    "read_daemon_metadata",
]
