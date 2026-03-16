from __future__ import annotations

import os
from pathlib import Path
import signal
import time
from dataclasses import dataclass

from mcp_memory.config import (
    GLOBAL_DAEMON_IDENTITY,
    resolve_daemon_lock_path,
    resolve_daemon_metadata_path,
)
from mcp_memory.daemon_app import create_daemon_app
from mcp_memory.daemon_lifecycle import DaemonLockTimeoutError, FilesystemLock
from mcp_memory.daemon_models import DaemonMetadata
from mcp_memory.daemon_process import (
    find_free_port as _find_free_port,
    is_daemon_healthy as _is_daemon_healthy,
    remove_metadata,
    read_daemon_metadata as _read_daemon_metadata,
    spawn_daemon_process as _spawn_daemon_process,
)
from mcp_memory.mcp.runtime import resolve_runtime_spec


@dataclass(frozen=True)
class _DaemonProcess:
    pid: int
    executable: str | None
    command: tuple[str, ...]


def ensure_daemon_started(
    workspace_root_override: str | None = None,
    cwd: Path | None = None,
) -> DaemonMetadata:
    spec = resolve_runtime_spec(workspace_root_override, cwd)
    metadata_path = resolve_daemon_metadata_path(GLOBAL_DAEMON_IDENTITY)
    lock = FilesystemLock(resolve_daemon_lock_path(GLOBAL_DAEMON_IDENTITY))
    timeout_seconds = spec.config.daemon.auto_start_timeout_seconds
    lock.acquire(timeout_seconds=timeout_seconds)
    try:
        existing = _read_daemon_metadata(metadata_path)
        if existing is not None and _is_daemon_healthy(existing):
            return existing
        if existing is None or not _is_process_running(existing.pid):
            if existing is not None:
                remove_metadata(metadata_path)
            _terminate_orphaned_daemon_processes(
                current_pid=os.getpid(),
                deadline=time.monotonic() + timeout_seconds,
                poll_interval_seconds=spec.config.daemon.healthcheck_interval_seconds,
            )
        elif existing is not None:
            try:
                os.kill(existing.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            _wait_for_process_exit(
                existing.pid,
                deadline=time.monotonic() + timeout_seconds,
                poll_interval_seconds=spec.config.daemon.healthcheck_interval_seconds,
            )

        daemon_port = _find_free_port()
        _spawn_daemon_process(spec.workspace_root, spec.config.daemon.host, daemon_port)
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            current = _read_daemon_metadata(metadata_path)
            if current is not None and _is_daemon_healthy(current):
                return current
            time.sleep(spec.config.daemon.healthcheck_interval_seconds)
        raise RuntimeError("Timed out waiting for global daemon startup")
    finally:
        lock.release()


def read_daemon_metadata(metadata_path: Path) -> DaemonMetadata | None:
    return _read_daemon_metadata(metadata_path)



def inspect_daemon(
    workspace_root_override: str | None = None,
    cwd: Path | None = None,
) -> tuple[str, DaemonMetadata | None, bool]:
    spec = resolve_runtime_spec(workspace_root_override, cwd)
    metadata = read_daemon_metadata(resolve_daemon_metadata_path(GLOBAL_DAEMON_IDENTITY))
    healthy = metadata is not None and _is_daemon_healthy(metadata)
    return spec.workspace_id, metadata, healthy


def stop_daemon(
    workspace_root_override: str | None = None,
    cwd: Path | None = None,
) -> DaemonMetadata | None:
    spec = resolve_runtime_spec(workspace_root_override, cwd)
    metadata_path = resolve_daemon_metadata_path(GLOBAL_DAEMON_IDENTITY)
    metadata = read_daemon_metadata(metadata_path)
    if metadata is None:
        return None

    if not _is_daemon_healthy(metadata):
        remove_metadata(metadata_path)
        return metadata

    try:
        os.kill(metadata.pid, signal.SIGTERM)
    except ProcessLookupError:
        remove_metadata(metadata_path)
        return metadata

    deadline = time.monotonic() + max(spec.config.daemon.shutdown_grace_seconds, 1.0)
    while time.monotonic() < deadline:
        healthy = _is_daemon_healthy(metadata)
        running = _is_process_running(metadata.pid)
        if not healthy and not running:
            remove_metadata(metadata_path)
            return metadata
        time.sleep(spec.config.daemon.healthcheck_interval_seconds)

    raise RuntimeError("Timed out waiting for global daemon shutdown")


def daemon_url(workspace_root_override: str | None = None, cwd: Path | None = None) -> str:
    return ensure_daemon_started(workspace_root_override, cwd).base_url


__all__ = [
    "DaemonLockTimeoutError",
    "DaemonMetadata",
    "create_daemon_app",
    "daemon_url",
    "ensure_daemon_started",
    "read_daemon_metadata",
    "inspect_daemon",
    "stop_daemon",
]


def _is_process_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_process_exit(
    pid: int,
    *,
    deadline: float,
    poll_interval_seconds: float,
) -> None:
    while time.monotonic() < deadline:
        if not _is_process_running(pid):
            return
        time.sleep(poll_interval_seconds)
    raise RuntimeError(f"Timed out waiting for daemon process exit: pid={pid}")


def _terminate_orphaned_daemon_processes(
    *,
    current_pid: int,
    deadline: float,
    poll_interval_seconds: float,
) -> None:
    for process in _list_daemon_processes(current_pid=current_pid):
        try:
            os.kill(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        _wait_for_process_exit(
            process.pid,
            deadline=deadline,
            poll_interval_seconds=poll_interval_seconds,
        )


def _list_daemon_processes(*, current_pid: int) -> list[_DaemonProcess]:
    processes: list[_DaemonProcess] = []
    proc_root = Path("/proc")
    if not proc_root.exists():
        return processes
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == current_pid:
            continue
        command = _read_process_command(entry)
        if not _is_daemon_command(command):
            continue
        processes.append(
            _DaemonProcess(
                pid=pid,
                executable=_read_process_executable(entry),
                command=command,
            )
        )
    return processes


def _read_process_command(proc_entry: Path) -> tuple[str, ...]:
    try:
        raw = (proc_entry / "cmdline").read_bytes()
    except OSError:
        return ()
    parts = [part.decode("utf-8", errors="ignore") for part in raw.split(b"\0") if part]
    return tuple(parts)


def _read_process_executable(proc_entry: Path) -> str | None:
    try:
        return str((proc_entry / "exe").resolve())
    except OSError:
        return None


def _is_daemon_command(command: tuple[str, ...]) -> bool:
    if not command:
        return False
    return len(command) >= 4 and command[1:4] == ("-m", "mcp_memory.cli", "daemon")
