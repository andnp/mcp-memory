from __future__ import annotations

import logging
import os
from pathlib import Path
import shlex
import signal
import time
from dataclasses import dataclass

from mcp_memory.config import (
    GLOBAL_DAEMON_IDENTITY,
    resolve_daemon_lock_path,
    resolve_daemon_metadata_path,
    resolve_daemon_socket_path,
    resolve_daemon_startup_log_path,
)
from mcp_memory.daemon_app import create_daemon_app
from mcp_memory.daemon_lifecycle import DaemonLockTimeoutError, FilesystemLock
from mcp_memory.daemon_models import DaemonMetadata
from mcp_memory.daemon_process import (
    DaemonSpawnDetails,
    find_free_port as _find_free_port,
    is_daemon_healthy as _is_daemon_healthy,
    remove_metadata,
    read_daemon_metadata as _read_daemon_metadata,
    spawn_daemon_process as _spawn_daemon_process,
)
from mcp_memory.daemon_transport import probe_daemon_socket as _probe_daemon_socket
from mcp_memory.daemon_transport import remove_daemon_socket as _remove_daemon_socket
from mcp_memory.mcp.runtime import resolve_runtime_spec
from mcp_memory.process_termination import ProcessTerminationResult as _DaemonTerminationResult
from mcp_memory.process_termination import send_process_signal as _send_process_signal
from mcp_memory.process_termination import terminate_process as _terminate_process_with_scope
from mcp_memory.process_termination import wait_for_process_exit as _shared_wait_for_process_exit


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _DaemonProcess:
    pid: int
    executable: str | None
    command: tuple[str, ...]


@dataclass(frozen=True)
class DaemonStopResult:
    metadata: DaemonMetadata
    stop_reason: str
    signal_sequence: tuple[str, ...] = ()
    process_group_id: int | None = None
    escalated_to_sigkill: bool = False
    stale_socket_removed: bool = False

    @property
    def pid(self) -> int:
        return self.metadata.pid


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
            logger.info("Reusing healthy global daemon: pid=%s endpoint=%s", existing.pid, existing.transport_endpoint)
            return existing

        probe_timeout_seconds = max(min(spec.config.daemon.healthcheck_interval_seconds, 0.1), 0.05)
        cleanup_deadline = time.monotonic() + timeout_seconds
        if existing is None or not _is_process_running(existing.pid):
            if existing is not None:
                logger.warning("Removing stale global daemon metadata: pid=%s", existing.pid)
                remove_metadata(metadata_path)
            _terminate_orphaned_daemon_processes(
                current_pid=os.getpid(),
                deadline=cleanup_deadline,
                poll_interval_seconds=spec.config.daemon.healthcheck_interval_seconds,
            )
            _cleanup_stale_daemon_socket(
                Path(existing.socket_path) if existing is not None and existing.socket_path else resolve_daemon_socket_path(),
                probe_timeout_seconds=probe_timeout_seconds,
                reason="owner_missing_before_spawn",
            )
        elif existing is not None:
            logger.warning("Stopping unhealthy global daemon before recovery: pid=%s", existing.pid)
            _terminate_daemon_process(
                existing.pid,
                deadline=cleanup_deadline,
                poll_interval_seconds=spec.config.daemon.healthcheck_interval_seconds,
            )
            remove_metadata(metadata_path)
            _cleanup_stale_daemon_socket(
                Path(existing.socket_path) if existing.socket_path else resolve_daemon_socket_path(),
                probe_timeout_seconds=probe_timeout_seconds,
                reason="owner_stopped_for_recovery",
            )

        daemon_port = spec.config.daemon.port
        if daemon_port == 0:
            daemon_port = _find_free_port()
        spawn_details = _spawn_daemon_process(spec.workspace_root, spec.config.daemon.host, daemon_port)
        poll_interval_seconds = spec.config.daemon.healthcheck_interval_seconds
        readiness_deadline = time.monotonic() + timeout_seconds
        current = None
        while time.monotonic() < readiness_deadline:
            current = _read_daemon_metadata(metadata_path)
            if current is not None and _is_daemon_healthy(current):
                return current
            if _spawned_daemon_exited_before_readiness(spawn_details, current):
                raise RuntimeError(
                    _format_startup_failure_message(
                        "Global daemon exited before becoming ready.",
                        spawn_details=spawn_details,
                        metadata=current,
                    )
                )
            time.sleep(poll_interval_seconds)

        if current is None:
            metadata_grace_deadline = time.monotonic() + max(20.0, timeout_seconds)
            while time.monotonic() < metadata_grace_deadline:
                current = _read_daemon_metadata(metadata_path)
                if current is not None:
                    break
                if _spawned_daemon_exited_before_readiness(spawn_details, current):
                    raise RuntimeError(
                        _format_startup_failure_message(
                            "Global daemon exited before publishing readiness metadata.",
                            spawn_details=spawn_details,
                            metadata=current,
                        )
                    )
                time.sleep(poll_interval_seconds)

        if current is not None and _is_process_running(current.pid):
            grace_seconds = max(5.0, timeout_seconds * 2)
            grace_deadline = time.monotonic() + grace_seconds
            while time.monotonic() < grace_deadline:
                latest = _read_daemon_metadata(metadata_path)
                if latest is not None and _is_daemon_healthy(latest):
                    return latest
                if _spawned_daemon_exited_before_readiness(spawn_details, latest):
                    raise RuntimeError(
                        _format_startup_failure_message(
                            "Global daemon exited before reaching a healthy state.",
                            spawn_details=spawn_details,
                            metadata=latest,
                        )
                    )
                time.sleep(poll_interval_seconds)
            latest = _read_daemon_metadata(metadata_path)
            if latest is not None and _is_process_running(latest.pid):
                raise RuntimeError(
                    _format_startup_failure_message(
                        "Timed out waiting for global daemon startup.",
                        spawn_details=spawn_details,
                        metadata=latest,
                    )
                )
        raise RuntimeError(
            _format_startup_failure_message(
                "Timed out waiting for global daemon startup.",
                spawn_details=spawn_details,
                metadata=current,
            )
        )
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
) -> DaemonStopResult | None:
    spec = resolve_runtime_spec(workspace_root_override, cwd)
    metadata_path = resolve_daemon_metadata_path(GLOBAL_DAEMON_IDENTITY)
    metadata = read_daemon_metadata(metadata_path)
    if metadata is None:
        return None

    deadline = time.monotonic() + max(spec.config.daemon.shutdown_grace_seconds, 1.0)
    poll_interval_seconds = spec.config.daemon.healthcheck_interval_seconds
    probe_timeout_seconds = max(min(spec.config.daemon.healthcheck_interval_seconds, 0.1), 0.05)
    if not _is_process_running(metadata.pid):
        remove_metadata(metadata_path)
        stale_socket_removed = _cleanup_stale_daemon_socket(
            Path(metadata.socket_path) if metadata.socket_path else resolve_daemon_socket_path(),
            probe_timeout_seconds=probe_timeout_seconds,
            reason="owner_missing_on_stop",
        )
        return DaemonStopResult(
            metadata=metadata,
            stop_reason="owner_missing_on_stop",
            stale_socket_removed=stale_socket_removed,
        )

    termination = _terminate_daemon_process(
        metadata.pid,
        deadline=deadline,
        poll_interval_seconds=poll_interval_seconds,
    )
    stale_socket_removed = _cleanup_stale_daemon_socket(
        Path(metadata.socket_path) if metadata.socket_path else resolve_daemon_socket_path(),
        probe_timeout_seconds=probe_timeout_seconds,
        reason="owner_stopped_on_stop",
    )
    remove_metadata(metadata_path)
    return DaemonStopResult(
        metadata=metadata,
        stop_reason="owner_stopped_on_stop",
        signal_sequence=termination.signal_sequence,
        process_group_id=termination.process_group_id,
        escalated_to_sigkill=termination.escalated_to_sigkill,
        stale_socket_removed=stale_socket_removed,
    )


def daemon_url(workspace_root_override: str | None = None, cwd: Path | None = None) -> str:
    return ensure_daemon_started(workspace_root_override, cwd).base_url


__all__ = [
    "DaemonLockTimeoutError",
    "DaemonMetadata",
    "DaemonStopResult",
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
    if _read_process_state(pid) == 'Z':
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_process_state(pid: int) -> str | None:
    if pid <= 0:
        return None
    stat_path = Path('/proc') / str(pid) / 'stat'
    try:
        raw = stat_path.read_text(encoding='utf-8')
    except OSError:
        return None
    fields = raw.split()
    if len(fields) < 3:
        return None
    return fields[2]


def _wait_for_process_exit(
    pid: int,
    *,
    deadline: float,
    poll_interval_seconds: float,
) -> None:
    _shared_wait_for_process_exit(
        pid,
        deadline=deadline,
        poll_interval_seconds=poll_interval_seconds,
        is_process_running=_is_process_running,
    )


def _terminate_daemon_process(
    pid: int,
    *,
    deadline: float,
    poll_interval_seconds: float,
) -> _DaemonTerminationResult:
    return _terminate_process_with_scope(
        pid,
        deadline=deadline,
        poll_interval_seconds=poll_interval_seconds,
        is_process_running=_is_process_running,
        send_signal=_signal_daemon_process,
        wait_for_exit=_wait_for_process_exit,
    )


def _signal_daemon_process(pid: int, sig: signal.Signals) -> int | None:
    scope = "pid" if sig == signal.SIGTERM else "process_group"
    process_group_id, _ = _send_process_signal(pid, sig, scope=scope)
    return process_group_id


def _cleanup_stale_daemon_socket(
    socket_path: Path,
    *,
    probe_timeout_seconds: float,
    reason: str,
) -> bool:
    if not _socket_path_exists(socket_path):
        return False
    if _probe_daemon_socket(socket_path, timeout_seconds=probe_timeout_seconds):
        logger.warning("Daemon socket still responded during %s; leaving it in place: %s", reason, socket_path)
        return False
    logger.warning("Cleaning stale daemon socket during %s: %s", reason, socket_path)
    _remove_daemon_socket(socket_path)
    return True


def _socket_path_exists(socket_path: Path) -> bool:
    try:
        return socket_path.exists() or socket_path.is_socket()
    except OSError:
        return socket_path.exists()


def _terminate_orphaned_daemon_processes(
    *,
    current_pid: int,
    deadline: float,
    poll_interval_seconds: float,
) -> None:
    for process in _list_daemon_processes(current_pid=current_pid):
        _terminate_daemon_process(
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


def _spawned_daemon_exited_before_readiness(
    spawn_details: DaemonSpawnDetails | None,
    metadata: DaemonMetadata | None,
) -> bool:
    if spawn_details is None:
        return False
    try:
        returncode = spawn_details.process.poll()
    except Exception:
        return False
    if returncode is None:
        return False
    return metadata is None or metadata.pid == spawn_details.pid


def _format_startup_failure_message(
    reason: str,
    *,
    spawn_details: DaemonSpawnDetails | None,
    metadata: DaemonMetadata | None,
) -> str:
    startup_log_path = resolve_daemon_startup_log_path()
    details: list[str] = [reason]
    if spawn_details is not None:
        details.append(f"spawn_pid={spawn_details.pid}")
        details.append(f"startup_log={spawn_details.startup_log_path}")
        try:
            returncode = spawn_details.process.poll()
        except Exception:
            returncode = None
        if returncode is not None:
            details.append(f"exit_code={returncode}")
        details.append(f"command={shlex.join(spawn_details.command)}")
        startup_log_path = spawn_details.startup_log_path
    else:
        details.append(f"startup_log={startup_log_path}")

    if metadata is None:
        details.append("metadata=missing")
    else:
        details.append(f"metadata_pid={metadata.pid}")
        details.append(f"metadata_endpoint={metadata.transport_endpoint}")
        details.append(f"metadata_status={metadata.status}")

    tail = _read_startup_log_tail(startup_log_path)
    if tail is None:
        return " ".join(details)
    return " ".join(details) + f"\nStartup log tail ({startup_log_path}):\n{tail}"


def _read_startup_log_tail(startup_log_path: Path, *, max_lines: int = 40, max_chars: int = 4000) -> str | None:
    try:
        content = startup_log_path.read_text(encoding="utf-8")
    except OSError:
        return None
    if not content.strip():
        return None
    tail = "\n".join(content.splitlines()[-max_lines:]).strip()
    if len(tail) > max_chars:
        tail = tail[-max_chars:]
    return tail
