from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from mcp_memory.config import Config, resolve_daemon_metadata_path
from mcp_memory.daemon import DaemonMetadata, DaemonStopResult, ensure_daemon_started, read_daemon_metadata, stop_daemon
from mcp_memory.daemon_process import is_daemon_healthy


pytestmark = pytest.mark.medium


@dataclass(frozen=True)
class _Spec:
    memory_path: Path
    config: Config
    workspace_id: str
    workspace_root: Path
    lock_path: Path


def test_ensure_daemon_started_reuses_healthy_metadata(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    spec = _Spec(
        memory_path=tmp_path / "memories",
        config=Config(),
        workspace_id="workspace-123",
        workspace_root=tmp_path / "workspace",
        lock_path=tmp_path / "workspace.lock",
    )
    metadata = DaemonMetadata(
        host="127.0.0.1",
        port=8123,
        pid=123,
        started_at=1.0,
        status="ready",
    )
    metadata_path = resolve_daemon_metadata_path()
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(__import__("json").dumps(metadata.__dict__), encoding="utf-8")

    monkeypatch.setattr("mcp_memory.daemon.resolve_runtime_spec", lambda workspace_root_override=None, cwd=None: spec)
    monkeypatch.setattr("mcp_memory.daemon._is_daemon_healthy", lambda current: True)
    monkeypatch.setattr("mcp_memory.daemon._spawn_daemon_process", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not spawn")))

    current = ensure_daemon_started()

    assert current.port == 8123
    assert current.daemon_scope == "global"


def test_read_daemon_metadata_returns_none_for_missing_file(tmp_path: Path) -> None:
    assert read_daemon_metadata(tmp_path / "missing.json") is None


def test_stop_daemon_waits_for_process_exit_after_healthcheck_fails(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    spec = _Spec(
        memory_path=tmp_path / "memories",
        config=Config(),
        workspace_id="workspace-stop",
        workspace_root=tmp_path / "workspace",
        lock_path=tmp_path / "workspace.lock",
    )
    metadata = DaemonMetadata(
        host="127.0.0.1",
        port=8124,
        pid=1234,
        started_at=1.0,
        status="ready",
    )
    metadata_path = resolve_daemon_metadata_path()
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(__import__("json").dumps(metadata.__dict__), encoding="utf-8")

    health_states = iter([True, False, False, False])
    process_states = iter([True, True, False])
    sent_signals: list[int] = []

    monkeypatch.setattr("mcp_memory.daemon.resolve_runtime_spec", lambda workspace_root_override=None, cwd=None: spec)
    monkeypatch.setattr("mcp_memory.daemon._is_daemon_healthy", lambda current: next(health_states))
    monkeypatch.setattr("mcp_memory.daemon._is_process_running", lambda pid: next(process_states))
    monkeypatch.setattr("mcp_memory.daemon.os.getpgid", lambda pid: (_ for _ in ()).throw(ProcessLookupError()))
    monkeypatch.setattr("mcp_memory.daemon.os.kill", lambda pid, sig: sent_signals.append(sig))
    monotonic_values = iter([0.0, 0.2, 0.4, 0.6, 0.8])
    monkeypatch.setattr("mcp_memory.daemon.time.sleep", lambda _: None)
    monkeypatch.setattr("mcp_memory.daemon.time.monotonic", lambda: next(monotonic_values))

    stopped = stop_daemon()

    assert stopped is not None
    assert stopped.pid == 1234
    assert isinstance(stopped, DaemonStopResult)
    assert stopped.signal_sequence == ("SIGTERM",)
    assert stopped.escalated_to_sigkill is False
    assert sent_signals == [15]
    assert not metadata_path.exists()


def test_stop_daemon_terminates_unhealthy_running_process_before_removing_metadata(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    spec = _Spec(
        memory_path=tmp_path / "memories",
        config=Config(),
        workspace_id="workspace-stop",
        workspace_root=tmp_path / "workspace",
        lock_path=tmp_path / "workspace.lock",
    )
    metadata = DaemonMetadata(
        host="127.0.0.1",
        port=8124,
        pid=2345,
        started_at=1.0,
        status="ready",
        transport="zmq",
        socket_path=str(tmp_path / "daemon.sock"),
    )
    metadata_path = resolve_daemon_metadata_path()
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(__import__("json").dumps(metadata.__dict__), encoding="utf-8")
    (tmp_path / "daemon.sock").write_text("stale", encoding="utf-8")

    sent_signals: list[int] = []
    process_states = iter([True, False])
    removed_sockets: list[Path] = []

    monkeypatch.setattr("mcp_memory.daemon.resolve_runtime_spec", lambda workspace_root_override=None, cwd=None: spec)
    monkeypatch.setattr("mcp_memory.daemon._is_daemon_healthy", lambda current: False)
    monkeypatch.setattr("mcp_memory.daemon._is_process_running", lambda pid: next(process_states))
    monkeypatch.setattr("mcp_memory.daemon.os.getpgid", lambda pid: (_ for _ in ()).throw(ProcessLookupError()))
    monkeypatch.setattr("mcp_memory.daemon.os.kill", lambda pid, sig: sent_signals.append(sig))
    monkeypatch.setattr("mcp_memory.daemon._probe_daemon_socket", lambda socket_path, timeout_seconds: False)
    monkeypatch.setattr("mcp_memory.daemon._remove_daemon_socket", lambda socket_path: removed_sockets.append(Path(socket_path)))
    monotonic_values = iter([0.0, 0.2, 0.4, 0.6, 0.8])
    monkeypatch.setattr("mcp_memory.daemon.time.sleep", lambda _: None)
    monkeypatch.setattr("mcp_memory.daemon.time.monotonic", lambda: next(monotonic_values))

    stopped = stop_daemon()

    assert stopped is not None
    assert stopped.pid == 2345
    assert isinstance(stopped, DaemonStopResult)
    assert stopped.stop_reason == "owner_stopped_on_stop"
    assert stopped.stale_socket_removed is True
    assert sent_signals == [15]
    assert removed_sockets == [tmp_path / "daemon.sock"]
    assert not metadata_path.exists()


def test_stop_daemon_reports_sigkill_escalation(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    config = Config()
    config.daemon.shutdown_grace_seconds = 0.1
    spec = _Spec(
        memory_path=tmp_path / "memories",
        config=config,
        workspace_id="workspace-stop",
        workspace_root=tmp_path / "workspace",
        lock_path=tmp_path / "workspace.lock",
    )
    metadata = DaemonMetadata(
        host="127.0.0.1",
        port=8124,
        pid=3456,
        started_at=1.0,
        status="ready",
    )
    metadata_path = resolve_daemon_metadata_path()
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(__import__("json").dumps(metadata.__dict__), encoding="utf-8")

    sent_signals: list[int] = []
    process_states = iter([True, True, True, False])

    monkeypatch.setattr("mcp_memory.daemon.resolve_runtime_spec", lambda workspace_root_override=None, cwd=None: spec)
    monkeypatch.setattr("mcp_memory.daemon._is_process_running", lambda pid: next(process_states))
    monkeypatch.setattr("mcp_memory.daemon.os.getpgid", lambda pid: (_ for _ in ()).throw(ProcessLookupError()))
    monkeypatch.setattr("mcp_memory.daemon.os.kill", lambda pid, sig: sent_signals.append(sig))
    wait_calls = iter([RuntimeError("timeout"), None])

    def _fake_wait(pid: int, deadline: float, poll_interval_seconds: float) -> None:
        result = next(wait_calls)
        if isinstance(result, Exception):
            raise result

    monkeypatch.setattr(
        "mcp_memory.daemon._wait_for_process_exit",
        _fake_wait,
    )

    stopped = stop_daemon()

    assert stopped is not None
    assert stopped.signal_sequence == ("SIGTERM", "SIGKILL")
    assert stopped.escalated_to_sigkill is True
    assert sent_signals == [15, 9]


def test_ensure_daemon_started_stops_unhealthy_running_process_before_spawn(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    spec = _Spec(
        memory_path=tmp_path / "memories",
        config=Config(),
        workspace_id="workspace-start",
        workspace_root=tmp_path / "workspace",
        lock_path=tmp_path / "workspace.lock",
    )
    metadata = DaemonMetadata(
        host="127.0.0.1",
        port=8125,
        pid=5678,
        started_at=1.0,
        status="ready",
    )
    fresh_metadata = DaemonMetadata(
        host="127.0.0.1",
        port=9001,
        pid=6789,
        started_at=2.0,
        status="ready",
    )
    metadata_path = resolve_daemon_metadata_path()
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(__import__("json").dumps(metadata.__dict__), encoding="utf-8")

    spawned: list[tuple[Path, str, int]] = []
    sent_signals: list[int] = []
    process_states = iter([True, False])

    monkeypatch.setattr("mcp_memory.daemon.resolve_runtime_spec", lambda workspace_root_override=None, cwd=None: spec)
    monkeypatch.setattr(
        "mcp_memory.daemon._read_daemon_metadata",
        lambda path: metadata if not spawned and metadata_path.exists() else fresh_metadata,
    )
    monkeypatch.setattr("mcp_memory.daemon._is_daemon_healthy", lambda current: current.pid == fresh_metadata.pid)
    monkeypatch.setattr("mcp_memory.daemon._is_process_running", lambda pid: next(process_states))
    monkeypatch.setattr("mcp_memory.daemon.os.getpgid", lambda pid: (_ for _ in ()).throw(ProcessLookupError()))
    monkeypatch.setattr("mcp_memory.daemon.os.kill", lambda pid, sig: sent_signals.append(sig))
    monkeypatch.setattr("mcp_memory.daemon._spawn_daemon_process", lambda workspace_root, host, port: spawned.append((workspace_root, host, port)))
    monkeypatch.setattr("mcp_memory.daemon._find_free_port", lambda: 9001)
    monotonic_values = iter([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    monkeypatch.setattr("mcp_memory.daemon.time.sleep", lambda _: None)
    monkeypatch.setattr("mcp_memory.daemon.time.monotonic", lambda: next(monotonic_values))

    current = ensure_daemon_started()

    assert current.daemon_scope == "global"
    assert sent_signals == [15]
    assert spawned == [(spec.workspace_root, spec.config.daemon.host, 9001)]


def test_ensure_daemon_started_terminates_orphaned_daemon_processes_when_metadata_missing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    spec = _Spec(
        memory_path=tmp_path / "memories",
        config=Config(),
        workspace_id="workspace-start",
        workspace_root=tmp_path / "workspace",
        lock_path=tmp_path / "workspace.lock",
    )
    metadata = DaemonMetadata(
        host="127.0.0.1",
        port=8126,
        pid=7777,
        started_at=1.0,
        status="ready",
    )
    spawned: list[tuple[Path, str, int]] = []
    killed: list[int] = []
    process_states = iter([False, False, False])
    monotonic_values = iter([0.0, 0.2, 0.4, 0.6, 0.8])

    monkeypatch.setattr("mcp_memory.daemon.resolve_runtime_spec", lambda workspace_root_override=None, cwd=None: spec)
    monkeypatch.setattr("mcp_memory.daemon._read_daemon_metadata", lambda path: None if not spawned else metadata)
    monkeypatch.setattr(
        "mcp_memory.daemon._list_daemon_processes",
        lambda current_pid: [
            __import__("mcp_memory.daemon").daemon._DaemonProcess(
                pid=4321,
                executable="/old/tool/python",
                command=("python", "-m", "mcp_memory.cli", "daemon"),
            )
        ],
    )
    monkeypatch.setattr("mcp_memory.daemon._is_process_running", lambda pid: next(process_states))
    monkeypatch.setattr("mcp_memory.daemon._is_daemon_healthy", lambda current: True)
    monkeypatch.setattr("mcp_memory.daemon.os.getpgid", lambda pid: (_ for _ in ()).throw(ProcessLookupError()))
    monkeypatch.setattr("mcp_memory.daemon.os.kill", lambda pid, sig: killed.append(pid))
    monkeypatch.setattr("mcp_memory.daemon._spawn_daemon_process", lambda workspace_root, host, port: spawned.append((workspace_root, host, port)))
    monkeypatch.setattr("mcp_memory.daemon._find_free_port", lambda: 9002)
    monkeypatch.setattr("mcp_memory.daemon.time.sleep", lambda _: None)
    monkeypatch.setattr("mcp_memory.daemon.time.monotonic", lambda: next(monotonic_values))

    current = ensure_daemon_started()

    assert current.port == 8126
    assert killed == [4321]
    assert spawned == [(spec.workspace_root, spec.config.daemon.host, 9002)]


def test_ensure_daemon_started_removes_stale_metadata_and_terminates_orphans(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    spec = _Spec(
        memory_path=tmp_path / "memories",
        config=Config(),
        workspace_id="workspace-start",
        workspace_root=tmp_path / "workspace",
        lock_path=tmp_path / "workspace.lock",
    )
    stale_metadata = DaemonMetadata(
        host="127.0.0.1",
        port=8127,
        pid=1111,
        started_at=1.0,
        status="ready",
        transport="http",
    )
    fresh_metadata = DaemonMetadata(
        host="127.0.0.1",
        port=8128,
        pid=2222,
        started_at=2.0,
        status="ready",
        transport="zmq",
        socket_path=str(tmp_path / "daemon.sock"),
    )
    metadata_path = resolve_daemon_metadata_path()
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(__import__("json").dumps(stale_metadata.__dict__), encoding="utf-8")

    spawned: list[tuple[Path, str, int]] = []
    killed: list[int] = []
    process_states = iter([False, False, False])
    monotonic_values = iter([0.0, 0.2, 0.4, 0.6, 0.8])

    monkeypatch.setattr("mcp_memory.daemon.resolve_runtime_spec", lambda workspace_root_override=None, cwd=None: spec)
    monkeypatch.setattr(
        "mcp_memory.daemon._read_daemon_metadata",
        lambda path: stale_metadata if not spawned and metadata_path.exists() else fresh_metadata,
    )
    monkeypatch.setattr("mcp_memory.daemon._is_daemon_healthy", lambda current: current.pid == fresh_metadata.pid)
    monkeypatch.setattr("mcp_memory.daemon._is_process_running", lambda pid: next(process_states))
    monkeypatch.setattr(
        "mcp_memory.daemon._list_daemon_processes",
        lambda current_pid: [
            __import__("mcp_memory.daemon").daemon._DaemonProcess(
                pid=3333,
                executable="/old/tool/python",
                command=("python", "-m", "mcp_memory.cli", "daemon"),
            )
        ],
    )
    monkeypatch.setattr("mcp_memory.daemon.os.getpgid", lambda pid: (_ for _ in ()).throw(ProcessLookupError()))
    monkeypatch.setattr("mcp_memory.daemon.os.kill", lambda pid, sig: killed.append(pid))
    monkeypatch.setattr("mcp_memory.daemon._spawn_daemon_process", lambda workspace_root, host, port: spawned.append((workspace_root, host, port)))
    monkeypatch.setattr("mcp_memory.daemon._find_free_port", lambda: 9003)
    monkeypatch.setattr("mcp_memory.daemon.time.sleep", lambda _: None)
    monkeypatch.setattr("mcp_memory.daemon.time.monotonic", lambda: next(monotonic_values))

    current = ensure_daemon_started()

    assert current.port == 8128
    assert killed == [3333]
    assert spawned == [(spec.workspace_root, spec.config.daemon.host, 9003)]


def test_is_daemon_healthy_requires_socket_path_for_zmq(monkeypatch) -> None:
    metadata = DaemonMetadata(
        host="127.0.0.1",
        port=8129,
        pid=9876,
        started_at=1.0,
        status="ready",
        transport="zmq",
        socket_path=None,
    )

    monkeypatch.setattr(
        "mcp_memory.daemon_process.request_daemon_json",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("request should not be attempted")),
    )

    assert is_daemon_healthy(metadata) is False


def test_ensure_daemon_started_cleans_stale_socket_before_spawn(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    spec = _Spec(
        memory_path=tmp_path / "memories",
        config=Config(),
        workspace_id="workspace-start",
        workspace_root=tmp_path / "workspace",
        lock_path=tmp_path / "workspace.lock",
    )
    stale_socket = tmp_path / "daemon.sock"
    stale_socket.write_text("stale", encoding="utf-8")
    stale_metadata = DaemonMetadata(
        host="127.0.0.1",
        port=8130,
        pid=1111,
        started_at=1.0,
        status="ready",
        transport="zmq",
        socket_path=str(stale_socket),
    )
    fresh_metadata = DaemonMetadata(
        host="127.0.0.1",
        port=8131,
        pid=2222,
        started_at=2.0,
        status="ready",
        transport="zmq",
        socket_path=str(stale_socket),
    )

    spawned: list[tuple[Path, str, int]] = []
    removed_sockets: list[Path] = []
    monotonic_values = iter([0.0, 0.2, 0.4, 0.6, 0.8])

    monkeypatch.setattr("mcp_memory.daemon.resolve_runtime_spec", lambda workspace_root_override=None, cwd=None: spec)
    monkeypatch.setattr(
        "mcp_memory.daemon._read_daemon_metadata",
        lambda path: stale_metadata if not spawned else fresh_metadata,
    )
    monkeypatch.setattr("mcp_memory.daemon._is_daemon_healthy", lambda current: current.pid == fresh_metadata.pid)
    monkeypatch.setattr("mcp_memory.daemon._is_process_running", lambda pid: False)
    monkeypatch.setattr("mcp_memory.daemon._terminate_orphaned_daemon_processes", lambda **kwargs: None)
    monkeypatch.setattr("mcp_memory.daemon._probe_daemon_socket", lambda socket_path, timeout_seconds: False)
    monkeypatch.setattr("mcp_memory.daemon._remove_daemon_socket", lambda socket_path: removed_sockets.append(Path(socket_path)))
    monkeypatch.setattr("mcp_memory.daemon._spawn_daemon_process", lambda workspace_root, host, port: spawned.append((workspace_root, host, port)))
    monkeypatch.setattr("mcp_memory.daemon._find_free_port", lambda: 9004)
    monkeypatch.setattr("mcp_memory.daemon.time.sleep", lambda _: None)
    monkeypatch.setattr("mcp_memory.daemon.time.monotonic", lambda: next(monotonic_values))

    current = ensure_daemon_started()

    assert current.port == 8131
    assert removed_sockets == [stale_socket]
    assert spawned == [(spec.workspace_root, spec.config.daemon.host, 9004)]
