from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from mcp_memory.config import Config, resolve_daemon_metadata_path
from mcp_memory.daemon import DaemonMetadata, ensure_daemon_started, read_daemon_metadata, stop_daemon


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
        workspace_id="global",
        workspace_root=str(spec.workspace_root),
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
    assert current.workspace_id == "global"


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
        workspace_id="global",
        workspace_root=str(spec.workspace_root),
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
    monkeypatch.setattr("mcp_memory.daemon.os.kill", lambda pid, sig: sent_signals.append(sig))
    monotonic_values = iter([0.0, 0.2, 0.4, 0.6, 0.8])
    monkeypatch.setattr("mcp_memory.daemon.time.sleep", lambda _: None)
    monkeypatch.setattr("mcp_memory.daemon.time.monotonic", lambda: next(monotonic_values))

    stopped = stop_daemon()

    assert stopped is not None
    assert stopped.pid == 1234
    assert sent_signals == [15]
    assert not metadata_path.exists()


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
        workspace_id="global",
        workspace_root=str(spec.workspace_root),
        host="127.0.0.1",
        port=8125,
        pid=5678,
        started_at=1.0,
        status="ready",
    )
    metadata_path = resolve_daemon_metadata_path()
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(__import__("json").dumps(metadata.__dict__), encoding="utf-8")

    spawned: list[tuple[Path, str, int]] = []
    sent_signals: list[int] = []
    process_states = iter([True, False])
    healthy_states = iter([False, True])

    monkeypatch.setattr("mcp_memory.daemon.resolve_runtime_spec", lambda workspace_root_override=None, cwd=None: spec)
    monkeypatch.setattr("mcp_memory.daemon._is_daemon_healthy", lambda current: next(healthy_states))
    monkeypatch.setattr("mcp_memory.daemon._is_process_running", lambda pid: next(process_states))
    monkeypatch.setattr("mcp_memory.daemon.os.kill", lambda pid, sig: sent_signals.append(sig))
    monkeypatch.setattr("mcp_memory.daemon._spawn_daemon_process", lambda workspace_root, host, port: spawned.append((workspace_root, host, port)))
    monkeypatch.setattr("mcp_memory.daemon._find_free_port", lambda: 9001)
    monotonic_values = iter([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    monkeypatch.setattr("mcp_memory.daemon.time.sleep", lambda _: None)
    monkeypatch.setattr("mcp_memory.daemon.time.monotonic", lambda: next(monotonic_values))

    current = ensure_daemon_started()

    assert current.workspace_id == "global"
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
        workspace_id="global",
        workspace_root=str(spec.workspace_root),
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
        workspace_id="global",
        workspace_root=str(spec.workspace_root),
        host="127.0.0.1",
        port=8127,
        pid=1111,
        started_at=1.0,
        status="ready",
    )
    fresh_metadata = DaemonMetadata(
        workspace_id="global",
        workspace_root=str(spec.workspace_root),
        host="127.0.0.1",
        port=8128,
        pid=2222,
        started_at=2.0,
        status="ready",
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
    monkeypatch.setattr("mcp_memory.daemon.os.kill", lambda pid, sig: killed.append(pid))
    monkeypatch.setattr("mcp_memory.daemon._spawn_daemon_process", lambda workspace_root, host, port: spawned.append((workspace_root, host, port)))
    monkeypatch.setattr("mcp_memory.daemon._find_free_port", lambda: 9003)
    monkeypatch.setattr("mcp_memory.daemon.time.sleep", lambda _: None)
    monkeypatch.setattr("mcp_memory.daemon.time.monotonic", lambda: next(monotonic_values))

    current = ensure_daemon_started()

    assert current.port == 8128
    assert killed == [3333]
    assert spawned == [(spec.workspace_root, spec.config.daemon.host, 9003)]
