from __future__ import annotations

import asyncio
import signal
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from click.testing import CliRunner

import mcp_memory.daemon as daemon_module
import mcp_memory.daemon_process as daemon_process_module
from mcp_memory.cli import main
from mcp_memory.config import Config, resolve_daemon_metadata_path
from mcp_memory.context import ApplicationContext
from mcp_memory.daemon import (
    DaemonMetadata,
    DaemonStopResult,
    ensure_daemon_started,
    inspect_daemon,
    read_daemon_metadata,
    stop_daemon,
)
from mcp_memory.daemon_ports import DaemonRoutesProvider
from mcp_memory.daemon_process import (
    DaemonHealthAssessment,
    DaemonSpawnDetails,
    assess_daemon_health,
    is_daemon_healthy,
    remove_metadata,
    spawn_daemon_process,
)
from mcp_memory.daemon_transport import DaemonZmqServer, request_daemon_json

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
    monkeypatch.setattr("mcp_memory.daemon.resolve_global_daemon_bootstrap_spec", lambda workspace_root_override=None, cwd=None: spec)
    assessments = iter(
        [
            DaemonHealthAssessment(False, "probe_timeout", True, 0.1),
            DaemonHealthAssessment(True, "healthy", False, 0.1),
        ]
    )
    observed_probes: list[tuple[int, float]] = []
    monkeypatch.setattr(
        "mcp_memory.daemon._assess_daemon_health",
        lambda current, timeout_seconds=3.0: observed_probes.append((current.pid, timeout_seconds)) or next(assessments),
    )
    monkeypatch.setattr("mcp_memory.daemon.time.sleep", lambda _: None)
    monkeypatch.setattr("mcp_memory.daemon._spawn_daemon_process", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not spawn")))

    current = ensure_daemon_started()

    assert current.port == 8123
    assert current.daemon_scope == "global"
    assert observed_probes == [
        (metadata.pid, pytest.approx(spec.config.daemon.auto_start_timeout_seconds / 3.0)),
        (metadata.pid, pytest.approx(spec.config.daemon.auto_start_timeout_seconds / 3.0)),
    ]


def test_ensure_daemon_started_uses_configured_auto_start_timeout(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    config = Config()
    config.daemon.port = 4242
    config.daemon.auto_start_timeout_seconds = 0.25
    spec = _Spec(
        memory_path=tmp_path / "memories",
        config=config,
        workspace_id="workspace-timeout",
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
    acquired_timeouts: list[float] = []
    released: list[bool] = []

    class _FakeLock:
        def __init__(self, path: Path) -> None:
            self.path = path

        def acquire(self, *, timeout_seconds: float) -> None:
            acquired_timeouts.append(timeout_seconds)

        def release(self) -> None:
            released.append(True)

    monkeypatch.setattr("mcp_memory.daemon.resolve_global_daemon_bootstrap_spec", lambda workspace_root_override=None, cwd=None: spec)
    monkeypatch.setattr("mcp_memory.daemon.FilesystemLock", _FakeLock)
    monkeypatch.setattr("mcp_memory.daemon._read_daemon_metadata", lambda path: metadata)
    monkeypatch.setattr(
        "mcp_memory.daemon._assess_daemon_health",
        lambda current, timeout_seconds=3.0: DaemonHealthAssessment(
            healthy=True,
            reason="healthy",
            retryable=False,
            timeout_seconds=timeout_seconds,
        ),
    )

    current = ensure_daemon_started()

    assert current.pid == metadata.pid
    assert acquired_timeouts == [pytest.approx(25.25)]
    assert released == [True]


def test_read_daemon_metadata_returns_none_for_missing_file(tmp_path: Path) -> None:
    assert read_daemon_metadata(tmp_path / "missing.json") is None


def test_remove_metadata_preserves_newer_daemon_registration(tmp_path: Path) -> None:
    metadata_path = tmp_path / "daemon-metadata.json"
    metadata_path.write_text(
        __import__("json").dumps(
            DaemonMetadata(
                host="127.0.0.1",
                port=8123,
                pid=222,
                started_at=2.0,
                status="ready",
            ).__dict__
        ),
        encoding="utf-8",
    )

    removed = remove_metadata(metadata_path, expected_pid=111)

    assert removed is False
    assert metadata_path.exists()
    preserved = read_daemon_metadata(metadata_path)
    assert preserved is not None
    assert preserved.pid == 222


def test_is_process_running_treats_zombies_as_stopped(monkeypatch) -> None:
    monkeypatch.setattr('mcp_memory.daemon._read_process_state', lambda pid: 'Z')
    monkeypatch.setattr('mcp_memory.daemon.os.kill', lambda pid, sig: (_ for _ in ()).throw(AssertionError('os.kill should not be called for zombies')))

    assert __import__('mcp_memory.daemon').daemon._is_process_running(1234) is False


def test_spawn_daemon_process_uses_global_bootstrap(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    captured: dict[str, object] = {}

    def _fake_popen(command, **kwargs):
        captured['command'] = command
        captured.update(kwargs)
        class _DummyProcess:
            pid = 4321
        return _DummyProcess()

    monkeypatch.setattr('mcp_memory.daemon_process.subprocess.Popen', _fake_popen)

    details = spawn_daemon_process('127.0.0.1', 8123)

    assert 'cwd' not in captured
    assert captured['stdin'] is not None
    assert captured['start_new_session'] is True
    assert details.pid == 4321
    assert details.command[0] == sys.executable
    assert details.command[1:4] == ('-m', 'mcp_memory.cli', 'daemon')
    assert details.startup_log_path.name == 'daemon.log'


def test_spawn_daemon_process_rotates_oversized_startup_log(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setattr(daemon_process_module, "DAEMON_STARTUP_LOG_MAX_BYTES", 8)
    monkeypatch.setattr(daemon_process_module, "DAEMON_STARTUP_LOG_BACKUP_COUNT", 2)

    def _fake_popen(command, **kwargs):
        class _DummyProcess:
            pid = 4321

        return _DummyProcess()

    monkeypatch.setattr(daemon_process_module.subprocess, "Popen", _fake_popen)
    log_path = tmp_path / "state" / "mcp-memory" / "daemons" / "daemon.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("x" * 8, encoding="utf-8")
    log_path.with_name("daemon.log.1").write_text("old", encoding="utf-8")
    log_path.with_name("daemon.log.2").write_text("older", encoding="utf-8")

    spawn_daemon_process("127.0.0.1", 8123)

    assert log_path.with_name("daemon.log.1").read_text(encoding="utf-8") == "x" * 8
    assert log_path.with_name("daemon.log.2").read_text(encoding="utf-8") == "old"
    assert "daemon spawn" in log_path.read_text(encoding="utf-8")


def test_ensure_daemon_started_includes_startup_log_tail_when_spawned_child_exits(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv('XDG_STATE_HOME', str(tmp_path / 'state'))
    config = Config()
    config.daemon.port = 4242
    config.daemon.auto_start_timeout_seconds = 0.2
    spec = _Spec(
        memory_path=tmp_path / 'memories',
        config=config,
        workspace_id='workspace-start',
        workspace_root=tmp_path / 'workspace',
        lock_path=tmp_path / 'workspace.lock',
    )
    startup_log_path = tmp_path / 'state' / 'mcp-memory' / 'daemons' / 'daemon.log'
    startup_log_path.parent.mkdir(parents=True, exist_ok=True)
    startup_log_path.write_text('booting\nTraceback: bind failed\n', encoding='utf-8')

    class _ExitedProcess:
        def poll(self) -> int:
            return 17

    monkeypatch.setattr('mcp_memory.daemon.resolve_global_daemon_bootstrap_spec', lambda workspace_root_override=None, cwd=None: spec)
    monkeypatch.setattr('mcp_memory.daemon._read_daemon_metadata', lambda path: None)
    monkeypatch.setattr('mcp_memory.daemon._terminate_orphaned_daemon_processes', lambda **kwargs: None)
    monkeypatch.setattr(
        'mcp_memory.daemon._spawn_daemon_process',
        lambda host, port: DaemonSpawnDetails(
            pid=4444,
            command=('python', '-m', 'mcp_memory.cli', 'daemon'),
            startup_log_path=startup_log_path,
            process=_ExitedProcess(),
        ),
    )

    with pytest.raises(RuntimeError, match='Startup log tail') as exc_info:
        ensure_daemon_started()

    message = str(exc_info.value)
    assert 'exit_code=17' in message
    assert f'startup_log={startup_log_path}' in message
    assert 'Traceback: bind failed' in message


def test_cli_dashboard_surfaces_startup_failure_diagnostics(monkeypatch) -> None:
    runner = CliRunner()

    monkeypatch.setattr(
        'mcp_memory.cli.ensure_daemon_started',
        lambda: (_ for _ in ()).throw(
            RuntimeError(
                'Timed out waiting for global daemon startup. '
                'startup_log=/tmp/daemon.log metadata=missing\n'
                'Startup log tail (/tmp/daemon.log):\n'
                'Traceback: bind failed'
            )
        ),
    )

    result = runner.invoke(main, ['admin', 'dashboard', 'open'])

    assert result.exit_code == 1
    assert 'Error:' in result.output
    assert 'Timed out waiting for global daemon startup.' in result.output
    assert 'Startup log tail (/tmp/daemon.log):' in result.output
    assert 'Traceback: bind failed' in result.output


def test_ensure_daemon_started_includes_startup_log_tail_on_timeout(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv('XDG_STATE_HOME', str(tmp_path / 'state'))
    config = Config()
    config.daemon.port = 4242
    config.daemon.auto_start_timeout_seconds = 0.2
    config.daemon.healthcheck_interval_seconds = 0.05
    spec = _Spec(
        memory_path=tmp_path / 'memories',
        config=config,
        workspace_id='workspace-start',
        workspace_root=tmp_path / 'workspace',
        lock_path=tmp_path / 'workspace.lock',
    )
    startup_log_path = tmp_path / 'state' / 'mcp-memory' / 'daemons' / 'daemon.log'
    startup_log_path.parent.mkdir(parents=True, exist_ok=True)
    startup_log_path.write_text('still starting\nlast warning\n', encoding='utf-8')
    metadata = DaemonMetadata(
        host='127.0.0.1',
        port=9007,
        pid=5555,
        started_at=1.0,
        status='starting',
    )

    class _RunningProcess:
        def poll(self) -> None:
            return None

    monotonic_state = {'value': -0.05}

    def _fake_monotonic() -> float:
        monotonic_state['value'] += 0.05
        return monotonic_state['value']

    monkeypatch.setattr('mcp_memory.daemon.resolve_global_daemon_bootstrap_spec', lambda workspace_root_override=None, cwd=None: spec)
    monkeypatch.setattr('mcp_memory.daemon._terminate_orphaned_daemon_processes', lambda **kwargs: None)
    monkeypatch.setattr(
        'mcp_memory.daemon._spawn_daemon_process',
        lambda host, port: DaemonSpawnDetails(
            pid=5555,
            command=('python', '-m', 'mcp_memory.cli', 'daemon'),
            startup_log_path=startup_log_path,
            process=_RunningProcess(),
        ),
    )
    read_count = {'count': 0}

    def _fake_read(_path):
        read_count['count'] += 1
        return None if read_count['count'] == 1 else metadata

    monkeypatch.setattr('mcp_memory.daemon._read_daemon_metadata', _fake_read)
    monkeypatch.setattr('mcp_memory.daemon._is_daemon_healthy', lambda current: False)
    monkeypatch.setattr('mcp_memory.daemon._is_process_running', lambda pid: True)
    monkeypatch.setattr('mcp_memory.daemon.time.sleep', lambda _: None)
    monkeypatch.setattr('mcp_memory.daemon.time.monotonic', _fake_monotonic)

    with pytest.raises(RuntimeError, match='Timed out waiting for global daemon startup') as exc_info:
        ensure_daemon_started()

    message = str(exc_info.value)
    assert 'metadata_status=starting' in message
    assert 'metadata_endpoint=http://127.0.0.1:9007' in message
    assert 'last warning' in message


def test_ensure_daemon_started_kills_spawned_process_on_timeout(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv('XDG_STATE_HOME', str(tmp_path / 'state'))
    config = Config()
    config.daemon.port = 4242
    config.daemon.auto_start_timeout_seconds = 0.2
    config.daemon.healthcheck_interval_seconds = 0.05
    config.daemon.shutdown_grace_seconds = 0.2
    spec = _Spec(
        memory_path=tmp_path / 'memories',
        config=config,
        workspace_id='workspace-start',
        workspace_root=tmp_path / 'workspace',
        lock_path=tmp_path / 'workspace.lock',
    )
    startup_log_path = tmp_path / 'state' / 'mcp-memory' / 'daemons' / 'daemon.log'
    startup_log_path.parent.mkdir(parents=True, exist_ok=True)
    startup_log_path.write_text('still starting\n', encoding='utf-8')

    class _StuckProcess:
        def poll(self) -> None:
            return None

    monotonic_state = {'value': -0.05}

    def _fake_monotonic() -> float:
        monotonic_state['value'] += 0.05
        return monotonic_state['value']

    monkeypatch.setattr('mcp_memory.daemon.resolve_global_daemon_bootstrap_spec', lambda workspace_root_override=None, cwd=None: spec)
    monkeypatch.setattr('mcp_memory.daemon._terminate_orphaned_daemon_processes', lambda **kwargs: None)
    monkeypatch.setattr(
        'mcp_memory.daemon._spawn_daemon_process',
        lambda host, port: DaemonSpawnDetails(
            pid=5555,
            command=('python', '-m', 'mcp_memory.cli', 'daemon'),
            startup_log_path=startup_log_path,
            process=_StuckProcess(),
        ),
    )
    monkeypatch.setattr('mcp_memory.daemon._read_daemon_metadata', lambda _path: None)
    monkeypatch.setattr('mcp_memory.daemon._is_daemon_healthy', lambda current: False)
    monkeypatch.setattr('mcp_memory.daemon._is_process_running', lambda pid: True)
    monkeypatch.setattr('mcp_memory.daemon.time.sleep', lambda _: None)
    monkeypatch.setattr('mcp_memory.daemon.time.monotonic', _fake_monotonic)

    terminate_calls: list[int] = []
    monkeypatch.setattr(
        'mcp_memory.daemon._terminate_daemon_process',
        lambda pid, **kwargs: terminate_calls.append(pid),
    )

    with pytest.raises(RuntimeError, match='Timed out waiting for global daemon startup'):
        ensure_daemon_started()

    assert terminate_calls == [5555]


def test_ensure_daemon_started_allows_short_grace_for_late_health(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv('XDG_STATE_HOME', str(tmp_path / 'state'))
    config = Config()
    config.daemon.port = 4242
    config.daemon.auto_start_timeout_seconds = 0.2
    config.daemon.healthcheck_interval_seconds = 0.05
    spec = _Spec(
        memory_path=tmp_path / 'memories',
        config=config,
        workspace_id='workspace-start',
        workspace_root=tmp_path / 'workspace',
        lock_path=tmp_path / 'workspace.lock',
    )
    metadata = DaemonMetadata(
        host='127.0.0.1',
        port=9005,
        pid=2468,
        started_at=2.0,
        status='ready',
        transport='zmq',
        socket_path=str(tmp_path / 'daemon.sock'),
    )

    read_count = {'count': 0}
    health_count = {'count': 0}
    spawned: list[tuple[str, int]] = []
    monotonic_values = iter([value * 0.05 for value in range(80)])

    monkeypatch.setattr('mcp_memory.daemon.resolve_global_daemon_bootstrap_spec', lambda workspace_root_override=None, cwd=None: spec)

    def _fake_read(_path):
        read_count['count'] += 1
        return None if read_count['count'] <= 4 else metadata

    def _fake_health(current):
        health_count['count'] += 1
        return health_count['count'] >= 2 and current.pid == metadata.pid

    monkeypatch.setattr('mcp_memory.daemon._read_daemon_metadata', _fake_read)
    monkeypatch.setattr('mcp_memory.daemon._is_daemon_healthy', _fake_health)
    monkeypatch.setattr('mcp_memory.daemon._is_process_running', lambda pid: True)
    monkeypatch.setattr('mcp_memory.daemon._terminate_orphaned_daemon_processes', lambda **kwargs: None)
    monkeypatch.setattr('mcp_memory.daemon._spawn_daemon_process', lambda host, port: spawned.append((host, port)))
    monkeypatch.setattr('mcp_memory.daemon.time.sleep', lambda _: None)
    monkeypatch.setattr('mcp_memory.daemon.time.monotonic', lambda: next(monotonic_values))

    current = ensure_daemon_started()

    assert current.pid == metadata.pid
    assert spawned == [(spec.config.daemon.host, 4242)]


def test_ensure_daemon_started_gives_readiness_a_fresh_timeout_after_cleanup(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv('XDG_STATE_HOME', str(tmp_path / 'state'))
    config = Config()
    config.daemon.port = 4242
    config.daemon.auto_start_timeout_seconds = 0.2
    config.daemon.healthcheck_interval_seconds = 0.05
    spec = _Spec(
        memory_path=tmp_path / 'memories',
        config=config,
        workspace_id='workspace-start',
        workspace_root=tmp_path / 'workspace',
        lock_path=tmp_path / 'workspace.lock',
    )
    stale_metadata = DaemonMetadata(
        host='127.0.0.1',
        port=8125,
        pid=1111,
        started_at=1.0,
        status='ready',
    )
    ready_metadata = DaemonMetadata(
        host='127.0.0.1',
        port=9008,
        pid=2222,
        started_at=2.0,
        status='ready',
    )

    read_count = {'count': 0}
    acquired_timeouts: list[float] = []
    cleanup_deadlines: list[float] = []
    spawned: list[tuple[str, int]] = []
    monotonic_values = iter([0.0, 0.19, 0.2, 0.25])

    class _FakeLock:
        def __init__(self, path: Path) -> None:
            self.path = path

        def acquire(self, *, timeout_seconds: float) -> None:
            acquired_timeouts.append(timeout_seconds)

        def release(self) -> None:
            pass

    monkeypatch.setattr('mcp_memory.daemon.FilesystemLock', _FakeLock)
    monkeypatch.setattr('mcp_memory.daemon.resolve_global_daemon_bootstrap_spec', lambda workspace_root_override=None, cwd=None: spec)

    def _fake_read(_path):
        read_count['count'] += 1
        if read_count['count'] == 1:
            return stale_metadata
        if read_count['count'] == 2:
            return None
        return ready_metadata

    monkeypatch.setattr('mcp_memory.daemon._read_daemon_metadata', _fake_read)
    monkeypatch.setattr('mcp_memory.daemon._is_daemon_healthy', lambda current: current.pid == ready_metadata.pid)
    monkeypatch.setattr('mcp_memory.daemon._is_process_running', lambda pid: False)
    monkeypatch.setattr('mcp_memory.daemon.remove_metadata', lambda path: None)
    monkeypatch.setattr(
        'mcp_memory.daemon._terminate_orphaned_daemon_processes',
        lambda **kwargs: cleanup_deadlines.append(kwargs['deadline']),
    )
    monkeypatch.setattr('mcp_memory.daemon._cleanup_stale_daemon_socket', lambda *args, **kwargs: False)
    monkeypatch.setattr(
        'mcp_memory.daemon._spawn_daemon_process',
        lambda host, port: spawned.append((host, port)),
    )
    monkeypatch.setattr('mcp_memory.daemon.time.sleep', lambda _: None)
    monkeypatch.setattr('mcp_memory.daemon.time.monotonic', lambda: next(monotonic_values))

    current = ensure_daemon_started()

    assert current.pid == ready_metadata.pid
    assert acquired_timeouts == [pytest.approx(25.2)]
    assert cleanup_deadlines == [0.2]
    assert spawned == [(spec.config.daemon.host, 4242)]


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

    monkeypatch.setattr("mcp_memory.daemon.resolve_global_daemon_bootstrap_spec", lambda workspace_root_override=None, cwd=None: spec)
    monkeypatch.setattr("mcp_memory.daemon._is_daemon_healthy", lambda current: next(health_states))
    monkeypatch.setattr("mcp_memory.daemon._is_process_running", lambda pid: next(process_states))
    monkeypatch.setattr("mcp_memory.daemon.os.getpgid", lambda pid: (_ for _ in ()).throw(ProcessLookupError()))
    monkeypatch.setattr("mcp_memory.daemon.os.kill", lambda pid, sig: sent_signals.append(sig))
    monotonic_values = iter([value * 0.2 for value in range(8)])
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

    monkeypatch.setattr("mcp_memory.daemon.resolve_global_daemon_bootstrap_spec", lambda workspace_root_override=None, cwd=None: spec)
    monkeypatch.setattr("mcp_memory.daemon._is_daemon_healthy", lambda current: False)
    monkeypatch.setattr("mcp_memory.daemon._is_process_running", lambda pid: next(process_states))
    monkeypatch.setattr("mcp_memory.daemon.os.getpgid", lambda pid: (_ for _ in ()).throw(ProcessLookupError()))
    monkeypatch.setattr("mcp_memory.daemon.os.kill", lambda pid, sig: sent_signals.append(sig))
    monkeypatch.setattr("mcp_memory.daemon._probe_daemon_socket", lambda socket_path, timeout_seconds: False)
    monkeypatch.setattr("mcp_memory.daemon._remove_daemon_socket", lambda socket_path: removed_sockets.append(Path(socket_path)))
    monotonic_values = iter([value * 0.2 for value in range(8)])
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

    monkeypatch.setattr("mcp_memory.daemon.resolve_global_daemon_bootstrap_spec", lambda workspace_root_override=None, cwd=None: spec)
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


def test_signal_daemon_process_uses_pid_scope_for_sigterm(monkeypatch) -> None:
    sent_pid_signals: list[tuple[int, signal.Signals]] = []

    monkeypatch.setattr(
        "mcp_memory.daemon.os.killpg",
        lambda pgid, sig: (_ for _ in ()).throw(AssertionError("process-group kill should not be used for graceful SIGTERM")),
    )
    monkeypatch.setattr("mcp_memory.daemon.os.kill", lambda pid, sig: sent_pid_signals.append((pid, sig)))

    process_group_id = daemon_module._signal_daemon_process(3456, signal.SIGTERM)

    assert process_group_id is None
    assert sent_pid_signals == [(3456, signal.SIGTERM)]


def test_stop_daemon_escalates_to_process_group_for_sigkill(monkeypatch, tmp_path: Path) -> None:
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

    sent_signals: list[tuple[str, int, signal.Signals]] = []
    process_states = iter([True, True, True, False])

    monkeypatch.setattr("mcp_memory.daemon.resolve_global_daemon_bootstrap_spec", lambda workspace_root_override=None, cwd=None: spec)
    monkeypatch.setattr("mcp_memory.daemon._is_process_running", lambda pid: next(process_states))
    monkeypatch.setattr("mcp_memory.daemon.os.getpgid", lambda pid: 777)
    monkeypatch.setattr("mcp_memory.daemon.os.kill", lambda pid, sig: sent_signals.append(("pid", pid, sig)))
    monkeypatch.setattr("mcp_memory.daemon.os.killpg", lambda pgid, sig: sent_signals.append(("pg", pgid, sig)))
    wait_calls = iter([RuntimeError("timeout"), None])

    def _fake_wait(pid: int, deadline: float, poll_interval_seconds: float) -> None:
        result = next(wait_calls)
        if isinstance(result, Exception):
            raise result

    monkeypatch.setattr("mcp_memory.daemon._wait_for_process_exit", _fake_wait)

    stopped = stop_daemon()

    assert stopped is not None
    assert stopped.signal_sequence == ("SIGTERM", "SIGKILL")
    assert stopped.process_group_id == 777
    assert stopped.escalated_to_sigkill is True
    assert sent_signals == [
        ("pid", 3456, signal.SIGTERM),
        ("pg", 777, signal.SIGKILL),
    ]


def test_terminate_daemon_process_uses_fresh_deadline_after_sigkill(monkeypatch) -> None:
    wait_deadlines: list[float] = []
    sent_signals: list[str] = []
    monotonic_values = iter([10.0, 20.0])

    monkeypatch.setattr("mcp_memory.daemon.time.monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr("mcp_memory.daemon._signal_daemon_process", lambda pid, sig: sent_signals.append(sig.name) or None)
    monkeypatch.setattr("mcp_memory.daemon._is_process_running", lambda pid: True)

    def _fake_wait(pid: int, *, deadline: float, poll_interval_seconds: float) -> None:
        wait_deadlines.append(deadline)
        if len(wait_deadlines) == 1:
            raise RuntimeError("timeout")

    monkeypatch.setattr("mcp_memory.daemon._wait_for_process_exit", _fake_wait)

    termination = daemon_module._terminate_daemon_process(
        3456,
        deadline=15.0,
        poll_interval_seconds=0.5,
    )

    assert wait_deadlines == [15.0, 25.0]
    assert sent_signals == ["SIGTERM", "SIGKILL"]
    assert termination.signal_sequence == ("SIGTERM", "SIGKILL")
    assert termination.escalated_to_sigkill is True


def test_ensure_daemon_started_stops_unhealthy_running_process_before_spawn(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    config = Config()
    config.daemon.port = 4242
    spec = _Spec(
        memory_path=tmp_path / "memories",
        config=config,
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

    spawned: list[tuple[str, int]] = []
    sent_signals: list[int] = []
    process_states = iter([True, False])

    monkeypatch.setattr("mcp_memory.daemon.resolve_global_daemon_bootstrap_spec", lambda workspace_root_override=None, cwd=None: spec)
    monkeypatch.setattr(
        "mcp_memory.daemon._read_daemon_metadata",
        lambda path: metadata if not spawned and metadata_path.exists() else fresh_metadata,
    )
    monkeypatch.setattr(
        "mcp_memory.daemon._assess_daemon_health",
        lambda current, timeout_seconds=3.0: DaemonHealthAssessment(
            healthy=False,
            reason="health_probe_timeout",
            retryable=False,
            timeout_seconds=timeout_seconds,
            error_type="TimeoutError",
            error_text="daemon_request_timed_out",
        ),
    )
    monkeypatch.setattr("mcp_memory.daemon._is_daemon_healthy", lambda current: current.pid == fresh_metadata.pid)
    monkeypatch.setattr("mcp_memory.daemon._is_process_running", lambda pid: next(process_states))
    monkeypatch.setattr("mcp_memory.daemon.os.getpgid", lambda pid: (_ for _ in ()).throw(ProcessLookupError()))
    monkeypatch.setattr("mcp_memory.daemon.os.kill", lambda pid, sig: sent_signals.append(sig))
    monkeypatch.setattr("mcp_memory.daemon._spawn_daemon_process", lambda host, port: spawned.append((host, port)))
    monotonic_values = iter([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    monkeypatch.setattr("mcp_memory.daemon.time.sleep", lambda _: None)
    monkeypatch.setattr("mcp_memory.daemon.time.monotonic", lambda: next(monotonic_values))

    current = ensure_daemon_started()

    assert current.daemon_scope == "global"
    assert sent_signals == [15]
    assert spawned == [(spec.config.daemon.host, 4242)]


def test_ensure_daemon_started_terminates_orphaned_daemon_processes_when_metadata_missing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    config = Config()
    config.daemon.port = 4242
    spec = _Spec(
        memory_path=tmp_path / "memories",
        config=config,
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
    spawned: list[tuple[str, int]] = []
    killed: list[int] = []
    process_states = iter([False, False, False])
    monotonic_values = iter([value * 0.2 for value in range(8)])

    monkeypatch.setattr("mcp_memory.daemon.resolve_global_daemon_bootstrap_spec", lambda workspace_root_override=None, cwd=None: spec)
    monkeypatch.setattr("mcp_memory.daemon._read_daemon_metadata", lambda path: None if not spawned else metadata)
    monkeypatch.setattr(
        "mcp_memory.daemon._list_daemon_processes",
        lambda current_pid: [
            __import__("mcp_memory.daemon").daemon._DaemonProcess(
                pid=4321,
                executable="/old/tool/python",
                command=("python", "-m", "mcp_memory.cli", "daemon"),
                state_dir=tmp_path / "state" / "mcp-memory",
            )
        ],
    )
    monkeypatch.setattr("mcp_memory.daemon._is_process_running", lambda pid: next(process_states))
    monkeypatch.setattr("mcp_memory.daemon._is_daemon_healthy", lambda current: True)
    monkeypatch.setattr("mcp_memory.daemon.os.getpgid", lambda pid: (_ for _ in ()).throw(ProcessLookupError()))
    monkeypatch.setattr("mcp_memory.daemon.os.kill", lambda pid, sig: killed.append(pid))
    monkeypatch.setattr("mcp_memory.daemon._spawn_daemon_process", lambda host, port: spawned.append((host, port)))
    monkeypatch.setattr("mcp_memory.daemon.time.sleep", lambda _: None)
    monkeypatch.setattr("mcp_memory.daemon.time.monotonic", lambda: next(monotonic_values))

    current = ensure_daemon_started()

    assert current.port == 8126
    assert killed == [4321]
    assert spawned == [(spec.config.daemon.host, 4242)]


def test_ensure_daemon_started_orphan_cleanup_ignores_other_state_roots(monkeypatch, tmp_path: Path) -> None:
    current_state_home = tmp_path / "state-current"
    monkeypatch.setenv("XDG_STATE_HOME", str(current_state_home))
    config = Config()
    config.daemon.port = 4242
    spec = _Spec(
        memory_path=tmp_path / "memories",
        config=config,
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
    spawned: list[tuple[str, int]] = []
    terminated: list[int] = []
    monotonic_values = iter([value * 0.2 for value in range(8)])

    monkeypatch.setattr("mcp_memory.daemon.resolve_global_daemon_bootstrap_spec", lambda workspace_root_override=None, cwd=None: spec)
    monkeypatch.setattr("mcp_memory.daemon._read_daemon_metadata", lambda path: None if not spawned else metadata)
    monkeypatch.setattr(
        "mcp_memory.daemon._list_daemon_processes",
        lambda current_pid: [
            __import__("mcp_memory.daemon").daemon._DaemonProcess(
                pid=4321,
                executable="/old/tool/python",
                command=("python", "-m", "mcp_memory.cli", "daemon"),
                state_dir=current_state_home / "mcp-memory",
            ),
            __import__("mcp_memory.daemon").daemon._DaemonProcess(
                pid=8765,
                executable="/other/tool/python",
                command=("python", "-m", "mcp_memory.cli", "daemon"),
                state_dir=tmp_path / "state-other" / "mcp-memory",
            ),
        ],
    )
    monkeypatch.setattr("mcp_memory.daemon._terminate_daemon_process", lambda pid, **kwargs: terminated.append(pid))
    monkeypatch.setattr("mcp_memory.daemon._cleanup_stale_daemon_socket", lambda *args, **kwargs: False)
    monkeypatch.setattr("mcp_memory.daemon._spawn_daemon_process", lambda host, port: spawned.append((host, port)))
    monkeypatch.setattr("mcp_memory.daemon._is_daemon_healthy", lambda current: True)
    monkeypatch.setattr("mcp_memory.daemon.time.sleep", lambda _: None)
    monkeypatch.setattr("mcp_memory.daemon.time.monotonic", lambda: next(monotonic_values))

    current = ensure_daemon_started()

    assert current.port == 8126
    assert terminated == [4321]
    assert spawned == [(spec.config.daemon.host, 4242)]


def test_ensure_daemon_started_removes_stale_metadata_and_terminates_orphans(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    config = Config()
    config.daemon.port = 4242
    spec = _Spec(
        memory_path=tmp_path / "memories",
        config=config,
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

    spawned: list[tuple[str, int]] = []
    killed: list[int] = []
    process_states = iter([False, False, False])
    monotonic_values = iter([value * 0.2 for value in range(8)])

    monkeypatch.setattr("mcp_memory.daemon.resolve_global_daemon_bootstrap_spec", lambda workspace_root_override=None, cwd=None: spec)
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
                state_dir=tmp_path / "state" / "mcp-memory",
            )
        ],
    )
    monkeypatch.setattr("mcp_memory.daemon.os.getpgid", lambda pid: (_ for _ in ()).throw(ProcessLookupError()))
    monkeypatch.setattr("mcp_memory.daemon.os.kill", lambda pid, sig: killed.append(pid))
    monkeypatch.setattr("mcp_memory.daemon._spawn_daemon_process", lambda host, port: spawned.append((host, port)))
    monkeypatch.setattr("mcp_memory.daemon.time.sleep", lambda _: None)
    monkeypatch.setattr("mcp_memory.daemon.time.monotonic", lambda: next(monotonic_values))

    current = ensure_daemon_started()

    assert current.port == 8128
    assert killed == [3333]
    assert spawned == [(spec.config.daemon.host, 4242)]


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


def test_assess_daemon_health_reports_probe_failure_details(monkeypatch, tmp_path: Path) -> None:
    metadata = DaemonMetadata(
        host="127.0.0.1",
        port=8129,
        pid=9876,
        started_at=1.0,
        status="ready",
        transport="zmq",
        socket_path=str(tmp_path / "daemon.sock"),
    )

    monkeypatch.setattr(
        "mcp_memory.daemon_process.request_daemon_json",
        lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError("daemon_request_timed_out")),
    )

    assessment = assess_daemon_health(metadata, timeout_seconds=1.25)

    assert assessment.healthy is False
    assert assessment.reason == "health_probe_timeout"
    assert assessment.retryable is True
    assert assessment.timeout_seconds == pytest.approx(1.25)
    assert assessment.error_type == "TimeoutError"
    assert assessment.error_text == "daemon_request_timed_out"


def test_ensure_daemon_started_does_not_kill_daemon_after_single_retryable_probe_failure(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    config = Config()
    config.daemon.port = 4242
    config.daemon.healthcheck_interval_seconds = 0.01
    spec = _Spec(
        memory_path=tmp_path / "memories",
        config=config,
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
        transport="zmq",
        socket_path=str(tmp_path / "daemon.sock"),
    )
    assessments = iter(
        [
            DaemonHealthAssessment(
                healthy=False,
                reason="health_probe_timeout",
                retryable=True,
                timeout_seconds=3.0,
                error_type="TimeoutError",
                error_text="daemon_request_timed_out",
            ),
            DaemonHealthAssessment(
                healthy=True,
                reason="healthy",
                retryable=False,
                timeout_seconds=3.0,
            ),
        ]
    )

    monkeypatch.setattr("mcp_memory.daemon.resolve_global_daemon_bootstrap_spec", lambda workspace_root_override=None, cwd=None: spec)
    monkeypatch.setattr("mcp_memory.daemon._read_daemon_metadata", lambda path: metadata)
    monkeypatch.setattr("mcp_memory.daemon._is_process_running", lambda pid: True)
    monkeypatch.setattr("mcp_memory.daemon._assess_daemon_health", lambda current, timeout_seconds=3.0: next(assessments))
    monkeypatch.setattr("mcp_memory.daemon.time.sleep", lambda _: None)
    monkeypatch.setattr(
        "mcp_memory.daemon._terminate_daemon_process",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("existing daemon should not be killed after a single retryable failure")),
    )
    monkeypatch.setattr(
        "mcp_memory.daemon._spawn_daemon_process",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("healthy daemon should be reused")),
    )

    current = ensure_daemon_started()

    assert current.pid == metadata.pid


def test_inspect_daemon_retries_transient_probe_failure(monkeypatch, tmp_path: Path) -> None:
    config = Config()
    config.daemon.healthcheck_interval_seconds = 0.01
    spec = _Spec(
        memory_path=tmp_path / "memories",
        config=config,
        workspace_id="workspace-inspect",
        workspace_root=tmp_path / "workspace",
        lock_path=tmp_path / "workspace.lock",
    )
    metadata = DaemonMetadata(
        host="127.0.0.1",
        port=8125,
        pid=5678,
        started_at=1.0,
        status="ready",
        transport="zmq",
        socket_path=str(tmp_path / "daemon.sock"),
    )
    assessments = iter(
        [
            DaemonHealthAssessment(False, "health_probe_timeout", True, 3.0),
            DaemonHealthAssessment(True, "healthy", False, 3.0),
        ]
    )

    monkeypatch.setattr("mcp_memory.daemon.resolve_global_daemon_bootstrap_spec", lambda: spec)
    monkeypatch.setattr("mcp_memory.daemon._read_daemon_metadata", lambda path: metadata)
    monkeypatch.setattr("mcp_memory.daemon._is_process_running", lambda pid: True)
    monkeypatch.setattr("mcp_memory.daemon._assess_daemon_health", lambda *args, **kwargs: next(assessments))
    monkeypatch.setattr("mcp_memory.daemon.time.sleep", lambda _: None)

    inspected_metadata, healthy = inspect_daemon()

    assert inspected_metadata == metadata
    assert healthy is True


def test_ensure_daemon_started_skips_health_probe_for_stale_metadata_pid(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    config = Config()
    config.daemon.port = 4242
    spec = _Spec(
        memory_path=tmp_path / "memories",
        config=config,
        workspace_id="workspace-start",
        workspace_root=tmp_path / "workspace",
        lock_path=tmp_path / "workspace.lock",
    )
    stale_metadata = DaemonMetadata(
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
    spawned: list[tuple[str, int]] = []

    monkeypatch.setattr("mcp_memory.daemon.resolve_global_daemon_bootstrap_spec", lambda workspace_root_override=None, cwd=None: spec)
    monkeypatch.setattr(
        "mcp_memory.daemon._read_daemon_metadata",
        lambda path: stale_metadata if not spawned else fresh_metadata,
    )
    monkeypatch.setattr("mcp_memory.daemon._is_process_running", lambda pid: pid == fresh_metadata.pid)
    monkeypatch.setattr(
        "mcp_memory.daemon._assess_daemon_health",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("stale metadata should be rejected before health probing")),
    )
    monkeypatch.setattr("mcp_memory.daemon.remove_metadata", lambda path: None)
    monkeypatch.setattr("mcp_memory.daemon._terminate_orphaned_daemon_processes", lambda **kwargs: None)
    monkeypatch.setattr("mcp_memory.daemon._cleanup_stale_daemon_socket", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        "mcp_memory.daemon._spawn_daemon_process",
        lambda host, port: spawned.append((host, port)),
    )
    monotonic_values = iter([0.0, 0.1, 0.2, 0.3])
    monkeypatch.setattr("mcp_memory.daemon.time.sleep", lambda _: None)
    monkeypatch.setattr("mcp_memory.daemon.time.monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr("mcp_memory.daemon._is_daemon_healthy", lambda current: current.pid == fresh_metadata.pid)

    current = ensure_daemon_started()

    assert current.pid == fresh_metadata.pid
    assert spawned == [(spec.config.daemon.host, 4242)]


@pytest.mark.asyncio
async def test_is_daemon_healthy_stays_true_while_request_pool_is_saturated(tmp_path: Path) -> None:
    blocking_started = asyncio.Event()
    release_blocking_request = asyncio.Event()
    blocking_request: asyncio.Task[dict] | None = None

    async def _blocking_hook(_payload: dict[str, object]) -> dict[str, object]:
        blocking_started.set()
        await release_blocking_request.wait()
        return {"status": "ok"}

    metadata = DaemonMetadata(
        host="127.0.0.1",
        port=8130,
        pid=4321,
        started_at=1.0,
        status="ready",
        transport="zmq",
        socket_path=str(tmp_path / "daemon.sock"),
    )
    server = DaemonZmqServer(
        context_factory=cast(Callable[[dict[str, object]], ApplicationContext], lambda _payload: None),
        hook_handlers={"/api/block": _blocking_hook},
        routes_provider=cast(DaemonRoutesProvider, lambda: None),
        socket_path=tmp_path / "daemon.sock",
        metadata_provider=lambda: metadata,
        max_concurrent_requests=1,
    )

    await server.start()
    try:
        blocking_request = asyncio.create_task(
            asyncio.to_thread(
                request_daemon_json,
                metadata,
                "/api/block",
                {},
                timeout_seconds=5.0,
            )
        )
        await asyncio.wait_for(blocking_started.wait(), timeout=1.0)

        assert await asyncio.to_thread(is_daemon_healthy, metadata) is True
    finally:
        release_blocking_request.set()
        if blocking_request is not None:
            await blocking_request
        await server.stop()


@pytest.mark.asyncio
async def test_is_daemon_healthy_stays_true_while_sync_management_request_runs(tmp_path: Path) -> None:
    overview_started = threading.Event()
    release_overview = threading.Event()
    overview_request: asyncio.Task[dict] | None = None

    class _OverviewPayload:
        def model_dump(self) -> dict[str, object]:
            return {"status": "ok"}

    class _RoutesService:
        def get_overview(self) -> _OverviewPayload:
            overview_started.set()
            release_overview.wait(timeout=5.0)
            return _OverviewPayload()

    metadata = DaemonMetadata(
        host="127.0.0.1",
        port=8131,
        pid=8765,
        started_at=1.0,
        status="ready",
        transport="zmq",
        socket_path=str(tmp_path / "daemon-overview.sock"),
    )
    server = DaemonZmqServer(
        context_factory=cast(Callable[[dict[str, object]], ApplicationContext], lambda _payload: None),
        hook_handlers={},
        routes_provider=cast(DaemonRoutesProvider, lambda: SimpleNamespace(service=_RoutesService())),
        socket_path=tmp_path / "daemon-overview.sock",
        metadata_provider=lambda: metadata,
        max_concurrent_requests=1,
    )

    await server.start()
    try:
        overview_request = asyncio.create_task(
            asyncio.to_thread(
                request_daemon_json,
                metadata,
                "/api/overview",
                {},
                timeout_seconds=5.0,
            )
        )
        assert await asyncio.to_thread(overview_started.wait, 1.0) is True

        assert await asyncio.to_thread(is_daemon_healthy, metadata) is True
    finally:
        release_overview.set()
        if overview_request is not None:
            await overview_request
        await server.stop()


@pytest.mark.asyncio
async def test_daemon_zmq_server_start_refuses_to_remove_live_socket(monkeypatch, tmp_path: Path) -> None:
    removed: list[Path] = []
    socket_path = tmp_path / "daemon.sock"
    socket_path.write_text("occupied", encoding="utf-8")

    monkeypatch.setattr("mcp_memory.daemon_transport.probe_daemon_socket", lambda path, timeout_seconds: True)
    monkeypatch.setattr("mcp_memory.daemon_transport._remove_stale_socket", lambda path: removed.append(Path(path)))

    server = DaemonZmqServer(
        context_factory=cast(Callable[[dict[str, object]], ApplicationContext], lambda _payload: None),
        hook_handlers={},
        routes_provider=cast(DaemonRoutesProvider, lambda: None),
        socket_path=socket_path,
        metadata_provider=lambda: DaemonMetadata(
            host="127.0.0.1",
            port=8131,
            pid=4321,
            started_at=1.0,
            status="ready",
            transport="zmq",
            socket_path=str(socket_path),
        ),
    )

    with pytest.raises(RuntimeError, match="daemon_socket_already_active"):
        await server.start()

    assert removed == []
    assert socket_path.exists()


def test_daemon_zmq_server_stop_preserves_live_rebound_socket(monkeypatch, tmp_path: Path) -> None:
    removed: list[Path] = []

    monkeypatch.setattr("mcp_memory.daemon_transport._socket_path_exists", lambda socket_path: True)
    monkeypatch.setattr("mcp_memory.daemon_transport.probe_daemon_socket", lambda socket_path, timeout_seconds: True)
    monkeypatch.setattr("mcp_memory.daemon_transport._remove_stale_socket", lambda socket_path: removed.append(Path(socket_path)))

    __import__("mcp_memory.daemon_transport").daemon_transport._cleanup_socket_path_after_stop(tmp_path / "daemon.sock")

    assert removed == []


def test_ensure_daemon_started_cleans_stale_socket_before_spawn(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    config = Config()
    config.daemon.port = 4242
    spec = _Spec(
        memory_path=tmp_path / "memories",
        config=config,
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

    spawned: list[tuple[str, int]] = []
    removed_sockets: list[Path] = []
    probe_timeouts: list[float] = []
    monotonic_values = iter([0.0, 0.2, 0.4, 0.6, 0.8])

    monkeypatch.setattr("mcp_memory.daemon.resolve_global_daemon_bootstrap_spec", lambda workspace_root_override=None, cwd=None: spec)
    monkeypatch.setattr(
        "mcp_memory.daemon._read_daemon_metadata",
        lambda path: stale_metadata if not spawned else fresh_metadata,
    )
    monkeypatch.setattr("mcp_memory.daemon._is_daemon_healthy", lambda current: current.pid == fresh_metadata.pid)
    monkeypatch.setattr("mcp_memory.daemon._is_process_running", lambda pid: False)
    monkeypatch.setattr("mcp_memory.daemon._terminate_orphaned_daemon_processes", lambda **kwargs: None)
    monkeypatch.setattr(
        "mcp_memory.daemon._probe_daemon_socket",
        lambda socket_path, timeout_seconds: probe_timeouts.append(timeout_seconds) or False,
    )
    monkeypatch.setattr("mcp_memory.daemon._remove_daemon_socket", lambda socket_path: removed_sockets.append(Path(socket_path)))
    monkeypatch.setattr("mcp_memory.daemon._spawn_daemon_process", lambda host, port: spawned.append((host, port)))
    monkeypatch.setattr("mcp_memory.daemon.time.sleep", lambda _: None)
    monkeypatch.setattr("mcp_memory.daemon.time.monotonic", lambda: next(monotonic_values))

    current = ensure_daemon_started()

    assert current.port == 8131
    assert removed_sockets == [stale_socket]
    assert len(probe_timeouts) == 1
    assert 0.05 <= probe_timeouts[0] <= 0.1
    assert spawned == [(spec.config.daemon.host, 4242)]
