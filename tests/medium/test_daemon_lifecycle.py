from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import signal

from click.testing import CliRunner
import pytest
import mcp_memory.daemon as daemon_module

from mcp_memory.cli import main
from mcp_memory.config import Config, resolve_daemon_metadata_path
from mcp_memory.daemon import DaemonMetadata, DaemonStopResult, ensure_daemon_started, read_daemon_metadata, stop_daemon
from mcp_memory.daemon_process import DaemonSpawnDetails, is_daemon_healthy, spawn_daemon_process


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


def test_ensure_daemon_started_uses_configured_auto_start_timeout(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    config = Config()
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

    monkeypatch.setattr("mcp_memory.daemon.resolve_runtime_spec", lambda workspace_root_override=None, cwd=None: spec)
    monkeypatch.setattr("mcp_memory.daemon.FilesystemLock", _FakeLock)
    monkeypatch.setattr("mcp_memory.daemon._read_daemon_metadata", lambda path: metadata)
    monkeypatch.setattr("mcp_memory.daemon._is_daemon_healthy", lambda current: True)

    current = ensure_daemon_started()

    assert current.pid == metadata.pid
    assert acquired_timeouts == [0.25]
    assert released == [True]


def test_read_daemon_metadata_returns_none_for_missing_file(tmp_path: Path) -> None:
    assert read_daemon_metadata(tmp_path / "missing.json") is None


def test_is_process_running_treats_zombies_as_stopped(monkeypatch) -> None:
    monkeypatch.setattr('mcp_memory.daemon._read_process_state', lambda pid: 'Z')
    monkeypatch.setattr('mcp_memory.daemon.os.kill', lambda pid, sig: (_ for _ in ()).throw(AssertionError('os.kill should not be called for zombies')))

    assert __import__('mcp_memory.daemon').daemon._is_process_running(1234) is False


def test_spawn_daemon_process_uses_workspace_root_as_cwd(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    captured: dict[str, object] = {}

    def _fake_popen(command, **kwargs):
        captured['command'] = command
        captured.update(kwargs)
        class _DummyProcess:
            pid = 4321
        return _DummyProcess()

    monkeypatch.setattr('mcp_memory.daemon_process.subprocess.Popen', _fake_popen)

    details = spawn_daemon_process(tmp_path / 'workspace', '127.0.0.1', 8123)

    assert captured['cwd'] == str(tmp_path / 'workspace')
    assert captured['stdin'] is not None
    assert captured['start_new_session'] is True
    assert details.pid == 4321
    assert details.command[1:4] == ('-m', 'mcp_memory.cli', 'daemon')
    assert details.startup_log_path.name == 'daemon.log'


def test_ensure_daemon_started_includes_startup_log_tail_when_spawned_child_exits(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv('XDG_STATE_HOME', str(tmp_path / 'state'))
    config = Config()
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

    monkeypatch.setattr('mcp_memory.daemon.resolve_runtime_spec', lambda workspace_root_override=None, cwd=None: spec)
    monkeypatch.setattr('mcp_memory.daemon._read_daemon_metadata', lambda path: None)
    monkeypatch.setattr('mcp_memory.daemon._terminate_orphaned_daemon_processes', lambda **kwargs: None)
    monkeypatch.setattr('mcp_memory.daemon._find_free_port', lambda: 9006)
    monkeypatch.setattr(
        'mcp_memory.daemon._spawn_daemon_process',
        lambda workspace_root, host, port: DaemonSpawnDetails(
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
        lambda workspace_root, cwd=None: (_ for _ in ()).throw(
            RuntimeError(
                'Timed out waiting for global daemon startup. '
                'startup_log=/tmp/daemon.log metadata=missing\n'
                'Startup log tail (/tmp/daemon.log):\n'
                'Traceback: bind failed'
            )
        ),
    )

    result = runner.invoke(main, ['daemon', 'dashboard'])

    assert result.exit_code == 1
    assert 'Error:' in result.output
    assert 'Timed out waiting for global daemon startup.' in result.output
    assert 'Startup log tail (/tmp/daemon.log):' in result.output
    assert 'Traceback: bind failed' in result.output


def test_ensure_daemon_started_includes_startup_log_tail_on_timeout(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv('XDG_STATE_HOME', str(tmp_path / 'state'))
    config = Config()
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

    monkeypatch.setattr('mcp_memory.daemon.resolve_runtime_spec', lambda workspace_root_override=None, cwd=None: spec)
    monkeypatch.setattr('mcp_memory.daemon._terminate_orphaned_daemon_processes', lambda **kwargs: None)
    monkeypatch.setattr('mcp_memory.daemon._find_free_port', lambda: 9007)
    monkeypatch.setattr(
        'mcp_memory.daemon._spawn_daemon_process',
        lambda workspace_root, host, port: DaemonSpawnDetails(
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


def test_ensure_daemon_started_allows_short_grace_for_late_health(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv('XDG_STATE_HOME', str(tmp_path / 'state'))
    config = Config()
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
    spawned: list[tuple[Path, str, int]] = []
    monotonic_values = iter([value * 0.05 for value in range(80)])

    monkeypatch.setattr('mcp_memory.daemon.resolve_runtime_spec', lambda workspace_root_override=None, cwd=None: spec)

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
    monkeypatch.setattr('mcp_memory.daemon._spawn_daemon_process', lambda workspace_root, host, port: spawned.append((workspace_root, host, port)))
    monkeypatch.setattr('mcp_memory.daemon._find_free_port', lambda: 9005)
    monkeypatch.setattr('mcp_memory.daemon.time.sleep', lambda _: None)
    monkeypatch.setattr('mcp_memory.daemon.time.monotonic', lambda: next(monotonic_values))

    current = ensure_daemon_started()

    assert current.pid == metadata.pid
    assert spawned == [(spec.workspace_root, spec.config.daemon.host, 9005)]


def test_ensure_daemon_started_gives_readiness_a_fresh_timeout_after_cleanup(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv('XDG_STATE_HOME', str(tmp_path / 'state'))
    config = Config()
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
    spawned: list[tuple[Path, str, int]] = []
    monotonic_values = iter([0.0, 0.19, 0.2, 0.25])

    class _FakeLock:
        def __init__(self, path: Path) -> None:
            self.path = path

        def acquire(self, *, timeout_seconds: float) -> None:
            acquired_timeouts.append(timeout_seconds)

        def release(self) -> None:
            pass

    monkeypatch.setattr('mcp_memory.daemon.FilesystemLock', _FakeLock)
    monkeypatch.setattr('mcp_memory.daemon.resolve_runtime_spec', lambda workspace_root_override=None, cwd=None: spec)

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
        lambda workspace_root, host, port: spawned.append((workspace_root, host, port)),
    )
    monkeypatch.setattr('mcp_memory.daemon._find_free_port', lambda: 9008)
    monkeypatch.setattr('mcp_memory.daemon.time.sleep', lambda _: None)
    monkeypatch.setattr('mcp_memory.daemon.time.monotonic', lambda: next(monotonic_values))

    current = ensure_daemon_started()

    assert current.pid == ready_metadata.pid
    assert acquired_timeouts == [0.2]
    assert cleanup_deadlines == [0.2]
    assert spawned == [(spec.workspace_root, spec.config.daemon.host, 9008)]


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

    monkeypatch.setattr("mcp_memory.daemon.resolve_runtime_spec", lambda workspace_root_override=None, cwd=None: spec)
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

    monkeypatch.setattr("mcp_memory.daemon.resolve_runtime_spec", lambda workspace_root_override=None, cwd=None: spec)
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
    monotonic_values = iter([value * 0.2 for value in range(8)])

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
    monotonic_values = iter([value * 0.2 for value in range(8)])

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
