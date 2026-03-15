from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from mcp_memory.config import Config, resolve_daemon_metadata_path
from mcp_memory.daemon import DaemonMetadata, ensure_daemon_started, read_daemon_metadata


pytestmark = pytest.mark.medium


@dataclass(frozen=True)
class _Spec:
    memory_path: Path
    config: Config
    workspace_id: str
    workspace_root: Path
    lock_path: Path


def test_ensure_daemon_started_reuses_healthy_metadata(monkeypatch, tmp_path: Path) -> None:
    spec = _Spec(
        memory_path=tmp_path / "memories",
        config=Config(),
        workspace_id="workspace-123",
        workspace_root=tmp_path / "workspace",
        lock_path=tmp_path / "workspace.lock",
    )
    metadata = DaemonMetadata(
        workspace_id=spec.workspace_id,
        workspace_root=str(spec.workspace_root),
        host="127.0.0.1",
        port=8123,
        pid=123,
        started_at=1.0,
        status="ready",
    )
    metadata_path = resolve_daemon_metadata_path(spec.workspace_id)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(__import__("json").dumps(metadata.__dict__), encoding="utf-8")

    monkeypatch.setattr("mcp_memory.daemon.resolve_runtime_spec", lambda workspace_root_override=None, cwd=None: spec)
    monkeypatch.setattr("mcp_memory.daemon._is_daemon_healthy", lambda current: True)
    monkeypatch.setattr("mcp_memory.daemon._spawn_daemon_process", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not spawn")))

    current = ensure_daemon_started()

    assert current.port == 8123
    assert current.workspace_id == spec.workspace_id


def test_read_daemon_metadata_returns_none_for_missing_file(tmp_path: Path) -> None:
    assert read_daemon_metadata(tmp_path / "missing.json") is None
