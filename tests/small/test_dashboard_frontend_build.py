from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from mcp_memory.management.frontend_build import ensure_dashboard_frontend_built


pytestmark = pytest.mark.small


def _seed_frontend_tree(tmp_path: Path) -> tuple[Path, Path]:
    static_root = tmp_path / "static"
    frontend_root = tmp_path / "frontend"
    dist_root = static_root / "dist"
    src_root = frontend_root / "src"
    dist_root.mkdir(parents=True)
    src_root.mkdir(parents=True)
    (frontend_root / "package.json").write_text('{"scripts":{"build":"vite build"}}', encoding="utf-8")
    (frontend_root / "index.html").write_text("<html></html>", encoding="utf-8")
    (src_root / "App.tsx").write_text("export const App = () => null;", encoding="utf-8")
    return static_root, dist_root / "index.html"


def test_ensure_dashboard_frontend_built_skips_when_bundle_is_current(tmp_path: Path) -> None:
    static_root, dist_index = _seed_frontend_tree(tmp_path)
    dist_index.write_text("built", encoding="utf-8")
    source_file = tmp_path / "frontend" / "src" / "App.tsx"
    source_file.touch()
    dist_index.touch()

    result = ensure_dashboard_frontend_built(static_root=static_root)

    assert result.status == "up_to_date"
    assert result.built is False


def test_ensure_dashboard_frontend_built_runs_build_when_bundle_is_stale(tmp_path: Path, monkeypatch) -> None:
    static_root, dist_index = _seed_frontend_tree(tmp_path)
    dist_index.write_text("old", encoding="utf-8")
    dist_index.touch()
    source_file = tmp_path / "frontend" / "src" / "App.tsx"
    source_file.write_text("export const App = () => <div />;", encoding="utf-8")
    source_file.touch()

    called: dict[str, object] = {}

    def _fake_run(command, *, cwd, check, capture_output, text):
        called["command"] = command
        called["cwd"] = cwd
        dist_index.write_text("fresh", encoding="utf-8")
        dist_index.touch()
        return SimpleNamespace(returncode=0, stdout="vite build ok", stderr="")

    monkeypatch.setattr("mcp_memory.management.frontend_build.subprocess.run", _fake_run)

    result = ensure_dashboard_frontend_built(static_root=static_root)

    assert result.status == "built"
    assert result.built is True
    assert called["command"] == ["npm", "run", "build"]
    assert called["cwd"] == tmp_path / "frontend"


def test_ensure_dashboard_frontend_built_reports_missing_npm(tmp_path: Path, monkeypatch) -> None:
    static_root, dist_index = _seed_frontend_tree(tmp_path)
    dist_index.write_text("old", encoding="utf-8")

    def _missing_run(command, *, cwd, check, capture_output, text):
        raise OSError("npm not found")

    monkeypatch.setattr("mcp_memory.management.frontend_build.subprocess.run", _missing_run)
    source_file = tmp_path / "frontend" / "src" / "App.tsx"
    source_file.write_text("export const App = () => <div />;", encoding="utf-8")
    source_file.touch()

    result = ensure_dashboard_frontend_built(static_root=static_root)

    assert result.status == "build_unavailable"
    assert result.message == "npm not found"