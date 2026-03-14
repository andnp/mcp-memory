from pathlib import Path

import pytest

from mcp_memory.config import (
    Config,
    IndexingConfig,
    MemoryConfig,
    ProjectConfig,
    detect_project,
    persist_project_to_config,
    resolve_documents_path,
    resolve_memory_path,
    resolve_workspace_id,
    resolve_workspace_root,
)


pytestmark = pytest.mark.medium


def test_persist_project_to_config_writes_project_entry(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    persist_project_to_config("demo-project", str(tmp_path / "workspace"))

    config_path = tmp_path / ".config" / "mcp-markdown-ragdocs" / "config.toml"
    config_text = config_path.read_text(encoding="utf-8")

    assert "demo-project" in config_text
    assert str(tmp_path / "workspace") in config_text


def test_detect_project_auto_registers_cwd(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    repo_path = tmp_path / "autonomous-memory"
    repo_path.mkdir(parents=True)

    detected = detect_project(cwd=repo_path, projects=[])

    assert detected == "autonomous-memory"
    config_path = tmp_path / "home" / ".config" / "mcp-markdown-ragdocs" / "config.toml"
    assert str(repo_path) in config_path.read_text(encoding="utf-8")


def test_resolve_memory_path_uses_xdg_data_home(tmp_path: Path, monkeypatch) -> None:
    data_home = tmp_path / "data"
    monkeypatch.setenv("XDG_DATA_HOME", str(data_home))
    config = Config(memory=MemoryConfig(storage_strategy="shared"))

    resolved = resolve_memory_path(config, detected_project="demo-project")

    assert resolved == data_home / "mcp-memory" / "memories"


def test_resolve_workspace_root_prefers_git_root(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    nested = repo_root / "src" / "feature"
    nested.mkdir(parents=True)
    (repo_root / ".git").mkdir()

    resolved = resolve_workspace_root(cwd=nested)

    assert resolved == repo_root


def test_resolve_workspace_id_is_stable_for_non_git_directory(tmp_path: Path) -> None:
    workspace = tmp_path / "scratch" / "notes"
    workspace.mkdir(parents=True)

    first = resolve_workspace_id(cwd=workspace)
    second = resolve_workspace_id(cwd=workspace)

    assert first == second
    assert first.startswith("notes-")


def test_resolve_documents_path_prefers_detected_project_path(tmp_path: Path) -> None:
    project_dir = tmp_path / "workspace"
    project_dir.mkdir()
    config = Config(indexing=IndexingConfig(documents_path="docs"))
    projects = [ProjectConfig(name="demo-project", path=str(project_dir))]

    resolved = resolve_documents_path(
        config,
        detected_project="demo-project",
        projects=projects,
    )

    assert resolved == str(project_dir)