from pathlib import Path

import pytest

from mcp_memory.config import (
    ensure_default_config_exists,
    load_config,
    resolve_daemon_metadata_path,
    resolve_memory_path,
    resolve_workspace_id,
    resolve_workspace_root,
)


pytestmark = pytest.mark.medium


def test_load_config_prefers_created_default(tmp_path: Path) -> None:
    config_path = tmp_path / "config" / "config.toml"
    ensure_default_config_exists(config_path)

    config = load_config(config_path)

    assert config.ai.provider == "none"
    assert config.daemon.host == "127.0.0.1"
    assert config.search_ranking.calibration_threshold == 0.035


def test_load_config_reads_search_ranking_overrides(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "[search_ranking]\n"
        "rrf_k = 42.0\n"
        "calibration_threshold = 0.05\n"
        "calibration_steepness = 180.0\n"
        "workspace_multiplier = 1.15\n"
        "degradation_multiplier = 0.25\n"
        "access_half_life_days = 5.0\n"
        "access_bonus_scale = 0.08\n"
        "authority_link_step = 0.03\n"
        "authority_link_cap = 8\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.search_ranking.rrf_k == 42.0
    assert config.search_ranking.calibration_threshold == 0.05
    assert config.search_ranking.calibration_steepness == 180.0
    assert config.search_ranking.workspace_multiplier == 1.15
    assert config.search_ranking.degradation_multiplier == 0.25
    assert config.search_ranking.access_half_life_days == 5.0
    assert config.search_ranking.access_bonus_scale == 0.08
    assert config.search_ranking.authority_link_step == 0.03
    assert config.search_ranking.authority_link_cap == 8


def test_resolve_workspace_root_prefers_git_root(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    nested = repo_root / "src" / "feature"
    nested.mkdir(parents=True)
    (repo_root / ".git").mkdir()

    assert resolve_workspace_root(cwd=nested) == repo_root


def test_resolve_workspace_id_is_stable_outside_git(tmp_path: Path) -> None:
    workspace = tmp_path / "scratch" / "notes"
    workspace.mkdir(parents=True)

    first = resolve_workspace_id(cwd=workspace)
    second = resolve_workspace_id(cwd=workspace)

    assert first == second
    assert first.startswith("notes-")


def test_resolve_memory_path_uses_shared_global_dir(tmp_path: Path, monkeypatch) -> None:
    data_home = tmp_path / "data"
    monkeypatch.setenv("XDG_DATA_HOME", str(data_home))
    config = load_config(tmp_path / "config.toml")

    resolved = resolve_memory_path(config)

    assert resolved == data_home / "mcp-memory" / "memories"


def test_resolve_daemon_metadata_path_uses_state_dir(tmp_path: Path, monkeypatch) -> None:
    state_home = tmp_path / "state"
    monkeypatch.setenv("XDG_STATE_HOME", str(state_home))

    metadata_path = resolve_daemon_metadata_path("workspace-123")

    assert metadata_path == state_home / "mcp-memory" / "daemons" / "daemon.json"
