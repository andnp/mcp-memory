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

    assert config.daemon.host == "127.0.0.1"
    assert config.search_ranking.calibration_threshold == 0.035
    assert config.provider_routing.task_routes["ingest-system1"] == ["copilot-mini", "gemini-cheap"]
    assert config.provider_routing.task_routes["deduplicator"] == ["copilot-mini", "gemini-cheap"]
    assert config.provider_routing.task_routes["memory-curator"] == ["copilot-strong", "gemini-strong", "gemini-cheap"]
    assert config.provider_routing.default_json_route == ["copilot-mini", "gemini-cheap"]
    assert config.provider_routing.default_agentic_route == ["copilot-mini", "gemini-cheap"]
    assert config.provider_routing.fallback_to_json_only is False
    assert config.provider_routing.task_classes["memory-curator"] == "premium_agentic"
    assert config.provider_routing.task_classes["summarize-memory"] == "deterministic"
    assert config.provider_routing.low_priority_task_names == ["graph-linker", "conflict-detector", "defragmenter", "taxonomist"]
    assert config.provider_routing.model_burst_call_limit == 2
    assert config.provider_routing.model_burst_window_seconds == 300.0
    assert config.provider_routing.profiles["gemini-strong"].model == "gemini-3.1-pro-preview"
    assert config.provider_routing.profiles["copilot-strong"].model == "gpt-5.4-mini"
    assert config.provider_routing.profiles["copilot-mini"].model == "gpt-5-mini"
    assert config.provider_routing.profiles["gemini-cheap"].model == "gemini-3-flash-preview"
    assert config.ingest_suppression.enabled is False
    assert config.ingest_escalation.enabled is True


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
        "authority_link_cap = 8\n"
        "adaptive_result_max = 12\n"
        "adaptive_result_score_ratio_floor = 0.75\n"
        "adaptive_result_min_score = 0.4\n"
        "adaptive_result_max_score_gap = 0.05\n",
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
    assert config.search_ranking.adaptive_result_max == 12
    assert config.search_ranking.adaptive_result_score_ratio_floor == 0.75
    assert config.search_ranking.adaptive_result_min_score == 0.4
    assert config.search_ranking.adaptive_result_max_score_gap == 0.05


def test_load_config_reads_provider_routing_overrides(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "[provider_routing]\n"
        'default_json_route = ["copilot-mini"]\n'
        'default_agentic_route = ["gemini-cheap"]\n'
        'fallback_to_json_only = false\n'
        'low_priority_task_names = ["deduplicator"]\n'
        'model_burst_call_limit = 1\n'
        'model_burst_window_seconds = 600.0\n'
        "\n"
        "[provider_routing.task_routes]\n"
        'ingest-system1 = ["copilot-mini", "gemini-cheap"]\n'
        'summarize-memory = ["copilot-mini"]\n'
        "\n"
        "[provider_routing.task_classes]\n"
        'ingest-system1 = "cheap_agentic"\n'
        'summarize-memory = "deterministic"\n'
        "\n"
        "[provider_routing.profile_daily_call_limits]\n"
        'copilot-mini = 50\n'
        'gemini-cheap = 20\n'
        "\n"
        "[provider_routing.profiles.copilot-mini]\n"
        'provider = "copilot-cli"\n'
        'model = "gpt-5-mini"\n'
        'max_retries = 0\n'
        "\n"
        "[provider_routing.profiles.gemini-cheap]\n"
        'provider = "gemini-cli"\n'
        'model = "gemini-3-flash-preview"\n'
        'max_retries = 0\n',
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.provider_routing.default_json_route == ["copilot-mini"]
    assert config.provider_routing.default_agentic_route == ["gemini-cheap"]
    assert config.provider_routing.fallback_to_json_only is False
    assert config.provider_routing.low_priority_task_names == ["deduplicator"]
    assert config.provider_routing.model_burst_call_limit == 1
    assert config.provider_routing.model_burst_window_seconds == 600.0
    assert config.provider_routing.task_routes["ingest-system1"] == ["copilot-mini", "gemini-cheap"]
    assert config.provider_routing.task_classes["ingest-system1"] == "cheap_agentic"
    assert config.provider_routing.task_classes["summarize-memory"] == "deterministic"
    assert config.provider_routing.profile_daily_call_limits["copilot-mini"] == 50
    assert config.provider_routing.profiles["copilot-mini"].provider == "copilot-cli"
    assert config.provider_routing.profiles["copilot-mini"].model == "gpt-5-mini"


def test_load_config_backfills_provider_routing_defaults_for_legacy_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "[ai]\n"
        'provider = "none"\n'
        'model = "gemini-3-flash-preview"\n',
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.provider_routing.task_routes["ingest-system1"] == ["gemini-cheap", "copilot-mini"]
    assert config.provider_routing.task_routes["memory-curator"] == ["gemini-strong", "copilot-strong", "gemini-cheap"]
    assert config.provider_routing.default_json_route == ["gemini-cheap", "copilot-mini"]
    assert config.provider_routing.default_agentic_route == ["gemini-cheap", "copilot-mini"]
    assert config.provider_routing.profiles["gemini-cheap"].provider == "gemini-cli"
    assert config.provider_routing.profiles["copilot-mini"].provider == "copilot-cli"


def test_load_config_merges_partial_provider_routing_override_with_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "[provider_routing]\n"
        'default_json_route = ["copilot-mini"]\n'
        "\n"
        "[provider_routing.task_routes]\n"
        'memory-curator = ["copilot-strong"]\n',
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.provider_routing.default_json_route == ["copilot-mini"]
    assert config.provider_routing.task_routes["memory-curator"] == ["copilot-strong"]
    assert config.provider_routing.task_routes["ingest-system1"] == ["gemini-cheap", "copilot-mini"]
    assert config.provider_routing.profiles["gemini-cheap"].provider == "gemini-cli"


def test_load_config_reads_ingest_control_overrides(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "[ingest_suppression]\n"
        "enabled = true\n"
        "\n"
        "[[ingest_suppression.windows]]\n"
        "start_hour = 22\n"
        "end_hour = 6\n"
        "days_of_week = [0, 1, 2, 3, 4]\n"
        "\n"
        "[ingest_escalation]\n"
        "enabled = true\n"
        "deterministic_first = true\n"
        "agentic_pending_count_threshold = 12\n"
        "novelty_threshold = 0.75\n"
        "preview_entry_limit = 5\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.ingest_suppression.enabled is True
    assert len(config.ingest_suppression.windows) == 1
    assert config.ingest_suppression.windows[0].start_hour == 22
    assert config.ingest_escalation.agentic_pending_count_threshold == 12
    assert config.ingest_escalation.novelty_threshold == 0.75
    assert config.ingest_escalation.preview_entry_limit == 5


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
