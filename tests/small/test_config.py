from pathlib import Path

import pytest

from mcp_memory.config import (
    AIConfig,
    BackupsConfig,
    DaemonConfig,
    MemoryConfig,
    SearchRankingConfig,
    SearchKernelConfig,
    ensure_default_config_exists,
    load_config,
    resolve_default_config_path,
    resolve_global_data_dir,
    resolve_memory_path,
    resolve_state_dir,
)


pytestmark = pytest.mark.small


def test_ai_config_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="ai.provider"):
        AIConfig(provider="mystery")


def test_memory_config_rejects_invalid_checkpoint_interval() -> None:
    with pytest.raises(ValueError, match="checkpoint_interval_ops"):
        MemoryConfig(checkpoint_interval_ops=0)


def test_searchkernel_defaults_to_lenient_failure_mode() -> None:
    config = SearchKernelConfig()
    assert config.failure_mode == "lenient"


def test_searchkernel_rejects_invalid_failure_mode() -> None:
    with pytest.raises(ValueError, match="searchkernel.failure_mode"):
        SearchKernelConfig(failure_mode="mystery")


def test_default_config_is_created_once(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"

    created = ensure_default_config_exists(config_path)
    loaded = load_config(config_path)

    assert created == config_path
    assert config_path.exists()
    assert loaded.provider_routing.task_routes["ingest-system1"] == ["copilot-mini"]
    assert loaded.provider_routing.task_routes["deduplicator"] == ["copilot-mini"]
    assert loaded.provider_routing.task_routes["memory-curator"] == ["copilot-strong", "copilot-mini"]
    assert loaded.provider_routing.fallback_to_json_only is False
    assert loaded.provider_routing.task_classes["memory-curator"] == "premium_agentic"
    assert loaded.provider_routing.task_classes["summarize-memory"] == "deterministic"
    assert loaded.provider_routing.low_priority_task_names == ["graph-linker", "conflict-detector", "defragmenter", "taxonomist"]
    assert loaded.provider_routing.profile_daily_call_limits["copilot-strong"] == 20
    assert loaded.provider_routing.profile_daily_call_limits["copilot-mini"] == 50
    assert loaded.provider_routing.model_burst_call_limit == 2
    assert loaded.provider_routing.model_burst_window_seconds == 300.0
    assert loaded.provider_routing.profiles["copilot-strong"].provider == "copilot-sdk"
    assert loaded.provider_routing.profiles["copilot-strong"].model == "gpt-5.4-mini"
    assert loaded.provider_routing.profiles["copilot-mini"].provider == "copilot-sdk"
    assert loaded.provider_routing.profiles["copilot-mini"].model == "gpt-5-mini"
    assert loaded.storage.backend == "sqlite"
    assert loaded.storage.sqlite.path == ""
    assert loaded.storage.postgres.pool_min == 1
    assert loaded.storage.postgres.pool_max == 10
    assert loaded.storage.cache.enabled is False
    assert loaded.storage.cache.mode == "readonly"
    assert loaded.daemon.port == 4242
    assert loaded.backups.enabled is True
    assert loaded.backups.interval_seconds == 3600.0
    assert loaded.backups.max_snapshots == 24
    assert loaded.embeddings.provider == "sentence-transformers"
    assert loaded.embeddings.model == "sentence-transformers/all-MiniLM-L6-v2"
    assert loaded.embeddings.ollama_base_url == "http://localhost:11434"
    assert loaded.search_ranking.rrf_k == 60.0
    assert loaded.search_ranking.adaptive_result_max == 15
    assert loaded.search_ranking.adaptive_result_score_ratio_floor == 0.7
    assert loaded.search_ranking.adaptive_result_min_score == 0.35
    assert loaded.search_ranking.adaptive_result_max_score_gap == 0.08
    assert loaded.searchkernel.failure_mode == "lenient"
    assert loaded.memory.recency_plan.max_boost_amount == 0.15
    assert loaded.memory.recency_plan.boost_decay_rate == 0.97
    assert loaded.memory.recency_fact.max_boost_amount == 0.05
    assert loaded.memory.recency_reflection.max_boost_amount == 0.05


def test_default_paths_resolve_inside_the_test_temp_area(tmp_path: Path) -> None:
    assert resolve_default_config_path() == tmp_path / "home" / ".config" / "mcp-memory" / "config.toml"
    assert resolve_global_data_dir() == tmp_path / "data"
    assert resolve_state_dir() == tmp_path / "state" / "mcp-memory"


def test_default_test_config_uses_ephemeral_daemon_port() -> None:
    assert load_config().daemon.port == 0


def test_load_config_ignores_legacy_searchkernel_keys(tmp_path: Path) -> None:
    config_path = tmp_path / "legacy-searchkernel-config.toml"
    config_path.write_text(
        """
[searchkernel_shadow]
enabled = true

[searchkernel_cutover]
enabled = true
failure_mode = "strict"
""",
        encoding="utf-8",
    )

    loaded = load_config(config_path)

    assert loaded.searchkernel.failure_mode == "lenient"
    assert not hasattr(loaded, "searchkernel_shadow")
    assert not hasattr(loaded, "searchkernel_cutover")


def test_daemon_config_rejects_invalid_port() -> None:
    with pytest.raises(ValueError, match="daemon.port"):
        DaemonConfig(port=-1)


def test_backups_config_rejects_invalid_interval() -> None:
    with pytest.raises(ValueError, match="backups.interval_seconds"):
        BackupsConfig(interval_seconds=0)


def test_search_ranking_config_rejects_invalid_rrf_k() -> None:
    with pytest.raises(ValueError, match="search_ranking.rrf_k"):
        SearchRankingConfig(rrf_k=0)


def test_search_ranking_config_rejects_invalid_semantic_only_abstain_threshold() -> None:
    with pytest.raises(ValueError, match="search_ranking.semantic_only_abstain_threshold"):
        SearchRankingConfig(semantic_only_abstain_threshold=1.5)


def test_search_ranking_config_rejects_invalid_graph_expansion_only_penalty() -> None:
    with pytest.raises(ValueError, match="search_ranking.graph_expansion_only_penalty"):
        SearchRankingConfig(graph_expansion_only_penalty=1.5)


def test_search_ranking_config_rejects_invalid_adaptive_result_max() -> None:
    with pytest.raises(ValueError, match="search_ranking.adaptive_result_max"):
        SearchRankingConfig(adaptive_result_max=0)


def test_storage_config_rejects_unknown_backend(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid-storage-config.toml"
    config_path.write_text(
        """
[storage]
backend = "mystery"
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="storage.backend"):
        load_config(config_path)


def test_load_config_rejects_postgres_backend_without_dsn(tmp_path: Path) -> None:
    config_path = tmp_path / "postgres-without-dsn.toml"
    config_path.write_text(
        """
[storage]
backend = "postgres"
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="storage.postgres.dsn"):
        load_config(config_path)


def test_load_config_rejects_postgres_backend_with_non_postgres_scheme(tmp_path: Path) -> None:
    config_path = tmp_path / "postgres-invalid-scheme.toml"
    config_path.write_text(
        """
[storage]
backend = "postgres"

[storage.postgres]
dsn = "sqlite:///tmp/memory.db"
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="postgres:// or postgresql://"):
        load_config(config_path)


def test_resolve_memory_path_uses_storage_sqlite_path(tmp_path: Path) -> None:
    configured_root = tmp_path / "custom-memories"
    config_path = tmp_path / "config.toml"
    config_path.write_text(
    f"""
[storage]
backend = "sqlite"

[storage.sqlite]
path = "{configured_root}"
""",
        encoding="utf-8",
    )

    loaded = load_config(config_path)

    assert resolve_memory_path(loaded) == configured_root
