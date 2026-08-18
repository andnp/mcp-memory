from pathlib import Path
from typing import Any

import pytest

from mcp_memory.config import (
    AIConfig,
    BackupsConfig,
    Config,
    DaemonConfig,
    MaintenanceConfig,
    MemoryConfig,
    SearchKernelConfig,
    SearchRankingConfig,
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


def test_ai_config_defaults_to_available_model() -> None:
    assert AIConfig().model == "gpt-5.6-luna"


def test_memory_config_rejects_invalid_checkpoint_interval() -> None:
    with pytest.raises(ValueError, match="checkpoint_interval_ops"):
        MemoryConfig(checkpoint_interval_ops=0)


def test_maintenance_config_has_safe_gc_defaults() -> None:
    config = MaintenanceConfig()

    assert config.archived_memory_retention_days == 90
    assert config.memory_gc_batch_size == 100
    assert config.dangling_link_gc_batch_size == 100
    assert config.memory_gc_mode == "report-only"


def test_load_config_reads_maintenance_settings(tmp_path: Path) -> None:
    config_path = tmp_path / "maintenance-config.toml"
    config_path.write_text(
        """
[maintenance]
archived_memory_retention_days = 30
memory_gc_batch_size = 25
dangling_link_gc_batch_size = 10
memory_gc_mode = "delete"
""",
        encoding="utf-8",
    )

    loaded = load_config(config_path)

    assert loaded.maintenance == MaintenanceConfig(30, 25, 10, "delete")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("archived_memory_retention_days", 0, "archived_memory_retention_days"),
        ("memory_gc_batch_size", 0, "memory_gc_batch_size"),
        ("dangling_link_gc_batch_size", 0, "dangling_link_gc_batch_size"),
        ("memory_gc_mode", "delete-now", "memory_gc_mode"),
    ],
)
def test_maintenance_config_rejects_invalid_values(field: str, value: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        invalid_config: dict[str, Any] = {field: value}
        MaintenanceConfig(**invalid_config)


def test_searchkernel_defaults_to_lenient_failure_mode() -> None:
    """Keep advanced searchkernel policies disabled by default."""
    config = SearchKernelConfig()

    assert config.failure_mode == "lenient"
    assert config.calibrated_fusion_enabled is False
    assert config.query_expansion_enabled is False
    assert config.query_expansion_policy == "vector"
    assert config.rerank_policy == "disabled"
    assert config.rerank_budget == 0
    assert config.active_feature_fingerprint() is None


def test_searchkernel_loads_advanced_policy_settings(tmp_path: Path) -> None:
    """Load explicitly enabled searchkernel policies from TOML."""
    config_path = tmp_path / "advanced-searchkernel-config.toml"
    config_path.write_text(
        """
[searchkernel]
calibrated_fusion_enabled = true
query_expansion_enabled = true
query_expansion_policy = "synonym"
rerank_policy = "default"
rerank_budget = 4
""",
        encoding="utf-8",
    )

    loaded = load_config(config_path)

    assert loaded.searchkernel.calibrated_fusion_enabled is True
    assert loaded.searchkernel.query_expansion_enabled is True
    assert loaded.searchkernel.query_expansion_policy == "synonym"
    assert loaded.searchkernel.rerank_policy == "default"
    assert loaded.searchkernel.rerank_budget == 4
    assert loaded.searchkernel.active_feature_fingerprint() == (
        "calibrated-fusion;query-expansion:synonym;rerank:default:4"
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("query_expansion_policy", "llm", "query_expansion_policy"),
        ("rerank_policy", "cross-encoder", "rerank_policy"),
        ("rerank_budget", -1, "rerank_budget"),
    ],
)
def test_searchkernel_rejects_invalid_advanced_policy_settings(
    field: str, value: object, message: str
) -> None:
    """Reject unsupported or unsafe advanced searchkernel values."""
    with pytest.raises(ValueError, match=message):
        invalid_config: dict[str, Any] = {field: value}
        SearchKernelConfig(**invalid_config)


def test_searchkernel_rejects_budget_for_disabled_reranking() -> None:
    """Prevent an inactive reranking policy from carrying a budget."""
    with pytest.raises(ValueError, match="rerank_budget"):
        SearchKernelConfig(rerank_budget=2)


def test_config_preserves_legacy_ingress_rollout_defaults() -> None:
    """New rollout controls preserve the pre-rollout ingress behavior."""
    config = Config()

    assert config.ingress_evidence_mode == "off"
    assert config.ingress_replay_policy == "legacy"
    assert config.ingress_quality_admission == "disabled"


def test_load_config_reads_ingress_rollout_controls(tmp_path: Path) -> None:
    """Configured ingress rollout controls load from top-level TOML keys."""
    config_path = tmp_path / "ingress-rollout-config.toml"
    config_path.write_text(
        '''
ingress_evidence_mode = "shadow"
ingress_replay_policy = "detect"
ingress_quality_admission = "canary"
''',
        encoding="utf-8",
    )

    loaded = load_config(config_path)

    assert loaded.ingress_evidence_mode == "shadow"
    assert loaded.ingress_replay_policy == "detect"
    assert loaded.ingress_quality_admission == "canary"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("ingress_evidence_mode", "observe", "ingress_evidence_mode"),
        ("ingress_replay_policy", "always", "ingress_replay_policy"),
        ("ingress_quality_admission", "automatic", "ingress_quality_admission"),
    ],
)
def test_config_rejects_invalid_ingress_rollout_controls(
    field: str, value: object, message: str
) -> None:
    """Invalid rollout control values fail configuration validation."""
    with pytest.raises(ValueError, match=message):
        invalid_config: dict[str, Any] = {field: value}
        Config(**invalid_config)


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
    assert loaded.provider_routing.profiles["copilot-strong"].model == "gpt-5.6-luna"
    assert loaded.provider_routing.profiles["copilot-mini"].provider == "copilot-sdk"
    assert loaded.provider_routing.profiles["copilot-mini"].model == "gpt-5.6-luna"
    assert loaded.storage.backend == "sqlite"
    assert loaded.storage.sqlite.path == ""
    assert loaded.storage.postgres.pool_min == 1
    assert loaded.storage.postgres.pool_max == 10
    assert loaded.storage.cache.enabled is False
    assert loaded.storage.cache.mode == "readonly"
    assert loaded.daemon.port == 4242
    assert loaded.maintenance.archived_memory_retention_days == 90
    assert loaded.maintenance.memory_gc_batch_size == 100
    assert loaded.maintenance.dangling_link_gc_batch_size == 100
    assert loaded.maintenance.memory_gc_mode == "report-only"
    assert loaded.backups.enabled is True
    assert loaded.backups.interval_seconds == 3600.0
    assert loaded.backups.max_snapshots == 24
    assert loaded.embeddings.provider == "sentence-transformers"
    assert loaded.embeddings.model == "sentence-transformers/all-MiniLM-L6-v2"
    assert loaded.embeddings.ollama_base_url == "http://localhost:11434"
    assert loaded.embeddings.ollama_max_concurrency == 1
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
