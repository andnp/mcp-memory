from pathlib import Path

import pytest

from mcp_memory.config import AIConfig, MemoryConfig, SearchRankingConfig, ensure_default_config_exists, load_config


pytestmark = pytest.mark.small


def test_ai_config_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="ai.provider"):
        AIConfig(provider="mystery")


def test_memory_config_rejects_invalid_checkpoint_interval() -> None:
    with pytest.raises(ValueError, match="checkpoint_interval_ops"):
        MemoryConfig(checkpoint_interval_ops=0)


def test_default_config_is_created_once(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"

    created = ensure_default_config_exists(config_path)
    loaded = load_config(config_path)

    assert created == config_path
    assert config_path.exists()
    assert loaded.ai.provider == "none"
    assert loaded.ai.max_retries == 0
    assert loaded.provider_routing.task_routes["ingest-system1"] == ["copilot-mini", "gemini-cheap"]
    assert loaded.provider_routing.task_routes["deduplicator"] == ["copilot-mini", "gemini-cheap"]
    assert loaded.provider_routing.task_routes["memory-curator"] == ["copilot-strong", "gemini-cheap"]
    assert loaded.provider_routing.fallback_to_json_only is False
    assert loaded.provider_routing.task_classes["memory-curator"] == "premium_agentic"
    assert loaded.provider_routing.task_classes["summarize-memory"] == "deterministic"
    assert loaded.provider_routing.low_priority_task_names == ["graph-linker", "conflict-detector", "defragmenter", "taxonomist"]
    assert loaded.provider_routing.profile_daily_call_limits["copilot-strong"] == 20
    assert loaded.provider_routing.profile_daily_call_limits["copilot-mini"] == 50
    assert loaded.provider_routing.profile_daily_call_limits["gemini-cheap"] == 100
    assert loaded.provider_routing.profiles["copilot-strong"].provider == "copilot-cli"
    assert loaded.provider_routing.profiles["copilot-strong"].model == "gpt-5.4"
    assert loaded.provider_routing.profiles["copilot-mini"].provider == "copilot-cli"
    assert loaded.provider_routing.profiles["copilot-mini"].model == "gpt-5-mini"
    assert loaded.provider_routing.profiles["gemini-cheap"].provider == "gemini-cli"
    assert loaded.provider_routing.profiles["gemini-cheap"].model == "gemini-3-flash-preview"
    assert loaded.gemini_cli.command == "gemini"
    assert loaded.copilot_cli.command == "copilot"
    assert loaded.opencode.command == "opencode"
    assert loaded.ollama.command == "ollama"
    assert loaded.embeddings.model == "sentence-transformers/all-MiniLM-L6-v2"
    assert loaded.search_ranking.rrf_k == 60.0
    assert loaded.memory.recency_plan.max_boost_amount == 0.15
    assert loaded.memory.recency_plan.boost_decay_rate == 0.97
    assert loaded.memory.recency_fact.max_boost_amount == 0.05
    assert loaded.memory.recency_reflection.max_boost_amount == 0.05


def test_search_ranking_config_rejects_invalid_rrf_k() -> None:
    with pytest.raises(ValueError, match="search_ranking.rrf_k"):
        SearchRankingConfig(rrf_k=0)
