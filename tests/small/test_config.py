from pathlib import Path

import pytest

from mcp_memory.config import AIConfig, MemoryConfig, ensure_default_config_exists, load_config


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
    assert loaded.gemini_cli.command == "gemini"
    assert loaded.copilot_cli.command == "copilot"
    assert loaded.opencode.command == "opencode"
    assert loaded.ollama.command == "ollama"
    assert loaded.embeddings.enabled is False
    assert loaded.embeddings.model == "sentence-transformers/all-MiniLM-L6-v2"
