from __future__ import annotations

from pathlib import Path
from typing import Protocol

from mcp_memory.config import AIConfig, Config, GeminiCLIConfig
from mcp_memory.core.providers._json_cli import AIResponse
from mcp_memory.core.providers.copilot_cli import CopilotCLIProvider
from mcp_memory.core.providers.gemini_cli import GeminiCLIProvider
from mcp_memory.core.providers.interfaces import AgenticRunResult
from mcp_memory.core.providers.interfaces import AgenticTaskProvider
from mcp_memory.core.providers.interfaces import JSONTaskProvider
from mcp_memory.core.providers.ollama_cli import OllamaCLIProvider
from mcp_memory.core.providers.opencode_cli import OpenCodeCLIProvider


class AIProvider(Protocol):
    async def ask(self, prompt: str) -> dict:
        ...


def build_ai_provider(ai_config: AIConfig, gemini_cli: GeminiCLIConfig, workspace_root: Path | None = None):
    if ai_config.provider == "none":
        return None
    if ai_config.provider == "gemini-cli":
        return GeminiCLIProvider(
            command=gemini_cli.command,
            model=ai_config.model,
            timeout_seconds=ai_config.timeout_seconds,
            max_retries=ai_config.max_retries,
            cwd=None if workspace_root is None else str(workspace_root),
        )
    raise ValueError(f"Unsupported AI provider: {ai_config.provider}")


def build_ai_provider_from_config(config: Config, workspace_root: Path | None = None):
    if config.ai.provider == "none":
        return None
    if config.ai.provider == "gemini-cli":
        return build_ai_provider(config.ai, config.gemini_cli, workspace_root)
    if config.ai.provider == "copilot-cli":
        return CopilotCLIProvider(
            command=config.copilot_cli.command,
            model=config.ai.model,
            timeout_seconds=config.ai.timeout_seconds,
            max_retries=config.ai.max_retries,
        )
    if config.ai.provider == "opencode":
        return OpenCodeCLIProvider(
            command=config.opencode.command,
            model=config.ai.model,
            timeout_seconds=config.ai.timeout_seconds,
            max_retries=config.ai.max_retries,
        )
    if config.ai.provider == "ollama":
        return OllamaCLIProvider(
            command=config.ollama.command,
            model=config.ai.model,
            timeout_seconds=config.ai.timeout_seconds,
            max_retries=config.ai.max_retries,
        )
    raise ValueError(f"Unsupported AI provider: {config.ai.provider}")


def build_json_ai_provider_from_config(config: Config, workspace_root: Path | None = None):
    return build_ai_provider_from_config(config, workspace_root)


def build_agentic_ai_provider_from_config(config: Config, workspace_root: Path | None = None):
    del config, workspace_root
    return None


__all__ = [
    "AIProvider",
    "AIResponse",
    "CopilotCLIProvider",
    "GeminiCLIProvider",
    "AgenticRunResult",
    "AgenticTaskProvider",
    "JSONTaskProvider",
    "OllamaCLIProvider",
    "OpenCodeCLIProvider",
    "build_agentic_ai_provider_from_config",
    "build_ai_provider",
    "build_ai_provider_from_config",
    "build_json_ai_provider_from_config",
]
