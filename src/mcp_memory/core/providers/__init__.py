from __future__ import annotations

from pathlib import Path
from typing import Protocol

from mcp_memory.config import AIConfig, Config
from mcp_memory.core.providers._json_cli import AIResponse
from mcp_memory.core.providers.copilot import CopilotCLIAgenticProvider
from mcp_memory.core.providers.copilot import CopilotCLIProvider
from mcp_memory.core.providers.gemini import GeminiCLIAgenticProvider
from mcp_memory.core.providers.gemini import GeminiCLIProvider
from mcp_memory.core.providers.interfaces import AgenticRunResult
from mcp_memory.core.providers.interfaces import AgenticTaskProvider
from mcp_memory.core.providers.interfaces import JSONTaskProvider
from mcp_memory.core.providers.small_json_cli import OllamaCLIProvider
from mcp_memory.core.providers.small_json_cli import OpenCodeCLIProvider


class AIProvider(Protocol):
    async def ask(self, prompt: str) -> dict:
        ...


def build_json_ai_provider(ai_config: AIConfig, config: Config, workspace_root: Path | None = None):
    if ai_config.provider == "none":
        return None
    if ai_config.provider == "gemini-cli":
        return GeminiCLIProvider(
            command=config.gemini_cli.command,
            model=ai_config.model,
            timeout_seconds=ai_config.timeout_seconds,
            max_retries=ai_config.max_retries,
            cwd=None if workspace_root is None else str(workspace_root),
        )
    if ai_config.provider == "copilot-cli":
        return CopilotCLIProvider(
            command=config.copilot_cli.command,
            model=ai_config.model,
            timeout_seconds=ai_config.timeout_seconds,
            max_retries=ai_config.max_retries,
            cwd=None if workspace_root is None else str(workspace_root),
        )
    if ai_config.provider == "opencode":
        return OpenCodeCLIProvider(
            command=config.opencode.command,
            model=ai_config.model,
            timeout_seconds=ai_config.timeout_seconds,
            max_retries=ai_config.max_retries,
        )
    if ai_config.provider == "ollama":
        return OllamaCLIProvider(
            command=config.ollama.command,
            model=ai_config.model,
            timeout_seconds=ai_config.timeout_seconds,
            max_retries=ai_config.max_retries,
        )
    raise ValueError(f"Unsupported AI provider: {ai_config.provider}")


def build_agentic_ai_provider(ai_config: AIConfig, config: Config, workspace_root: Path | None = None):
    if ai_config.provider == "gemini-cli":
        return GeminiCLIAgenticProvider(
            command=config.gemini_cli.command,
            model=ai_config.model,
            timeout_seconds=ai_config.timeout_seconds,
            max_retries=ai_config.max_retries,
            cwd=None if workspace_root is None else str(workspace_root),
        )
    if ai_config.provider == "copilot-cli":
        return CopilotCLIAgenticProvider(
            command=config.copilot_cli.command,
            model=ai_config.model,
            timeout_seconds=ai_config.timeout_seconds,
            max_retries=ai_config.max_retries,
            cwd=None if workspace_root is None else str(workspace_root),
        )
    return None


__all__ = [
    "AIProvider",
    "AIResponse",
    "CopilotCLIAgenticProvider",
    "CopilotCLIProvider",
    "GeminiCLIAgenticProvider",
    "GeminiCLIProvider",
    "AgenticRunResult",
    "AgenticTaskProvider",
    "JSONTaskProvider",
    "OllamaCLIProvider",
    "OpenCodeCLIProvider",
    "build_agentic_ai_provider",
    "build_json_ai_provider",
]
