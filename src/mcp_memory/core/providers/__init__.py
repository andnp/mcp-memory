from __future__ import annotations

from pathlib import Path
from typing import Protocol

from mcp_memory.config import AIConfig
from mcp_memory.core.providers._json_cli import AIResponse
from mcp_memory.core.providers.copilot_sdk import CopilotSDKAgenticProvider
from mcp_memory.core.providers.copilot_sdk import CopilotSDKProvider
from mcp_memory.core.providers.interfaces import AgenticRunResult
from mcp_memory.core.providers.interfaces import AgenticTaskProvider
from mcp_memory.core.providers.interfaces import JSONTaskProvider
from mcp_memory.core.providers.interfaces import ProviderJSONCall


class AIProvider(Protocol):
    async def ask(self, prompt: str) -> dict:
        ...


def build_json_ai_provider(ai_config: AIConfig, workspace_root: Path | None = None):
    if ai_config.provider == "none":
        return None
    if ai_config.provider == "copilot-sdk":
        return CopilotSDKProvider(
            model=ai_config.model,
            timeout_seconds=ai_config.timeout_seconds,
            max_retries=ai_config.max_retries,
            cwd=None if workspace_root is None else str(workspace_root),
        )
    raise ValueError(f"Unsupported AI provider: {ai_config.provider}")


def build_agentic_ai_provider(ai_config: AIConfig, workspace_root: Path | None = None):
    if ai_config.provider == "copilot-sdk":
        return CopilotSDKAgenticProvider(
            model=ai_config.model,
            timeout_seconds=ai_config.timeout_seconds,
            max_retries=ai_config.max_retries,
            cwd=None if workspace_root is None else str(workspace_root),
        )
    return None


__all__ = [
    "AIProvider",
    "AIResponse",
    "CopilotSDKAgenticProvider",
    "CopilotSDKProvider",
    "AgenticRunResult",
    "AgenticTaskProvider",
    "JSONTaskProvider",
    "ProviderJSONCall",
    "build_agentic_ai_provider",
    "build_json_ai_provider",
]
