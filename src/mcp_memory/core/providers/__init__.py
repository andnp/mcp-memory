from __future__ import annotations

from typing import Protocol

from mcp_memory.config import AIConfig, GeminiCLIConfig
from mcp_memory.core.providers.gemini_cli import AIResponse, GeminiCLIProvider


class AIProvider(Protocol):
    async def ask(self, prompt: str) -> dict:
        ...


def build_ai_provider(ai_config: AIConfig, gemini_cli: GeminiCLIConfig):
    if ai_config.provider == "none":
        return None
    if ai_config.provider == "gemini-cli":
        return GeminiCLIProvider(
            command=gemini_cli.command,
            model=ai_config.model,
            timeout_seconds=ai_config.timeout_seconds,
            max_retries=ai_config.max_retries,
        )
    raise ValueError(f"Unsupported AI provider: {ai_config.provider}")


__all__ = ["AIProvider", "AIResponse", "GeminiCLIProvider", "build_ai_provider"]
