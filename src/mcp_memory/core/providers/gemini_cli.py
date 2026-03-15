from __future__ import annotations

from mcp_memory.core.providers._json_cli import JSONCLIProvider


class GeminiCLIProvider(JSONCLIProvider):
    provider_name = "Gemini CLI"

    def __init__(
        self,
        *,
        command: str = "gemini",
        model: str = "gemini-3-flash-preview",
        timeout_seconds: float = 60.0,
        max_retries: int = 1,
        cwd: str | None = None,
    ) -> None:
        super().__init__(
            command=command,
            model=model,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            cwd=cwd,
        )

    def build_command(self, prompt: str) -> tuple[str, ...]:
        return (
            self._command,
            "--model",
            self._model,
            "ask",
            "--json",
            prompt,
        )
