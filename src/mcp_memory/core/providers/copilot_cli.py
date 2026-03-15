from __future__ import annotations

from mcp_memory.core.providers._json_cli import JSONCLIProvider


class CopilotCLIProvider(JSONCLIProvider):
    provider_name = "Copilot CLI"

    def __init__(
        self,
        *,
        command: str = "copilot",
        model: str = "gpt-5.4",
        timeout_seconds: float = 60.0,
        max_retries: int = 1,
    ) -> None:
        super().__init__(
            command=command,
            model=model,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )

    def build_command(self, prompt: str) -> tuple[str, ...]:
        return (
            self._command,
            "ask",
            "--model",
            self._model,
            "--json",
            prompt,
        )