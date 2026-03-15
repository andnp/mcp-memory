from __future__ import annotations

import json

from mcp_memory.core.providers._json_cli import AIResponse
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
            "--output-format",
            "json",
            "--prompt",
            prompt,
        )

    def _parse_response(self, text: str) -> AIResponse:
        parsed_response = super()._parse_response(text)
        if parsed_response.parsed is None:
            return parsed_response

        envelope = parsed_response.parsed.get("response")
        if not isinstance(envelope, str):
            return parsed_response

        try:
            unwrapped = json.loads(envelope)
        except json.JSONDecodeError:
            return parsed_response
        if not isinstance(unwrapped, dict):
            return parsed_response
        return AIResponse(raw_text=text, parsed=unwrapped)
