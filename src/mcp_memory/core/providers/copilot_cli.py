from __future__ import annotations

import json
from typing import Any

from mcp_memory.core.providers._json_cli import AIResponse
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
            "--stream",
            "off",
            "--silent",
            "--no-ask-user",
            "--no-custom-instructions",
            "--prompt",
            prompt,
        )

    def _parse_response(self, text: str) -> AIResponse:
        content = extract_copilot_message_content(text)
        if content is None:
            return AIResponse(raw_text=text, parsed=None, error="No assistant.message content found")
        parsed = _extract_json_object(content)
        if parsed is None:
            return AIResponse(raw_text=text, parsed=None, error="Assistant message did not contain a JSON object")
        return AIResponse(raw_text=text, parsed=parsed)


def extract_copilot_message_content(text: str) -> str | None:
    stripped = text.strip()
    if not stripped:
        return None
    direct = _extract_json_object(stripped)
    if direct is not None and "type" not in direct:
        return stripped

    content: str | None = None
    json_content: str | None = None
    for line in stripped.splitlines():
        line = line.strip()
        if not line:
            continue
        event = _extract_json_object(line)
        if not isinstance(event, dict):
            continue
        if event.get("type") != "assistant.message":
            continue
        data = event.get("data")
        if not isinstance(data, dict):
            continue
        maybe_content = data.get("content")
        if isinstance(maybe_content, str) and maybe_content.strip():
            content = maybe_content.strip()
            if _extract_json_object(content) is not None:
                json_content = content
    return json_content or content


def _extract_json_object(text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        json_start = text.find("{")
        json_end = text.rfind("}") + 1
        if json_start < 0 or json_end <= json_start:
            return None
        try:
            parsed = json.loads(text[json_start:json_end])
        except json.JSONDecodeError:
            return None
    if isinstance(parsed, dict):
        return parsed
    return None