from __future__ import annotations

import json
from typing import Any

from mcp_memory.core.providers._json_cli import AIResponse
from mcp_memory.core.providers.gemini_cli import GeminiCLIProvider
from mcp_memory.core.providers.interfaces import AgenticRunResult


class GeminiCLIAgenticProvider(GeminiCLIProvider):
    provider_name = "Gemini CLI Agentic"

    def __init__(
        self,
        *,
        command: str = "gemini",
        model: str = "gemini-3-flash-preview",
        timeout_seconds: float = 60.0,
        max_retries: int = 1,
        cwd: str | None = None,
        approval_mode: str = "yolo",
        allowed_mcp_server_names: tuple[str, ...] = ("mcp-memory-internal",),
    ) -> None:
        super().__init__(
            command=command,
            model=model,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            cwd=cwd,
        )
        self._approval_mode = approval_mode
        self._allowed_mcp_server_names = tuple(
            name.strip() for name in allowed_mcp_server_names if isinstance(name, str) and name.strip()
        )

    def build_command(self, prompt: str) -> tuple[str, ...]:
        command = [
            self._command,
            "--model",
            self._model,
            "--prompt",
            prompt,
            "--output-format",
            "json",
            "--approval-mode",
            self._approval_mode,
        ]
        for server_name in self._allowed_mcp_server_names:
            command.extend(["--allowed-mcp-server-names", server_name])
        return tuple(command)

    async def run_agent(self, prompt: str) -> AgenticRunResult:
        last_response: AIResponse | None = None
        for attempt in range(self._max_retries + 1):
            response = await self._execute(prompt, attempt=attempt + 1)
            last_response = response
            if response.success:
                parsed = response.parsed if isinstance(response.parsed, dict) else None
                return AgenticRunResult(
                    status="success",
                    summary=_extract_summary(parsed),
                    raw_text=response.raw_text,
                    parsed=parsed,
                    subprocess_pid=response.subprocess_pid,
                )
        assert last_response is not None
        return AgenticRunResult(
            status="error",
            summary=None,
            raw_text=last_response.raw_text,
            parsed=last_response.parsed if isinstance(last_response.parsed, dict) else None,
            subprocess_pid=last_response.subprocess_pid,
        )


def _extract_summary(parsed: dict[str, Any] | None) -> str | None:
    if not isinstance(parsed, dict):
        return None
    summary = parsed.get("summary")
    if isinstance(summary, str) and summary.strip():
        return summary.strip()
    result = parsed.get("result")
    if isinstance(result, str) and result.strip():
        return result.strip()
    response = parsed.get("response")
    if isinstance(response, str) and response.strip():
        nested = _extract_json_object(response)
        if isinstance(nested, dict):
            nested_summary = nested.get("summary")
            if isinstance(nested_summary, str) and nested_summary.strip():
                return nested_summary.strip()
        return response.strip()
    return None


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