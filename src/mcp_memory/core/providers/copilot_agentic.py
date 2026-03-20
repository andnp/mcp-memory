from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mcp_memory.core.providers._json_cli import AIResponse
from mcp_memory.core.providers._json_cli import build_cli_failure_exception
from mcp_memory.core.providers.copilot_cli import CopilotCLIProvider
from mcp_memory.core.providers.copilot_cli import extract_copilot_message_content
from mcp_memory.core.providers.interfaces import AgenticRunResult


class CopilotCLIAgenticProvider(CopilotCLIProvider):
    provider_name = "Copilot CLI Agentic"
    _DEFAULT_INTERNAL_TOOL_NAMES = (
        "internal_search_memory_records",
        "internal_read_memory_record",
        "internal_list_memory_records",
        "internal_get_next_dedup_batch",
        "internal_get_next_curator_batch",
        "internal_get_next_ingest_batch",
        "internal_ingest_append_memory",
        "internal_ingest_create_memory",
        "internal_append_memory_content",
        "internal_archive_memory_record",
        "internal_merge_memory_into_canonical",
        "internal_split_memory_record",
        "internal_create_memory_record",
        "internal_update_memory_record",
        "internal_delete_memory_record",
        "internal_create_memory_link",
        "internal_delete_memory_link",
    )

    def __init__(
        self,
        *,
        command: str = "copilot",
        model: str = "gpt-5.4",
        timeout_seconds: float = 60.0,
        max_retries: int = 1,
        cwd: str | None = None,
        max_autopilot_continues: int = 12,
        internal_server_name: str = "mcp-memory-internal",
        allowed_tool_names: tuple[str, ...] | None = None,
    ) -> None:
        super().__init__(
            command=command,
            model=model,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            cwd=cwd,
        )
        self._max_autopilot_continues = max_autopilot_continues
        self._internal_server_name = internal_server_name
        self._allowed_tool_names = tuple(allowed_tool_names or self._DEFAULT_INTERNAL_TOOL_NAMES)

    def build_command(self, prompt: str) -> tuple[str, ...]:
        command = [
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
            "--autopilot",
            "--allow-all-tools",
            "--max-autopilot-continues",
            str(self._max_autopilot_continues),
            "--available-tools",
            self._available_tools_value(),
            "--additional-mcp-config",
            json.dumps(self._mcp_config(), separators=(",", ":"), sort_keys=True),
            "--prompt",
            prompt,
        ]
        return tuple(command)

    async def run_agent(self, prompt: str) -> AgenticRunResult:
        last_response: AIResponse | None = None
        for attempt in range(self._max_retries + 1):
            response = await self._execute(prompt, attempt=attempt + 1)
            last_response = response
            if response.error is not None:
                continue
            content = extract_copilot_message_content(response.raw_text)
            parsed = _extract_agentic_payload(content)
            return AgenticRunResult(
                status="success",
                summary=_extract_summary(content, parsed),
                raw_text=response.raw_text,
                parsed=parsed,
                subprocess_pid=response.subprocess_pid,
            )
        assert last_response is not None
        raise build_cli_failure_exception(
            self.provider_name,
            self._max_retries + 1,
            last_response.error,
        )

    def _mcp_config(self) -> dict[str, object]:
        workspace_root = self._cwd or str(Path.cwd())
        return {
            "mcpServers": {
                self._internal_server_name: {
                    "type": "stdio",
                    "command": "uv",
                    "args": ["run", "mcp-memory", "internal-run", "--workspace-root", workspace_root],
                    "cwd": workspace_root,
                    "env": {},
                    "tools": ["*"],
                    "timeout": int(max(self._timeout_seconds, 1.0) * 1000),
                }
            }
        }

    def _available_tools_value(self) -> str:
        return ",".join(f"{self._internal_server_name}-{tool_name}" for tool_name in self._allowed_tool_names)


def _extract_agentic_payload(content: str | None) -> dict[str, Any] | None:
    if not isinstance(content, str) or not content.strip():
        return None
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        json_start = content.find("{")
        json_end = content.rfind("}") + 1
        if json_start < 0 or json_end <= json_start:
            return None
        try:
            parsed = json.loads(content[json_start:json_end])
        except json.JSONDecodeError:
            return None
    if isinstance(parsed, dict):
        return parsed
    return None


def _extract_summary(content: str | None, parsed: dict[str, Any] | None) -> str | None:
    if isinstance(parsed, dict):
        for key in ("summary", "result", "response"):
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    if isinstance(content, str) and content.strip():
        return content.strip()
    return None