from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class AIResponse:
    raw_text: str
    parsed: dict | None
    error: str | None = None

    @property
    def success(self) -> bool:
        return self.parsed is not None and self.error is None


class JSONCLIProvider:
    provider_name = "JSON CLI"

    def __init__(
        self,
        *,
        command: str,
        model: str,
        timeout_seconds: float,
        max_retries: int,
        cwd: str | None = None,
    ) -> None:
        self._command = command
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._cwd = cwd

    async def ask(self, prompt: str) -> dict:
        last_error: str | None = None
        for attempt in range(self._max_retries + 1):
            response = await self._execute(prompt)
            if response.success:
                assert response.parsed is not None
                return response.parsed
            last_error = response.error
            if attempt < self._max_retries:
                logger.warning(
                    "%s attempt %d failed: %s. Retrying...",
                    self.provider_name,
                    attempt + 1,
                    last_error,
                )
        raise RuntimeError(
            f"{self.provider_name} failed after {self._max_retries + 1} attempts: {last_error}"
        )

    def build_command(self, prompt: str) -> tuple[str, ...]:
        raise NotImplementedError

    async def _execute(self, prompt: str) -> AIResponse:
        try:
            if self._cwd is None:
                proc = await asyncio.create_subprocess_exec(
                    *self.build_command(prompt),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            else:
                proc = await asyncio.create_subprocess_exec(
                    *self.build_command(prompt),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=self._cwd,
                )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=self._timeout_seconds,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return AIResponse(raw_text="", parsed=None, error="Command timed out")

            stdout_text = stdout_bytes.decode("utf-8", errors="replace").strip()
            stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip()
            if proc.returncode != 0:
                return AIResponse(
                    raw_text=stdout_text,
                    parsed=None,
                    error=f"Exit code {proc.returncode}: {stderr_text or stdout_text}",
                )
            return self._parse_response(stdout_text)
        except FileNotFoundError:
            return AIResponse(raw_text="", parsed=None, error=f"Command not found: {self._command}")
        except OSError as exc:
            return AIResponse(raw_text="", parsed=None, error=f"OS error: {exc}")

    def _parse_response(self, text: str) -> AIResponse:
        if not text:
            return AIResponse(raw_text=text, parsed=None, error="Empty response")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            json_start = text.find("{")
            json_end = text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                try:
                    parsed = json.loads(text[json_start:json_end])
                except json.JSONDecodeError:
                    return AIResponse(raw_text=text, parsed=None, error=f"Invalid JSON: {exc}")
            else:
                return AIResponse(raw_text=text, parsed=None, error=f"Invalid JSON: {exc}")

        if not isinstance(parsed, dict):
            return AIResponse(
                raw_text=text,
                parsed=None,
                error=f"Expected JSON object, got {type(parsed).__name__}",
            )
        return AIResponse(raw_text=text, parsed=parsed)