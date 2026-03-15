from __future__ import annotations

import asyncio
import copy
import json
import logging
from dataclasses import dataclass
import time
from typing import Any, Callable


logger = logging.getLogger(__name__)


@dataclass
class AIResponse:
    raw_text: str
    parsed: dict | None
    error: str | None = None
    subprocess_pid: int | None = None
    returncode: int | None = None

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
        observer: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._command = command
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries
        self._cwd = cwd
        self._observer = observer

    def with_observer(self, observer: Callable[[dict[str, Any]], None]):
        clone = copy.copy(self)
        existing = getattr(self, "_observer", None)
        if existing is None:
            clone._observer = observer
        else:
            clone._observer = _chain_observers(existing, observer)
        return clone

    async def ask(self, prompt: str) -> dict:
        last_error: str | None = None
        for attempt in range(self._max_retries + 1):
            response = await self._execute(prompt, attempt=attempt + 1)
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

    async def _execute(self, prompt: str, *, attempt: int) -> AIResponse:
        started_at = time.time()
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
            self._notify(
                {
                    "event": "started",
                    "attempt": attempt,
                    "prompt": prompt,
                    "subprocess_pid": proc.pid,
                    "started_at": started_at,
                }
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=self._timeout_seconds,
                )
            except asyncio.CancelledError:
                proc.kill()
                await proc.wait()
                completed_at = time.time()
                self._notify(
                    {
                        "event": "finished",
                        "attempt": attempt,
                        "status": "cancelled",
                        "prompt": prompt,
                        "subprocess_pid": proc.pid,
                        "returncode": proc.returncode,
                        "raw_text": "",
                        "parsed": None,
                        "error": "Command cancelled",
                        "started_at": started_at,
                        "completed_at": completed_at,
                        "duration_seconds": max(completed_at - started_at, 0.0),
                    }
                )
                raise
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                completed_at = time.time()
                response = AIResponse(raw_text="", parsed=None, error="Command timed out", subprocess_pid=proc.pid)
                self._notify(
                    {
                        "event": "finished",
                        "attempt": attempt,
                        "status": "timeout",
                        "prompt": prompt,
                        "subprocess_pid": proc.pid,
                        "returncode": proc.returncode,
                        "raw_text": response.raw_text,
                        "parsed": response.parsed,
                        "error": response.error,
                        "started_at": started_at,
                        "completed_at": completed_at,
                        "duration_seconds": max(completed_at - started_at, 0.0),
                    }
                )
                return response

            stdout_text = stdout_bytes.decode("utf-8", errors="replace").strip()
            stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip()
            if proc.returncode != 0:
                response = AIResponse(
                    raw_text=stdout_text,
                    parsed=None,
                    error=f"Exit code {proc.returncode}: {stderr_text or stdout_text}",
                    subprocess_pid=proc.pid,
                    returncode=proc.returncode,
                )
                completed_at = time.time()
                self._notify(
                    {
                        "event": "finished",
                        "attempt": attempt,
                        "status": "error",
                        "prompt": prompt,
                        "subprocess_pid": proc.pid,
                        "returncode": proc.returncode,
                        "raw_text": response.raw_text,
                        "parsed": response.parsed,
                        "error": response.error,
                        "started_at": started_at,
                        "completed_at": completed_at,
                        "duration_seconds": max(completed_at - started_at, 0.0),
                    }
                )
                return response
            response = self._parse_response(stdout_text)
            response.subprocess_pid = proc.pid
            response.returncode = proc.returncode
            completed_at = time.time()
            self._notify(
                {
                    "event": "finished",
                    "attempt": attempt,
                    "status": "success" if response.success else "parse_error",
                    "prompt": prompt,
                    "subprocess_pid": proc.pid,
                    "returncode": proc.returncode,
                    "raw_text": response.raw_text,
                    "parsed": response.parsed,
                    "error": response.error,
                    "started_at": started_at,
                    "completed_at": completed_at,
                    "duration_seconds": max(completed_at - started_at, 0.0),
                }
            )
            return response
        except FileNotFoundError:
            completed_at = time.time()
            response = AIResponse(raw_text="", parsed=None, error=f"Command not found: {self._command}")
            self._notify(
                {
                    "event": "finished",
                    "attempt": attempt,
                    "status": "error",
                    "prompt": prompt,
                    "subprocess_pid": None,
                    "returncode": None,
                    "raw_text": response.raw_text,
                    "parsed": response.parsed,
                    "error": response.error,
                    "started_at": started_at,
                    "completed_at": completed_at,
                    "duration_seconds": max(completed_at - started_at, 0.0),
                }
            )
            return response
        except OSError as exc:
            completed_at = time.time()
            response = AIResponse(raw_text="", parsed=None, error=f"OS error: {exc}")
            self._notify(
                {
                    "event": "finished",
                    "attempt": attempt,
                    "status": "error",
                    "prompt": prompt,
                    "subprocess_pid": None,
                    "returncode": None,
                    "raw_text": response.raw_text,
                    "parsed": response.parsed,
                    "error": response.error,
                    "started_at": started_at,
                    "completed_at": completed_at,
                    "duration_seconds": max(completed_at - started_at, 0.0),
                }
            )
            return response

    def _notify(self, payload: dict[str, Any]) -> None:
        observer = getattr(self, "_observer", None)
        if observer is None:
            return
        try:
            observer(payload)
        except Exception:
            logger.exception("Provider observer failed")

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


def _chain_observers(*observers: Callable[[dict[str, Any]], None]):
    def _notify(payload: dict[str, Any]) -> None:
        for observer in observers:
            observer(payload)

    return _notify