from __future__ import annotations

import asyncio
import copy
import json
import logging
import re
import signal
from dataclasses import dataclass
import time
from typing import Any, Callable


logger = logging.getLogger(__name__)
PROVIDER_SUBPROCESS_HEARTBEAT_SECONDS = 20.0
_RETRY_DELAY_MS_RE = re.compile(r"retryDelayMs:\s*([0-9]+(?:\.[0-9]+)?)")
_RESET_AFTER_RE = re.compile(
    r"reset after\s*(?:(?P<hours>\d+)h)?\s*(?:(?P<minutes>\d+)m)?\s*(?:(?P<seconds>\d+)s)?",
    re.IGNORECASE,
)


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


class ProviderBackoffError(RuntimeError):
    def __init__(self, message: str, *, retry_delay_seconds: float | None = None) -> None:
        super().__init__(message)
        self.retry_delay_seconds = retry_delay_seconds


def looks_like_interactive_auth_prompt(text: str | None) -> bool:
    if not isinstance(text, str):
        return False
    lowered = text.lower()
    return (
        "opening authentication page in your browser" in lowered
        or "do you want to continue? [y/n]" in lowered
        or "interactive authentication required" in lowered
    )


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

    async def ask_json(self, prompt: str) -> dict:
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
        raise build_cli_failure_exception(
            self.provider_name,
            self._max_retries + 1,
            last_error,
        )

    async def ask(self, prompt: str) -> dict:
        return await self.ask_json(prompt)

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
            communicate_task = asyncio.create_task(proc.communicate())
            deadline = time.monotonic() + self._timeout_seconds
            try:
                while True:
                    remaining_seconds = deadline - time.monotonic()
                    if remaining_seconds <= 0:
                        raise asyncio.TimeoutError
                    try:
                        stdout_bytes, stderr_bytes = await asyncio.wait_for(
                            asyncio.shield(communicate_task),
                            timeout=min(PROVIDER_SUBPROCESS_HEARTBEAT_SECONDS, remaining_seconds),
                        )
                        break
                    except asyncio.TimeoutError:
                        if communicate_task.done():
                            stdout_bytes, stderr_bytes = await communicate_task
                            break
                        heartbeat_at = time.time()
                        self._notify(
                            {
                                "event": "heartbeat",
                                "attempt": attempt,
                                "prompt": prompt,
                                "subprocess_pid": proc.pid,
                                "started_at": started_at,
                                "heartbeat_at": heartbeat_at,
                                "elapsed_seconds": max(heartbeat_at - started_at, 0.0),
                            }
                        )
            except asyncio.CancelledError:
                communicate_task.cancel()
                proc.kill()
                await proc.wait()
                await asyncio.gather(communicate_task, return_exceptions=True)
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
                communicate_task.cancel()
                proc.kill()
                await proc.wait()
                await asyncio.gather(communicate_task, return_exceptions=True)
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
            returncode = proc.returncode if proc.returncode is not None else 0
            if returncode != 0:
                response = AIResponse(
                    raw_text=stdout_text,
                    parsed=None,
                    error=_format_subprocess_failure(
                        returncode,
                        stderr_text=stderr_text,
                        stdout_text=stdout_text,
                    ),
                    subprocess_pid=proc.pid,
                    returncode=returncode,
                )
                completed_at = time.time()
                self._notify(
                    {
                        "event": "finished",
                        "attempt": attempt,
                        "status": "error",
                        "prompt": prompt,
                        "subprocess_pid": proc.pid,
                        "returncode": returncode,
                        "raw_text": response.raw_text,
                        "parsed": response.parsed,
                        "error": response.error,
                        "started_at": started_at,
                        "completed_at": completed_at,
                        "duration_seconds": max(completed_at - started_at, 0.0),
                    }
                )
                return response
            if looks_like_interactive_auth_prompt(stdout_text) or looks_like_interactive_auth_prompt(stderr_text):
                response = AIResponse(
                    raw_text=stdout_text or stderr_text,
                    parsed=None,
                    error="Interactive authentication required",
                    subprocess_pid=proc.pid,
                    returncode=returncode,
                )
                completed_at = time.time()
                self._notify(
                    {
                        "event": "finished",
                        "attempt": attempt,
                        "status": "auth_required",
                        "prompt": prompt,
                        "subprocess_pid": proc.pid,
                        "returncode": returncode,
                        "raw_text": response.raw_text,
                        "parsed": response.parsed,
                        "error": response.error,
                        "reason_category": "auth",
                        "reason_code": "interactive_auth_required",
                        "started_at": started_at,
                        "completed_at": completed_at,
                        "duration_seconds": max(completed_at - started_at, 0.0),
                    }
                )
                return response
            response = self._parse_response(stdout_text)
            response.subprocess_pid = proc.pid
            response.returncode = returncode
            completed_at = time.time()
            self._notify(
                {
                    "event": "finished",
                    "attempt": attempt,
                    "status": "success" if response.success else "parse_error",
                    "prompt": prompt,
                    "subprocess_pid": proc.pid,
                    "returncode": returncode,
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


def _format_subprocess_failure(
    returncode: int,
    *,
    stderr_text: str,
    stdout_text: str,
) -> str:
    output = stderr_text or stdout_text
    if returncode < 0:
        signal_number = -returncode
        try:
            signal_label = f"signal {signal.Signals(signal_number).name} ({signal_number})"
        except ValueError:
            signal_label = f"signal {signal_number}"
        prefix = f"Terminated by {signal_label}"
        return f"{prefix}: {output}" if output else prefix
    prefix = f"Exit code {returncode}"
    return f"{prefix}: {output}" if output else prefix


def build_cli_failure_exception(
    provider_name: str,
    attempts: int,
    last_error: str | None,
) -> RuntimeError:
    from mcp_memory.core.providers.interfaces import ProviderAuthenticationRequired

    if looks_like_interactive_auth_prompt(last_error):
        return ProviderAuthenticationRequired(provider_name, error_text=last_error)
    message = f"{provider_name} failed after {attempts} attempts: {last_error}"
    retry_delay_seconds = recommended_retry_delay_seconds(last_error)
    if retry_delay_seconds is None:
        return RuntimeError(message)
    return ProviderBackoffError(message, retry_delay_seconds=retry_delay_seconds)


def recommended_retry_delay_seconds(error_text: str | None) -> float | None:
    if not error_text:
        return None
    retry_delay_match = _RETRY_DELAY_MS_RE.search(error_text)
    if retry_delay_match is not None:
        return max(float(retry_delay_match.group(1)) / 1000.0, 0.0)
    if "QUOTA_EXHAUSTED" not in error_text and "quota will reset after" not in error_text.lower():
        return None
    reset_match = _RESET_AFTER_RE.search(error_text)
    if reset_match is None:
        return 900.0
    hours = int(reset_match.group("hours") or 0)
    minutes = int(reset_match.group("minutes") or 0)
    seconds = int(reset_match.group("seconds") or 0)
    total_seconds = (hours * 3600) + (minutes * 60) + seconds
    return float(total_seconds if total_seconds > 0 else 900)
