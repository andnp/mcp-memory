from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from mcp_memory.core.providers.interfaces import ProviderObserver
from mcp_memory.core.providers.interfaces import ProviderObserverEvent
from mcp_memory.core.providers.interfaces import ProviderTokenUsage


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
    parsed: dict[str, Any] | None
    error: str | None = None
    subprocess_pid: int | None = None
    returncode: int | None = None
    token_usage: ProviderTokenUsage | None = None

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


def chain_observers(*observers: ProviderObserver) -> ProviderObserver:
    def _notify(payload: ProviderObserverEvent) -> None:
        for observer in observers:
            observer(payload)

    return _notify


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
