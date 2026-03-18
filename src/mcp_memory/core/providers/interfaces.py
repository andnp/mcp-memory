from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(slots=True)
class AgenticRunResult:
    status: str
    summary: str | None = None
    raw_text: str | None = None
    parsed: dict | None = None
    subprocess_pid: int | None = None


class ProviderBudgetExceeded(RuntimeError):
    def __init__(self, provider_key: str, *, calls_last_day: int, daily_call_limit: int) -> None:
        super().__init__(
            f"provider_budget_exceeded: provider_key={provider_key} calls_last_day={calls_last_day} daily_call_limit={daily_call_limit}"
        )
        self.provider_key = provider_key
        self.calls_last_day = calls_last_day
        self.daily_call_limit = daily_call_limit


class JSONTaskProvider(Protocol):
    async def ask_json(self, prompt: str) -> dict:
        ...


class AgenticTaskProvider(Protocol):
    async def run_agent(self, prompt: str) -> AgenticRunResult:
        ...
