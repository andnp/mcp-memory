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


class JSONTaskProvider(Protocol):
    async def ask_json(self, prompt: str) -> dict:
        ...


class AgenticTaskProvider(Protocol):
    async def run_agent(self, prompt: str) -> AgenticRunResult:
        ...