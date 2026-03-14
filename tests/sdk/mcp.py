from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FakeAsyncContextManager:
    payload: tuple[Any, Any]

    async def __aenter__(self) -> tuple[Any, Any]:
        return self.payload

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


@dataclass
class FakeToolRuntime:
    tools: list[Any] = field(default_factory=list)
    responses: dict[str, list[Any]] = field(default_factory=dict)
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def get_memory_tools(self) -> list[Any]:
        return list(self.tools)

    async def call_memory_tool(self, ctx: Any, name: str, arguments: dict[str, Any]) -> list[Any]:
        self.calls.append((name, arguments))
        return list(self.responses.get(name, []))