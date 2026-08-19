"""Provider-neutral dispatch capabilities for internal maintenance tools."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from mcp_memory.context import ApplicationContext


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    input_schema: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class ToolResponsePart:
    text: str | None


class ToolDispatchPort(Protocol):
    def available_tools(self) -> Mapping[str, ToolDefinition]: ...

    async def dispatch(
        self,
        ctx: ApplicationContext,
        name: str,
        arguments: dict[str, object],
    ) -> Sequence[ToolResponsePart]: ...
