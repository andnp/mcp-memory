"""Provider-neutral dispatch capabilities for internal maintenance tools."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, TypeVar

CtxT = TypeVar("CtxT", contravariant=True)


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    input_schema: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class ToolResponsePart:
    text: str | None


class ToolDispatchPort(Protocol[CtxT]):
    def available_tools(self) -> Mapping[str, ToolDefinition]: ...

    async def dispatch(
        self,
        ctx: CtxT,
        name: str,
        arguments: dict[str, object],
    ) -> Sequence[ToolResponsePart]: ...
