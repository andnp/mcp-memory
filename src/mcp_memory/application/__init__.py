"""Application use cases shared by protocol and internal adapters."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp_memory.application.memory_use_cases import (
        ReadMemoryRecordUseCase,
        RecordThoughtUseCase,
        SearchMemoryRecordsUseCase,
    )

__all__ = [
    "ReadMemoryRecordUseCase",
    "RecordThoughtUseCase",
    "SearchMemoryRecordsUseCase",
]


def __getattr__(name: str):
    if name in __all__:
        from mcp_memory.application import memory_use_cases

        return getattr(memory_use_cases, name)
    raise AttributeError(name)
