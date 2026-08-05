"""Compatibility exports for the memory-owned federation source."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, cast

from searchkernel.ports.federation import SearchSource
from searchkernel.runtime.federation import FederationConfig, FederationExecutor

from mcp_memory.integrations.federation_source import MemoryFederationSource
from mcp_memory.integrations.memory_retrieval import MemoryRetrievalPort


class MemorySearchableSource(MemoryFederationSource):
    """Backward-compatible name for the modern memory federation adapter."""


@dataclass
class _MemorySearchKernelCompatibility:
    registry: dict[str, SearchSource]
    executor: FederationExecutor


def build_memory_search_kernel(
    retrieval: MemoryRetrievalPort,
    *,
    extra_sources: Iterable[SearchSource] = (),
    per_source_timeout_s: float = 5.0,
    side_effect_free: bool = False,
) -> Any:
    """Retain the legacy factory while preserving federation controls."""
    memory_source = MemorySearchableSource(
        retrieval,
        side_effect_free=side_effect_free,
    )
    registry: dict[str, SearchSource] = {"memory": memory_source}
    for source in extra_sources:
        source_kind = getattr(source, "source_kind", None)
        if isinstance(source_kind, str):
            registry[source_kind] = source
    return _MemorySearchKernelCompatibility(
        registry=registry,
        executor=FederationExecutor(
            cast(Any, registry),
            config=FederationConfig(per_source_timeout_s=per_source_timeout_s),
        ),
    )


__all__ = [
    "MemorySearchableSource",
    "build_memory_search_kernel",
]
