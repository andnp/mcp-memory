from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
from searchkernel.ports.federation import SearchRequest

from mcp_memory.integrations.searchkernel_source import (
    MemorySearchableSource,
    build_memory_search_kernel,
)


pytestmark = pytest.mark.small


class RetrievalDouble:
    async def search(self, query: str, **kwargs: object) -> Any:
        return SimpleNamespace(
            results=(),
            diagnostics=(),
            cache_diagnostics=(),
            failures=(),
            degraded=False,
        )


@pytest.mark.asyncio
async def test_compatibility_source_uses_v1_contract() -> None:
    source = MemorySearchableSource(cast(Any, RetrievalDouble()))

    response = await source.search(SearchRequest(query="query"))

    assert source.source_kind == "memory"
    assert response.source == source.source_identity
    assert response.contract_version == "v1"


def test_compatibility_factory_keeps_memory_registry_entry() -> None:
    kernel = build_memory_search_kernel(cast(Any, RetrievalDouble()))

    assert isinstance(kernel.registry["memory"], MemorySearchableSource)


def test_compatibility_factory_preserves_federation_controls() -> None:
    kernel = build_memory_search_kernel(
        cast(Any, RetrievalDouble()),
        per_source_timeout_s=1.25,
        side_effect_free=True,
    )

    assert kernel.executor.config.per_source_timeout_s == 1.25
    assert kernel.registry["memory"]._side_effect_free is True
