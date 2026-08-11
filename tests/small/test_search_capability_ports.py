from dataclasses import dataclass

from mcp_memory.core.ports import (
    EmbeddingMaintenancePort,
    MemoryIDResolutionPort,
    ReadCacheValidationPort,
    SearchHealthPort,
    StartupHealthPort,
)


@dataclass
class _Health:
    available: bool = True


class _SearchCapabilities:
    def get_health(self) -> _Health:
        return _Health()

    def run_startup_health_check(self) -> _Health:
        return _Health()

    def rebuild_semantic_index(self, *, limit: int = 10_000) -> dict[str, int]:
        return {"limit": limit}

    def get_read_cache_validation_tokens(self, memory_ids: list[str]) -> dict[str, str]:
        return {memory_id: f"token:{memory_id}" for memory_id in memory_ids}

    def resolve_memory_id(self, memory_id: str) -> str | None:
        return memory_id


def test_search_capability_ports_are_independently_structural() -> None:
    capabilities = _SearchCapabilities()

    assert isinstance(capabilities, SearchHealthPort)
    assert isinstance(capabilities, StartupHealthPort)
    assert isinstance(capabilities, EmbeddingMaintenancePort)
    assert isinstance(capabilities, ReadCacheValidationPort)
    assert isinstance(capabilities, MemoryIDResolutionPort)


def test_search_capability_contracts_preserve_small_typed_operations() -> None:
    capabilities = _SearchCapabilities()

    assert capabilities.get_health().available
    assert capabilities.run_startup_health_check().available
    assert capabilities.rebuild_semantic_index(limit=3) == {"limit": 3}
    assert capabilities.get_read_cache_validation_tokens(["mem-1"]) == {
        "mem-1": "token:mem-1"
    }
    assert capabilities.resolve_memory_id("mem-1") == "mem-1"
