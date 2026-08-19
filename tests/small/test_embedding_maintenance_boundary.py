from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pytest

from mcp_memory.application.memory_embedding_maintenance import MemoryEmbeddingMaintenance
from mcp_memory.config import Config
from mcp_memory.core.ports.embedding_maintenance import EmbeddingDatabaseHealthError

pytestmark = pytest.mark.small


@dataclass
class _Embedder:
    model_name: str = "test-embedding-model"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class _VectorStore:
    def __init__(self) -> None:
        self.search_count = 0

    def search(self, **_: object) -> list[tuple[str, float]]:
        self.search_count += 1
        return []


class _DatabaseHealth:
    def __init__(self, *, failures: int = 0) -> None:
        self.failures = failures
        self.integrity_check_count = 0
        self.retry_count = 0

    def run_integrity_check(self, operation: Callable[[], None]) -> None:
        self.integrity_check_count += 1
        if self.integrity_check_count <= self.failures:
            raise EmbeddingDatabaseHealthError("database unavailable")
        operation()

    def retry_after_reopen(self, operation: Callable[[], None]) -> bool:
        self.retry_count += 1
        try:
            operation()
        except EmbeddingDatabaseHealthError:
            return False
        return True


def _maintenance(database_health: _DatabaseHealth) -> tuple[MemoryEmbeddingMaintenance, _VectorStore]:
    vector_store = _VectorStore()
    maintenance = MemoryEmbeddingMaintenance(
        object(),
        Config(),
        embedder=_Embedder(),
        vector_store=vector_store,
        database_health=database_health,
    )
    return maintenance, vector_store


def test_startup_health_uses_database_health_capability() -> None:
    """A healthy persistence boundary
    performs one embedding health check.
    """
    maintenance, vector_store = _maintenance(_DatabaseHealth())

    health = maintenance.run_startup_health_check()

    assert health.available is True
    assert health.integrity_check_error is None
    assert vector_store.search_count == 1


def test_startup_health_preserves_recovery_after_reopen_retry() -> None:
    """A transient persistence failure
    recovers without marking semantic health failed.
    """
    database_health = _DatabaseHealth(failures=1)
    maintenance, vector_store = _maintenance(database_health)

    health = maintenance.run_startup_health_check()

    assert health.available is True
    assert health.last_error is None
    assert health.last_recovery_at is not None
    assert database_health.retry_count == 1
    assert vector_store.search_count == 1


def test_startup_health_reports_original_error_when_retry_fails() -> None:
    """A persistent database failure
    records the original health-check error.
    """
    database_health = _DatabaseHealth(failures=2)
    maintenance, vector_store = _maintenance(database_health)

    health = maintenance.run_startup_health_check()

    assert health.available is False
    assert health.last_error == "database unavailable"
    assert health.integrity_check_error == "database unavailable"
    assert health.last_failure_at is not None
    assert database_health.retry_count == 1
    assert vector_store.search_count == 0
