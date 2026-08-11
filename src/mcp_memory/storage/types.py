from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp_memory.application.ports import MemorySearchPort
from mcp_memory.config import Config
from mcp_memory.core.ports import (
    EmbeddingMaintenancePort,
    MemoryIDResolutionPort,
    ReadCacheValidationPort,
    SearchHealthPort,
    StartupHealthPort,
)


class PostgresBackendNotImplementedError(NotImplementedError):
    pass


@dataclass(frozen=True)
class StorageBackendResources:
    backend: str
    db_manager: Any
    journal: Any
    repository: Any
    relational_search: MemorySearchPort | None
    read_cache: Any
    task_queue: Any
    provider_usage: Any
    runtime_logs: Any
    provider_policy_events: Any
    embedding_integrity_events: Any
    task_execution_attempts: Any
    work_items: Any
    embedding_repair_queue: Any
    vector_store: Any
    mutation_history: Any = None
    curation: Any = None
    curation_action_store: Any = None
    direct_mutation_evidence: Any = None
    ingress_batch_evidence: Any = None
    ingress_action_receipts: Any = None
    source_coverage: Any = None
    ingress_quality_evidence: Any = None
    ingress_mutation_transaction: Any = None
    embedding_maintenance: EmbeddingMaintenancePort | None = None
    search_health: SearchHealthPort | None = None
    startup_health: StartupHealthPort | None = None
    read_cache_validation: ReadCacheValidationPort | None = None
    memory_id_resolution: MemoryIDResolutionPort | None = None


@dataclass(frozen=True)
class StorageBootstrapSpec:
    memory_path: Path
    config: Config
    workspace_id: str | None
