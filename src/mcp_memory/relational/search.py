from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, cast

from searchkernel.ports import EmbeddingBatchProvider
from searchkernel.runtime import QueryEmbeddingCache
from searchkernel.search.record_pipeline import RecordSearchResult

from mcp_memory.config import Config
from mcp_memory.core.ports.memory import (
    MemoryLink,
    MemoryMaintenanceReadPort,
    MemoryReadContext,
    MemoryReadPort,
    MemoryRecord,
    MemoryRepositoryPort,
    parse_memory_ref,
)
from mcp_memory.core.ports.search import (
    EmbeddingMaintenancePort,
    MemoryIDResolutionPort,
    ReadCacheValidationPort,
    SearchHealthPort,
    StartupHealthPort,
)
from mcp_memory.integrations.memory_retrieval import (
    MemoryRetrievalFacade,
    SearchExecutionDiagnostics,
)
from mcp_memory.integrations.searchkernel_record_pipeline import (
    MEMORY_SEMANTIC_ABSTENTION_DIAGNOSTIC_PREFIX,
    build_memory_record_pipeline,
)
from mcp_memory.utils.db import DatabaseManager

if TYPE_CHECKING:
    from mcp_memory.application.memory_embedding_maintenance import (
        MemoryEmbeddingMaintenance,
    )

ACCESS_HALF_LIFE_DAYS = 7

logger = logging.getLogger(__name__)


SearchRepositoryLike = MemoryReadPort
MaintenanceReadRepositoryLike = MemoryMaintenanceReadPort
RelationalMemoryRecord = MemoryRecord
RelationalMemoryReadContext = MemoryReadContext


@dataclass(slots=True)
class RelationalSearchResult:
    memory_id: str
    title: str
    summary: str
    memory_type: str
    status: str
    tags: list[str] = field(default_factory=list)
    workspace_ids: list[str] = field(default_factory=list)
    score: float = 0.0
    ranking_debug: dict[str, Any] | None = None
    memory_ref: int | None = None
    created_at: str | None = None
    updated_at: str | None = None


@dataclass(slots=True)
class RelationalReadResult:
    record: RelationalMemoryRecord
    relationships: dict[str, list[MemoryLink]]
    superseded: list[RelationalMemoryRecord]


def _to_relational_read_result(
    context: RelationalMemoryReadContext | None,
) -> RelationalReadResult | None:
    if context is None:
        return None
    return RelationalReadResult(
        record=context.record,
        relationships=context.relationships,
        superseded=context.superseded,
    )


def build_search_scope_diagnostics(
    *,
    workspace_id: str | None,
    ranking_workspace_id: str | None,
) -> dict[str, object]:
    return {
        "mode": "filtered" if workspace_id is not None else "global",
        "workspace_filter": workspace_id,
        "ranking_workspace_id": ranking_workspace_id,
    }


def _semantic_abstention_counts(
    diagnostics: list[str] | tuple[str, ...],
) -> tuple[int, int, int]:
    diagnostic = next(
        (
            value
            for value in diagnostics
            if value.startswith(MEMORY_SEMANTIC_ABSTENTION_DIAGNOSTIC_PREFIX)
        ),
        None,
    )
    if diagnostic is None:
        return 0, 0, 0
    values: dict[str, int] = {}
    for field_value in diagnostic[
        len(MEMORY_SEMANTIC_ABSTENTION_DIAGNOSTIC_PREFIX) :
    ].split(";"):
        name, separator, raw_value = field_value.partition("=")
        if not separator:
            continue
        try:
            value = int(raw_value)
        except ValueError:
            continue
        if value >= 0:
            values[name] = value
    return (
        values.get("semantic_candidates", 0),
        values.get("semantic_only_candidates", 0),
        values.get("rejected", 0),
    )


@dataclass(slots=True)
class SearchHealthStatus:
    semantic_enabled: bool
    available: bool
    degraded: bool = False
    fallback_count: int = 0
    rebuild_count: int = 0
    background_repair_enabled: bool = False
    background_repair_wait_seconds: float = 0.0
    queued_repair_backlog_count: int = 0
    running_repair_count: int = 0
    oldest_queued_repair_age_seconds: float | None = None
    repair_wait_count: int = 0
    partial_semantic_search_count: int = 0
    last_partial_semantic_at: str | None = None
    last_repair_wait_seconds: float = 0.0
    last_repair_candidate_count: int = 0
    last_repair_pending_count: int = 0
    last_error: str | None = None
    last_failure_at: str | None = None
    last_recovery_at: str | None = None
    last_integrity_check_at: str | None = None
    integrity_check_error: str | None = None


class RelationalMemorySearchService(
    SearchHealthPort,
    StartupHealthPort,
    EmbeddingMaintenancePort,
    ReadCacheValidationPort,
    MemoryIDResolutionPort,
):
    """Application compatibility boundary around authoritative memory services.

    Retrieval orchestration belongs to searchkernel. This service retains the
    stable application API for reads, maintenance, embedding lifecycle, health,
    and compatibility result payloads.
    """

    def __init__(
        self,
        repository: SearchRepositoryLike,
        config: Config,
        *,
        embedder: EmbeddingBatchProvider | None = None,
        vector_store: Any | None = None,
        db_manager: DatabaseManager | None = None,
        task_queue=None,
        work_items=None,
        embedding_repair_queue=None,
        background_repair_wait_seconds: float = 0.0,
        embedding_maintenance: MemoryEmbeddingMaintenance | None = None,
        query_embedding_cache: QueryEmbeddingCache | None = None,
    ) -> None:
        if embedding_maintenance is None:
            from mcp_memory.application.memory_embedding_maintenance import (
                MemoryEmbeddingMaintenance,
            )
            embedding_maintenance = MemoryEmbeddingMaintenance(
                repository,
                config,
                embedder=embedder,
                vector_store=vector_store,
                db_manager=db_manager,
                task_queue=task_queue,
                work_items=work_items,
                embedding_repair_queue=embedding_repair_queue,
                background_repair_wait_seconds=background_repair_wait_seconds,
            )

        self._repository = repository
        self._config = config
        self._embedder = embedder
        self._vector_store = vector_store
        self._db_manager = db_manager
        self._embedding_maintenance = embedding_maintenance
        self._query_embedding_cache = query_embedding_cache or QueryEmbeddingCache()
        memory_repository = cast(MemoryRepositoryPort, repository)
        self._health = SearchHealthStatus(
            semantic_enabled=embedder is not None and vector_store is not None,
            available=embedder is not None and vector_store is not None,
            background_repair_enabled=(
                self._embedding_maintenance.get_health().background_repair_enabled
            ),
            background_repair_wait_seconds=(
                self._embedding_maintenance.get_health().background_repair_wait_seconds
            ),
        )
        self._retrieval_facade = MemoryRetrievalFacade(
            memory_repository,
            query_embedding_cache=self._query_embedding_cache,
            native_search=self,
            pipeline_factory=lambda adaptive_enabled: build_memory_record_pipeline(
                memory_repository,
                vector_store=self._vector_store,
                embedder=self._embedder,
                embedding_maintenance=self._embedding_maintenance,
                query_embedding_cache=self._query_embedding_cache,
                adaptive_enabled=adaptive_enabled,
                config=self._config,
            ),
        )

    def _sync_embedding_maintenance_health(self) -> None:
        maintenance_health = self._embedding_maintenance.get_health()
        for field_name in SearchHealthStatus.__dataclass_fields__:
            if hasattr(maintenance_health, field_name):
                setattr(self._health, field_name, getattr(maintenance_health, field_name))

    def get_health(self) -> SearchHealthStatus:
        self._sync_embedding_maintenance_health()
        return replace(self._health)

    def run_startup_health_check(self) -> SearchHealthStatus:
        self._embedding_maintenance.run_startup_health_check()
        self._sync_embedding_maintenance_health()
        return self.get_health()

    def rebuild_semantic_index(
        self,
        *,
        limit: int = 10_000,
    ) -> dict[str, int | bool | str | None]:
        result = self._embedding_maintenance.rebuild_semantic_index(limit=limit)
        self._sync_embedding_maintenance_health()
        return result

    def search_memories(
        self,
        query: str,
        workspace_id: str | None = None,
        limit: int = 5,
        *,
        ranking_workspace_id: str | None = None,
        adaptive_limit: bool = False,
        memory_type: str | None = None,
        status: str | None = None,
        include_superseded: bool = False,
        debug: bool = False,
        side_effect_free: bool = False,
    ) -> list[RelationalSearchResult]:
        results, _ = self.search_memories_with_diagnostics(
            query,
            workspace_id=workspace_id,
            ranking_workspace_id=ranking_workspace_id,
            limit=limit,
            adaptive_limit=adaptive_limit,
            memory_type=memory_type,
            status=status,
            include_superseded=include_superseded,
            debug=debug,
            side_effect_free=side_effect_free,
        )
        return results

    def search_memories_with_diagnostics(
        self,
        query: str,
        workspace_id: str | None = None,
        limit: int = 5,
        *,
        ranking_workspace_id: str | None = None,
        adaptive_limit: bool = False,
        memory_type: str | None = None,
        status: str | None = None,
        include_superseded: bool = False,
        debug: bool = False,
        side_effect_free: bool = False,
    ) -> tuple[list[RelationalSearchResult], SearchExecutionDiagnostics]:
        """Delegate retrieval to searchkernel while preserving the legacy payload."""
        outcome, diagnostics = self._retrieval_facade.search_sync_with_diagnostics(
            query,
            limit=limit,
            adaptive_limit=adaptive_limit,
            workspace_id=workspace_id,
            ranking_workspace_id=ranking_workspace_id,
            memory_type=memory_type,
            status=status,
            include_superseded=include_superseded,
            side_effect_free=side_effect_free,
            debug=debug,
        )
        results = [_to_relational_search_result(result) for result in outcome.results]
        for result in results:
            if result.ranking_debug is not None:
                result.ranking_debug["final_duplicate"] = (
                    result.memory_id in (diagnostics.final_duplicate_ids or [])
                )
        if results and not side_effect_free:
            self._repository.touch_last_surfaced(
                [result.memory_id for result in results],
                _utc_now(),
                best_effort=True,
            )
        return results, diagnostics

    def read_memory(self, memory_id: str) -> RelationalReadResult | None:
        record = self._repository.get_memory(memory_id)
        if record is None:
            return None

        decayed_score = _decayed_access_score(
            record.access_score,
            record.last_accessed_at,
        )
        updated_record = self._repository.record_access(
            memory_id,
            decayed_score + 1.0,
            _utc_now(),
            increment_read_count=True,
        )
        current = updated_record or record

        outgoing = self._repository.get_links(memory_id, direction="outgoing")
        incoming = self._repository.get_links(memory_id, direction="incoming")
        superseded: list[RelationalMemoryRecord] = []
        for link in outgoing:
            if link.link_type != "SUPERSEDES":
                continue
            target = self._repository.get_memory(link.target_id)
            if target is not None:
                superseded.append(target)

        return RelationalReadResult(
            record=current,
            relationships={
                "outgoing": outgoing,
                "incoming": incoming,
            },
            superseded=superseded,
        )

    def peek_memory(self, memory_id: str) -> RelationalReadResult | None:
        """Return authoritative memory context without changing retrieval telemetry."""
        repository = cast(MaintenanceReadRepositoryLike, self._repository)
        return _to_relational_read_result(repository.peek_memory(memory_id))

    def search_memories_for_maintenance(
        self,
        query: str,
        workspace_id: str | None = None,
        limit: int = 50,
        *,
        memory_type: str | None = None,
        status: str | None = None,
        include_superseded: bool = False,
    ) -> list[RelationalReadResult]:
        """Search authoritative records without surfacing or accessing them."""
        repository = cast(MaintenanceReadRepositoryLike, self._repository)
        contexts = repository.search_memories_for_maintenance(
            query,
            workspace_id=workspace_id,
            memory_type=memory_type,
            status=status,
            include_superseded=include_superseded,
            limit=limit,
        )
        return [
            result
            for context in contexts
            if (result := _to_relational_read_result(context)) is not None
        ]

    def get_read_cache_validation_tokens(self, memory_ids: list[str]) -> dict[str, str]:
        return self._repository.get_read_cache_validation_tokens(memory_ids)

    def resolve_memory_id(self, memory_id: str) -> str | None:
        return self._repository.resolve_memory_id(memory_id)


def _to_relational_search_result(result: RecordSearchResult) -> RelationalSearchResult:
    record = result.record
    metadata = record.metadata
    memory_ref_value = metadata.get("memory_ref")
    memory_ref = (
        memory_ref_value
        if isinstance(memory_ref_value, int) and memory_ref_value > 0
        else parse_memory_ref(memory_ref_value)
        if isinstance(memory_ref_value, str)
        else None
    )
    summary_value = metadata.get("summary")
    summary = (
        summary_value
        if isinstance(summary_value, str) and summary_value
        else _smart_truncate(record.body)
    )
    memory_type = metadata.get("memory_type")
    status = metadata.get("memory_status")
    status_value = getattr(record.status, "value", record.status)
    created_at = getattr(record, "created_at", None)
    updated_at = getattr(record, "updated_at", None)
    return RelationalSearchResult(
        memory_id=record.source_id,
        memory_ref=memory_ref,
        title=record.title,
        summary=summary,
        memory_type=memory_type if isinstance(memory_type, str) else "",
        status=status if isinstance(status, str) else str(status_value),
        created_at=(
            created_at.isoformat()
            if isinstance(created_at, datetime)
            else created_at
            if isinstance(created_at, str)
            else None
        ),
        updated_at=(
            updated_at.isoformat()
            if isinstance(updated_at, datetime)
            else updated_at
            if isinstance(updated_at, str)
            else None
        ),
        tags=[
            str(tag)
            for tag in metadata.get("tags", [])
            if isinstance(tag, str)
        ],
    workspace_ids=[
            str(workspace_id)
            for workspace_id in metadata.get("workspace_ids", [])
            if isinstance(workspace_id, str)
        ],
        score=round(result.score, 6),
        ranking_debug={
            "provenance": result.provenance.to_dict(),
            "canonical_id": record.storage_key,
            "duplicate_candidate": _is_duplicate_candidate(result),
            "multi_lane_provenance": _is_duplicate_candidate(result),
            "final_duplicate": False,
        },
    )


def _is_duplicate_candidate(result: RecordSearchResult) -> bool:
    """Flag one record found by multiple retrieval strategies without merging it."""
    strategies = getattr(result.provenance, "strategies", None)
    if strategies is None:
        strategies = result.provenance.to_dict().get("strategies", ())
    return len(strategies) > 1


def _smart_truncate(text: str, *, max_chars: int = 200) -> str:
    normalized = text.strip()
    if len(normalized) <= max_chars:
        return normalized

    truncated = normalized[:max_chars].rstrip()
    sentence_end = max(
        truncated.rfind("."),
        truncated.rfind("?"),
        truncated.rfind("!"),
    )
    if sentence_end >= 0:
        candidate = truncated[: sentence_end + 1].rstrip()
        if candidate:
            return candidate
    return f"{truncated}…"


def _decayed_access_score(
    access_score: float,
    last_accessed_at: str | None,
) -> float:
    return _decayed_access_score_for_half_life(
        access_score,
        last_accessed_at,
        half_life_days=ACCESS_HALF_LIFE_DAYS,
    )


def _decayed_access_score_for_half_life(
    access_score: float,
    last_accessed_at: str | None,
    *,
    half_life_days: float,
) -> float:
    if access_score <= 0 or not last_accessed_at:
        return max(access_score, 0.0)

    try:
        accessed_at = datetime.fromisoformat(last_accessed_at)
    except ValueError:
        return max(access_score, 0.0)

    if accessed_at.tzinfo is None:
        accessed_at = accessed_at.replace(tzinfo=UTC)

    elapsed = datetime.now(UTC) - accessed_at
    elapsed_days = max(elapsed / timedelta(days=1), 0.0)
    return access_score * (0.5 ** (elapsed_days / half_life_days))


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()
