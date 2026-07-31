from __future__ import annotations

import logging
import inspect
import sqlite3
import time
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from typing import Any, TypedDict, cast

from mcp_memory.config import Config
from mcp_memory.embeddings import is_fallback_embedding_model
from mcp_memory.core.ports.memory import (
    FTS_QUERY_TOKEN_PATTERN,
    MemoryLink,
    MemoryMaintenanceReadPort,
    MemoryReadContext,
    MemoryReadPort,
    RankedMemoryCandidate,
    MemoryRecord,
)
from mcp_memory.utils.db import DatabaseManager
from mcp_memory.core.search_graph import GraphCandidateExpander, GraphExpansionInfo
from mcp_memory.core.search_repair import EmbeddingRepairScheduler
from mcp_memory.core.search_ranking import (
    RankingEngine as PureRankingEngine,
    RankingSignals,
    ScoringWeights,
)
from mcp_memory.relational.semantic import RelationalSemanticSearchAdapter
from searchkernel.ingestion import EmbeddingInput, embed_and_upsert
from searchkernel.ports import EmbeddingBatchProvider
from searchkernel.search.adaptive_limit import resolve_adaptive_result_limit


ACCESS_HALF_LIFE_DAYS = 7
DEGRADATION_PENALTY = 0.3
WORKSPACE_BOOST = 1.2
GRAPH_EXPANSION_MAX_SEEDS = 3
GRAPH_EXPANSION_MAX_NEIGHBORS_PER_SEED = 10
LEXICAL_STRENGTH_EVAL_LIMIT = 5
STRONG_KEYWORD_BOUNDED_CANDIDATE_MULTIPLIER = 4
MIN_STRONG_KEYWORD_BOUNDED_CANDIDATES = 20
GRAPH_EXPANSION_DISCOUNTS = {
    "DEPENDS_ON": 0.7,
    "AMENDS": 0.6,
    "CONTRADICTS": 0.35,
}
INLINE_EMBEDDING_REPAIR_LIMIT = 64
BACKGROUND_REPAIR_POLL_INTERVAL_SECONDS = 0.05
DEFAULT_BACKGROUND_REPAIR_BATCH_SIZE = 32
DEFAULT_BACKGROUND_REPAIR_MAX_BATCHES_PER_RUN = 8


class _SemanticScoreKwargs(TypedDict, total=False):
    candidate_ids: Sequence[str] | None
    semantic_timing_ms: dict[str, float] | None
    vector_search_diagnostics: dict[str, object] | None
    limit: int


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
    ranking_debug: dict[str, float | str | bool] | None = None


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


@dataclass(slots=True)
class SearchExecutionDiagnostics:
    timing_ms: dict[str, float] = field(default_factory=dict)
    keyword_candidate_count: int = 0
    semantic_candidate_count: int = 0
    semantic_candidate_strategy: str = "global"
    vector_search: dict[str, object] | None = None

    def to_payload(self) -> dict[str, object]:
        return {
            "timing_ms": dict(self.timing_ms),
            "keyword_candidate_count": self.keyword_candidate_count,
            "semantic_candidate_count": self.semantic_candidate_count,
            "semantic_candidate_strategy": self.semantic_candidate_strategy,
            "vector_search": None if self.vector_search is None else dict(self.vector_search),
        }


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


@dataclass(slots=True)
class SemanticCandidatePool:
    candidates: list[RelationalMemoryRecord]
    candidate_ids: list[str] | None = None
    speculative: bool = False
    strategy: str = "global"
    skip_semantic_scoring: bool = False


class RankingEngine(PureRankingEngine):
    """Compatibility facade that enriches raw records from the memory port."""

    def __init__(self, repository: SearchRepositoryLike, config: Config, *, weights: ScoringWeights | None = None) -> None:
        super().__init__(config, weights=weights)
        self._repository = repository

    def _candidate_with_authority(self, candidate: RankedMemoryCandidate | RelationalMemoryRecord) -> RankedMemoryCandidate:
        if isinstance(candidate, RankedMemoryCandidate):
            return candidate
        links = self._repository.get_links(candidate.id, direction="incoming")
        return RankedMemoryCandidate(
            record=candidate,
            incoming_links_count=len(links),
            incoming_link_type_counts=_count_links_by_type(links),
        )

    def rank_records(self, records, rrf_scores, workspace_id=None, *, ranking_signals=None, keyword_candidates_present=False):
        return super().rank_records(
            [self._candidate_with_authority(candidate) for candidate in records],
            rrf_scores, workspace_id, ranking_signals=ranking_signals,
            keyword_candidates_present=keyword_candidates_present,
        )

    def explain_candidate(self, candidate, rrf_score, workspace_id=None, *, signals=None, keyword_candidates_present=False):
        return super().explain_candidate(
            self._candidate_with_authority(candidate), rrf_score, workspace_id,
            signals=signals, keyword_candidates_present=keyword_candidates_present,
        )

    def authority_multiplier(self, record):
        return super().authority_multiplier_for_candidate(self._candidate_with_authority(record))

    def score_from_rrf(self, record, rrf_score, workspace_id=None):
        return self.rank_records([record], {record.id: rrf_score}, workspace_id)[0][1]


class RelationalMemorySearchService:
    def __init__(
        self,
        repository: SearchRepositoryLike,
        config: Config,
        *,
        embedder: EmbeddingBatchProvider | None = None,
        vector_store: Any | None = None,
        db_manager: DatabaseManager | None = None,
        task_queue = None,
        work_items = None,
        embedding_repair_queue = None,
        background_repair_wait_seconds: float = 0.0,
    ) -> None:
        self._repository = repository
        self._config = config
        self._embedder = embedder
        self._vector_store = vector_store
        self._db_manager = db_manager
        self._task_queue = task_queue
        self._work_items = work_items
        self._embedding_repair_queue = embedding_repair_queue
        self._background_repair_wait_seconds = max(background_repair_wait_seconds, 0.0)
        self._semantic_adapter = (
            RelationalSemanticSearchAdapter(embedder, vector_store)
            if embedder is not None and vector_store is not None
            else None
        )
        self._repair_scheduler = (
            EmbeddingRepairScheduler(
                config=config,
                task_queue=task_queue,
                work_items=work_items,
                embedding_repair_queue=embedding_repair_queue,
                wait_seconds=self._background_repair_wait_seconds,
            )
            if task_queue is not None and (embedding_repair_queue is not None or work_items is not None)
            else None
        )
        self._graph_expander = GraphCandidateExpander(
            lambda memory_id: self._repository.get_links(memory_id, direction="outgoing")
        )
        semantic_enabled = embedder is not None and vector_store is not None
        self._health = SearchHealthStatus(
            semantic_enabled=semantic_enabled,
            available=semantic_enabled,
            background_repair_enabled=background_repair_wait_seconds > 0 and task_queue is not None and (embedding_repair_queue is not None or work_items is not None),
            background_repair_wait_seconds=max(background_repair_wait_seconds, 0.0),
        )

    def get_health(self) -> SearchHealthStatus:
        self._refresh_background_repair_health_snapshot()
        return replace(self._health)

    def run_startup_health_check(self) -> SearchHealthStatus:
        check_timestamp = _utc_now()
        self._health.last_integrity_check_at = check_timestamp
        self._health.integrity_check_error = None
        if not self._health.semantic_enabled:
            self._health.available = False
            self._health.degraded = False
            return self.get_health()

        try:
            self._run_integrity_check_once()
            self._mark_semantic_recovered()
            self._health.last_recovery_at = check_timestamp
        except (OSError, sqlite3.Error, ValueError) as exc:
            recovered = self._retry_after_reopen(
                "startup health check",
                lambda: self._run_integrity_check_once() or True,
            )
            if not recovered:
                self._mark_semantic_failure(exc, fallback=False)
                self._health.integrity_check_error = str(exc)
        return self.get_health()

    def rebuild_semantic_index(self, *, limit: int = 10_000) -> dict[str, int | bool | str | None]:
        if not self._health.semantic_enabled or self._embedder is None or self._vector_store is None:
            return {
                "semantic_enabled": False,
                "rebuilt": False,
                "records_indexed": 0,
                "reason": "semantic_search_disabled",
            }

        blocked_reason = self._blocked_embedding_persistence_reason()
        if blocked_reason is not None:
            self._mark_semantic_failure(ValueError(blocked_reason), fallback=True)
            return {
                "semantic_enabled": True,
                "rebuilt": False,
                "records_indexed": 0,
                "reason": "fallback_embedding_persistence_blocked",
            }

        candidates = [
            record
            for record in self._repository.list_memories(limit=limit)
            if record.status != "archived"
        ]
        if self._db_manager is None:
            return {
                "semantic_enabled": True,
                "rebuilt": False,
                "records_indexed": 0,
                "reason": "db_manager_unavailable",
            }

        conn = self._db_manager.get_connection()
        with conn:
            conn.execute(
                "DELETE FROM embeddings WHERE source_kind = ? AND model_name = ?",
                ("memory", self._embedder.model_name),
            )
        self._ensure_memory_embeddings(candidates)
        self._health.rebuild_count += 1
        self._health.last_recovery_at = _utc_now()
        self._health.last_error = None
        self._health.last_failure_at = None
        self._health.available = True
        self._health.degraded = False
        self._health.integrity_check_error = None
        return {
            "semantic_enabled": True,
            "rebuilt": True,
            "records_indexed": len(candidates),
            "reason": None,
        }

    def search_memories(
        self,
        query: str,
        workspace_id: str | None = None,
        limit: int = 5,
        *,
        adaptive_limit: bool = False,
        memory_type: str | None = None,
        status: str | None = None,
        include_superseded: bool = False,
        debug: bool = False,
    ):
        return self.search_memories_with_diagnostics(
            query,
            workspace_id=workspace_id,
            limit=limit,
            adaptive_limit=adaptive_limit,
            memory_type=memory_type,
            status=status,
            include_superseded=include_superseded,
            debug=debug,
        )[0]

    def search_memories_with_diagnostics(
        self,
        query: str,
        workspace_id: str | None = None,
        limit: int = 5,
        *,
        adaptive_limit: bool = False,
        memory_type: str | None = None,
        status: str | None = None,
        include_superseded: bool = False,
        debug: bool = False,
    ) -> tuple[list[RelationalSearchResult], SearchExecutionDiagnostics]:
        diagnostics = SearchExecutionDiagnostics()
        started_at = time.perf_counter()
        if not query.strip():
            diagnostics.timing_ms["total"] = round((time.perf_counter() - started_at) * 1000.0, 3)
            return [], diagnostics

        candidate_limit = (
            max(limit, self._config.search_ranking.adaptive_result_max)
            if adaptive_limit
            else limit
        )
        query_tokens = _query_tokens(query)
        keyword_started = time.perf_counter()
        keyword_ids = self._repository.search_keyword_memory_ids(
            query,
            status=status,
            include_superseded=include_superseded,
            limit=max(50, candidate_limit),
        )
        diagnostics.timing_ms["keyword_lookup"] = round((time.perf_counter() - keyword_started) * 1000.0, 3)
        diagnostics.keyword_candidate_count = len(keyword_ids)
        semantic_started = time.perf_counter()
        semantic_ids, semantic_scores = self._semantic_candidate_ids(
            query,
            workspace_id=workspace_id,
            status=status,
            include_superseded=include_superseded,
            keyword_ids=keyword_ids,
            query_tokens=query_tokens,
            requested_limit=candidate_limit,
            limit=max(50, candidate_limit),
            diagnostics=diagnostics if debug else None,
        )
        diagnostics.timing_ms["semantic_selection"] = round((time.perf_counter() - semantic_started) * 1000.0, 3)
        diagnostics.semantic_candidate_count = len(semantic_ids)
        engine = RankingEngine(self._repository, self._config)
        rrf_scores = engine.fuse_reciprocal_rank(semantic_ids, keyword_ids)
        if not rrf_scores:
            diagnostics.timing_ms["total"] = round((time.perf_counter() - started_at) * 1000.0, 3)
            return [], diagnostics
        graph_started = time.perf_counter()
        graph_expansion = self._expand_graph_candidate_scores(rrf_scores)
        diagnostics.timing_ms["graph_expansion"] = round((time.perf_counter() - graph_started) * 1000.0, 3)
        for memory_id, expansion in graph_expansion.items():
            rrf_scores[memory_id] = max(rrf_scores.get(memory_id, 0.0), expansion.rrf_score)

        candidate_hydration_started = time.perf_counter()
        if debug:
            diagnostics.timing_ms.setdefault("candidate_hydration_query_execution", 0.0)
            diagnostics.timing_ms.setdefault("candidate_hydration_row_fetch", 0.0)
            diagnostics.timing_ms.setdefault("candidate_hydration_candidate_build", 0.0)
        candidates = self._get_ranking_candidates_with_optional_diagnostics(
            list(rrf_scores.keys()),
            status=status,
            include_superseded=include_superseded,
            timing_ms=diagnostics.timing_ms if debug else None,
        )
        diagnostics.timing_ms["candidate_hydration"] = round((time.perf_counter() - candidate_hydration_started) * 1000.0, 3)
        if status is None:
            candidates = [
                candidate
                for candidate in candidates
                if (candidate.record.status if isinstance(candidate, RankedMemoryCandidate) else candidate.status) != "archived"
            ]
        candidate_by_id = {
            (candidate.record.id if isinstance(candidate, RankedMemoryCandidate) else candidate.id): candidate
            for candidate in candidates
        }
        keyword_id_set = set(keyword_ids)
        semantic_id_set = set(semantic_ids)
        signal_preparation_started = time.perf_counter()
        ranking_signals = {
            memory_id: RankingSignals(
                matched_by_keyword=memory_id in keyword_id_set,
                matched_by_semantic=memory_id in semantic_id_set,
                semantic_score=semantic_scores.get(memory_id, 0.0),
                expanded_by_graph=memory_id in graph_expansion,
                keyword_token_coverage=_keyword_token_coverage(
                    query_tokens,
                    candidate_by_id[memory_id].record if isinstance(candidate_by_id[memory_id], RankedMemoryCandidate) else candidate_by_id[memory_id],
                ) if memory_id in candidate_by_id else 0.0,
            )
            for memory_id in rrf_scores
        }
        diagnostics.timing_ms["signal_preparation"] = round((time.perf_counter() - signal_preparation_started) * 1000.0, 3)
        ranking_started = time.perf_counter()
        ranked = [
            RelationalSearchResult(
                memory_id=record.id,
                title=record.title,
                summary=_search_result_summary(record),
                memory_type=record.type,
                status=record.status,
                tags=list(record.tags),
                workspace_ids=list(record.workspace_ids),
                score=round(score, 6),
                ranking_debug=(
                    engine.explain_candidate(
                        candidate_by_id[record.id],
                        rrf_scores[record.id],
                        workspace_id,
                        signals=ranking_signals.get(record.id),
                        keyword_candidates_present=bool(keyword_ids),
                    )
                    | {
                        "matched_by_keyword": record.id in keyword_id_set,
                        "matched_by_semantic": record.id in semantic_id_set,
                        "expanded_by_graph": record.id in graph_expansion,
                        "graph_seed_id": graph_expansion[record.id].seed_id if record.id in graph_expansion else "",
                        "graph_link_type": graph_expansion[record.id].link_type if record.id in graph_expansion else "",
                        "graph_rrf_score": round(graph_expansion[record.id].rrf_score, 6) if record.id in graph_expansion else 0.0,
                    }
                )
                if debug
                else None,
            )
            for record, score in engine.rank_records(
                candidates,
                rrf_scores,
                workspace_id,
                ranking_signals=ranking_signals,
                keyword_candidates_present=bool(keyword_ids),
            )
        ]
        ranked.sort(
            key=lambda item: (
                item.score,
                _direct_match_rank(ranking_signals.get(item.memory_id)),
                ranking_signals.get(item.memory_id, RankingSignals()).keyword_token_coverage,
                ranking_signals.get(item.memory_id, RankingSignals()).semantic_score,
            ),
            reverse=True,
        )
        diagnostics.timing_ms["ranking"] = round((time.perf_counter() - ranking_started) * 1000.0, 3)
        result_limit = self._resolved_result_limit(
            ranked,
            requested_limit=limit,
            adaptive_limit=adaptive_limit,
        )
        ranked = ranked[:result_limit]

        if not keyword_ids and ranked:
            top_semantic_score = semantic_scores.get(ranked[0].memory_id, 0.0)
            if top_semantic_score < self._config.search_ranking.semantic_only_abstain_threshold:
                diagnostics.timing_ms["total"] = round((time.perf_counter() - started_at) * 1000.0, 3)
                return [], diagnostics

        surfaced_ids = [result.memory_id for result in ranked]
        if surfaced_ids:
            surfaced_writeback_started = time.perf_counter()
            self._repository.touch_last_surfaced(surfaced_ids, _utc_now(), best_effort=True)
            diagnostics.timing_ms["surfaced_writeback"] = round((time.perf_counter() - surfaced_writeback_started) * 1000.0, 3)
        diagnostics.timing_ms["total"] = round((time.perf_counter() - started_at) * 1000.0, 3)
        return ranked, diagnostics

    def _resolved_result_limit(
        self,
        ranked: Sequence[RelationalSearchResult],
        *,
        requested_limit: int,
        adaptive_limit: bool,
    ) -> int:
        ranking_config = self._config.search_ranking
        return resolve_adaptive_result_limit(
            [result.score for result in ranked],
            requested_limit=requested_limit,
            adaptive_enabled=adaptive_limit,
            maximum_limit=ranking_config.adaptive_result_max,
            score_ratio_floor=ranking_config.adaptive_result_score_ratio_floor,
            minimum_score=ranking_config.adaptive_result_min_score,
            maximum_score_gap=ranking_config.adaptive_result_max_score_gap,
        )

    def read_memory(self, memory_id: str):
        record = self._repository.get_memory(memory_id)
        if record is None:
            return None

        decayed_score = _decayed_access_score(record.access_score, record.last_accessed_at)
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
        context = repository.peek_memory(memory_id)
        return _to_relational_read_result(context)

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
        return [result for context in contexts if (result := _to_relational_read_result(context)) is not None]

    def get_read_cache_validation_tokens(self, memory_ids: list[str]) -> dict[str, str]:
        return self._repository.get_read_cache_validation_tokens(memory_ids)

    def _blocked_embedding_persistence_reason(self) -> str | None:
        if self._embedder is None or self._vector_store is None:
            return None
        if not is_fallback_embedding_model(self._embedder.model_name):
            return None

        get_write_policy_state = getattr(self._vector_store, "get_write_policy_state", None)
        if not callable(get_write_policy_state):
            return None

        policy_state = get_write_policy_state()
        if getattr(policy_state, "fallback_persistence_policy", None) != "blocked":
            return None

        return (
            "Fallback/hash embeddings cannot be persisted in Postgres/shared mode; "
            f"blocked model_name {self._embedder.model_name!r}"
        )

    def _get_query_embedding(self, query: str) -> list[float]:
        adapter = self._get_semantic_adapter()
        return adapter.get_query_embedding(query)

    def _get_semantic_adapter(self) -> RelationalSemanticSearchAdapter:
        assert self._embedder is not None
        assert self._vector_store is not None
        if self._semantic_adapter is None or not self._semantic_adapter.matches(self._embedder, self._vector_store):
            self._semantic_adapter = RelationalSemanticSearchAdapter(self._embedder, self._vector_store)
        return self._semantic_adapter

    def _semantic_scores(
        self,
        query: str,
        candidates: list[RelationalMemoryRecord],
        workspace_id: str | None,
        *,
        candidate_ids: Sequence[str] | None = None,
        semantic_timing_ms: dict[str, float] | None = None,
        vector_search_diagnostics: dict[str, object] | None = None,
        limit: int,
    ) -> dict[str, float]:
        if self._embedder is None or self._vector_store is None or not candidates:
            return {}
        try:
            matches = self._compute_semantic_matches(
                query,
                candidates,
                workspace_id,
                candidate_ids=candidate_ids,
                semantic_timing_ms=semantic_timing_ms,
                vector_search_diagnostics=vector_search_diagnostics,
                limit=limit,
            )
        except (OSError, sqlite3.Error, ValueError) as exc:
            recovered_matches = self._retry_after_reopen(
                "semantic search",
                lambda: self._compute_semantic_matches(
                    query,
                    candidates,
                    workspace_id,
                    candidate_ids=candidate_ids,
                    semantic_timing_ms=semantic_timing_ms,
                    vector_search_diagnostics=vector_search_diagnostics,
                    limit=limit,
                ),
            )
            if recovered_matches is not None:
                return {
                    memory_id: max(min((score + 1.0) / 2.0, 1.0), 0.0)
                    for memory_id, score in recovered_matches
                    if score > 0
                }
            self._mark_semantic_failure(exc, fallback=True)
            logger.warning(
                "Semantic search unavailable; falling back to keyword-only ranking: %s",
                exc,
            )
            return {}
        return {
            memory_id: max(min((score + 1.0) / 2.0, 1.0), 0.0)
            for memory_id, score in matches
            if score > 0
        }

    def _semantic_scores_with_optional_diagnostics(
        self,
        query: str,
        candidates: list[RelationalMemoryRecord],
        workspace_id: str | None,
        *,
        candidate_ids: Sequence[str] | None = None,
        semantic_timing_ms: dict[str, float] | None = None,
        vector_search_diagnostics: dict[str, object] | None = None,
        limit: int,
    ) -> dict[str, float]:
        signature = inspect.signature(self._semantic_scores)
        accepts_var_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        optional_kwargs: _SemanticScoreKwargs = {
            "candidate_ids": candidate_ids,
            "semantic_timing_ms": semantic_timing_ms,
            "vector_search_diagnostics": vector_search_diagnostics,
            "limit": limit,
        }
        filtered_kwargs = (
            optional_kwargs
            if accepts_var_kwargs
            else cast(
                "_SemanticScoreKwargs",
                {
                    key: value
                    for key, value in optional_kwargs.items()
                    if key in signature.parameters
                },
            )
        )
        return self._semantic_scores(
            query,
            candidates,
            workspace_id,
            **filtered_kwargs,
        )

    def _compute_semantic_matches(
        self,
        query: str,
        candidates: list[RelationalMemoryRecord],
        workspace_id: str | None,
        *,
        candidate_ids: Sequence[str] | None = None,
        semantic_timing_ms: dict[str, float] | None = None,
        vector_search_diagnostics: dict[str, object] | None = None,
        limit: int,
    ) -> list[tuple[str, float]]:
        _ = workspace_id
        adapter = self._get_semantic_adapter()
        self._ensure_searchable_memory_embeddings(candidates, semantic_timing_ms=semantic_timing_ms)
        return adapter.search(
            query,
            candidates,
            candidate_ids=candidate_ids,
            limit=limit,
            semantic_timing_ms=semantic_timing_ms,
            vector_search_diagnostics=vector_search_diagnostics,
        )

    def _run_integrity_check_once(self) -> None:
        if self._db_manager is None or self._embedder is None or self._vector_store is None:
            return
        connection = self._db_manager.get_connection()
        quick_check = connection.execute("PRAGMA quick_check").fetchone()
        if quick_check is not None and str(quick_check[0]).lower() != "ok":
            raise sqlite3.DatabaseError(f"quick_check_failed:{quick_check[0]}")
        self._vector_store.search(
            source_kind="memory",
            model_name=self._embedder.model_name,
            query_embedding=[0.0],
            limit=1,
        )

    def _retry_after_reopen(self, reason: str, operation):
        if self._db_manager is None:
            return None
        try:
            self._db_manager.close()
            result = operation()
        except (OSError, sqlite3.Error, ValueError) as retry_exc:
            logger.warning("Semantic search recovery after %s failed: %s", reason, retry_exc)
            return None
        self._mark_semantic_recovered()
        logger.info("Semantic search recovered after %s by reopening the SQLite connection", reason)
        return result

    def _mark_semantic_failure(self, exc: Exception, *, fallback: bool) -> None:
        self._health.available = False
        self._health.degraded = fallback or self._health.degraded
        self._health.last_error = str(exc)
        self._health.last_failure_at = _utc_now()
        if fallback:
            self._health.fallback_count += 1

    def _mark_semantic_recovered(self) -> None:
        self._health.available = self._health.semantic_enabled
        self._health.degraded = False
        self._health.last_error = None
        self._health.last_failure_at = None
        self._health.last_recovery_at = _utc_now()
        self._health.integrity_check_error = None

    def _semantic_candidate_ids(
        self,
        query: str,
        workspace_id: str | None,
        *,
        status: str | None,
        include_superseded: bool,
        keyword_ids: Sequence[str],
        query_tokens: Sequence[str],
        requested_limit: int,
        diagnostics: SearchExecutionDiagnostics | None,
        limit: int,
    ) -> tuple[list[str], dict[str, float]]:
        if diagnostics is not None:
            diagnostics.timing_ms.setdefault("semantic_candidate_pool", 0.0)
            diagnostics.timing_ms.setdefault("semantic_embedding_stale_check", 0.0)
            diagnostics.timing_ms.setdefault("semantic_embedding_refresh", 0.0)
            diagnostics.timing_ms.setdefault("semantic_query_embedding", 0.0)
            diagnostics.timing_ms.setdefault("semantic_vector_search", 0.0)
            diagnostics.timing_ms.setdefault("semantic_ranking", 0.0)
        pool_started = time.perf_counter()
        candidate_pool = self._semantic_candidate_pool(
            keyword_ids,
            query_tokens,
            status=status,
            include_superseded=include_superseded,
            requested_limit=requested_limit,
        )
        if diagnostics is not None:
            _record_timing_ms(diagnostics.timing_ms, "semantic_candidate_pool", pool_started)
            diagnostics.semantic_candidate_strategy = candidate_pool.strategy
        if candidate_pool.skip_semantic_scoring:
            return ([], {})
        candidates = candidate_pool.candidates
        bounded_candidate_ids = candidate_pool.candidate_ids
        semantic_timing_ms = diagnostics.timing_ms if diagnostics is not None else None
        if bounded_candidate_ids is not None:
            vector_search_diagnostics = {} if diagnostics is not None else None
            semantic_scores = self._semantic_scores_with_optional_diagnostics(
                query,
                candidates,
                None,
                candidate_ids=bounded_candidate_ids,
                semantic_timing_ms=semantic_timing_ms,
                vector_search_diagnostics=vector_search_diagnostics,
                limit=limit,
            )
            if diagnostics is not None:
                diagnostics.vector_search = vector_search_diagnostics
            if candidate_pool.speculative and self._should_broaden_semantic_search(
                candidates,
                semantic_scores,
                workspace_id=workspace_id,
                requested_limit=requested_limit,
            ):
                fallback_started = time.perf_counter()
                fallback_candidate_ids = self._repository.list_memory_ids(
                    status=status,
                    limit=_strong_keyword_bounded_candidate_cap(requested_limit),
                )
                candidates = self._repository.get_searchable_memories(
                    fallback_candidate_ids,
                    status=status,
                    include_superseded=True,
                )
                vector_search_diagnostics = {} if diagnostics is not None else None
                semantic_scores = self._semantic_scores_with_optional_diagnostics(
                    query,
                    candidates,
                    None,
                    candidate_ids=fallback_candidate_ids,
                    semantic_timing_ms=semantic_timing_ms,
                    vector_search_diagnostics=vector_search_diagnostics,
                    limit=limit,
                )
                if diagnostics is not None:
                    diagnostics.semantic_candidate_strategy = "global-fallback"
                    diagnostics.vector_search = vector_search_diagnostics
                    _record_timing_ms(diagnostics.timing_ms, "semantic_speculative_fallback", fallback_started)
        else:
            vector_search_diagnostics = {} if diagnostics is not None else None
            semantic_scores = self._semantic_scores_with_optional_diagnostics(
                query,
                candidates,
                None,
                semantic_timing_ms=semantic_timing_ms,
                vector_search_diagnostics=vector_search_diagnostics,
                limit=limit,
            )
            if diagnostics is not None:
                diagnostics.vector_search = vector_search_diagnostics
        ranking_started = time.perf_counter()
        ranked_ids = _rank_semantic_candidate_ids(
            candidates,
            semantic_scores,
            workspace_id=workspace_id,
            limit=limit,
            workspace_multiplier=self._config.search_ranking.workspace_multiplier,
        )
        if diagnostics is not None:
            _record_timing_ms(diagnostics.timing_ms, "semantic_ranking", ranking_started)
        return (ranked_ids, semantic_scores)

    def _semantic_candidate_pool(
        self,
        keyword_ids: Sequence[str],
        query_tokens: Sequence[str],
        *,
        status: str | None,
        include_superseded: bool,
        requested_limit: int,
    ) -> SemanticCandidatePool:
        bounded_candidates, speculative = self._bounded_semantic_candidates(
            keyword_ids,
            query_tokens,
            status=status,
            include_superseded=include_superseded,
            requested_limit=requested_limit,
        )
        if bounded_candidates is not None:
            keyword_only_bounded = _is_technical_single_token_query(query_tokens) and not speculative
            return SemanticCandidatePool(
                candidates=bounded_candidates,
                candidate_ids=[candidate.id for candidate in bounded_candidates],
                speculative=speculative,
                skip_semantic_scoring=keyword_only_bounded,
                strategy="keyword-only-bounded" if keyword_only_bounded else ("speculative-bounded" if speculative else "bounded"),
            )
        global_candidate_ids = self._repository.list_memory_ids(status=status, limit=500)
        return SemanticCandidatePool(
            candidates=self._repository.get_searchable_memories(
                global_candidate_ids,
                status=status,
                include_superseded=True,
            ),
            candidate_ids=global_candidate_ids,
            strategy="global",
        )

    def _get_ranking_candidates_with_optional_diagnostics(
        self,
        memory_ids: list[str],
        *,
        status: str | None,
        include_superseded: bool,
        timing_ms: dict[str, float] | None,
    ) -> list[RankedMemoryCandidate]:
        signature = inspect.signature(self._repository.get_ranking_candidates)
        accepts_var_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        optional_kwargs = {"timing_ms": timing_ms}
        filtered_kwargs = (
            optional_kwargs
            if accepts_var_kwargs
            else {
                key: value
                for key, value in optional_kwargs.items()
                if key in signature.parameters
            }
        )
        return self._repository.get_ranking_candidates(
            memory_ids,
            status=status,
            include_superseded=include_superseded,
            **filtered_kwargs,
        )

    def _bounded_semantic_candidates(
        self,
        keyword_ids: Sequence[str],
        query_tokens: Sequence[str],
        *,
        status: str | None,
        include_superseded: bool,
        requested_limit: int,
    ) -> tuple[list[RelationalMemoryRecord] | None, bool]:
        if not keyword_ids or not getattr(self._vector_store, "supports_candidate_filtering", False):
            return None, False

        keyword_candidates = self._repository.get_searchable_memories(
            list(keyword_ids),
            status=status,
            include_superseded=include_superseded,
        )
        if not keyword_candidates:
            return None, False

        candidate_cap = _strong_keyword_bounded_candidate_cap(requested_limit)
        if _is_technical_single_token_query(query_tokens) and len(keyword_candidates) >= max(requested_limit, 1):
            return keyword_candidates[:candidate_cap], False

        evaluated_candidates = keyword_candidates[:LEXICAL_STRENGTH_EVAL_LIMIT]
        strongest_keyword_coverage = max(
            (_keyword_token_coverage(query_tokens, keyword_candidate) for keyword_candidate in evaluated_candidates),
            default=0.0,
        )
        if strongest_keyword_coverage < self._config.search_ranking.keyword_coverage_floor:
            speculative_keyword_floor = (
                self._config.search_ranking.keyword_coverage_floor
                * self._config.search_ranking.adaptive_result_score_ratio_floor
            )
            if (
                strongest_keyword_coverage < speculative_keyword_floor
                or len(keyword_candidates) < max(requested_limit, 1)
            ):
                return None, False
            return keyword_candidates, True
        return keyword_candidates[:candidate_cap], False

    def _should_broaden_semantic_search(
        self,
        candidates: Sequence[RelationalMemoryRecord],
        semantic_scores: dict[str, float],
        *,
        workspace_id: str | None,
        requested_limit: int,
    ) -> bool:
        if not semantic_scores:
            return True
        effective_limit = max(requested_limit, 1)
        ranked_ids = _rank_semantic_candidate_ids(
            list(candidates),
            semantic_scores,
            workspace_id=workspace_id,
            limit=effective_limit,
            workspace_multiplier=self._config.search_ranking.workspace_multiplier,
        )
        if len(ranked_ids) < effective_limit:
            return True

        top_score = semantic_scores.get(ranked_ids[0], 0.0)
        minimum_score = max(
            self._config.search_ranking.adaptive_result_min_score,
            top_score * self._config.search_ranking.adaptive_result_score_ratio_floor,
        )
        cutoff_score = semantic_scores.get(ranked_ids[effective_limit - 1], 0.0)
        return top_score < self._config.search_ranking.adaptive_result_min_score or cutoff_score < minimum_score

    def _expand_graph_candidate_scores(
        self,
        rrf_scores: dict[str, float],
    ) -> dict[str, GraphExpansionInfo]:
        return self._graph_expander.expand(rrf_scores)

    def _ensure_searchable_memory_embeddings(
        self,
        candidates: list[RelationalMemoryRecord],
        *,
        semantic_timing_ms: dict[str, float] | None = None,
    ) -> None:
        blocked_reason = self._blocked_embedding_persistence_reason()
        if blocked_reason is not None:
            raise ValueError(blocked_reason)

        stale_check_started = time.perf_counter()
        stale_or_missing = self._stale_or_missing_embedding_candidates(candidates)
        _record_timing_ms(semantic_timing_ms, "semantic_embedding_stale_check", stale_check_started)
        if not stale_or_missing:
            return

        if self._background_repair_wait_seconds > 0 and self._task_queue is not None and (self._embedding_repair_queue is not None or self._work_items is not None):
            refresh_started = time.perf_counter()
            self._queue_memory_embedding_repairs(stale_or_missing)
            self._wait_for_background_repairs(stale_or_missing)
            _record_timing_ms(semantic_timing_ms, "semantic_embedding_refresh", refresh_started)
            return

        refresh_started = time.perf_counter()
        self._ensure_memory_embeddings(stale_or_missing)
        _record_timing_ms(semantic_timing_ms, "semantic_embedding_refresh", refresh_started)

    def _stale_or_missing_embedding_candidates(self, candidates: list[RelationalMemoryRecord]) -> list[RelationalMemoryRecord]:
        assert self._embedder is not None
        assert self._vector_store is not None

        get_memory_updated_at_map = getattr(self._vector_store, "get_memory_updated_at_map", None)
        memory_updated_at_by_id: dict[str, str | None] | None = None
        if callable(get_memory_updated_at_map):
            memory_updated_at_by_id = cast(dict[str, str | None], get_memory_updated_at_map(
                source_kind="memory",
                model_name=self._embedder.model_name,
                source_ids=[candidate.id for candidate in candidates],
            ))
        get_updated_at_map = getattr(self._vector_store, "get_updated_at_map", None)
        updated_at_by_id: dict[str, float] | None = None
        if callable(get_updated_at_map):
            updated_at_by_id = cast(dict[str, float], get_updated_at_map(
                source_kind="memory",
                model_name=self._embedder.model_name,
                source_ids=[candidate.id for candidate in candidates],
            ))

        stale_or_missing: list[RelationalMemoryRecord] = []
        for candidate in candidates:
            if memory_updated_at_by_id is not None:
                if memory_updated_at_by_id.get(candidate.id) == candidate.updated_at:
                    continue
                if memory_updated_at_by_id.get(candidate.id) is not None:
                    stale_or_missing.append(candidate)
                    continue
            existing_updated_at = (
                updated_at_by_id.get(candidate.id)
                if updated_at_by_id is not None
                else None
            )
            if updated_at_by_id is None:
                existing = self._vector_store.get(
                    source_kind="memory",
                    source_id=candidate.id,
                    model_name=self._embedder.model_name,
                )
                existing_updated_at = None if existing is None else cast(Any, existing).updated_at
            if existing_updated_at is None or _embedding_is_stale(existing_updated_at, candidate.updated_at):
                stale_or_missing.append(candidate)

        return stale_or_missing

    def _ensure_memory_embeddings(self, stale_or_missing: list[RelationalMemoryRecord]) -> None:
        assert self._embedder is not None
        assert self._vector_store is not None

        if not stale_or_missing:
            return

        if len(stale_or_missing) > INLINE_EMBEDDING_REPAIR_LIMIT:
            prioritized_candidates = _prioritize_embedding_repairs(stale_or_missing)
            logger.info(
                "Semantic search deferred %s embedding refreshes; repairing the freshest %s inline",
                len(stale_or_missing) - INLINE_EMBEDDING_REPAIR_LIMIT,
                INLINE_EMBEDDING_REPAIR_LIMIT,
            )
            stale_or_missing = prioritized_candidates[:INLINE_EMBEDDING_REPAIR_LIMIT]

        embed_and_upsert(
            [
                EmbeddingInput(
                    source_kind="memory",
                    source_id=candidate.id,
                    text=_memory_embedding_text(candidate),
                )
                for candidate in stale_or_missing
            ],
            provider=self._embedder,
            sink=self._vector_store,
            batch_size=len(stale_or_missing),
        )

    def _queue_memory_embedding_repairs(self, candidates: list[RelationalMemoryRecord]) -> None:
        if self._repair_scheduler is None or self._embedder is None:
            return
        self._repair_scheduler.schedule(candidates, self._embedder.model_name)

    def _wait_for_background_repairs(self, candidates: list[RelationalMemoryRecord]) -> None:
        if self._repair_scheduler is None:
            return
        result = self._repair_scheduler.wait_for(candidates, self._stale_or_missing_embedding_candidates)
        self._health.repair_wait_count += 1
        self._health.last_repair_wait_seconds = result.elapsed_seconds
        self._health.last_repair_candidate_count = len(candidates)
        self._health.last_repair_pending_count = result.pending_count
        if result.pending_count:
            self._health.partial_semantic_search_count += 1
            self._health.last_partial_semantic_at = _utc_now()
        self._refresh_background_repair_health_snapshot()

    def _refresh_background_repair_health_snapshot(self) -> None:
        if not self._health.semantic_enabled or self._repair_scheduler is None:
            self._health.queued_repair_backlog_count = 0
            self._health.running_repair_count = 0
            self._health.oldest_queued_repair_age_seconds = None
            return
        snapshot = self._repair_scheduler.backlog_snapshot()
        self._health.queued_repair_backlog_count = snapshot.queued_count
        self._health.running_repair_count = snapshot.running_count
        self._health.oldest_queued_repair_age_seconds = snapshot.oldest_queued_age_seconds



def _memory_embedding_text(record: RelationalMemoryRecord) -> str:
    tag_text = ", ".join(record.tags)
    return "\n".join(
        part
        for part in [
            record.title,
            record.summary or "",
            record.content,
            f"tags: {tag_text}" if tag_text else "",
            f"type: {record.type}",
        ]
        if part
    )


def _search_result_summary(record: RelationalMemoryRecord):
    if record.summary:
        return record.summary
    return _smart_truncate(record.content)


def _smart_truncate(text: str, *, max_chars: int = 200):
    normalized = text.strip()
    if len(normalized) <= max_chars:
        return normalized

    truncated = normalized[:max_chars].rstrip()
    sentence_end = max(truncated.rfind("."), truncated.rfind("?"), truncated.rfind("!"))
    if sentence_end >= 0:
        candidate = truncated[: sentence_end + 1].rstrip()
        if candidate:
            return candidate
    return f"{truncated}…"


def _query_tokens(query: str) -> list[str]:
    return [match.group(0).lower() for match in FTS_QUERY_TOKEN_PATTERN.finditer(query)]


def _is_technical_single_token_query(query_tokens: Sequence[str]) -> bool:
    return len(query_tokens) == 1 and any(character in query_tokens[0] for character in "_:-")


def _keyword_token_coverage(
    query_tokens: Sequence[str],
    record: RankedMemoryCandidate | RelationalMemoryRecord,
) -> float:
    if not query_tokens:
        return 0.0
    resolved_record = record.record if isinstance(record, RankedMemoryCandidate) else record
    title_coverage = _token_presence_ratio(query_tokens, resolved_record.title)
    summary_coverage = _token_presence_ratio(query_tokens, resolved_record.summary or "")
    content_coverage = _token_presence_ratio(query_tokens, resolved_record.content)
    tag_coverage = _token_presence_ratio(query_tokens, " ".join(resolved_record.tags))
    coverage = (
        (0.25 * title_coverage)
        + (0.4 * summary_coverage)
        + (0.25 * content_coverage)
        + (0.1 * tag_coverage)
    )
    if title_coverage >= 0.75:
        coverage += 0.1
    if summary_coverage >= 0.75:
        coverage += 0.15
    if content_coverage >= 0.75:
        coverage += 0.05
    return min(coverage, 1.0)


def _token_presence_ratio(query_tokens: Sequence[str], text: str) -> float:
    if not query_tokens or not text.strip():
        return 0.0
    haystack = text.lower()
    matched_tokens = sum(1 for token in query_tokens if token in haystack)
    return matched_tokens / len(query_tokens)


def _strong_keyword_bounded_candidate_cap(requested_limit: int) -> int:
    effective_limit = max(requested_limit, 1)
    return max(effective_limit * STRONG_KEYWORD_BOUNDED_CANDIDATE_MULTIPLIER, MIN_STRONG_KEYWORD_BOUNDED_CANDIDATES)


def _direct_match_rank(signals: RankingSignals | None) -> int:
    if signals is None:
        return 0
    if signals.matched_by_keyword and signals.matched_by_semantic:
        return 3
    if signals.matched_by_keyword:
        return 2
    if signals.matched_by_semantic:
        return 1
    return 0


def _decayed_access_score(access_score: float, last_accessed_at: str | None):
    return _decayed_access_score_for_half_life(
        access_score,
        last_accessed_at,
        half_life_days=ACCESS_HALF_LIFE_DAYS,
    )


def _decayed_access_score_for_half_life(access_score: float, last_accessed_at: str | None, *, half_life_days: float):
    if access_score <= 0 or not last_accessed_at:
        return max(access_score, 0.0)

    try:
        accessed_at = datetime.fromisoformat(last_accessed_at)
    except ValueError:
        return max(access_score, 0.0)

    if accessed_at.tzinfo is None:
        accessed_at = accessed_at.replace(tzinfo=timezone.utc)

    elapsed = datetime.now(timezone.utc) - accessed_at
    elapsed_days = max(elapsed / timedelta(days=1), 0.0)
    return access_score * (0.5 ** (elapsed_days / half_life_days))


def _embedding_is_stale(embedding_updated_at: float, memory_updated_at: str | None) -> bool:
    if not memory_updated_at:
        return False
    try:
        updated_at = datetime.fromisoformat(memory_updated_at)
    except ValueError:
        return False
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=timezone.utc)
    return embedding_updated_at + 1e-6 < updated_at.timestamp()


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _rank_semantic_candidate_ids(
    candidates: list[RelationalMemoryRecord],
    semantic_scores: dict[str, float],
    *,
    workspace_id: str | None,
    limit: int,
    workspace_multiplier: float,
) -> list[str]:
    candidate_by_id = {candidate.id: candidate for candidate in candidates}

    def _semantic_rank_key(item: tuple[str, float]) -> tuple[float, float, str, str]:
        memory_id, score = item
        candidate = candidate_by_id.get(memory_id)
        workspace_boost = workspace_multiplier if candidate is not None and workspace_id and workspace_id in candidate.workspace_ids else 1.0
        updated_at = candidate.updated_at if candidate is not None else ""
        return (score * workspace_boost, score, updated_at, memory_id)

    ranked = sorted(
        semantic_scores.items(),
        key=_semantic_rank_key,
        reverse=True,
    )
    return [memory_id for memory_id, _ in ranked[:limit]]


def _count_links_by_type(links: Sequence[MemoryLink]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for link in links:
        counts[link.link_type] = counts.get(link.link_type, 0) + 1
    return counts


def _prioritize_embedding_repairs(candidates: Sequence[RelationalMemoryRecord]) -> list[RelationalMemoryRecord]:
    def _repair_priority(record: RelationalMemoryRecord) -> tuple[float, float, str]:
        return (
            _iso_timestamp_to_sortable_float(record.updated_at),
            _iso_timestamp_to_sortable_float(record.created_at),
            record.id,
        )

    return sorted(candidates, key=_repair_priority, reverse=True)


def _iso_timestamp_to_sortable_float(value: str | None) -> float:
    if not value:
        return float("-inf")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return float("-inf")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _weighted_incoming_link_count(counts: dict[str, int]) -> float:
    return float(counts.get("DEPENDS_ON", 0) + counts.get("AMENDS", 0)) + (0.5 * float(counts.get("CONTRADICTS", 0)))


def _record_timing_ms(timing_ms: dict[str, float] | None, key: str, started_at: float) -> None:
    if timing_ms is None:
        return
    elapsed_ms = round((time.perf_counter() - started_at) * 1000.0, 3)
    timing_ms[key] = round(timing_ms.get(key, 0.0) + elapsed_ms, 3)
