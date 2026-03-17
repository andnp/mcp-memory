from __future__ import annotations

import logging
import math
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone

from mcp_memory.config import Config
from mcp_memory.embeddings import Embedder, SQLiteVectorStore
from mcp_memory.relational.repository import MemoryLink, RankedMemoryCandidate, RelationalMemoryRecord, RelationalMemoryRepository
from mcp_memory.utils.db import DatabaseManager


ACCESS_HALF_LIFE_DAYS = 7
DEGRADATION_PENALTY = 0.3
WORKSPACE_BOOST = 1.2
GRAPH_EXPANSION_MAX_SEEDS = 3
GRAPH_EXPANSION_MAX_NEIGHBORS_PER_SEED = 10
GRAPH_EXPANSION_DISCOUNTS = {
    "DEPENDS_ON": 0.7,
    "AMENDS": 0.6,
    "CONTRADICTS": 0.35,
}


logger = logging.getLogger(__name__)


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


@dataclass(slots=True)
class SearchHealthStatus:
    semantic_enabled: bool
    available: bool
    degraded: bool = False
    fallback_count: int = 0
    rebuild_count: int = 0
    last_error: str | None = None
    last_failure_at: str | None = None
    last_recovery_at: str | None = None
    last_integrity_check_at: str | None = None
    integrity_check_error: str | None = None


@dataclass(slots=True)
class GraphExpansionInfo:
    rrf_score: float
    seed_id: str
    link_type: str


@dataclass(slots=True)
class ScoringWeights:
    rrf_k: float = 60.0
    calibration_threshold: float = 0.035
    calibration_steepness: float = 150.0
    workspace_multiplier: float = WORKSPACE_BOOST
    degradation_multiplier: float = DEGRADATION_PENALTY
    access_half_life_days: float = ACCESS_HALF_LIFE_DAYS
    access_bonus_scale: float = 0.1
    authority_link_step: float = 0.02
    authority_link_cap: int = 10

    @classmethod
    def from_config(cls, config: Config):
        ranking = config.search_ranking
        return cls(
            rrf_k=ranking.rrf_k,
            calibration_threshold=ranking.calibration_threshold,
            calibration_steepness=ranking.calibration_steepness,
            workspace_multiplier=ranking.workspace_multiplier,
            degradation_multiplier=ranking.degradation_multiplier,
            access_half_life_days=ranking.access_half_life_days,
            access_bonus_scale=ranking.access_bonus_scale,
            authority_link_step=ranking.authority_link_step,
            authority_link_cap=ranking.authority_link_cap,
        )


class RankingEngine:
    def __init__(
        self,
        repository: RelationalMemoryRepository,
        config: Config,
        *,
        weights: ScoringWeights | None = None,
    ) -> None:
        self._repository = repository
        self._config = config
        self._weights = weights or ScoringWeights.from_config(config)

    def fuse_reciprocal_rank(
        self,
        vector_ranked_ids: list[str],
        keyword_ranked_ids: list[str],
    ) -> dict[str, float]:
        fused: dict[str, float] = {}
        for ranked_ids in (vector_ranked_ids, keyword_ranked_ids):
            for rank, memory_id in enumerate(ranked_ids, start=1):
                fused[memory_id] = fused.get(memory_id, 0.0) + (1.0 / (self._weights.rrf_k + rank))
        return fused

    def calibrate_score(self, rrf_score: float) -> float:
        exponent = -self._weights.calibration_steepness * (rrf_score - self._weights.calibration_threshold)
        return 1.0 / (1.0 + math.exp(exponent))

    def type_aware_recency_bonus(self, record: RelationalMemoryRecord) -> float:
        try:
            created_at = datetime.fromisoformat(record.created_at)
        except ValueError:
            return 0.0

        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)

        age_days = max((datetime.now(timezone.utc) - created_at).days, 0)
        recency = self._config.memory.get_recency_config(record.type)
        return recency.max_boost_amount * (recency.boost_decay_rate ** age_days)

    def workspace_multiplier(self, record: RelationalMemoryRecord, workspace_id: str | None) -> float:
        if workspace_id is None or workspace_id not in record.workspace_ids:
            return 1.0
        return self._weights.workspace_multiplier

    def access_bonus(self, record: RelationalMemoryRecord) -> float:
        current_access_score = _decayed_access_score_for_half_life(
            record.access_score,
            record.last_accessed_at,
            half_life_days=self._weights.access_half_life_days,
        )
        if current_access_score <= 0:
            return 0.0
        return self._weights.access_bonus_scale * math.log10(current_access_score + 1.0)

    def authority_multiplier(self, record: RelationalMemoryRecord) -> float:
        incoming_links_count = self._repository.count_incoming_links(record.id)
        capped_links = min(incoming_links_count, self._weights.authority_link_cap)
        return 1.0 + (capped_links * self._weights.authority_link_step)

    def authority_multiplier_for_candidate(self, candidate: RankedMemoryCandidate | RelationalMemoryRecord) -> float:
        if isinstance(candidate, RankedMemoryCandidate):
            weighted_links = _weighted_incoming_link_count(candidate.incoming_link_type_counts)
            incoming_links_count = weighted_links if weighted_links > 0 else float(candidate.incoming_links_count)
        else:
            incoming_links = self._repository.get_links(candidate.id, direction="incoming")
            incoming_links_count = _weighted_incoming_link_count(_count_links_by_type(incoming_links))
        capped_links = min(incoming_links_count, self._weights.authority_link_cap)
        return 1.0 + (capped_links * self._weights.authority_link_step)

    def degradation_multiplier(self, record: RelationalMemoryRecord) -> float:
        if record.status not in {"stale", "degraded"}:
            return 1.0
        return self._weights.degradation_multiplier

    def score_from_rrf(
        self,
        record: RelationalMemoryRecord,
        rrf_score: float,
        workspace_id: str | None = None,
    ) -> float:
        score = self.calibrate_score(rrf_score)
        score = min(score + self.type_aware_recency_bonus(record), 1.0)
        score *= self.workspace_multiplier(record, workspace_id)
        score += self.access_bonus(record)
        score *= self.authority_multiplier(record)
        score *= self.degradation_multiplier(record)
        return min(max(score, 0.0), 1.0)

    def rank_records(
        self,
        records: Sequence[RankedMemoryCandidate | RelationalMemoryRecord],
        rrf_scores: dict[str, float],
        workspace_id: str | None = None,
    ) -> list[tuple[RelationalMemoryRecord, float]]:
        ranked: list[tuple[RelationalMemoryRecord, float]] = []
        for candidate in records:
            record = candidate.record if isinstance(candidate, RankedMemoryCandidate) else candidate
            if record.id not in rrf_scores:
                continue
            score = self.calibrate_score(rrf_scores[record.id])
            score = min(score + self.type_aware_recency_bonus(record), 1.0)
            score *= self.workspace_multiplier(record, workspace_id)
            score += self.access_bonus(record)
            score *= self.authority_multiplier_for_candidate(candidate)
            score *= self.degradation_multiplier(record)
            ranked.append((record, min(max(score, 0.0), 1.0)))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked

    def explain_candidate(
        self,
        candidate: RankedMemoryCandidate | RelationalMemoryRecord,
        rrf_score: float,
        workspace_id: str | None = None,
    ) -> dict[str, float | str | bool]:
        record = candidate.record if isinstance(candidate, RankedMemoryCandidate) else candidate
        calibrated_score = self.calibrate_score(rrf_score)
        recency_bonus = self.type_aware_recency_bonus(record)
        workspace_multiplier = self.workspace_multiplier(record, workspace_id)
        access_bonus = self.access_bonus(record)
        authority_multiplier = self.authority_multiplier_for_candidate(candidate)
        authority_counts = (
            candidate.incoming_link_type_counts
            if isinstance(candidate, RankedMemoryCandidate)
            else _count_links_by_type(self._repository.get_links(record.id, direction="incoming"))
        )
        degradation_multiplier = self.degradation_multiplier(record)
        score_after_recency = min(calibrated_score + recency_bonus, 1.0)
        final_score = score_after_recency
        final_score *= workspace_multiplier
        final_score += access_bonus
        final_score *= authority_multiplier
        final_score *= degradation_multiplier
        final_score = min(max(final_score, 0.0), 1.0)
        return {
            "rrf_score": round(rrf_score, 6),
            "calibrated_score": round(calibrated_score, 6),
            "recency_bonus": round(recency_bonus, 6),
            "workspace_multiplier": round(workspace_multiplier, 6),
            "access_bonus": round(access_bonus, 6),
            "authority_multiplier": round(authority_multiplier, 6),
            "authority_supporting_links": round(
                authority_counts.get("DEPENDS_ON", 0) + authority_counts.get("AMENDS", 0),
                6,
            ),
            "authority_contradicting_links": round(authority_counts.get("CONTRADICTS", 0), 6),
            "authority_superseding_links": round(authority_counts.get("SUPERSEDES", 0), 6),
            "degradation_multiplier": round(degradation_multiplier, 6),
            "final_score": round(final_score, 6),
            "memory_type": record.type,
            "status": record.status,
            "workspace_match": bool(workspace_id and workspace_id in record.workspace_ids),
        }


class RelationalMemorySearchService:
    def __init__(
        self,
        repository: RelationalMemoryRepository,
        config: Config,
        *,
        embedder: Embedder | None = None,
        vector_store: SQLiteVectorStore | None = None,
        db_manager: DatabaseManager | None = None,
    ) -> None:
        self._repository = repository
        self._config = config
        self._embedder = embedder
        self._vector_store = vector_store
        self._db_manager = db_manager
        semantic_enabled = embedder is not None and vector_store is not None
        self._health = SearchHealthStatus(
            semantic_enabled=semantic_enabled,
            available=semantic_enabled,
        )

    def get_health(self) -> SearchHealthStatus:
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
        memory_type: str | None = None,
        status: str | None = None,
        include_superseded: bool = False,
        debug: bool = False,
    ):
        if not query.strip():
            return []

        keyword_ids = self._repository.search_keyword_memory_ids(
            query,
            status=status,
            include_superseded=include_superseded,
            limit=50,
        )
        semantic_ids = self._semantic_candidate_ids(
            query,
            workspace_id=workspace_id,
            status=status,
            limit=50,
        )
        engine = RankingEngine(self._repository, self._config)
        rrf_scores = engine.fuse_reciprocal_rank(semantic_ids, keyword_ids)
        if not rrf_scores:
            return []
        graph_expansion = self._expand_graph_candidate_scores(rrf_scores)
        for memory_id, expansion in graph_expansion.items():
            rrf_scores[memory_id] = max(rrf_scores.get(memory_id, 0.0), expansion.rrf_score)

        candidates = self._repository.get_ranking_candidates(
            list(rrf_scores.keys()),
            status=status,
            include_superseded=include_superseded,
        )
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
                    engine.explain_candidate(candidate_by_id[record.id], rrf_scores[record.id], workspace_id)
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
            for record, score in engine.rank_records(candidates, rrf_scores, workspace_id)
        ]
        ranked.sort(key=lambda item: item.score, reverse=True)
        ranked = ranked[:limit]

        surfaced_ids = [result.memory_id for result in ranked]
        if surfaced_ids:
            self._repository.touch_last_surfaced(surfaced_ids, _utc_now())
        return ranked

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

    def _semantic_scores(
        self,
        query: str,
        candidates: list[RelationalMemoryRecord],
        workspace_id: str | None,
        *,
        limit: int,
    ) -> dict[str, float]:
        if self._embedder is None or self._vector_store is None or not candidates:
            return {}
        try:
            matches = self._compute_semantic_matches(query, candidates, workspace_id, limit=limit)
        except (OSError, sqlite3.Error, ValueError) as exc:
            recovered_matches = self._retry_after_reopen(
                "semantic search",
                lambda: self._compute_semantic_matches(query, candidates, workspace_id, limit=limit),
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

    def _compute_semantic_matches(
        self,
        query: str,
        candidates: list[RelationalMemoryRecord],
        workspace_id: str | None,
        *,
        limit: int,
    ) -> list[tuple[str, float]]:
        assert self._embedder is not None
        assert self._vector_store is not None
        self._ensure_memory_embeddings(candidates)
        query_embedding = self._embedder.embed([query])[0]
        return self._vector_store.search(
            source_kind="memory",
            model_name=self._embedder.model_name,
            query_embedding=query_embedding,
            workspace_id=workspace_id,
            limit=limit,
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
        limit: int,
    ) -> list[str]:
        candidates = self._repository.list_memories(status=status, limit=500)
        semantic_scores = self._semantic_scores(query, candidates, None, limit=limit)
        return _rank_semantic_candidate_ids(
            candidates,
            semantic_scores,
            workspace_id=workspace_id,
            limit=limit,
            workspace_multiplier=self._config.search_ranking.workspace_multiplier,
        )

    def _expand_graph_candidate_scores(
        self,
        rrf_scores: dict[str, float],
    ) -> dict[str, GraphExpansionInfo]:
        expanded: dict[str, GraphExpansionInfo] = {}
        seed_items = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)[:GRAPH_EXPANSION_MAX_SEEDS]
        for seed_id, seed_score in seed_items:
            links = self._repository.get_links(seed_id, direction="outgoing")
            expanded_neighbors = 0
            for link in links:
                discount = GRAPH_EXPANSION_DISCOUNTS.get(link.link_type)
                if discount is None:
                    continue
                expanded_neighbors += 1
                if expanded_neighbors > GRAPH_EXPANSION_MAX_NEIGHBORS_PER_SEED:
                    break
                expanded_score = seed_score * discount
                current = expanded.get(link.target_id)
                if current is None or expanded_score > current.rrf_score:
                    expanded[link.target_id] = GraphExpansionInfo(
                        rrf_score=expanded_score,
                        seed_id=seed_id,
                        link_type=link.link_type,
                    )
        return expanded

    def _ensure_memory_embeddings(self, candidates: list[RelationalMemoryRecord]) -> None:
        assert self._embedder is not None
        assert self._vector_store is not None

        stale_or_missing: list[RelationalMemoryRecord] = []
        for candidate in candidates:
            existing = self._vector_store.get(
                source_kind="memory",
                source_id=candidate.id,
                model_name=self._embedder.model_name,
            )
            if existing is None or _embedding_is_stale(existing.updated_at, candidate.updated_at):
                stale_or_missing.append(candidate)

        if not stale_or_missing:
            return

        payloads = [_memory_embedding_text(candidate) for candidate in stale_or_missing]
        embeddings = self._embedder.embed(payloads)
        for candidate, embedding in zip(stale_or_missing, embeddings, strict=False):
            self._vector_store.upsert(
                source_kind="memory",
                source_id=candidate.id,
                workspace_id=candidate.workspace_ids[0] if candidate.workspace_ids else None,
                model_name=self._embedder.model_name,
                embedding=embedding,
            )

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


def _weighted_incoming_link_count(counts: dict[str, int]) -> float:
    return float(counts.get("DEPENDS_ON", 0) + counts.get("AMENDS", 0)) + (0.5 * float(counts.get("CONTRADICTS", 0)))
