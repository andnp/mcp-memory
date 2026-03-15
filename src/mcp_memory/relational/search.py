from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from mcp_memory.config import Config
from mcp_memory.embeddings import Embedder, SQLiteVectorStore
from mcp_memory.relational.repository import MemoryLink, RankedMemoryCandidate, RelationalMemoryRecord, RelationalMemoryRepository


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_:-]+")
ACCESS_HALF_LIFE_DAYS = 7
DEGRADATION_PENALTY = 0.3
WORKSPACE_BOOST = 1.2
MEMORY_TYPE_BOOST = 1.15


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


@dataclass(slots=True)
class RelationalReadResult:
    record: RelationalMemoryRecord
    relationships: dict[str, list[MemoryLink]]
    superseded: list[RelationalMemoryRecord]


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
        self._weights = weights or ScoringWeights()

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
            incoming_links_count = candidate.incoming_links_count
        else:
            incoming_links_count = self._repository.count_incoming_links(candidate.id)
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
        records: list[RankedMemoryCandidate | RelationalMemoryRecord],
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


class RelationalMemorySearchService:
    def __init__(
        self,
        repository: RelationalMemoryRepository,
        config: Config,
        *,
        embedder: Embedder | None = None,
        vector_store: SQLiteVectorStore | None = None,
    ) -> None:
        self._repository = repository
        self._config = config
        self._embedder = embedder
        self._vector_store = vector_store

    def search_memories(
        self,
        query: str,
        workspace_id: str | None = None,
        limit: int = 5,
        memory_type: str | None = None,
        status: str | None = None,
        include_superseded: bool = False,
    ):
        if not query.strip():
            return []

        tokens = _tokenize(query)
        keyword_ids = self._repository.search_keyword_memory_ids(
            query,
            status=status,
            include_superseded=include_superseded,
            limit=50,
        )
        semantic_ids = self._semantic_candidate_ids(
            query,
            status=status,
            limit=50,
        )
        lexical_ids = self._lexical_candidate_ids(
            tokens,
            status=status,
            include_superseded=include_superseded,
            limit=50,
        )
        engine = RankingEngine(self._repository, self._config)
        rrf_scores = engine.fuse_reciprocal_rank(semantic_ids, keyword_ids)
        for memory_id in lexical_ids:
            rrf_scores.setdefault(memory_id, 0.0)
        if not rrf_scores:
            return []

        candidates = self._repository.get_ranking_candidates(
            list(rrf_scores.keys()),
            status=status,
            include_superseded=include_superseded,
        )
        ranked = [
            RelationalSearchResult(
                memory_id=record.id,
                title=record.title,
                summary=record.summary or "",
                memory_type=record.type,
                status=record.status,
                tags=list(record.tags),
                workspace_ids=list(record.workspace_ids),
                score=round(
                    _apply_memory_type_boost(
                        max(score, _normalize_base_score(_base_match_score(record, tokens))),
                        record,
                        memory_type,
                    ),
                    6,
                ),
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

        self._ensure_memory_embeddings(candidates)
        query_embedding = self._embedder.embed([query])[0]
        matches = self._vector_store.search(
            source_kind="memory",
            model_name=self._embedder.model_name,
            query_embedding=query_embedding,
            workspace_id=workspace_id,
            limit=limit,
        )
        return {
            memory_id: max(min((score + 1.0) / 2.0, 1.0), 0.0)
            for memory_id, score in matches
            if score > 0
        }

    def _semantic_candidate_ids(
        self,
        query: str,
        *,
        status: str | None,
        limit: int,
    ) -> list[str]:
        candidates = self._repository.list_memories(status=status, limit=500)
        semantic_scores = self._semantic_scores(query, candidates, None, limit=limit)
        ranked = sorted(semantic_scores.items(), key=lambda item: item[1], reverse=True)
        return [memory_id for memory_id, _ in ranked[:limit]]

    def _lexical_candidate_ids(
        self,
        tokens: list[str],
        *,
        status: str | None,
        include_superseded: bool,
        limit: int,
    ) -> list[str]:
        candidates = self._repository.list_memories(status=status, limit=500)
        scored: list[tuple[str, float]] = []
        for candidate in candidates:
            if not include_superseded and self._repository.has_incoming_link(candidate.id, "SUPERSEDES"):
                continue
            base_score = _base_match_score(candidate, tokens)
            if base_score <= 0:
                continue
            scored.append((candidate.id, base_score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [memory_id for memory_id, _ in scored[:limit]]

    def _ensure_memory_embeddings(self, candidates: list[RelationalMemoryRecord]) -> None:
        assert self._embedder is not None
        assert self._vector_store is not None

        missing: list[RelationalMemoryRecord] = []
        for candidate in candidates:
            existing = self._vector_store.get(
                source_kind="memory",
                source_id=candidate.id,
                model_name=self._embedder.model_name,
            )
            if existing is None:
                missing.append(candidate)

        if not missing:
            return

        payloads = [_memory_embedding_text(candidate) for candidate in missing]
        embeddings = self._embedder.embed(payloads)
        for candidate, embedding in zip(missing, embeddings, strict=False):
            self._vector_store.upsert(
                source_kind="memory",
                source_id=candidate.id,
                workspace_id=candidate.workspace_ids[0] if candidate.workspace_ids else None,
                model_name=self._embedder.model_name,
                embedding=embedding,
            )


def _tokenize(query: str):
    return [token.lower() for token in TOKEN_PATTERN.findall(query)]


def _base_match_score(record: RelationalMemoryRecord, tokens: list[str]):
    title = record.title.lower()
    summary = (record.summary or "").lower()
    content = record.content.lower()
    tags = {tag.lower() for tag in record.tags}

    score = 0.0
    for token in tokens:
        if token in title:
            score += 3.0
        if token in summary:
            score += 2.0
        if token in content:
            score += 1.0
        if token in tags:
            score += 1.5
    return score


def _normalize_base_score(score: float):
    return score / (score + 3.0)


def _apply_recency_boost(score: float, record: RelationalMemoryRecord, config: Config):
    try:
        created_at = datetime.fromisoformat(record.created_at)
    except ValueError:
        return score

    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)

    age_days = (datetime.now(timezone.utc) - created_at).days
    recency = config.memory.get_recency_config(record.type)
    if age_days > recency.boost_window_days:
        return score

    bonus = (recency.boost_decay_rate ** age_days) * recency.max_boost_amount
    return min(1.0, score + bonus)


def _apply_access_boost(score: float, record: RelationalMemoryRecord):
    decayed = _decayed_access_score(record.access_score, record.last_accessed_at)
    if decayed <= 0:
        return score
    return min(1.0, score + min(0.2, math.log1p(decayed) / 10.0))


def _apply_graph_authority_boost(
    score: float,
    record: RelationalMemoryRecord,
    repository: RelationalMemoryRepository,
):
    incoming_links = repository.count_incoming_links(record.id)
    if incoming_links <= 0:
        return score
    multiplier = 1.0 + min(0.15, math.log1p(incoming_links) / 10.0)
    return min(1.0, score * multiplier)


def _apply_memory_type_boost(score: float, record: RelationalMemoryRecord, memory_type: str | None):
    if memory_type is None or record.type != memory_type:
        return score
    return score * MEMORY_TYPE_BOOST


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


def _utc_now():
    return datetime.now(timezone.utc).isoformat()
