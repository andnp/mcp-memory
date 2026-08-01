"""Pure ranking and scoring for memory search.

This module deliberately contains no repository or storage dependencies.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from mcp_memory.config import Config
from mcp_memory.core.ports.memory import (
    FTS_QUERY_TOKEN_PATTERN,
    RankedMemoryCandidate,
    MemoryRecord,
)
from searchkernel.search.fusion import fuse_reciprocal_rank

ACCESS_HALF_LIFE_DAYS = 7
DEGRADATION_PENALTY = 0.3
WORKSPACE_BOOST = 1.2
RelationalMemoryRecord = MemoryRecord


def _query_tokens(query: str) -> list[str]:
    return [match.group(0).lower() for match in FTS_QUERY_TOKEN_PATTERN.finditer(query)]


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

@dataclass(slots=True)
class RankingSignals:
    matched_by_keyword: bool = False
    matched_by_semantic: bool = False
    semantic_score: float = 0.0
    keyword_token_coverage: float = 0.0
    expanded_by_graph: bool = False


@dataclass(slots=True)
class ScoringWeights:
    rrf_k: float = 60.0
    calibration_threshold: float = 0.035
    calibration_steepness: float = 150.0
    workspace_multiplier: float = WORKSPACE_BOOST
    semantic_only_abstain_threshold: float = 0.8
    semantic_only_keyword_penalty: float = 0.65
    keyword_coverage_floor: float = 0.6
    keyword_low_coverage_penalty: float = 0.7
    graph_expansion_only_penalty: float = 0.35
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
            semantic_only_abstain_threshold=ranking.semantic_only_abstain_threshold,
            semantic_only_keyword_penalty=ranking.semantic_only_keyword_penalty,
            keyword_coverage_floor=ranking.keyword_coverage_floor,
            keyword_low_coverage_penalty=ranking.keyword_low_coverage_penalty,
            graph_expansion_only_penalty=ranking.graph_expansion_only_penalty,
            degradation_multiplier=ranking.degradation_multiplier,
            access_half_life_days=ranking.access_half_life_days,
            access_bonus_scale=ranking.access_bonus_scale,
            authority_link_step=ranking.authority_link_step,
            authority_link_cap=ranking.authority_link_cap,
        )


class RankingEngine:
    def __init__(
        self,
        config: Config,
        *,
        weights: ScoringWeights | None = None,
    ) -> None:
        self._config = config
        self._weights = weights or ScoringWeights.from_config(config)

    def fuse_reciprocal_rank(
        self,
        vector_ranked_ids: list[str],
        keyword_ranked_ids: list[str],
    ) -> dict[str, float]:
        return fuse_reciprocal_rank(
            (vector_ranked_ids, keyword_ranked_ids),
            k=self._weights.rrf_k,
        )

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

    def adjusted_access_bonus(self, record: RelationalMemoryRecord, authority_counts: dict[str, int]) -> float:
        bonus = self.access_bonus(record)
        supporting_links = authority_counts.get("DEPENDS_ON", 0) + authority_counts.get("AMENDS", 0)
        if supporting_links <= 0 and record.type in {"journal", "plan"}:
            return bonus * 0.5
        return bonus

    def authority_multiplier(self, record: RelationalMemoryRecord) -> float:
        return 1.0

    def authority_multiplier_for_candidate(self, candidate: RankedMemoryCandidate | RelationalMemoryRecord) -> float:
        if isinstance(candidate, RankedMemoryCandidate):
            weighted_links = _weighted_incoming_link_count(candidate.incoming_link_type_counts)
            incoming_links_count = weighted_links if weighted_links > 0 else float(candidate.incoming_links_count)
        else:
            incoming_links_count = 0.0
        capped_links = min(incoming_links_count, self._weights.authority_link_cap)
        return 1.0 + (capped_links * self._weights.authority_link_step)

    def degradation_multiplier(self, record: RelationalMemoryRecord) -> float:
        if record.status not in {"stale", "degraded"}:
            return 1.0
        return self._weights.degradation_multiplier

    def graph_support_bonus(self, record: RelationalMemoryRecord, authority_counts: dict[str, int]) -> float:
        supporting_links = authority_counts.get("DEPENDS_ON", 0) + authority_counts.get("AMENDS", 0)
        if supporting_links <= 0 or record.type not in {"fact", "observation", "reflection"}:
            return 0.0
        return min((supporting_links * 0.08) + 0.04, 0.2)

    def score_from_rrf(
        self,
        record: RelationalMemoryRecord,
        rrf_score: float,
        workspace_id: str | None = None,
    ) -> float:
        authority_counts = {}
        score = self.calibrate_score(rrf_score)
        score = min(score + self.type_aware_recency_bonus(record) + self.graph_support_bonus(record, authority_counts), 1.0)
        score *= self.workspace_multiplier(record, workspace_id)
        score += self.adjusted_access_bonus(record, authority_counts)
        score *= self.authority_multiplier(record)
        score *= self.degradation_multiplier(record)
        return min(max(score, 0.0), 1.0)

    def _signal_adjustment_multiplier(
        self,
        signals: RankingSignals,
        *,
        keyword_candidates_present: bool,
    ) -> float:
        if not keyword_candidates_present:
            if signals.expanded_by_graph and not signals.matched_by_semantic:
                return self._weights.graph_expansion_only_penalty
            return 1.0
        if signals.expanded_by_graph and not signals.matched_by_keyword and not signals.matched_by_semantic:
            return self._weights.graph_expansion_only_penalty
        if signals.matched_by_semantic and not signals.matched_by_keyword:
            return self._weights.semantic_only_keyword_penalty
        if not signals.matched_by_keyword:
            return 1.0
        if signals.keyword_token_coverage >= self._weights.keyword_coverage_floor:
            return 1.0
        return max(self._weights.keyword_low_coverage_penalty, signals.keyword_token_coverage)

    def rank_records(
        self,
        records: Sequence[RankedMemoryCandidate | RelationalMemoryRecord],
        rrf_scores: dict[str, float],
        workspace_id: str | None = None,
        *,
        ranking_signals: dict[str, RankingSignals] | None = None,
        keyword_candidates_present: bool = False,
    ) -> list[tuple[RelationalMemoryRecord, float]]:
        ranked: list[tuple[RelationalMemoryRecord, float]] = []
        for candidate in records:
            record = candidate.record if isinstance(candidate, RankedMemoryCandidate) else candidate
            if record.id not in rrf_scores:
                continue
            authority_counts = (
                candidate.incoming_link_type_counts
                if isinstance(candidate, RankedMemoryCandidate)
                else {}
            )
            score = self.calibrate_score(rrf_scores[record.id])
            score = min(score + self.type_aware_recency_bonus(record) + self.graph_support_bonus(record, authority_counts), 1.0)
            score *= self.workspace_multiplier(record, workspace_id)
            score += self.adjusted_access_bonus(record, authority_counts)
            score *= self.authority_multiplier_for_candidate(candidate)
            score *= self.degradation_multiplier(record)
            signals = ranking_signals.get(record.id, RankingSignals()) if ranking_signals is not None else RankingSignals()
            score *= self._signal_adjustment_multiplier(signals, keyword_candidates_present=keyword_candidates_present)
            ranked.append((record, min(max(score, 0.0), 1.0)))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked

    def explain_candidate(
        self,
        candidate: RankedMemoryCandidate | RelationalMemoryRecord,
        rrf_score: float,
        workspace_id: str | None = None,
        *,
        signals: RankingSignals | None = None,
        keyword_candidates_present: bool = False,
    ) -> dict[str, float | str | bool]:
        record = candidate.record if isinstance(candidate, RankedMemoryCandidate) else candidate
        calibrated_score = self.calibrate_score(rrf_score)
        recency_bonus = self.type_aware_recency_bonus(record)
        workspace_multiplier = self.workspace_multiplier(record, workspace_id)
        authority_counts = (
            candidate.incoming_link_type_counts
            if isinstance(candidate, RankedMemoryCandidate)
            else {}
        )
        support_bonus = self.graph_support_bonus(record, authority_counts)
        access_bonus = self.adjusted_access_bonus(record, authority_counts)
        authority_multiplier = self.authority_multiplier_for_candidate(candidate)
        degradation_multiplier = self.degradation_multiplier(record)
        score_after_recency = min(calibrated_score + recency_bonus + support_bonus, 1.0)
        final_score = score_after_recency
        final_score *= workspace_multiplier
        final_score += access_bonus
        final_score *= authority_multiplier
        final_score *= degradation_multiplier
        active_signals = signals or RankingSignals()
        signal_multiplier = self._signal_adjustment_multiplier(
            active_signals,
            keyword_candidates_present=keyword_candidates_present,
        )
        final_score *= signal_multiplier
        final_score = min(max(final_score, 0.0), 1.0)
        return {
            "rrf_score": round(rrf_score, 6),
            "calibrated_score": round(calibrated_score, 6),
            "recency_bonus": round(recency_bonus, 6),
            "graph_support_bonus": round(support_bonus, 6),
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
            "keyword_token_coverage": round(active_signals.keyword_token_coverage, 6),
            "semantic_score": round(active_signals.semantic_score, 6),
            "ranking_signal_multiplier": round(signal_multiplier, 6),
            "final_score": round(final_score, 6),
            "memory_type": record.type,
            "status": record.status,
            "workspace_match": bool(workspace_id and workspace_id in record.workspace_ids),
        }


def _decayed_access_score_for_half_life(access_score: float, last_accessed_at: str | None, *, half_life_days: float) -> float:
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


def _count_links_by_type(links: Sequence[object]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for link in links:
        link_type = getattr(link, "link_type")
        counts[link_type] = counts.get(link_type, 0) + 1
    return counts


def _weighted_incoming_link_count(counts: dict[str, int]) -> float:
    return float(counts.get("DEPENDS_ON", 0) + counts.get("AMENDS", 0)) + (0.5 * float(counts.get("CONTRADICTS", 0)))


__all__ = [
    "RankingEngine",
    "RankingSignals",
    "ScoringWeights",
    "_keyword_token_coverage",
    "_query_tokens",
]
