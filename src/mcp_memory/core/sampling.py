from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from random import Random
import time
import re
from statistics import median
from typing import Any, Callable, Generic, Protocol, TypeVar


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_:-]+")

CURATOR_TASK_NAME = "memory-curator"
DEDUPLICATOR_TASK_NAME = "deduplicator"
TAXONOMIST_TASK_NAME = "taxonomist"
GRAPH_LINKER_TASK_NAME = "graph-linker"
CONFLICT_DETECTOR_TASK_NAME = "conflict-detector"
DEFRAGMENTER_TASK_NAME = "defragmenter"
CURATOR_COLD_TAIL_SECONDS = 30 * 24 * 60 * 60
TAXONOMIST_COLD_TAIL_SECONDS = 30 * 24 * 60 * 60
GRAPH_LINKER_CLUSTER_THRESHOLD = 0.2
GRAPH_LINKER_BRIDGE_THRESHOLD = 0.12
CONFLICT_FRONTIER_THRESHOLD = 0.18
CURATOR_LOW_SUPPORT_THRESHOLD = 1
CURATOR_LOW_READ_THRESHOLD = 3
CURATOR_LARGE_CANDIDATE_MIN_CHARS = 1200
CURATOR_OVERSIZED_MULTIPLIER = 1.75
CURATOR_LENGTH_OUTLIER_RATIO = 0.75
CURATOR_SEMANTIC_CLUSTER_THRESHOLD = 0.2
DEDUPLICATOR_LARGE_FACT_MIN_CHARS = 1200
DEDUPLICATOR_OVERSIZED_MULTIPLIER = 1.6
DEDUPLICATOR_LENGTH_OUTLIER_RATIO = 0.7
DEDUPLICATOR_SEMANTIC_CLUSTER_THRESHOLD = 0.2
SELECTION_UTILITY_PRIOR_BLEND_WEIGHT = 0.35
SELECTION_UTILITY_PRIOR_MAX_SCORE_SHIFT = 0.12
SELECTION_EXPLORATION_BONUS = 0.02

SEMANTIC_STRATEGY = "semantic"
COLD_STORAGE_STRATEGY = "cold-storage"
NEVER_SURFACED_STRATEGY = "never-surfaced"
ANOMALY_STRATEGY = "anomaly"
BOUNDED_NOISE_STRATEGY = "bounded-noise"
QUALITY_SIGNAL_STRATEGY = "quality-signal"
GRAPH_BRIDGE_STRATEGY = "graph-bridge"
ORPHAN_LOW_SUPPORT_STRATEGY = "orphan/low-support"
COOLDOWN_ESCAPE_STRATEGY = "cooldown-escape"
CONFLICT_FRONTIER_STRATEGY = "conflict-frontier"

ALL_STRATEGIES = (
    SEMANTIC_STRATEGY,
    COLD_STORAGE_STRATEGY,
    NEVER_SURFACED_STRATEGY,
    ANOMALY_STRATEGY,
    BOUNDED_NOISE_STRATEGY,
    QUALITY_SIGNAL_STRATEGY,
    GRAPH_BRIDGE_STRATEGY,
    ORPHAN_LOW_SUPPORT_STRATEGY,
    COOLDOWN_ESCAPE_STRATEGY,
    CONFLICT_FRONTIER_STRATEGY,
)


class SamplingCandidate(Protocol):
    @property
    def id(self) -> str: ...

    @property
    def title(self) -> str: ...

    @property
    def content(self) -> str: ...

    @property
    def type(self) -> str: ...

    @property
    def read_count(self) -> int: ...

    @property
    def access_score(self) -> float: ...

    @property
    def updated_at(self) -> str: ...

    @property
    def last_accessed_at(self) -> str | None: ...

    @property
    def last_surfaced_at(self) -> str | None: ...

    @property
    def tags(self) -> list[str]: ...


T = TypeVar("T", bound=SamplingCandidate)


@dataclass(frozen=True)
class SamplingBatch(Generic[T]):
    requested_strategy: str | None
    strategy_used: str
    strategy_fallback_reason: str | None
    candidate_count: int
    records: list[T]
    strategy_selection_mode: str | None = None
    strategy_selection_reason: str | None = None
    strategy_selection_scores: dict[str, float] | None = None
    selector_feature_snapshot: dict[str, Any] | None = None
    sampler_priority_score: float | None = None
    sampler_priority_explanation: str | None = None


class RouletteProvider(Generic[T]):
    def __init__(
        self,
        *,
        task_name: str,
        task_id: str,
        candidates: Sequence[T],
        support_counts: dict[str, int] | None = None,
        strategy_prior_scores: dict[str, float] | None = None,
        cooldown_window_seconds: float = 21600.0,
        now_timestamp: float | None = None,
    ) -> None:
        self._task_name = task_name
        self._task_id = task_id
        self._candidates = list(candidates)
        self._support_counts = support_counts or {}
        self._strategy_prior_scores = dict(strategy_prior_scores or {})
        self._rng = Random(f"{task_name}:{task_id}")
        self._cooldown_window_seconds = max(cooldown_window_seconds, 0.0)
        self._now_timestamp = time.time() if now_timestamp is None else now_timestamp

    def choose_strategy(
        self,
        allowed_strategies: tuple[str, ...],
        strategy_weights: dict[str, int] | None = None,
    ) -> str:
        if not allowed_strategies:
            raise ValueError("allowed_strategies must be non-empty")
        weights = [max(int((strategy_weights or {}).get(strategy, 1)), 0) for strategy in allowed_strategies]
        if not any(weights):
            return self._rng.choice(list(allowed_strategies))
        return self._rng.choices(list(allowed_strategies), weights=weights, k=1)[0]

    def get_batch(
        self,
        *,
        strategy: str | None,
        allowed_strategies: tuple[str, ...],
        strategy_weights: dict[str, int] | None = None,
        limit: int,
    ) -> SamplingBatch[T]:
        if not self._candidates or limit <= 0:
            return SamplingBatch(
                requested_strategy=strategy,
                strategy_used=strategy or "none",
                strategy_fallback_reason=None,
                candidate_count=len(self._candidates),
                records=[],
            )

        requested_strategy = strategy.strip() if isinstance(strategy, str) and strategy.strip() else None
        strategy_used = requested_strategy
        fallback_reason = None
        selection_mode = None
        selection_reason = None
        selection_scores = None
        selection_signals = None
        if strategy_used is None:
            strategy_used, selection_mode, selection_reason, selection_scores, selection_signals = self._select_strategy(
                allowed_strategies,
                strategy_weights,
            )
        elif strategy_used not in ALL_STRATEGIES:
            fallback_reason = "unknown_requested_strategy"
            strategy_used, selection_mode, selection_reason, selection_scores, selection_signals = self._select_strategy(
                allowed_strategies,
                strategy_weights,
            )
        elif strategy_used not in allowed_strategies:
            fallback_reason = "disallowed_requested_strategy"
            strategy_used, selection_mode, selection_reason, selection_scores, selection_signals = self._select_strategy(
                allowed_strategies,
                strategy_weights,
            )
        else:
            selection_mode = "requested_strategy"
            selection_reason = "requested_strategy_preserved"

        ranked = self._rank_candidates(strategy_used)
        selected_records = ranked[:limit]
        return SamplingBatch(
            requested_strategy=requested_strategy,
            strategy_used=strategy_used,
            strategy_fallback_reason=fallback_reason,
            candidate_count=len(self._candidates),
            records=selected_records,
            strategy_selection_mode=selection_mode,
            strategy_selection_reason=selection_reason,
            strategy_selection_scores=selection_scores,
            selector_feature_snapshot=self._build_selector_feature_snapshot(
                selected_records,
                strategy_signals=selection_signals,
            ),
        )

    def _select_strategy(
        self,
        allowed_strategies: tuple[str, ...],
        strategy_weights: dict[str, int] | None,
    ) -> tuple[str, str | None, str | None, dict[str, float] | None, dict[str, float] | None]:
        if self._task_name == CURATOR_TASK_NAME:
            return self._choose_curator_strategy(allowed_strategies)
        if self._task_name == DEDUPLICATOR_TASK_NAME:
            return self._choose_deduplicator_strategy(allowed_strategies)
        if self._task_name == TAXONOMIST_TASK_NAME:
            return self._choose_taxonomist_strategy(allowed_strategies)
        if self._task_name == DEFRAGMENTER_TASK_NAME:
            return self._choose_defragmenter_strategy(allowed_strategies)
        if self._task_name == GRAPH_LINKER_TASK_NAME:
            return self._choose_graph_linker_strategy(allowed_strategies)
        if self._task_name == CONFLICT_DETECTOR_TASK_NAME:
            return self._choose_conflict_detector_strategy(allowed_strategies)
        return (
            self.choose_strategy(allowed_strategies, strategy_weights),
            "seeded_random",
            None,
            None,
            None,
        )

    def _choose_curator_strategy(
        self,
        allowed_strategies: tuple[str, ...],
    ) -> tuple[str, str, str, dict[str, float], dict[str, float]]:
        scores, signals, applied_utility_priors = self._curator_strategy_scores(allowed_strategies)
        return self._finalize_deterministic_strategy_choice(
            allowed_strategies,
            scores=scores,
            signals=signals,
            applied_utility_priors=applied_utility_priors,
        )

    def _choose_deduplicator_strategy(
        self,
        allowed_strategies: tuple[str, ...],
    ) -> tuple[str, str, str, dict[str, float], dict[str, float]]:
        scores, signals, applied_utility_priors = self._deduplicator_strategy_scores(allowed_strategies)
        return self._finalize_deterministic_strategy_choice(
            allowed_strategies,
            scores=scores,
            signals=signals,
            applied_utility_priors=applied_utility_priors,
        )

    def _choose_taxonomist_strategy(
        self,
        allowed_strategies: tuple[str, ...],
    ) -> tuple[str, str, str, dict[str, float], dict[str, float]]:
        scores, signals, applied_utility_priors = self._taxonomist_strategy_scores(allowed_strategies)
        return self._finalize_deterministic_strategy_choice(
            allowed_strategies,
            scores=scores,
            signals=signals,
            applied_utility_priors=applied_utility_priors,
        )

    def _choose_graph_linker_strategy(
        self,
        allowed_strategies: tuple[str, ...],
    ) -> tuple[str, str, str, dict[str, float], dict[str, float]]:
        scores, signals = self._graph_linker_strategy_scores(allowed_strategies)
        return self._finalize_deterministic_strategy_choice(
            allowed_strategies,
            scores=scores,
            signals=signals,
            applied_utility_priors={},
        )

    def _choose_defragmenter_strategy(
        self,
        allowed_strategies: tuple[str, ...],
    ) -> tuple[str, str, str, dict[str, float], dict[str, float]]:
        scores, signals = self._defragmenter_strategy_scores(allowed_strategies)
        return self._finalize_deterministic_strategy_choice(
            allowed_strategies,
            scores=scores,
            signals=signals,
            applied_utility_priors={},
        )

    def _choose_conflict_detector_strategy(
        self,
        allowed_strategies: tuple[str, ...],
    ) -> tuple[str, str, str, dict[str, float], dict[str, float]]:
        scores, signals = self._conflict_detector_strategy_scores(allowed_strategies)
        return self._finalize_deterministic_strategy_choice(
            allowed_strategies,
            scores=scores,
            signals=signals,
            applied_utility_priors={},
        )

    def _finalize_deterministic_strategy_choice(
        self,
        allowed_strategies: tuple[str, ...],
        *,
        scores: dict[str, float],
        signals: dict[str, float],
        applied_utility_priors: dict[str, float],
    ) -> tuple[str, str, str, dict[str, float], dict[str, float]]:
        exploration_scores = (
            {
                strategy: min(
                    max(scores.get(strategy, 0.0) + self._rng.random() * SELECTION_EXPLORATION_BONUS, 0.0),
                    1.0,
                )
                for strategy in allowed_strategies
            }
            if applied_utility_priors
            else scores
        )
        strategy_used = min(
            allowed_strategies,
            key=lambda strategy: (-exploration_scores[strategy], allowed_strategies.index(strategy), strategy),
        )
        rounded_scores = {strategy: round(exploration_scores[strategy], 3) for strategy in allowed_strategies}
        rounded_signals = {signal_name: round(signal_value, 3) for signal_name, signal_value in sorted(signals.items())}
        dominant_signals = ", ".join(
            f"{signal_name}={signal_value:.3f}"
            for signal_name, signal_value in sorted(signals.items(), key=lambda item: (-item[1], item[0]))[:3]
        )
        selection_mode = "deterministic_scores"
        selection_reason = f"selected={strategy_used}; signals={dominant_signals}"
        if applied_utility_priors:
            selection_mode = "deterministic_scores_with_utility_priors"
            compact_priors = ",".join(
                f"{strategy}:{score:.2f}"
                for strategy, score in sorted(
                    applied_utility_priors.items(),
                    key=lambda item: (-item[1], item[0]),
                )[:3]
            )
            selection_reason = f"{selection_reason}; utility_priors={compact_priors}"
        return (
            strategy_used,
            selection_mode,
            selection_reason,
            rounded_scores,
            rounded_signals,
        )

    def _build_selector_feature_snapshot(
        self,
        selected_records: Sequence[T],
        *,
        strategy_signals: dict[str, float] | None,
    ) -> dict[str, Any]:
        return {
            "strategy_signals": {
                signal_name: round(signal_value, 3)
                for signal_name, signal_value in sorted((strategy_signals or {}).items())
            },
            "candidate_population": self._build_population_feature_snapshot(self._candidates),
            "selected_population": self._build_population_feature_snapshot(selected_records),
        }

    def _build_population_feature_snapshot(self, records: Sequence[T]) -> dict[str, Any]:
        record_list = list(records)
        metrics: dict[str, dict[str, float | int | None]] = {}

        self._add_metric_summary(
            metrics,
            "content_chars",
            [float(len(item.content.strip())) for item in record_list],
        )
        self._add_metric_summary(
            metrics,
            "updated_age_seconds",
            [age for item in record_list if (age := self._age_seconds(item.updated_at)) is not None],
        )
        self._add_metric_summary(
            metrics,
            "last_access_age_seconds",
            [age for item in record_list if (age := self._age_seconds(item.last_accessed_at)) is not None],
        )
        self._add_metric_summary(
            metrics,
            "last_surfaced_age_seconds",
            [age for item in record_list if (age := self._age_seconds(item.last_surfaced_at)) is not None],
        )
        self._add_metric_summary(metrics, "read_count", [float(item.read_count) for item in record_list])
        self._add_metric_summary(
            metrics,
            "support_count",
            [float(self._support_counts.get(item.id, 0)) for item in record_list],
        )

        return {
            "count": len(record_list),
            "metrics": metrics,
            "shares": {
                "never_surfaced_share": self._share_for_records(record_list, lambda item: item.last_surfaced_at is None),
                "never_accessed_share": self._share_for_records(record_list, lambda item: item.last_accessed_at is None),
                "cooldown_share": self._share_for_records(record_list, self._is_in_cooldown),
                "low_support_share": self._share_for_records(
                    record_list,
                    lambda item: self._support_counts.get(item.id, 0) <= CURATOR_LOW_SUPPORT_THRESHOLD,
                ),
            },
        }

    def _add_metric_summary(
        self,
        metrics: dict[str, dict[str, float | int | None]],
        key: str,
        values: Sequence[float],
    ) -> None:
        summary = self._summarize_numeric_values(values)
        if summary is not None:
            metrics[key] = summary

    def _summarize_numeric_values(self, values: Sequence[float]) -> dict[str, float | int | None] | None:
        if not values:
            return None
        ordered = sorted(values)
        return {
            "count": len(ordered),
            "min": round(ordered[0], 3),
            "p50": round(median(ordered), 3),
            "p90": round(_percentile(ordered, 0.9), 3),
            "max": round(ordered[-1], 3),
            "mean": round(sum(ordered) / len(ordered), 3),
        }

    def _age_seconds(self, value: str | None) -> float | None:
        parsed = _parse_iso_timestamp(value)
        if parsed is None:
            return None
        return max(self._now_timestamp - parsed, 0.0)

    def _share_for_records(self, records: Sequence[T], predicate: Callable[[T], bool]) -> float:
        if not records:
            return 0.0
        matches = 0
        for record in records:
            if predicate(record):
                matches += 1
        return round(matches / len(records), 4)

    def _curator_strategy_scores(
        self,
        allowed_strategies: tuple[str, ...],
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        candidate_count = len(self._candidates)
        lengths = [len(item.content.strip()) for item in self._candidates]
        median_length = median(lengths) if lengths else 0.0
        oversized_threshold = max(median_length * CURATOR_OVERSIZED_MULTIPLIER, CURATOR_LARGE_CANDIDATE_MIN_CHARS)
        outlier_threshold = max(median_length * CURATOR_LENGTH_OUTLIER_RATIO, 400.0)

        oversized_share = self._share(
            1
            for item in self._candidates
            if len(item.content.strip()) >= oversized_threshold
        )
        length_outlier_share = self._share(
            1
            for item in self._candidates
            if abs(len(item.content.strip()) - median_length) >= outlier_threshold
        ) if median_length > 0 else 0.0
        never_surfaced_share = self._share(1 for item in self._candidates if item.last_surfaced_at is None)
        never_accessed_share = self._share(1 for item in self._candidates if item.last_accessed_at is None)
        cold_tail_share = self._share(
            1
            for item in self._candidates
            if self._is_older_than(item.last_accessed_at, CURATOR_COLD_TAIL_SECONDS)
        )
        low_support_share = self._share(
            1
            for item in self._candidates
            if self._support_counts.get(item.id, 0) <= CURATOR_LOW_SUPPORT_THRESHOLD
        )
        retrieval_friction_share = self._share(
            1
            for item in self._candidates
            if item.last_surfaced_at is not None and item.read_count <= CURATOR_LOW_READ_THRESHOLD
        )
        from mcp_memory.core.task_handlers.curator_support import retrieval_friction_flags

        quality_signal_share = self._share(
            1 for item in self._candidates if retrieval_friction_flags(item)
        )
        semantic_cluster_share = self._semantic_cluster_share() if candidate_count > 1 else 0.0

        signals = {
            "never_surfaced_share": never_surfaced_share,
            "low_support_share": low_support_share,
            "never_accessed_share": never_accessed_share,
            "cold_tail_share": cold_tail_share,
            "oversized_share": oversized_share,
            "length_outlier_share": length_outlier_share,
            "retrieval_friction_share": retrieval_friction_share,
            "quality_signal_share": quality_signal_share,
            "semantic_cluster_share": semantic_cluster_share,
        }

        dominant_signal = max(signals.values(), default=0.0)
        scores: dict[str, float] = {}
        for strategy in allowed_strategies:
            if strategy == ANOMALY_STRATEGY:
                scores[strategy] = (
                    0.55 * oversized_share
                    + 0.35 * length_outlier_share
                    + 0.10 * retrieval_friction_share
                )
            elif strategy == COLD_STORAGE_STRATEGY:
                scores[strategy] = (
                    0.60 * cold_tail_share
                    + 0.25 * never_accessed_share
                    + 0.15 * retrieval_friction_share
                )
            elif strategy == NEVER_SURFACED_STRATEGY:
                scores[strategy] = 0.80 * never_surfaced_share + 0.20 * retrieval_friction_share
            elif strategy == ORPHAN_LOW_SUPPORT_STRATEGY:
                scores[strategy] = (
                    0.65 * low_support_share
                    + 0.20 * retrieval_friction_share
                    + 0.15 * never_surfaced_share
                )
            elif strategy == SEMANTIC_STRATEGY:
                scores[strategy] = (
                    0.75 * semantic_cluster_share
                    + 0.15 * retrieval_friction_share
                    + 0.10 * (1.0 - min(oversized_share, 1.0))
                )
            elif strategy == BOUNDED_NOISE_STRATEGY:
                scores[strategy] = 0.08 + 0.22 * (1.0 - dominant_signal)
            elif strategy == QUALITY_SIGNAL_STRATEGY:
                scores[strategy] = (
                    0.70 * quality_signal_share
                    + 0.20 * retrieval_friction_share
                    + 0.10 * oversized_share
                )
            else:
                scores[strategy] = 0.0
        applied_utility_priors = {
            strategy: min(max(self._strategy_prior_scores.get(strategy, 0.0), 0.0), 1.0)
            for strategy in allowed_strategies
            if strategy in self._strategy_prior_scores
        }
        if not applied_utility_priors:
            return scores, signals, {}

        blended_scores = {
            strategy: self._blend_utility_prior_score(scores.get(strategy, 0.0), applied_utility_priors.get(strategy))
            for strategy in allowed_strategies
        }
        return blended_scores, signals, applied_utility_priors

    def _deduplicator_strategy_scores(
        self,
        allowed_strategies: tuple[str, ...],
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        fact_candidates = [item for item in self._candidates if item.type == "fact"]
        fact_lengths = [len(item.content.strip()) for item in fact_candidates]
        median_length = median(fact_lengths) if fact_lengths else 0.0
        oversized_threshold = max(median_length * DEDUPLICATOR_OVERSIZED_MULTIPLIER, DEDUPLICATOR_LARGE_FACT_MIN_CHARS)
        outlier_threshold = max(median_length * DEDUPLICATOR_LENGTH_OUTLIER_RATIO, 300.0)

        semantic_overlap_share = (
            self._semantic_cluster_share(threshold=DEDUPLICATOR_SEMANTIC_CLUSTER_THRESHOLD)
            if len(self._candidates) > 1
            else 0.0
        )
        large_fact_share = self._typed_share(
            fact_candidates,
            lambda item: len(item.content.strip()) >= oversized_threshold,
        )
        fact_length_outlier_share = self._typed_share(
            fact_candidates,
            lambda item: abs(len(item.content.strip()) - median_length) >= outlier_threshold,
        ) if median_length > 0 else 0.0
        size_anomaly_pressure = 0.65 * large_fact_share + 0.35 * fact_length_outlier_share
        cooldown_pressure_share = self._share(1 for item in self._candidates if self._is_in_cooldown(item))

        signals = {
            "semantic_overlap_share": semantic_overlap_share,
            "size_anomaly_pressure": size_anomaly_pressure,
            "cooldown_pressure_share": cooldown_pressure_share,
        }

        scores: dict[str, float] = {}
        for strategy in allowed_strategies:
            if strategy == SEMANTIC_STRATEGY:
                scores[strategy] = (
                    0.72 * semantic_overlap_share
                    + 0.18 * (1.0 - min(size_anomaly_pressure, 1.0))
                    + 0.10 * (1.0 - min(cooldown_pressure_share, 1.0))
                )
            elif strategy == ANOMALY_STRATEGY:
                scores[strategy] = (
                    0.70 * size_anomaly_pressure
                    + 0.20 * semantic_overlap_share
                    + 0.10 * (1.0 - min(cooldown_pressure_share, 1.0))
                )
            elif strategy == COOLDOWN_ESCAPE_STRATEGY:
                scores[strategy] = (
                    0.75 * cooldown_pressure_share
                    + 0.15 * semantic_overlap_share
                    + 0.10 * (1.0 - min(size_anomaly_pressure, 1.0))
                )
            else:
                scores[strategy] = 0.0

        applied_utility_priors = {
            strategy: min(max(self._strategy_prior_scores.get(strategy, 0.0), 0.0), 1.0)
            for strategy in allowed_strategies
            if strategy in self._strategy_prior_scores
        }
        if not applied_utility_priors:
            return scores, signals, {}

        blended_scores = {
            strategy: self._blend_utility_prior_score(scores.get(strategy, 0.0), applied_utility_priors.get(strategy))
            for strategy in allowed_strategies
        }
        return blended_scores, signals, applied_utility_priors

    def _taxonomist_strategy_scores(
        self,
        allowed_strategies: tuple[str, ...],
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        never_surfaced_share = self._share(1 for item in self._candidates if item.last_surfaced_at is None)
        never_accessed_share = self._share(1 for item in self._candidates if item.last_accessed_at is None)
        cold_tail_or_never_accessed_share = self._share(
            1
            for item in self._candidates
            if item.last_accessed_at is None
            or self._is_older_than(item.last_accessed_at, TAXONOMIST_COLD_TAIL_SECONDS)
        )

        dominant_signal = max(never_surfaced_share, cold_tail_or_never_accessed_share)
        frontier_gap = abs(never_surfaced_share - cold_tail_or_never_accessed_share)
        low_signal_share = 1.0 - dominant_signal
        flat_frontier_share = 1.0 - frontier_gap

        signals = {
            "never_surfaced_share": never_surfaced_share,
            "cold_tail_or_never_accessed_share": cold_tail_or_never_accessed_share,
            "never_accessed_share": never_accessed_share,
            "low_signal_share": low_signal_share,
            "flat_frontier_share": flat_frontier_share,
        }

        scores: dict[str, float] = {}
        for strategy in allowed_strategies:
            if strategy == NEVER_SURFACED_STRATEGY:
                scores[strategy] = 0.85 * never_surfaced_share + 0.15 * never_accessed_share
            elif strategy == COLD_STORAGE_STRATEGY:
                scores[strategy] = (
                    0.75 * cold_tail_or_never_accessed_share
                    + 0.25 * never_accessed_share
                )
            elif strategy == BOUNDED_NOISE_STRATEGY:
                scores[strategy] = 0.05 + 0.35 * low_signal_share * flat_frontier_share
            else:
                scores[strategy] = 0.0

        applied_utility_priors = {
            strategy: min(max(self._strategy_prior_scores.get(strategy, 0.0), 0.0), 1.0)
            for strategy in allowed_strategies
            if strategy in self._strategy_prior_scores
        }
        if not applied_utility_priors:
            return scores, signals, {}

        blended_scores = {
            strategy: self._blend_utility_prior_score(scores.get(strategy, 0.0), applied_utility_priors.get(strategy))
            for strategy in allowed_strategies
        }
        return blended_scores, signals, applied_utility_priors

    def _graph_linker_strategy_scores(
        self,
        allowed_strategies: tuple[str, ...],
    ) -> tuple[dict[str, float], dict[str, float]]:
        semantic_cluster_share = self._semantic_cluster_share(threshold=GRAPH_LINKER_CLUSTER_THRESHOLD)
        low_support_adjacent_share = self._share(
            1
            for item in self._candidates
            if self._best_neighbor_similarity(item) >= GRAPH_LINKER_BRIDGE_THRESHOLD
            and self._support_counts.get(item.id, 0) <= CURATOR_LOW_SUPPORT_THRESHOLD
        )
        low_signal_share = 1.0 - max(semantic_cluster_share, low_support_adjacent_share)

        signals = {
            "semantic_cluster_share": semantic_cluster_share,
            "low_support_adjacent_share": low_support_adjacent_share,
            "low_signal_share": low_signal_share,
        }

        scores: dict[str, float] = {}
        for strategy in allowed_strategies:
            if strategy == SEMANTIC_STRATEGY:
                scores[strategy] = (
                    0.70 * semantic_cluster_share
                    + 0.20 * (1.0 - min(low_support_adjacent_share, 1.0))
                    + 0.10 * (1.0 - min(low_signal_share, 1.0))
                )
            elif strategy == GRAPH_BRIDGE_STRATEGY:
                scores[strategy] = (
                    0.75 * low_support_adjacent_share
                    + 0.15 * semantic_cluster_share
                    + 0.10 * (1.0 - min(low_signal_share, 1.0))
                )
            elif strategy == BOUNDED_NOISE_STRATEGY:
                scores[strategy] = 0.06 + 0.28 * low_signal_share
            else:
                scores[strategy] = 0.0
        return scores, signals
    def _defragmenter_strategy_scores(
        self,
        allowed_strategies: tuple[str, ...],
    ) -> tuple[dict[str, float], dict[str, float]]:
        semantic_cluster_share = self._semantic_cluster_share()
        low_support_share = self._share(
            1
            for item in self._candidates
            if self._support_counts.get(item.id, 0) <= CURATOR_LOW_SUPPORT_THRESHOLD
        )
        never_accessed_share = self._share(1 for item in self._candidates if item.last_accessed_at is None)
        cold_tail_share = self._share(
            1
            for item in self._candidates
            if self._is_older_than(item.last_accessed_at, CURATOR_COLD_TAIL_SECONDS)
        )

        signals = {
            "semantic_cluster_share": semantic_cluster_share,
            "low_support_share": low_support_share,
            "never_accessed_share": never_accessed_share,
            "cold_tail_share": cold_tail_share,
        }

        scores: dict[str, float] = {}
        for strategy in allowed_strategies:
            if strategy == SEMANTIC_STRATEGY:
                scores[strategy] = 0.80 * semantic_cluster_share + 0.20 * (1.0 - min(low_support_share, 1.0))
            elif strategy == ORPHAN_LOW_SUPPORT_STRATEGY:
                scores[strategy] = 0.75 * low_support_share + 0.15 * semantic_cluster_share + 0.10 * never_accessed_share
            elif strategy == COLD_STORAGE_STRATEGY:
                scores[strategy] = 0.70 * cold_tail_share + 0.30 * never_accessed_share
            else:
                scores[strategy] = 0.0
        return scores, signals

    def _conflict_detector_strategy_scores(
        self,
        allowed_strategies: tuple[str, ...],
    ) -> tuple[dict[str, float], dict[str, float]]:
        semantic_overlap_share = self._semantic_cluster_share(threshold=DEDUPLICATOR_SEMANTIC_CLUSTER_THRESHOLD)
        conflict_frontier_share = self._share(
            1
            for item in self._candidates
            if self._best_conflict_signal(item) >= CONFLICT_FRONTIER_THRESHOLD
        )
        never_surfaced_share = self._share(1 for item in self._candidates if item.last_surfaced_at is None)

        signals = {
            "conflict_frontier_share": conflict_frontier_share,
            "semantic_overlap_share": semantic_overlap_share,
            "never_surfaced_share": never_surfaced_share,
        }

        scores: dict[str, float] = {}
        for strategy in allowed_strategies:
            if strategy == CONFLICT_FRONTIER_STRATEGY:
                scores[strategy] = (
                    0.75 * conflict_frontier_share
                    + 0.15 * semantic_overlap_share
                    + 0.10 * never_surfaced_share
                )
            elif strategy == SEMANTIC_STRATEGY:
                scores[strategy] = (
                    0.65 * semantic_overlap_share
                    + 0.25 * (1.0 - min(conflict_frontier_share, 1.0))
                    + 0.10 * (1.0 - min(never_surfaced_share, 1.0))
                )
            elif strategy == NEVER_SURFACED_STRATEGY:
                scores[strategy] = (
                    0.80 * never_surfaced_share
                    + 0.10 * (1.0 - min(conflict_frontier_share, 1.0))
                    + 0.10 * (1.0 - min(semantic_overlap_share, 1.0))
                )
            else:
                scores[strategy] = 0.0
        return scores, signals

    def _blend_utility_prior_score(self, live_score: float, prior_score: float | None) -> float:
        if prior_score is None:
            return live_score
        shift = (prior_score - 0.5) * SELECTION_UTILITY_PRIOR_BLEND_WEIGHT
        shift = min(max(shift, -SELECTION_UTILITY_PRIOR_MAX_SCORE_SHIFT), SELECTION_UTILITY_PRIOR_MAX_SCORE_SHIFT)
        return live_score + shift

    def _rank_candidates(self, strategy: str) -> list[T]:
        if strategy == SEMANTIC_STRATEGY:
            return self._semantic_candidates()
        if strategy == COLD_STORAGE_STRATEGY:
            return sorted(self._candidates, key=lambda item: self._timestamp_sort_key(item.last_accessed_at))
        if strategy == NEVER_SURFACED_STRATEGY:
            return sorted(self._candidates, key=lambda item: self._timestamp_sort_key(item.last_surfaced_at))
        if strategy == ANOMALY_STRATEGY:
            return self._anomaly_candidates()
        if strategy == BOUNDED_NOISE_STRATEGY:
            return self._bounded_noise_candidates()
        if strategy == QUALITY_SIGNAL_STRATEGY:
            from mcp_memory.core.task_handlers.curator_support import retrieval_friction_flags

            return sorted(
                self._candidates,
                key=lambda item: (
                    len(retrieval_friction_flags(item)),
                    item.last_surfaced_at is not None,
                    -item.read_count,
                    len(item.content.strip()),
                    item.updated_at,
                    item.id,
                ),
                reverse=True,
            )
        if strategy == GRAPH_BRIDGE_STRATEGY:
            return self._graph_bridge_candidates()
        if strategy == ORPHAN_LOW_SUPPORT_STRATEGY:
            return self._orphan_low_support_candidates()
        if strategy == COOLDOWN_ESCAPE_STRATEGY:
            return sorted(
                self._candidates,
                key=lambda item: (
                    1 if self._is_in_cooldown(item) else 0,
                    item.updated_at,
                    self._support_counts.get(item.id, 0),
                    item.read_count,
                    item.id,
                ),
            )
        if strategy == CONFLICT_FRONTIER_STRATEGY:
            return self._conflict_frontier_candidates()
        return list(self._candidates)

    def _semantic_candidates(self) -> list[T]:
        seed = self._rng.choice(self._candidates)
        return sorted(
            self._candidates,
            key=lambda item: (
                -self._lexical_similarity(seed, item),
                self._timestamp_sort_key(item.last_surfaced_at),
                item.id,
            ),
        )

    def _anomaly_candidates(self) -> list[T]:
        lengths = [len(item.content.strip()) for item in self._candidates]
        pivot = median(lengths) if lengths else 0
        return sorted(
            self._candidates,
            key=lambda item: (
                -abs(len(item.content.strip()) - pivot),
                self._timestamp_sort_key(item.last_surfaced_at),
                item.id,
            ),
        )

    def _bounded_noise_candidates(self) -> list[T]:
        cooler_candidates = [item for item in self._candidates if not self._is_in_cooldown(item)]
        ordered = sorted(cooler_candidates or self._candidates, key=lambda item: (item.updated_at, item.id))
        if len(ordered) <= 2:
            pool = ordered
        else:
            recent_cutoff = max(len(ordered) // 4, 1)
            pool = ordered[:-recent_cutoff] or ordered
        shuffled = list(pool)
        self._rng.shuffle(shuffled)
        if len(pool) == len(self._candidates):
            return shuffled
        omitted_ids = {item.id for item in shuffled}
        remainder = [item for item in self._candidates if item.id not in omitted_ids]
        return [*shuffled, *remainder]

    def _graph_bridge_candidates(self) -> list[T]:
        return sorted(
            self._candidates,
            key=lambda item: (
                -self._best_neighbor_similarity(item),
                self._support_counts.get(item.id, 0),
                self._timestamp_sort_key(item.last_surfaced_at),
                item.id,
            ),
        )

    def _orphan_low_support_candidates(self) -> list[T]:
        return sorted(
            self._candidates,
            key=lambda item: (
                self._support_counts.get(item.id, 0),
                item.read_count,
                item.access_score,
                self._timestamp_sort_key(item.last_surfaced_at),
                item.id,
            ),
        )

    def _conflict_frontier_candidates(self) -> list[T]:
        return sorted(
            [item for item in self._candidates if item.type in {"fact", "plan"}],
            key=lambda item: (
                -self._best_conflict_signal(item),
                self._timestamp_sort_key(item.last_surfaced_at),
                item.id,
            ),
        )

    def _best_neighbor_similarity(self, item: T) -> float:
        return max(
            (self._lexical_similarity(item, other) for other in self._candidates if other.id != item.id),
            default=0.0,
        )

    def _semantic_cluster_share(self, *, threshold: float = CURATOR_SEMANTIC_CLUSTER_THRESHOLD) -> float:
        return self._share(
            1
            for item in self._candidates
            if self._best_neighbor_similarity(item) >= threshold
        )

    def _typed_share(self, items: Sequence[T], predicate: Callable[[T], bool]) -> float:
        if not items:
            return 0.0
        matches = 0
        for item in items:
            if predicate(item):
                matches += 1
        return matches / len(items)

    def _best_conflict_signal(self, item: T) -> float:
        return max(
            (
                self._lexical_similarity(item, other)
                for other in self._candidates
                if other.id != item.id and other.type == item.type and other.content.strip() != item.content.strip()
            ),
            default=0.0,
        )

    def _lexical_similarity(self, left: T, right: T) -> float:
        left_tokens = self._candidate_tokens(left)
        right_tokens = self._candidate_tokens(right)
        if not left_tokens or not right_tokens:
            return 0.0
        return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)

    def _candidate_tokens(self, item: T) -> set[str]:
        payload = " ".join([item.title, item.content, " ".join(item.tags)])
        return {token.lower() for token in TOKEN_PATTERN.findall(payload)}

    def _timestamp_sort_key(self, value: str | None) -> tuple[int, str]:
        if value is None:
            return (0, "")
        return (1, value)

    def _is_in_cooldown(self, item: T) -> bool:
        updated_timestamp = _parse_iso_timestamp(item.updated_at)
        if updated_timestamp is None:
            return False
        return self._now_timestamp - updated_timestamp < self._cooldown_window_seconds

    def _is_older_than(self, value: str | None, age_seconds: float) -> bool:
        parsed = _parse_iso_timestamp(value)
        if parsed is None:
            return False
        return self._now_timestamp - parsed >= age_seconds

    def _share(self, matches: Iterable[object]) -> float:
        count = 0
        for _ in matches:
            count += 1
        if not self._candidates:
            return 0.0
        return count / len(self._candidates)


def _parse_iso_timestamp(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return datetime.fromisoformat(value).astimezone(UTC).timestamp()
    except ValueError:
        return None


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("values must be non-empty")
    index = max(0, min(len(values) - 1, int(round((len(values) - 1) * quantile))))
    return values[index]