from __future__ import annotations

import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from random import Random
from statistics import median
from typing import Any, Generic, Protocol, TypeVar


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_:-]+")

CURATOR_LOW_SUPPORT_THRESHOLD = 1
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
        self._candidates = list(candidates)
        self._support_counts = support_counts or {}
        self._strategy_prior_scores = dict(strategy_prior_scores or {})
        self._rng = Random(f"{task_name}:{task_id}")
        self._cooldown_window_seconds = max(cooldown_window_seconds, 0.0)
        self._now_timestamp = time.time() if now_timestamp is None else now_timestamp

    def get_batch(
        self,
        *,
        strategy: str | None,
        allowed_strategies: tuple[str, ...],
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
            )
        elif strategy_used not in ALL_STRATEGIES:
            fallback_reason = "unknown_requested_strategy"
            strategy_used, selection_mode, selection_reason, selection_scores, selection_signals = self._select_strategy(
                allowed_strategies,
            )
        elif strategy_used not in allowed_strategies:
            fallback_reason = "disallowed_requested_strategy"
            strategy_used, selection_mode, selection_reason, selection_scores, selection_signals = self._select_strategy(
                allowed_strategies,
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
    ) -> tuple[str, str | None, str | None, dict[str, float] | None, dict[str, float] | None]:
        priority_scores = {
            strategy: round(min(max(self._strategy_prior_scores[strategy], 0.0), 1.0), 4)
            for strategy in allowed_strategies
            if strategy in self._strategy_prior_scores
        }
        if priority_scores:
            selection_scores = {
                strategy: priority_scores.get(strategy, SELECTION_EXPLORATION_BONUS)
                for strategy in allowed_strategies
            }
            strategy_used = min(
                allowed_strategies,
                key=lambda strategy: (-selection_scores[strategy], allowed_strategies.index(strategy)),
            )
            is_exploration = strategy_used not in priority_scores
            selection_mode = "priority_scores_with_exploration" if is_exploration else "priority_scores"
            selection_reason = (
                f"selected={strategy_used}; priority_score={selection_scores[strategy_used]:.4f}; "
                f"exploration={'bounded' if is_exploration else 'none'}"
            )
            return (
                strategy_used,
                selection_mode,
                selection_reason,
                {strategy: round(selection_scores[strategy], 4) for strategy in allowed_strategies},
                None,
            )

        strategy_used = BOUNDED_NOISE_STRATEGY if BOUNDED_NOISE_STRATEGY in allowed_strategies else allowed_strategies[0]
        return (
            strategy_used,
            "bounded_exploration",
            f"selected={strategy_used}; fallback=no_priority_feedback; exploration=bounded",
            {strategy: SELECTION_EXPLORATION_BONUS for strategy in allowed_strategies},
            None,
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
        return round(sum(predicate(record) for record in records) / len(records), 4)


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