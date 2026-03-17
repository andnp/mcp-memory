from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from random import Random
import time
import re
from statistics import median
from typing import Generic, Protocol, TypeVar


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_:-]+")

SEMANTIC_STRATEGY = "semantic"
COLD_STORAGE_STRATEGY = "cold-storage"
NEVER_SURFACED_STRATEGY = "never-surfaced"
ANOMALY_STRATEGY = "anomaly"
BOUNDED_NOISE_STRATEGY = "bounded-noise"
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


class RouletteProvider(Generic[T]):
    def __init__(
        self,
        *,
        task_name: str,
        task_id: str,
        candidates: Sequence[T],
        support_counts: dict[str, int] | None = None,
        cooldown_window_seconds: float = 21600.0,
        now_timestamp: float | None = None,
    ) -> None:
        self._task_name = task_name
        self._task_id = task_id
        self._candidates = list(candidates)
        self._support_counts = support_counts or {}
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
        if strategy_used is None:
            strategy_used = self.choose_strategy(allowed_strategies, strategy_weights)
        elif strategy_used not in ALL_STRATEGIES:
            fallback_reason = "unknown_requested_strategy"
            strategy_used = self.choose_strategy(allowed_strategies, strategy_weights)
        elif strategy_used not in allowed_strategies:
            fallback_reason = "disallowed_requested_strategy"
            strategy_used = self.choose_strategy(allowed_strategies, strategy_weights)

        ranked = self._rank_candidates(strategy_used)
        return SamplingBatch(
            requested_strategy=requested_strategy,
            strategy_used=strategy_used,
            strategy_fallback_reason=fallback_reason,
            candidate_count=len(self._candidates),
            records=ranked[:limit],
        )

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