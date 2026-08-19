from __future__ import annotations

from dataclasses import dataclass

import pytest

from mcp_memory.toolkit.maintenance_batch_selector import MaintenanceBatchSelector

pytestmark = pytest.mark.small


@dataclass(frozen=True)
class _FakeBatch:
    strategy_used: str
    strategy_selection_scores: dict[str, float] | None = None
    strategy_selection_reason: str | None = None
    sampler_priority_score: float | None = None
    sampler_priority_explanation: str | None = None


class _FakeBatchProvider:
    def __init__(self, batch: _FakeBatch) -> None:
        self._batch = batch
        self.calls: list[dict[str, object]] = []

    def get_batch(
        self,
        *,
        strategy: str | None,
        allowed_strategies: tuple[str, ...],
        limit: int,
    ) -> _FakeBatch:
        self.calls.append({"strategy": strategy, "allowed_strategies": allowed_strategies, "limit": limit})
        return self._batch


def test_select_returns_empty_batch_without_calling_priority_feedback_when_no_candidates() -> None:
    feedback_calls: list[None] = []

    def priority_feedback_source() -> dict[str, tuple[float, str]]:
        feedback_calls.append(None)
        return {}

    def empty_batch_factory(strategy: str | None) -> _FakeBatch:
        return _FakeBatch(strategy_used=strategy or "none")

    selector: MaintenanceBatchSelector[str, _FakeBatch] = MaintenanceBatchSelector(
        requested_strategy="semantic",
        candidates=[],
        batch_provider_factory=lambda _scores: _FakeBatchProvider(_FakeBatch(strategy_used="unused")),
        priority_feedback_source=priority_feedback_source,
        empty_batch_factory=empty_batch_factory,
    )

    result = selector.select(allowed_strategies=("semantic",), limit=5)

    assert result == _FakeBatch(strategy_used="semantic")
    assert feedback_calls == []


def test_select_merges_priority_feedback_score_for_chosen_strategy() -> None:
    provider = _FakeBatchProvider(_FakeBatch(strategy_used="anomaly"))

    selector: MaintenanceBatchSelector[str, _FakeBatch] = MaintenanceBatchSelector(
        requested_strategy=None,
        candidates=["a", "b"],
        batch_provider_factory=lambda _scores: provider,
        priority_feedback_source=lambda: {"anomaly": (0.82, "quality_pass_rate=0.900")},
        empty_batch_factory=lambda strategy: _FakeBatch(strategy_used=strategy or "none"),
    )

    result = selector.select(allowed_strategies=("anomaly", "semantic"), limit=3)

    assert result.sampler_priority_score == 0.82
    assert result.sampler_priority_explanation == "quality_pass_rate=0.900"
    assert provider.calls == [{"strategy": None, "allowed_strategies": ("anomaly", "semantic"), "limit": 3}]


def test_select_falls_back_to_batch_selection_scores_without_priority_feedback() -> None:
    batch = _FakeBatch(
        strategy_used="cold-storage",
        strategy_selection_scores={"cold-storage": 0.4},
        strategy_selection_reason="selected=cold-storage",
    )
    provider = _FakeBatchProvider(batch)

    selector: MaintenanceBatchSelector[str, _FakeBatch] = MaintenanceBatchSelector(
        requested_strategy=None,
        candidates=["a"],
        batch_provider_factory=lambda _scores: provider,
        priority_feedback_source=lambda: {},
        empty_batch_factory=lambda strategy: _FakeBatch(strategy_used=strategy or "none"),
    )

    result = selector.select(allowed_strategies=("cold-storage",), limit=1)

    assert result.sampler_priority_score == 0.4
    assert result.sampler_priority_explanation == "selected=cold-storage"


def test_select_passes_priority_feedback_scores_into_batch_provider_factory() -> None:
    seen_scores: list[dict[str, float]] = []

    def batch_provider_factory(scores: dict[str, float]) -> _FakeBatchProvider:
        seen_scores.append(scores)
        return _FakeBatchProvider(_FakeBatch(strategy_used="semantic"))

    selector: MaintenanceBatchSelector[str, _FakeBatch] = MaintenanceBatchSelector(
        requested_strategy=None,
        candidates=["a"],
        batch_provider_factory=batch_provider_factory,
        priority_feedback_source=lambda: {"semantic": (0.6, "reason")},
        empty_batch_factory=lambda strategy: _FakeBatch(strategy_used=strategy or "none"),
    )

    selector.select(allowed_strategies=("semantic",), limit=1)

    assert seen_scores == [{"semantic": 0.6}]
