from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any, ClassVar, Generic, Protocol, TypeVar

T = TypeVar("T")


class SelectionBatch(Protocol):
    __dataclass_fields__: ClassVar[dict[str, Any]]

    @property
    def strategy_used(self) -> str: ...
    @property
    def strategy_selection_scores(self) -> dict[str, float] | None: ...
    @property
    def strategy_selection_reason(self) -> str | None: ...
    @property
    def sampler_priority_score(self) -> float | None: ...
    @property
    def sampler_priority_explanation(self) -> str | None: ...


B = TypeVar("B", bound=SelectionBatch)
B_co = TypeVar("B_co", bound=SelectionBatch, covariant=True)


class BatchProvider(Protocol[B_co]):
    def get_batch(
        self,
        *,
        strategy: str | None,
        allowed_strategies: tuple[str, ...],
        limit: int,
    ) -> B_co: ...


BatchProviderFactory = Callable[[dict[str, float]], "BatchProvider[B]"]
PriorityFeedbackSource = Callable[[], dict[str, tuple[float, str]]]
EmptyBatchFactory = Callable[[str | None], B]


class MaintenanceBatchSelector(Generic[T, B]):
    """Generic stats-driven maintenance batch selector.

    Combines live candidate-set signal scoring (via ``batch_provider_factory``),
    historical-run priority feedback (via ``priority_feedback_source``), and a
    deterministic empty-candidate fallback into one reusable selection
    contract, decoupled from any single maintenance task's candidate or batch
    types. See docs/adrs/2026-04-01-stats-driven-batch-selection-for-agentic-maintenance.md
    for the rationale this generalizes.
    """

    def __init__(
        self,
        *,
        requested_strategy: str | None,
        candidates: list[T],
        batch_provider_factory: BatchProviderFactory[B],
        priority_feedback_source: PriorityFeedbackSource,
        empty_batch_factory: EmptyBatchFactory[B],
    ) -> None:
        self._requested_strategy = requested_strategy
        self._candidates = candidates
        self._batch_provider_factory = batch_provider_factory
        self._priority_feedback_source = priority_feedback_source
        self._empty_batch_factory = empty_batch_factory

    def select(self, *, allowed_strategies: tuple[str, ...], limit: int) -> B:
        if not self._candidates:
            return self._empty_batch_factory(self._requested_strategy)

        priority_feedback = self._priority_feedback_source()
        batch_provider = self._batch_provider_factory(
            {strategy: score for strategy, (score, _) in priority_feedback.items()},
        )
        batch = batch_provider.get_batch(
            strategy=self._requested_strategy,
            allowed_strategies=allowed_strategies,
            limit=limit,
        )
        priority_score, priority_explanation = priority_feedback.get(
            batch.strategy_used,
            (None, None),
        )
        if priority_score is None and batch.strategy_selection_scores is not None:
            priority_score = batch.strategy_selection_scores.get(batch.strategy_used)
            priority_explanation = batch.strategy_selection_reason
        return replace(
            batch,
            sampler_priority_score=priority_score,
            sampler_priority_explanation=priority_explanation,
        )
