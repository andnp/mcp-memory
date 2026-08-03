"""Provider-free, deterministic evaluation of captured retrieval snapshots.

This module is intentionally standalone.  Replay reports are produced by
offline tooling or tests and are not part of the public retrieval path.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ReplayResult:
    """One ordered result from a captured search response."""

    memory_id: str
    payload: Mapping[str, object] = field(default_factory=dict)
    payload_size: int | None = None

    def size(self) -> int:
        if self.payload_size is not None:
            if self.payload_size < 0:
                raise ValueError("payload_size must be non-negative")
            return self.payload_size
        encoded = json.dumps(
            self.payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return len(encoded)


@dataclass(frozen=True, slots=True)
class ReplaySnapshot:
    """A captured, already-ranked response; no search is performed."""

    results: tuple[ReplayResult, ...] = ()

    def payload_size(self) -> int:
        return sum(result.size() for result in self.results)


@dataclass(frozen=True, slots=True)
class ReplayCase:
    """Before/after snapshots and the memories expected for a query."""

    query_id: str
    intended_memory_ids: tuple[str, ...]
    before: ReplaySnapshot
    after: ReplaySnapshot
    top_k: int = 5
    query_text: str | None = None
    expected_query: str | None = None
    trusted_query: bool = True
    replay_complete: bool = True

    def __post_init__(self) -> None:
        if not self.query_id.strip():
            raise ValueError("query_id must be non-empty")
        if self.top_k < 1:
            raise ValueError("top_k must be positive")


@dataclass(frozen=True, slots=True)
class ReplayCaseReport:
    query_id: str
    intended_rank_before: int | None
    intended_rank_after: int | None
    irrelevant_top_results_before: int
    irrelevant_top_results_after: int
    zero_results_before: bool
    zero_results_after: bool
    payload_size_before: int
    payload_size_after: int
    retrieval_utility_before: float = 0.0
    retrieval_utility_after: float = 0.0
    neutral_reason: str | None = None
    intended_top_k_before: int = 0
    intended_top_k_after: int = 0
    intended_top_k_loss: bool = False

    @property
    def intended_rank_change(self) -> int | None:
        if self.intended_rank_before is None or self.intended_rank_after is None:
            return None
        return self.intended_rank_after - self.intended_rank_before

    @property
    def retrieval_regression(self) -> bool:
        if self.neutral_reason is not None or self.intended_rank_before is None:
            return False
        return self.intended_rank_after is None or self.intended_rank_after > self.intended_rank_before

    @property
    def zero_result_change(self) -> int:
        return int(self.zero_results_after) - int(self.zero_results_before)

    @property
    def payload_size_change(self) -> int:
        return self.payload_size_after - self.payload_size_before

    @property
    def retrieval_utility_delta(self) -> float:
        return self.retrieval_utility_after - self.retrieval_utility_before

    @property
    def rank_improvement(self) -> float:
        before = 0.0 if self.intended_rank_before is None else 1.0 / self.intended_rank_before
        after = 0.0 if self.intended_rank_after is None else 1.0 / self.intended_rank_after
        return after - before

    @property
    def top_k_improvement(self) -> float:
        return float(
            self.irrelevant_top_results_before - self.irrelevant_top_results_after
        )

    @property
    def zero_results_improvement(self) -> float:
        return float(self.zero_results_before) - float(self.zero_results_after)


@dataclass(frozen=True, slots=True)
class ReplayEvaluationReport:
    cases: tuple[ReplayCaseReport, ...]
    zero_results_before: int
    zero_results_after: int
    payload_size_before: int
    payload_size_after: int

    @property
    def zero_result_change(self) -> int:
        return self.zero_results_after - self.zero_results_before

    @property
    def retrieval_regression_count(self) -> int:
        return sum(case.retrieval_regression for case in self.cases)

    @property
    def useful_work_count(self) -> int:
        return sum(
            case.neutral_reason is None
            and (
                case.retrieval_utility_delta > 0
                or case.irrelevant_top_results_after < case.irrelevant_top_results_before
                or case.zero_result_change < 0
            )
            for case in self.cases
        )

    @property
    def payload_size_change(self) -> int:
        return self.payload_size_after - self.payload_size_before

    @property
    def retrieval_utility_delta(self) -> float:
        return sum(case.retrieval_utility_delta for case in self.cases)


def evaluate_query_replay(cases: tuple[ReplayCase, ...] | list[ReplayCase]) -> ReplayEvaluationReport:
    """Evaluate captured cases in input order without querying or mutating anything."""
    reports = tuple(_evaluate_case(case) for case in cases)
    return ReplayEvaluationReport(
        cases=reports,
        zero_results_before=sum(report.zero_results_before for report in reports),
        zero_results_after=sum(report.zero_results_after for report in reports),
        payload_size_before=sum(report.payload_size_before for report in reports),
        payload_size_after=sum(report.payload_size_after for report in reports),
    )


def _evaluate_case(case: ReplayCase) -> ReplayCaseReport:
    intended = frozenset(case.intended_memory_ids)
    before_ids = tuple(result.memory_id for result in case.before.results)
    after_ids = tuple(result.memory_id for result in case.after.results)
    intended_top_k_before = set(before_ids[: case.top_k]) & intended
    intended_top_k_after = set(after_ids[: case.top_k]) & intended
    neutral_reason = _neutral_reason(case, intended, before_ids, after_ids)
    return ReplayCaseReport(
        query_id=case.query_id,
        intended_rank_before=_first_rank(before_ids, intended),
        intended_rank_after=_first_rank(after_ids, intended),
        irrelevant_top_results_before=sum(memory_id not in intended for memory_id in before_ids[: case.top_k]),
        irrelevant_top_results_after=sum(memory_id not in intended for memory_id in after_ids[: case.top_k]),
        zero_results_before=not case.before.results,
        zero_results_after=not case.after.results,
        payload_size_before=case.before.payload_size(),
        payload_size_after=case.after.payload_size(),
        retrieval_utility_before=_case_utility(
            intended_rank=_first_rank(before_ids, intended),
            irrelevant_top_results=sum(
                memory_id not in intended for memory_id in before_ids[: case.top_k]
            ),
            zero_results=not case.before.results,
            top_k=case.top_k,
            neutral=neutral_reason is not None,
        ),
        retrieval_utility_after=_case_utility(
            intended_rank=_first_rank(after_ids, intended),
            irrelevant_top_results=sum(
                memory_id not in intended for memory_id in after_ids[: case.top_k]
            ),
            zero_results=not case.after.results,
            top_k=case.top_k,
            neutral=neutral_reason is not None,
        ),
        neutral_reason=neutral_reason,
        intended_top_k_before=len(intended_top_k_before),
        intended_top_k_after=len(intended_top_k_after),
        intended_top_k_loss=bool(intended_top_k_before - intended_top_k_after),
    )


def _neutral_reason(
    case: ReplayCase,
    intended: frozenset[str],
    before_ids: tuple[str, ...],
    after_ids: tuple[str, ...],
) -> str | None:
    if not case.trusted_query:
        return "no_trusted_query"
    if case.query_text is not None and not case.query_text.strip():
        return "blank_query"
    if (
        case.expected_query
        and not _query_terms_overlap(case.query_text or "", case.expected_query)
    ):
        return "irrelevant_query"
    if not case.replay_complete:
        return "incomplete_replay"
    if intended and not (set(before_ids) | set(after_ids)) & intended:
        return "irrelevant_query"
    return None


def _query_terms_overlap(query: str, expected_query: str) -> bool:
    query_terms = set(query.casefold().split())
    expected_terms = set(expected_query.casefold().split())
    return bool(query_terms & expected_terms)


def _case_utility(
    *,
    intended_rank: int | None,
    irrelevant_top_results: int,
    zero_results: bool,
    top_k: int,
    neutral: bool,
) -> float:
    if neutral:
        return 0.0
    rank_utility = 0.0 if intended_rank is None else 1.0 / intended_rank
    return rank_utility - (irrelevant_top_results / top_k) - float(zero_results)


def _first_rank(result_ids: tuple[str, ...], intended: frozenset[str]) -> int | None:
    for rank, memory_id in enumerate(result_ids, start=1):
        if memory_id in intended:
            return rank
    return None
