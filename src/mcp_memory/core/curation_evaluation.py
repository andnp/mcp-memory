"""Provider-free, deterministic evaluation of captured retrieval snapshots.

This module is intentionally standalone.  Replay reports are produced by
offline tooling or tests and are not part of the public retrieval path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Mapping


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

    @property
    def intended_rank_change(self) -> int | None:
        if self.intended_rank_before is None or self.intended_rank_after is None:
            return None
        return self.intended_rank_after - self.intended_rank_before

    @property
    def zero_result_change(self) -> int:
        return int(self.zero_results_after) - int(self.zero_results_before)

    @property
    def payload_size_change(self) -> int:
        return self.payload_size_after - self.payload_size_before


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
    def payload_size_change(self) -> int:
        return self.payload_size_after - self.payload_size_before


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
    )


def _first_rank(result_ids: tuple[str, ...], intended: frozenset[str]) -> int | None:
    for rank, memory_id in enumerate(result_ids, start=1):
        if memory_id in intended:
            return rank
    return None
