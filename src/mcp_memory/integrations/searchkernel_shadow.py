"""Feature-flagged native/searchkernel parity diagnostics."""

from __future__ import annotations

import json
import inspect
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, cast

from searchkernel.search.record_pipeline import RecordSearchOutcome


@dataclass(frozen=True, slots=True)
class GoldenQuery:
    query_id: str
    query: str
    limit: int = 5


@dataclass(frozen=True, slots=True)
class SearchKernelShadowDiagnostics:
    query: str
    requested_limit: int
    source_kind: str
    native_result_count: int
    kernel_result_count: int
    overlap_count: int
    overlap_ratio: float
    ordered_id_agreement: bool
    native_ids: tuple[str, ...]
    kernel_ids: tuple[str, ...]
    rank_deltas: dict[str, int]
    rank_mismatches: tuple[dict[str, int | str], ...]
    score_deltas: dict[str, float]
    elapsed_ms: float
    error: str | None = None

    def to_payload(self) -> dict[str, object]:
        return {
            "query": self.query,
            "requested_limit": self.requested_limit,
            "source_kind": self.source_kind,
            "native_result_count": self.native_result_count,
            "kernel_result_count": self.kernel_result_count,
            "overlap_count": self.overlap_count,
            "overlap_ratio": self.overlap_ratio,
            "ordered_id_agreement": self.ordered_id_agreement,
            "native_ids": list(self.native_ids),
            "kernel_ids": list(self.kernel_ids),
            "rank_deltas": dict(self.rank_deltas),
            "rank_mismatches": [dict(item) for item in self.rank_mismatches],
            "score_deltas": dict(self.score_deltas),
            "elapsed_ms": self.elapsed_ms,
            "error": self.error,
        }


def load_golden_queries(path: Path) -> list[GoldenQuery]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("golden query fixture must contain a list")

    queries: list[GoldenQuery] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("golden query fixture entries must be objects")
        query_id = item.get("id")
        query = item.get("query")
        limit = item.get("limit", 5)
        if (
            not isinstance(query_id, str)
            or not query_id
            or not isinstance(query, str)
            or not query
            or not isinstance(limit, int)
            or isinstance(limit, bool)
            or limit < 1
        ):
            raise ValueError("golden query fixture entry is invalid")
        queries.append(GoldenQuery(query_id=query_id, query=query, limit=limit))
    return queries


def compare_ranked_results(
    *,
    query: str,
    requested_limit: int,
    native_results: Iterable[Any],
    kernel_results: Iterable[Any],
    source_kind: str = "memory",
    elapsed_ms: float = 0.0,
    max_results: int = 20,
) -> SearchKernelShadowDiagnostics:
    native = list(native_results)[:max_results]
    kernel = list(kernel_results)[:max_results]
    native_ids = tuple(_result_id(result) for result in native)
    kernel_ids = tuple(_result_id(result) for result in kernel)
    native_positions = {memory_id: index for index, memory_id in enumerate(native_ids, start=1)}
    kernel_positions = {memory_id: index for index, memory_id in enumerate(kernel_ids, start=1)}
    overlap = set(native_positions) & set(kernel_positions)
    rank_deltas = {
        memory_id: kernel_positions[memory_id] - native_positions[memory_id]
        for memory_id in sorted(overlap)
    }
    rank_mismatches = tuple(
        {
            "memory_id": memory_id,
            "native_rank": native_positions[memory_id],
            "kernel_rank": kernel_positions[memory_id],
        }
        for memory_id in sorted(overlap)
        if native_positions[memory_id] != kernel_positions[memory_id]
    )
    native_scores = {_result_id(result): float(result.score) for result in native}
    kernel_scores = {_result_id(result): float(result.score) for result in kernel}
    score_deltas = {
        memory_id: round(kernel_scores[memory_id] - native_scores[memory_id], 6)
        for memory_id in sorted(overlap)
    }
    denominator = max(len(native_ids), len(kernel_ids), 1)
    return SearchKernelShadowDiagnostics(
        query=query,
        requested_limit=requested_limit,
        source_kind=source_kind,
        native_result_count=len(native_ids),
        kernel_result_count=len(kernel_ids),
        overlap_count=len(overlap),
        overlap_ratio=round(len(overlap) / denominator, 6),
        ordered_id_agreement=native_ids == kernel_ids,
        native_ids=native_ids,
        kernel_ids=kernel_ids,
        rank_deltas=rank_deltas,
        rank_mismatches=rank_mismatches,
        score_deltas=score_deltas,
        elapsed_ms=round(elapsed_ms, 3),
    )


async def run_searchkernel_shadow(
    kernel: Any,
    *,
    query: str,
    requested_limit: int,
    native_results: Iterable[Any],
    filters: dict[str, object] | None = None,
    source_kind: str = "memory",
    max_results: int = 20,
) -> SearchKernelShadowDiagnostics:
    started_at = perf_counter()
    pipeline_search = getattr(kernel, "search", None)
    if callable(pipeline_search) and not callable(
        getattr(kernel, "search_anything", None)
    ):
        try:
            outcome = pipeline_search(
                query,
                limit=requested_limit,
                filters=filters,
            )
            if inspect.isawaitable(outcome):
                outcome = await outcome
            outcome = cast(RecordSearchOutcome, outcome)
            kernel_results = outcome.results
            errors: list[str] = []
            composition_diagnostics = getattr(kernel, "diagnostics", None)
            reasons = getattr(composition_diagnostics, "reasons", ())
            errors.extend(str(reason) for reason in reasons)
            if getattr(outcome, "degraded", False):
                errors.extend(
                    f"{failure.stage}: {failure.message}"
                    for failure in outcome.failures
                )
                if outcome.missing_record_ids:
                    errors.append(
                        "hydration missing records: "
                        + ", ".join(outcome.missing_record_ids)
                    )
            diagnostics = compare_ranked_results(
                query=query,
                requested_limit=requested_limit,
                native_results=native_results,
                kernel_results=kernel_results,
                source_kind=source_kind,
                elapsed_ms=(perf_counter() - started_at) * 1000.0,
                max_results=max_results,
            )
            if errors:
                diagnostics = replace(diagnostics, error="; ".join(errors))
            return diagnostics
        except Exception as exc:
            return replace(
                compare_ranked_results(
                    query=query,
                    requested_limit=requested_limit,
                    native_results=native_results,
                    kernel_results=(),
                    source_kind=source_kind,
                    elapsed_ms=(perf_counter() - started_at) * 1000.0,
                    max_results=max_results,
                ),
                error=f"{type(exc).__name__}: {exc}",
            )
    try:
        kernel_results = await kernel.search_anything(
            query,
            sources=[source_kind],
            filters=filters,
            k=requested_limit,
        )
    except Exception as exc:
        return replace(
            compare_ranked_results(
                query=query,
                requested_limit=requested_limit,
                native_results=native_results,
                kernel_results=(),
                source_kind=source_kind,
                elapsed_ms=(perf_counter() - started_at) * 1000.0,
                max_results=max_results,
            ),
            error=f"{type(exc).__name__}: {exc}",
        )
    return compare_ranked_results(
        query=query,
        requested_limit=requested_limit,
        native_results=native_results,
        kernel_results=kernel_results,
        source_kind=source_kind,
        elapsed_ms=(perf_counter() - started_at) * 1000.0,
        max_results=max_results,
    )


def _result_id(result: Any) -> str:
    memory_id = getattr(result, "memory_id", None)
    if isinstance(memory_id, str):
        return memory_id
    record_id = getattr(result, "record_id", None)
    if isinstance(record_id, str):
        return record_id
    raise TypeError("ranked result does not expose a memory_id or record_id")


__all__ = [
    "GoldenQuery",
    "SearchKernelShadowDiagnostics",
    "compare_ranked_results",
    "load_golden_queries",
    "run_searchkernel_shadow",
]
