"""Contract coverage for the deterministic search performance harness."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

from benchmarks.searchkernel_ingest_search import _run_benchmark


pytestmark = pytest.mark.small


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise AssertionError(f"expected mapping, got {type(value).__name__}")
    return value


def _number(value: object) -> float:
    if not isinstance(value, (int, float)):
        raise AssertionError(f"expected number, got {type(value).__name__}")
    return float(value)


@pytest.mark.asyncio
async def test_benchmark_report_separates_cold_and_warm_measurements(
    tmp_path: Path,
) -> None:
    """Report setup, first-search, and repeated warm metrics per corpus size."""
    report = await _run_benchmark(
        tmp_path / "embeddings.db",
        record_counts=(7, 14),
        warmups=1,
        repetitions=2,
    )

    benchmarks = report["benchmarks"]
    assert isinstance(benchmarks, list)
    assert [benchmark["record_count"] for benchmark in benchmarks] == [7, 14]
    for benchmark in benchmarks:
        benchmark_mapping = _mapping(benchmark)
        cold = _mapping(benchmark_mapping["cold"])
        warm = _mapping(benchmark_mapping["warm"])
        overall = _mapping(warm["overall"])
        by_case = _mapping(warm["by_case"])
        assert _number(cold["startup_ms"]) >= 0.0
        assert _number(cold["first_search_ms"]) >= 0.0
        assert overall["count"] == 6
        assert set(by_case) == {
            "authentication policy:workspace-a",
            "database migration:workspace-a",
            "database migration:workspace-b",
        }

    ingest = _mapping(report["ingest"])
    search = _mapping(report["search"])
    authentication = _mapping(search["authentication policy:workspace-a"])
    assert ingest["committed"] == 7
    assert authentication["result_ids"] == (
        "auth-current",
    )


@pytest.mark.asyncio
async def test_benchmark_writes_a_warmed_profile_artifact(tmp_path: Path) -> None:
    """Write a reusable cProfile artifact while keeping timing output structured."""
    profile_path = tmp_path / "search.prof"

    report = await _run_benchmark(
        tmp_path / "embeddings.db",
        record_counts=(7,),
        warmups=0,
        repetitions=1,
        profile_path=profile_path,
    )

    assert profile_path.is_file()
    assert profile_path.stat().st_size > 0
    benchmarks = report["benchmarks"]
    assert isinstance(benchmarks, list)
    benchmark = _mapping(benchmarks[0])
    warm = _mapping(benchmark["warm"])
    overall = _mapping(warm["overall"])
    assert overall["count"] == 3
