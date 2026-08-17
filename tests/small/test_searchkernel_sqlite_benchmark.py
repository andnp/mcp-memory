"""Contract coverage for the SQLite-backed search performance workload."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

from benchmarks.searchkernel_sqlite_search import _run_benchmark


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
async def test_sqlite_benchmark_uses_real_fts5_search_path(tmp_path: Path) -> None:
    """Report setup, cold search, and warmed latency for SQLite FTS5."""
    report = await _run_benchmark(
        tmp_path / "memory.db",
        record_counts=(7, 14),
        warmups=1,
        repetitions=2,
    )

    benchmarks = report["benchmarks"]
    assert isinstance(benchmarks, list)
    assert [benchmark["record_count"] for benchmark in benchmarks] == [7, 14]
    for benchmark in benchmarks:
        benchmark_mapping = _mapping(benchmark)
        setup = _mapping(benchmark_mapping["setup"])
        cold = _mapping(benchmark_mapping["cold"])
        warm = _mapping(benchmark_mapping["warm"])
        assert setup["backend"] == "sqlite-fts5"
        assert setup["records_inserted"] == benchmark_mapping["record_count"]
        assert _number(cold["setup_ms"]) >= 0.0
        assert _number(cold["first_search_ms"]) >= 0.0
        assert _mapping(warm["overall"])["count"] == 6

    first_search = _mapping(_mapping(benchmarks[0])["search"])
    authentication = _mapping(first_search["authentication policy:workspace-a"])
    assert authentication["result_ids"] == ("auth-current",)


@pytest.mark.asyncio
async def test_sqlite_benchmark_writes_a_profile_artifact(tmp_path: Path) -> None:
    """Write a warmed cProfile artifact for the real SQLite workload."""
    profile_path = tmp_path / "sqlite-search.prof"

    report = await _run_benchmark(
        tmp_path / "memory.db",
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
    assert _mapping(_mapping(benchmark["warm"])["overall"])["count"] == 3


@pytest.mark.asyncio
async def test_sqlite_benchmark_can_repeat_with_one_database_prefix(tmp_path: Path) -> None:
    """Repeated runs isolate generated databases instead of colliding on IDs."""
    database_prefix = tmp_path / "repeatable.db"

    first = await _run_benchmark(database_prefix, record_counts=(7,), repetitions=1)
    second = await _run_benchmark(database_prefix, record_counts=(7,), repetitions=1)

    first_benchmarks = first["benchmarks"]
    second_benchmarks = second["benchmarks"]
    assert isinstance(first_benchmarks, list)
    assert isinstance(second_benchmarks, list)
    assert len(first_benchmarks) == 1
    assert len(second_benchmarks) == 1
