from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from mcp_memory.application import memory_use_cases
from mcp_memory.config import Config, SearchKernelShadowConfig
from mcp_memory.context import ApplicationContext
from mcp_memory.integrations.searchkernel_shadow import (
    SearchKernelShadowDiagnostics,
    compare_ranked_results,
    load_golden_queries,
)


pytestmark = pytest.mark.small


@dataclass
class _Result:
    memory_id: str
    score: float


class _RuntimeLogs:
    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []

    def write_log(self, **kwargs: object) -> None:
        self.rows.append(kwargs)


def _context(enabled: bool, runtime_logs: _RuntimeLogs | None = None) -> ApplicationContext:
    return ApplicationContext(
        config=Config(searchkernel_shadow=SearchKernelShadowConfig(enabled=enabled)),
        relational_search=object(),
        runtime_logs=runtime_logs,
    )


def test_shadow_comparison_is_disabled_without_feature_flag(monkeypatch) -> None:
    called = False

    def build_kernel(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("disabled shadow comparison built a kernel")

    monkeypatch.setattr(memory_use_cases, "build_memory_search_kernel", build_kernel)

    memory_use_cases._run_searchkernel_shadow(
        _context(False),
        query="query",
        limit=3,
        native_results=[_Result("native", 1.0)],
        workspace_id=None,
        memory_type=None,
        status=None,
        include_superseded=False,
    )

    assert called is False


def test_enabled_shadow_comparison_logs_parity_diagnostics(monkeypatch) -> None:
    runtime_logs = _RuntimeLogs()
    diagnostics = compare_ranked_results(
        query="query",
        requested_limit=3,
        native_results=[_Result("a", 0.9), _Result("b", 0.8)],
        kernel_results=[_Result("b", 0.7), _Result("a", 0.6)],
    )

    async def run_shadow(*args, **kwargs):
        return diagnostics

    monkeypatch.setattr(memory_use_cases, "build_memory_search_kernel", lambda *args, **kwargs: object())
    monkeypatch.setattr(memory_use_cases, "run_searchkernel_shadow", run_shadow)

    memory_use_cases._run_searchkernel_shadow(
        _context(True, runtime_logs),
        query="query",
        limit=3,
        native_results=[_Result("a", 0.9), _Result("b", 0.8)],
        workspace_id="workspace",
        memory_type="fact",
        status=None,
        include_superseded=False,
    )

    assert len(runtime_logs.rows) == 1
    payload = runtime_logs.rows[0]["data"]
    assert isinstance(payload, dict)
    assert payload["overlap_count"] == 2
    assert payload["ordered_id_agreement"] is False
    assert payload["rank_deltas"] == {"a": 1, "b": -1}


def test_shadow_failure_is_isolated_from_search_response(monkeypatch) -> None:
    def build_kernel(*args, **kwargs):
        raise RuntimeError("kernel unavailable")

    monkeypatch.setattr(memory_use_cases, "build_memory_search_kernel", build_kernel)

    memory_use_cases._run_searchkernel_shadow(
        _context(True),
        query="query",
        limit=3,
        native_results=[_Result("native", 1.0)],
        workspace_id=None,
        memory_type=None,
        status=None,
        include_superseded=False,
    )


def test_golden_queries_compare_deterministically() -> None:
    fixture = Path(__file__).parents[1] / "fixtures" / "searchkernel" / "golden_queries.json"

    queries = load_golden_queries(fixture)
    first = compare_ranked_results(
        query=queries[0].query,
        requested_limit=queries[0].limit,
        native_results=[_Result("a", 0.9), _Result("b", 0.8)],
        kernel_results=[_Result("a", 0.7), _Result("b", 0.6)],
    )
    second = compare_ranked_results(
        query=queries[0].query,
        requested_limit=queries[0].limit,
        native_results=[_Result("a", 0.9), _Result("b", 0.8)],
        kernel_results=[_Result("a", 0.7), _Result("b", 0.6)],
    )

    assert [query.query_id for query in queries] == [
        "searchkernel-adapter",
        "native-authority",
    ]
    assert first == second
    assert first.score_deltas == {"a": -0.2, "b": -0.2}


@pytest.mark.asyncio
async def test_kernel_shadow_failure_is_reported_without_raising() -> None:
    class Kernel:
        async def search_anything(self, *args, **kwargs):
            raise RuntimeError("broken kernel")

    result = await memory_use_cases.run_searchkernel_shadow(
        Kernel(),
        query="query",
        requested_limit=2,
        native_results=[_Result("native", 1.0)],
    )

    assert isinstance(result, SearchKernelShadowDiagnostics)
    assert result.error == "RuntimeError: broken kernel"
    assert result.native_ids == ("native",)
