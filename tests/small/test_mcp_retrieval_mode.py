from types import SimpleNamespace
from typing import cast

import pytest

from mcp_memory.application.cache_policy import build_search_cache_request
from mcp_memory.application.memory_use_cases import SearchMemoryRecordsUseCase
from mcp_memory.application.ports import (
    MemoryReadDependencies,
    MemorySearchPort,
    RetrievalTelemetryPort,
)
from mcp_memory.core.ports import SearchHealthPort
from mcp_memory.core.retrieval import RetrievalRequest, RetrievalResult
from mcp_memory.mcp.adapters import parse_search_arguments
from mcp_memory.mcp.tools import get_memory_tools

pytestmark = pytest.mark.small


def test_search_tool_schema_advertises_retrieval_modes() -> None:
    """Advertise the opt-in retrieval mode contract to MCP clients.

    The schema must preserve hybrid as the protocol default.
    """
    tool = next(tool for tool in get_memory_tools() if tool.name == "search_memory_records")

    assert tool.input_schema["properties"]["retrieval_mode"] == {
        "type": "string",
        "enum": ["keyword", "semantic", "hybrid"],
        "default": "hybrid",
        "description": "Optional retrieval strategy. Defaults to hybrid.",
    }


@pytest.mark.parametrize("mode", ["keyword", "semantic", "hybrid"])
def test_parse_search_arguments_accepts_retrieval_modes(mode: str) -> None:
    """Parse every supported retrieval mode without changing other defaults.

    Omitted mode behavior is covered separately as the compatibility case.
    """
    parsed = parse_search_arguments({"query": "retrieval", "retrieval_mode": mode})

    assert parsed["retrieval_mode"] == mode
    assert parsed["limit"] == 5
    assert parsed["adaptive_limit"] is True


def test_parse_search_arguments_defaults_to_hybrid() -> None:
    """Keep searches that omit retrieval_mode on the hybrid path.

    Existing callers should receive the same parsed search defaults.
    """
    assert parse_search_arguments({"query": "retrieval"})["retrieval_mode"] == "hybrid"


def test_parse_search_arguments_rejects_invalid_retrieval_mode() -> None:
    """Reject unsupported retrieval modes deterministically at the MCP boundary.

    Invalid values must not reach retrieval or storage code.
    """
    with pytest.raises(ValueError, match="retrieval_mode must be keyword, semantic, or hybrid"):
        parse_search_arguments({"query": "retrieval", "retrieval_mode": "graph"})


def test_search_cache_identity_separates_retrieval_modes() -> None:
    """Include non-default retrieval modes in response-cache identity.

    Hybrid retains the existing feature fingerprint for cache compatibility.
    """
    ctx = MemoryReadDependencies(workspace_id="workspace-1")
    arguments = {
        "query": "retrieval",
        "workspace_id": None,
        "limit": 5,
        "adaptive_limit": True,
        "memory_type": None,
        "status": None,
        "include_superseded": False,
    }

    hybrid = build_search_cache_request(
        ctx,
        arguments | {"retrieval_mode": "hybrid"},
        feature_fingerprint="features-v1",
    )
    semantic = build_search_cache_request(
        ctx,
        arguments | {"retrieval_mode": "semantic"},
        feature_fingerprint="features-v1",
    )

    assert hybrid.normalized_params()["feature_fingerprint"] == "features-v1"
    assert semantic.normalized_params()["feature_fingerprint"] == (
        "features-v1|retrieval_mode=semantic"
    )
    assert hybrid.normalized_params() != semantic.normalized_params()


class _SemanticRetrieval:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def search_sync(self, query: str, **kwargs: object) -> SimpleNamespace:
        self.calls.append({"query": query, **kwargs})
        record = SimpleNamespace(
            source_id="memory-1",
            title="Semantic result",
            body="semantic summary",
            metadata={
                "memory_ref": 1,
                "summary": "semantic summary",
                "memory_type": "fact",
                "memory_status": "active",
                "tags": [],
                "workspace_ids": [],
            },
            status=SimpleNamespace(value="active"),
            storage_key="memory:memory-1",
        )
        provenance = SimpleNamespace(strategies=("vector",), to_dict=lambda: {})
        return SimpleNamespace(results=[SimpleNamespace(record=record, score=0.9, provenance=provenance)])


class _DebugSemanticRetrieval:
    def search_sync_with_diagnostics(self, query: str, **kwargs: object) -> tuple[SimpleNamespace, SimpleNamespace]:
        _ = query, kwargs
        record = SimpleNamespace(
            source_id="memory-1",
            title="Semantic result",
            body="semantic body",
            metadata={
                "memory_ref": 1,
                "summary": "semantic summary",
                "memory_type": "fact",
                "memory_status": "active",
                "tags": [],
                "workspace_ids": [],
            },
            status=SimpleNamespace(value="active"),
            storage_key="memory:memory-1",
        )
        provenance = SimpleNamespace(
            strategies=("keyword", "vector"),
            to_dict=lambda: {"strategies": ["keyword", "vector"]},
        )
        result = SimpleNamespace(
            record=record,
            score=0.9,
            normalized_score=0.75,
            provenance=provenance,
            chunk_matches=(SimpleNamespace(score=0.8, content="x" * 400),),
        )
        outcome = SimpleNamespace(results=(result,))
        diagnostics = SimpleNamespace(
            timing_ms={"total": 1.0},
            final_duplicate_ids=[],
            to_payload=lambda: {"timing_ms": {"total": 1.0}},
        )
        return outcome, diagnostics


class _AsyncFacade:
    def __init__(self) -> None:
        self.calls: list[tuple[str, RetrievalRequest, dict[str, object]]] = []
        self.outcome = _DebugSemanticRetrieval().search_sync_with_diagnostics("query")[0]

    async def retrieve(
        self,
        request: RetrievalRequest,
        *,
        filters: dict[str, object],
    ) -> RetrievalResult[SimpleNamespace]:
        self.calls.append(("retrieve", request, filters))
        return RetrievalResult(items=(self.outcome,))

    async def search_with_diagnostics(
        self,
        request: RetrievalRequest,
        *,
        filters: dict[str, object],
        debug: bool,
    ) -> tuple[SimpleNamespace, SimpleNamespace]:
        self.calls.append(("search_with_diagnostics", request, filters))
        assert debug is True
        diagnostics = SimpleNamespace(
            timing_ms={"total": 1.0, "search": 0.5},
            to_payload=lambda: {
                "timing_ms": {"total": 1.0, "search": 0.5},
                "degraded": False,
            },
        )
        return self.outcome, diagnostics


def test_search_use_case_debug_includes_bounded_search_evidence() -> None:
    """Carry raw SearchKernel evidence into debug results without changing compact mode."""
    retrieval = _DebugSemanticRetrieval()
    ctx = MemoryReadDependencies(
        workspace_id="workspace-1",
        relational_search=cast(
            MemorySearchPort,
            SimpleNamespace(get_health=lambda: None),
        ),
        memory_retrieval=retrieval,
    )
    telemetry = cast(RetrievalTelemetryPort, SimpleNamespace(record_search=lambda **_: None))

    payload = SearchMemoryRecordsUseCase(ctx, telemetry).execute(
        {
            "query": "semantic retrieval",
            "workspace_id": None,
            "limit": 5,
            "adaptive_limit": True,
            "memory_type": None,
            "status": None,
            "tags": (),
            "include_superseded": False,
            "debug": True,
            "retrieval_mode": "semantic",
        }
    )

    evidence = payload["results"][0]["ranking_debug"]["evidence"]
    assert evidence["lanes"] == ["keyword", "semantic"]
    assert evidence["normalized_score"] == 0.75
    assert len(evidence["excerpts"]) == 1
    assert len(evidence["excerpts"][0]["content"]) == 240


def test_search_use_case_propagates_non_default_mode_to_canonical_filters() -> None:
    """Pass an opt-in retrieval mode through the canonical SearchKernel filters.

    The normal response remains the compact MCP result shape.
    """
    retrieval = _SemanticRetrieval()
    ctx = MemoryReadDependencies(
        workspace_id="workspace-1",
        relational_search=cast(
            MemorySearchPort,
            SimpleNamespace(get_health=lambda: None),
        ),
        memory_retrieval=retrieval,
    )
    telemetry = cast(RetrievalTelemetryPort, SimpleNamespace(record_search=lambda **_: None))

    payload = SearchMemoryRecordsUseCase(ctx, telemetry).execute(
        {
            "query": "semantic retrieval",
            "workspace_id": None,
            "limit": 5,
            "adaptive_limit": True,
            "memory_type": None,
            "status": None,
            "tags": (),
            "include_superseded": False,
            "debug": False,
            "retrieval_mode": "semantic",
        }
    )

    assert retrieval.calls[0]["filters"] == {"retrieval_mode": "semantic"}
    assert payload["status"] == "ok"
    assert payload["results"][0]["summary"] == "semantic summary"
    assert "timing_ms" not in payload


@pytest.mark.asyncio
async def test_async_search_service_uses_typed_facade_for_compact_and_debug_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Route both public async response modes through the typed facade contract."""
    facade = _AsyncFacade()
    monkeypatch.setattr(
        "mcp_memory.application.memory_use_cases.build_memory_retrieval_facade",
        lambda *args, **kwargs: facade,
    )
    typed_health = SimpleNamespace(
        get_health=lambda: SimpleNamespace(
            repair_wait_count=0,
            last_repair_wait_seconds=0.0,
        )
    )
    legacy_search = SimpleNamespace(
        get_health=lambda: pytest.fail("legacy search health should not be called")
    )
    ctx = cast(
        MemoryReadDependencies,
        SimpleNamespace(
            workspace_id="workspace-1",
            repository=object(),
            relational_search=cast(MemorySearchPort, legacy_search),
            search_health=cast(SearchHealthPort, typed_health),
            memory_retrieval=None,
            config=None,
            read_cache=None,
            read_cache_validation=None,
            memory_id_resolution=None,
            vector_store=None,
            embedder=None,
            embedding_maintenance=None,
            surface_tracker=None,
        ),
    )
    telemetry = cast(RetrievalTelemetryPort, SimpleNamespace(record_search=lambda **_: None))
    use_case = SearchMemoryRecordsUseCase(ctx, telemetry)
    arguments = {
        "query": "semantic retrieval",
        "workspace_id": None,
        "limit": 5,
        "adaptive_limit": True,
        "memory_type": None,
        "status": None,
        "tags": (),
        "include_superseded": False,
        "retrieval_mode": "semantic",
        "debug": False,
    }

    compact_payload = await use_case.execute_async(arguments)

    assert facade.calls[0][0] == "retrieve"
    assert facade.calls[0][1] == RetrievalRequest(
        query="semantic retrieval",
        limit=5,
        adaptive_limit=True,
    )
    assert facade.calls[0][2] == {
        "_ranking_workspace_id": "workspace-1",
        "retrieval_mode": "semantic",
    }
    assert compact_payload["results"][0]["summary"] == "semantic summary"
    assert "timing_ms" not in compact_payload

    arguments["debug"] = True
    debug_payload = await use_case.execute_async(arguments)

    assert facade.calls[1][0] == "search_with_diagnostics"
    assert debug_payload["search_diagnostics"]["degraded"] is False
    assert debug_payload["search_diagnostics"]["timing_ms"] == debug_payload["timing_ms"]
