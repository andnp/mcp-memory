from mcp_memory.relational.semantic import RelationalSemanticSearchAdapter


class _FakeEmbedder:
    model_name = "fake-mini"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] for _ in texts]


class _RecordingVectorStore:
    def __init__(self) -> None:
        self.search_kwargs: dict[str, object] | None = None

    def search(self, **kwargs: object) -> list[tuple[str, float]]:
        self.search_kwargs = kwargs
        return []


class _CandidateFilteringVectorStore(_RecordingVectorStore):
    supports_candidate_filtering = True


def test_candidate_ids_require_formal_filtering_capability() -> None:
    capable_store = _CandidateFilteringVectorStore()
    RelationalSemanticSearchAdapter(_FakeEmbedder(), capable_store).search(
        "query",
        [],
        candidate_ids=["memory-a"],
        limit=5,
        semantic_timing_ms=None,
        vector_search_diagnostics=None,
    )

    basic_store = _RecordingVectorStore()
    RelationalSemanticSearchAdapter(_FakeEmbedder(), basic_store).search(
        "query",
        [],
        candidate_ids=["memory-a"],
        limit=5,
        semantic_timing_ms=None,
        vector_search_diagnostics=None,
    )

    assert capable_store.search_kwargs is not None
    assert capable_store.search_kwargs["candidate_ids"] == ["memory-a"]
    assert basic_store.search_kwargs is not None
    assert "candidate_ids" not in basic_store.search_kwargs
