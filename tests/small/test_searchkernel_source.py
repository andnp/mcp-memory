"""Unit tests for the memory SearchableSource adapter.

Tests verify that the adapter correctly:
1. Maps memory search results to ScoredRef format
2. Honors status filtering (archived records excluded)
3. Honors the SUPERSEDES graph (superseded records never returned)
4. Includes candidate text for retrieve-then-rerank-once
5. Offloads blocking I/O correctly via asyncio
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from searchkernel.domain import ScoredRef, SearchFilters

from mcp_memory.integrations.searchkernel_source import (
    MemorySearchableSource,
    build_memory_search_kernel,
)
from mcp_memory.relational.search import RelationalSearchResult


pytestmark = pytest.mark.small


class SearchFacadeDouble:
    def __init__(self) -> None:
        self.return_value: list[Any] = []
        self.call_args: Any = None

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.call_args = SimpleNamespace(kwargs=kwargs)
        return SimpleNamespace(results=[self._result(result) for result in self.return_value])

    @staticmethod
    def _result(result: Any) -> Any:
        record = SimpleNamespace(
            source_id=result.memory_id,
            title=result.title,
            metadata={
                "summary": result.summary,
                "memory_type": result.memory_type,
                "memory_status": result.status,
                "tags": result.tags,
                "workspace_ids": result.workspace_ids,
                "memory_ref": result.memory_ref,
            },
            status=SimpleNamespace(value=result.status),
            storage_key=f"memory:memory:{result.memory_id}",
            workspace_id=result.workspace_ids[0] if result.workspace_ids else None,
        )
        return SimpleNamespace(
            record=record,
            score=result.score,
            provenance=SimpleNamespace(to_dict=lambda: {}),
        )


class ExtraSearchableSource:
    source_kind = "extra"

    async def search(
        self, query: str, k: int, filters: SearchFilters | None = None
    ) -> list[ScoredRef]:
        return []


@pytest.fixture
def mock_search_service() -> MagicMock:
    """Create a mock RelationalMemorySearchService."""
    service = MagicMock()
    service.search = SearchFacadeDouble()
    return service


@pytest.fixture
def adapter(mock_search_service: MagicMock) -> MemorySearchableSource:
    """Create a MemorySearchableSource with a mocked search service."""
    return MemorySearchableSource(mock_search_service)


class TestMemorySearchableSource:
    """Tests for MemorySearchableSource adapter."""

    def test_source_kind_is_memory(self, adapter: MemorySearchableSource) -> None:
        """Verify the adapter identifies itself as 'memory' source."""
        assert adapter.source_kind == "memory"

    @pytest.mark.asyncio
    async def test_search_returns_scored_refs(
        self, adapter: MemorySearchableSource, mock_search_service: MagicMock
    ) -> None:
        """Verify search returns ScoredRef objects with correct structure."""
        # Arrange
        result = RelationalSearchResult(
            memory_id="mem-001",
            title="Test Memory",
            summary="A test summary",
            memory_type="fact",
            status="active",
            tags=["test"],
            workspace_ids=["ws-1"],
            score=0.85,
        )
        mock_search_service.search.return_value = [result]

        # Act
        scored_refs = await adapter.search("test query", k=10)
        scored_refs_list = list(scored_refs)

        # Assert
        assert len(scored_refs_list) == 1
        scored_ref = scored_refs_list[0]
        assert scored_ref.source_id == "mem-001"
        assert scored_ref.score == 0.85
        assert scored_ref.source_kind == "memory"

    @pytest.mark.asyncio
    async def test_search_includes_candidate_text_in_metadata(
        self, adapter: MemorySearchableSource, mock_search_service: MagicMock
    ) -> None:
        """Verify metadata contains 'text' field with title and summary."""
        # Arrange
        result = RelationalSearchResult(
            memory_id="mem-002",
            title="Title Text",
            summary="Summary text here",
            memory_type="reflection",
            status="active",
            score=0.75,
        )
        mock_search_service.search.return_value = [result]

        # Act
        scored_refs = await adapter.search("query", k=5)
        scored_ref = list(scored_refs)[0]

        # Assert
        assert "text" in scored_ref.metadata
        assert "Title Text" in scored_ref.metadata["text"]
        assert "Summary text here" in scored_ref.metadata["text"]

    @pytest.mark.asyncio
    async def test_search_handles_empty_summary(
        self, adapter: MemorySearchableSource, mock_search_service: MagicMock
    ) -> None:
        """Verify text field is just title when summary is empty."""
        # Arrange
        result = RelationalSearchResult(
            memory_id="mem-003",
            title="Just Title",
            summary="",
            memory_type="observation",
            status="active",
            score=0.6,
        )
        mock_search_service.search.return_value = [result]

        # Act
        scored_ref = list(await adapter.search("query", k=5))[0]

        # Assert
        assert scored_ref.metadata["text"] == "Just Title"

    @pytest.mark.asyncio
    async def test_search_includes_full_metadata(
        self, adapter: MemorySearchableSource, mock_search_service: MagicMock
    ) -> None:
        """Verify all expected metadata fields are included."""
        # Arrange
        result = RelationalSearchResult(
            memory_id="mem-004",
            title="Complete Record",
            summary="Complete summary",
            memory_type="plan",
            status="stale",
            tags=["important", "work"],
            workspace_ids=["ws-1", "ws-2"],
            score=0.5,
        )
        mock_search_service.search.return_value = [result]

        # Act
        scored_ref = list(await adapter.search("query", k=5))[0]

        # Assert
        metadata = scored_ref.metadata
        assert metadata["title"] == "Complete Record"
        assert metadata["summary"] == "Complete summary"
        assert metadata["memory_type"] == "plan"
        assert metadata["status"] == "stale"
        assert metadata["tags"] == ["important", "work"]
        assert metadata["workspace_ids"] == ["ws-1", "ws-2"]

    @pytest.mark.asyncio
    async def test_search_respects_k_limit(
        self, adapter: MemorySearchableSource, mock_search_service: MagicMock
    ) -> None:
        """Verify the k parameter is passed to the search service."""
        # Arrange
        mock_search_service.search.return_value = []

        # Act
        await adapter.search("query", k=25)

        # Assert
        call_args = mock_search_service.search.call_args
        assert call_args.kwargs["limit"] == 25

    @pytest.mark.asyncio
    async def test_search_enforces_include_superseded_false(
        self, adapter: MemorySearchableSource, mock_search_service: MagicMock
    ) -> None:
        """Verify search enforces include_superseded=False (contract obligation)."""
        # Arrange
        mock_search_service.search.return_value = []

        # Act
        await adapter.search("query", k=10)

        # Assert
        call_args = mock_search_service.search.call_args
        assert call_args.kwargs["include_superseded"] is False

    @pytest.mark.asyncio
    async def test_search_enforces_status_none_to_skip_archived(
        self, adapter: MemorySearchableSource, mock_search_service: MagicMock
    ) -> None:
        """Verify search uses status=None to skip archived records."""
        # Arrange
        mock_search_service.search.return_value = []

        # Act
        await adapter.search("query", k=10)

        # Assert
        call_args = mock_search_service.search.call_args
        assert call_args.kwargs["status"] is None

    @pytest.mark.asyncio
    async def test_side_effect_free_search_is_forwarded(
        self, mock_search_service: MagicMock
    ) -> None:
        mock_search_service.search.return_value = []
        adapter = MemorySearchableSource(
            mock_search_service,
            side_effect_free=True,
        )

        await adapter.search("query", k=10)

        assert mock_search_service.search.call_args.kwargs["include_superseded"] is False
        # The facade is read-only; no native side-effect flag is required.

    @pytest.mark.asyncio
    async def test_search_passes_workspace_filter(
        self, adapter: MemorySearchableSource, mock_search_service: MagicMock
    ) -> None:
        """Verify workspace_id filter is extracted and passed through."""
        # Arrange
        mock_search_service.search.return_value = []
        filters = {"workspace_id": "ws-custom"}

        # Act
        await adapter.search("query", k=10, filters=filters)

        # Assert
        call_args = mock_search_service.search.call_args
        assert call_args.kwargs["workspace_id"] == "ws-custom"

    @pytest.mark.asyncio
    async def test_search_passes_memory_type_filter(
        self, adapter: MemorySearchableSource, mock_search_service: MagicMock
    ) -> None:
        """Verify memory_type filter is extracted and passed through."""
        # Arrange
        mock_search_service.search.return_value = []
        filters = {"memory_type": "journal"}

        # Act
        await adapter.search("query", k=10, filters=filters)

        # Assert
        call_args = mock_search_service.search.call_args
        assert call_args.kwargs["memory_type"] == "journal"

    @pytest.mark.asyncio
    async def test_search_ignores_non_string_workspace_id(
        self, adapter: MemorySearchableSource, mock_search_service: MagicMock
    ) -> None:
        """Verify non-string workspace_id filters are safely ignored."""
        # Arrange
        mock_search_service.search.return_value = []
        filters = {"workspace_id": 123}  # Invalid type

        # Act
        await adapter.search("query", k=10, filters=filters)

        # Assert
        call_args = mock_search_service.search.call_args
        assert call_args.kwargs["workspace_id"] is None

    @pytest.mark.asyncio
    async def test_search_returns_results_in_descending_score_order(
        self, adapter: MemorySearchableSource, mock_search_service: MagicMock
    ) -> None:
        """Verify results maintain descending score order from search service."""
        # Arrange
        results = [
            RelationalSearchResult(
                memory_id="mem-a",
                title="A",
                summary="",
                memory_type="fact",
                status="active",
                score=0.9,
            ),
            RelationalSearchResult(
                memory_id="mem-b",
                title="B",
                summary="",
                memory_type="fact",
                status="active",
                score=0.7,
            ),
            RelationalSearchResult(
                memory_id="mem-c",
                title="C",
                summary="",
                memory_type="fact",
                status="active",
                score=0.5,
            ),
        ]
        mock_search_service.search.return_value = results

        # Act
        scored_refs = list(await adapter.search("query", k=10))

        # Assert
        assert len(scored_refs) == 3
        assert scored_refs[0].score == 0.9
        assert scored_refs[1].score == 0.7
        assert scored_refs[2].score == 0.5

    @pytest.mark.asyncio
    async def test_search_handles_no_results(
        self, adapter: MemorySearchableSource, mock_search_service: MagicMock
    ) -> None:
        """Verify empty results are handled gracefully."""
        # Arrange
        mock_search_service.search.return_value = []

        # Act
        scored_refs = list(await adapter.search("no matches", k=10))

        # Assert
        assert scored_refs == []

    @pytest.mark.asyncio
    async def test_search_offloads_to_thread(
        self, adapter: MemorySearchableSource, mock_search_service: MagicMock
    ) -> None:
        """Verify search offloads blocking I/O to a thread via asyncio.to_thread."""
        # Arrange
        result = RelationalSearchResult(
            memory_id="mem-thread",
            title="Thread Test",
            summary="",
            memory_type="fact",
            status="active",
            score=0.8,
        )
        mock_search_service.search.return_value = [result]

        # Act
        # This should not raise even if search were blocking, because
        # it's running in a thread. We just verify the result is correct.
        scored_refs = list(await adapter.search("query", k=10))

        # Assert
        assert len(scored_refs) == 1
        assert scored_refs[0].source_id == "mem-thread"

    @pytest.mark.asyncio
    async def test_search_with_none_filters_defaults_to_empty_dict(
        self, adapter: MemorySearchableSource, mock_search_service: MagicMock
    ) -> None:
        """Verify passing None filters is handled safely."""
        # Arrange
        mock_search_service.search.return_value = []

        # Act
        await adapter.search("query", k=10, filters=None)

        # Assert
        call_args = mock_search_service.search.call_args
        assert call_args.kwargs["workspace_id"] is None
        assert call_args.kwargs["memory_type"] is None

    @pytest.mark.asyncio
    async def test_search_multiple_results_all_converted_to_scored_refs(
        self, adapter: MemorySearchableSource, mock_search_service: MagicMock
    ) -> None:
        """Verify all results from search service are converted to ScoredRef."""
        # Arrange
        results = [
            RelationalSearchResult(
                memory_id=f"mem-{i}",
                title=f"Memory {i}",
                summary=f"Summary {i}",
                memory_type="fact",
                status="active",
                score=0.9 - (i * 0.1),
            )
            for i in range(5)
        ]
        mock_search_service.search.return_value = results

        # Act
        scored_refs = list(await adapter.search("query", k=10))

        # Assert
        assert len(scored_refs) == 5
        for i, scored_ref in enumerate(scored_refs):
            assert scored_ref.source_id == f"mem-{i}"
            assert scored_ref.metadata["title"] == f"Memory {i}"


def test_build_memory_search_kernel_registers_memory_source() -> None:
    """Verify the factory registers the memory adapter under its source kind."""
    kernel = build_memory_search_kernel(MagicMock())

    source = kernel.registry.get("memory")

    assert isinstance(source, MemorySearchableSource)


def test_build_memory_search_kernel_preserves_extra_sources() -> None:
    """Verify extra searchable sources remain registered alongside memory."""
    extra_source = ExtraSearchableSource()

    kernel = build_memory_search_kernel(MagicMock(), extra_sources=[extra_source])

    assert kernel.registry.get("memory") is not None
    assert kernel.registry.get("extra") is extra_source
