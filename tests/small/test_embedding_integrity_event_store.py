import pytest

from mcp_memory.embedding_integrity_event_store import (
    EMBEDDING_INTEGRITY_EVENT_KIND_BLOCKED_FALLBACK_WRITE,
    EMBEDDING_INTEGRITY_EVENT_KIND_SCAN_SUMMARY,
    EmbeddingIntegrityEventRepository,
)


pytestmark = pytest.mark.small


def test_embedding_integrity_event_repository_records_and_summarizes_events(db_manager) -> None:
    repository = EmbeddingIntegrityEventRepository(db_manager, workspace_id="workspace-a")

    repository.record_event(
        event_kind=EMBEDDING_INTEGRITY_EVENT_KIND_SCAN_SUMMARY,
        model_name="mini-embed",
        scanned_row_count=4,
        invalid_row_count=2,
        mixed_dimension_group_count=1,
        details={"expected_dimension": 2},
        created_at=100.0,
    )
    repository.record_event(
        event_kind=EMBEDDING_INTEGRITY_EVENT_KIND_BLOCKED_FALLBACK_WRITE,
        model_name="hash:sentence-transformers/all-MiniLM-L6-v2",
        source_kind="memory",
        source_id="memory-1",
        details={"reason": "fallback_embedding_persistence_blocked"},
        created_at=101.0,
    )

    summary = repository.summarize_events()
    global_summary = repository.summarize_events(workspace_id=None)
    workspace_summary = repository.summarize_events(workspace_id="workspace-a")

    assert summary.total == 2
    assert summary.by_kind == {
        EMBEDDING_INTEGRITY_EVENT_KIND_BLOCKED_FALLBACK_WRITE: 1,
        EMBEDDING_INTEGRITY_EVENT_KIND_SCAN_SUMMARY: 1,
    }
    assert summary.last_scan is not None
    assert summary.last_scan.model_name == "mini-embed"
    assert summary.last_scan.scanned_row_count == 4
    assert summary.last_scan.invalid_row_count == 2
    assert summary.last_scan.mixed_dimension_group_count == 1
    assert summary.last_scan.details == {"expected_dimension": 2}
    assert summary.last_blocked_fallback_write is not None
    assert summary.last_blocked_fallback_write.model_name == "hash:sentence-transformers/all-MiniLM-L6-v2"
    assert summary.last_blocked_fallback_write.source_kind == "memory"
    assert summary.last_blocked_fallback_write.source_id == "memory-1"
    assert summary.last_scan.workspace_id is None
    assert summary.last_blocked_fallback_write.workspace_id is None
    assert global_summary.total == 2
    assert workspace_summary.total == 0
