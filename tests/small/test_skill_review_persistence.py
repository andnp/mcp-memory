import sqlite3
from pathlib import Path
from typing import cast
from uuid import uuid4

import pytest

from mcp_memory.application.skill_review_contract import (
    SkillReviewCommitRequest,
    SkillReviewDisposition,
    SkillReviewEvidence,
    Resolution,
)
from mcp_memory.relational.repository import SQLiteRelationalMemoryRepository
from mcp_memory.storage.skill_review import (
    SQLiteSkillReviewCommitStore,
    SkillReviewCommitConflict,
    SkillReviewCommitRejected,
)
from mcp_memory.utils.db import DatabaseManager


pytestmark = pytest.mark.small


def _request(*memory_ids: str, resolution: Resolution = "verified") -> SkillReviewCommitRequest:
    """Build a valid batch for the selected skill observations."""
    return SkillReviewCommitRequest(
        review_run_id=str(uuid4()),
        workspace_id="skill-manager-workspace",
        evidence=SkillReviewEvidence(
            skills=("delegate-code-review",),
            source_hashes={"delegate-code-review": "a" * 64},
            deployment_status="passed",
            deployment_receipt_hash="b" * 64,
        ),
        dispositions=tuple(
            SkillReviewDisposition(memory_id, resolution, "Reviewed with deployed evidence.")
            for memory_id in memory_ids
        ),
    )


def _store(tmp_path: Path) -> tuple[DatabaseManager, SQLiteRelationalMemoryRepository, SQLiteSkillReviewCommitStore]:
    """Create a real SQLite repository and batch store for each test."""
    manager = DatabaseManager(tmp_path / "memory.db")
    repository = SQLiteRelationalMemoryRepository(manager)
    return manager, repository, SQLiteSkillReviewCommitStore(manager, repository)


def _observation(repository: SQLiteRelationalMemoryRepository, memory_id: str, **overrides: object) -> None:
    """Insert one eligible skill observation with optional state overrides."""
    tags_value = overrides.get("tags", ["skill-observation"])
    metadata_value = overrides.get(
        "metadata",
        {"skill": "delegate-code-review", "review_status": "open"},
    )
    assert isinstance(tags_value, list)
    assert isinstance(metadata_value, dict)
    repository.create_memory(
        memory_id=memory_id,
        title="Review observation",
        content="The observed skill behavior is documented.",
        summary="Review observation",
        memory_type=str(overrides.get("memory_type", "observation")),
        status=str(overrides.get("status", "active")),
        workspace_ids=[str(overrides.get("workspace_id", "skill-manager-workspace"))],
        tags=cast(list[str], tags_value),
        metadata=cast(dict[str, object], metadata_value),
    )


def test_commit_skill_review_updates_all_records_and_audits_evidence(tmp_path: Path) -> None:
    """Commit a valid batch and retain resolution plus deployment evidence."""
    manager, repository, store = _store(tmp_path)
    _observation(repository, "mem-1")
    _observation(repository, "mem-2")

    response = store.commit(_request("mem-1", "mem-2"))

    assert response.as_dict()["status"] == "committed"
    assert [item.status for item in response.dispositions] == ["stale", "stale"]
    for memory_id in ("mem-1", "mem-2"):
        record = repository.get_memory(memory_id)
        assert record is not None
        assert record.status == "stale"
        assert record.metadata["review_status"] == "verified"
        evidence = cast(dict[str, object], record.metadata["review_evidence"])
        assert evidence["deployment_status"] == "passed"
    audit = cast(sqlite3.Row | None, manager.get_connection().execute(
        "SELECT request_digest, request_json FROM skill_review_commits WHERE review_run_id = ?",
        (response.review_run_id,),
    ).fetchone())
    assert audit is not None
    assert len(audit["request_digest"]) == 64
    assert "deployment_receipt_hash" in audit["request_json"]


@pytest.mark.parametrize(
    ("override", "error"),
    [
        ({"workspace_id": "other-workspace"}, "workspace_mismatch"),
        ({"memory_type": "fact", "tags": []}, "not_skill_observation"),
        ({"status": "stale"}, "observation_not_active"),
    ],
)
def test_commit_skill_review_rejects_ineligible_records(
    tmp_path: Path, override: dict[str, object], error: str
) -> None:
    """Reject workspace, type, and active-state violations without mutation."""
    _manager, repository, store = _store(tmp_path)
    _observation(repository, "mem-1", **override)

    with pytest.raises(SkillReviewCommitRejected, match=error):
        store.commit(_request("mem-1"))

    record = repository.get_memory("mem-1")
    assert record is not None
    assert record.status == str(override.get("status", "active"))


def test_commit_skill_review_rolls_back_when_a_later_update_fails(tmp_path: Path) -> None:
    """Roll back an earlier disposition when a later database update aborts."""
    manager, repository, store = _store(tmp_path)
    _observation(repository, "mem-1")
    _observation(repository, "mem-2")
    manager.get_connection().execute(
        """
        CREATE TRIGGER fail_skill_review_second
        BEFORE UPDATE OF status ON memories
        WHEN NEW.id = 'mem-2'
        BEGIN SELECT RAISE(ABORT, 'forced failure'); END
        """
    )

    with pytest.raises(sqlite3.IntegrityError, match="forced failure"):
        store.commit(_request("mem-1", "mem-2"))

    first = repository.get_memory("mem-1")
    second = repository.get_memory("mem-2")
    assert first is not None
    assert second is not None
    assert first.status == "active"
    assert second.status == "active"
    count_row = cast(
        sqlite3.Row | None,
        manager.get_connection().execute("SELECT COUNT(*) FROM skill_review_commits").fetchone(),
    )
    assert count_row is not None
    assert count_row[0] == 0


def test_commit_skill_review_replays_identical_request_without_mutating_again(tmp_path: Path) -> None:
    """Return an idempotent response for an identical review-run retry."""
    _manager, repository, store = _store(tmp_path)
    _observation(repository, "mem-1")
    request = _request("mem-1")

    first = store.commit(request)
    second = store.commit(request)

    assert first.idempotent is False
    assert second.idempotent is True
    assert second.dispositions == first.dispositions


def test_commit_skill_review_rejects_conflicting_review_run_reuse(tmp_path: Path) -> None:
    """Reject reuse of a review-run identifier for different request content."""
    _manager, repository, store = _store(tmp_path)
    _observation(repository, "mem-1")
    _observation(repository, "mem-2")
    request = _request("mem-1")
    conflicting = SkillReviewCommitRequest(
        review_run_id=request.review_run_id,
        workspace_id=request.workspace_id,
        evidence=request.evidence,
        dispositions=(SkillReviewDisposition("mem-2", "verified", "Different content."),),
    )

    store.commit(request)
    with pytest.raises(SkillReviewCommitConflict, match="review_run_conflict"):
        store.commit(conflicting)
