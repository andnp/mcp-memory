from __future__ import annotations

from typing import cast
from uuid import uuid4

import psycopg
import pytest

from mcp_memory.application.memory_use_cases import CommitSkillReviewUseCase
from mcp_memory.application.skill_review_contract import (
    SkillReviewCommitRequest,
    SkillReviewDisposition,
    SkillReviewEvidence,
    ReviewOutcome,
)
from mcp_memory.context import ApplicationContext
from mcp_memory.storage.postgres import ensure_postgres_schema
from mcp_memory.storage.postgres_connection import PostgresConnectionManager
from mcp_memory.storage.postgres_repository import PostgresRelationalMemoryRepository
from mcp_memory.storage.skill_review import (
    PostgresSkillReviewCommitStore,
    SkillReviewCommitConflict,
    SkillReviewCommitRejected,
)


pytestmark = pytest.mark.medium


def _request(*memory_ids: str, outcome: ReviewOutcome = "done") -> SkillReviewCommitRequest:
    """Build one valid review batch for Postgres contract tests."""
    return SkillReviewCommitRequest(
        review_run_id=str(uuid4()),
        workspace_id="skill-manager-workspace",
        evidence=SkillReviewEvidence(
            skills=("delegate-code-review",),
            source_hashes={"delegate-code-review": "a" * 64},
            deployment_status="passed",
            deployment_receipt_hash="b" * 64,
            ledger_snapshot_id="c" * 64,
        ),
        dispositions=tuple(
            SkillReviewDisposition(memory_id, outcome, "Reviewed with deployed evidence.")
            for memory_id in memory_ids
        ),
    )


def _observation(
    repository: PostgresRelationalMemoryRepository,
    memory_id: str,
    *,
    workspace_id: str = "skill-manager-workspace",
) -> None:
    """Insert one eligible skill observation into the real Postgres repository."""
    repository.create_memory(
        memory_id=memory_id,
        title="Review observation",
        content="The observed skill behavior is documented.",
        summary="Review observation",
        memory_type="observation",
        status="active",
        workspace_ids=[workspace_id],
        tags=["skill-observation"],
        metadata={"skill": "delegate-code-review", "review_status": "open"},
    )


def _database(postgres_storage_config):
    """Bootstrap an isolated Postgres database and its skill-review collaborators."""
    ensure_postgres_schema(postgres_storage_config)
    manager = PostgresConnectionManager(postgres_storage_config)
    repository = PostgresRelationalMemoryRepository(manager)
    return manager, repository, PostgresSkillReviewCommitStore(manager, repository)


def test_postgres_skill_review_routes_through_context_and_records_evidence(postgres_storage_config) -> None:
    """Route the public use case to Postgres and persist the review evidence.

    The assertion crosses the backend-selection boundary that failed in live dogfooding.
    """
    manager, repository, _store = _database(postgres_storage_config)
    try:
        _observation(repository, "mem-1")
        response = CommitSkillReviewUseCase(
            ApplicationContext(storage_backend="postgres", db_manager=manager, repository=repository)
        ).execute(_request("mem-1"))

        assert response["status"] == "committed"
        record = repository.get_memory("mem-1")
        assert record is not None
        assert record.status == "archived"
        assert record.metadata["review_status"] == "done"
        evidence = cast(dict[str, object], record.metadata["review_evidence"])
        assert evidence["deployment_status"] == "passed"
    finally:
        manager.close()


def test_postgres_skill_review_replays_identical_request_and_rejects_conflicts(postgres_storage_config) -> None:
    """Preserve idempotent replay and conflict detection in Postgres.

    A review-run identifier remains the stable idempotency key across retries.
    """
    manager, repository, store = _database(postgres_storage_config)
    try:
        _observation(repository, "mem-1")
        _observation(repository, "mem-2")
        request = _request("mem-1")

        first = store.commit(request)
        second = store.commit(request)
        conflicting = SkillReviewCommitRequest(
            review_run_id=request.review_run_id,
            workspace_id=request.workspace_id,
            evidence=request.evidence,
            dispositions=(SkillReviewDisposition("mem-2", "bad", "Different content."),),
        )

        assert first.idempotent is False
        assert second.idempotent is True
        assert second.dispositions == first.dispositions
        with pytest.raises(SkillReviewCommitConflict, match="review_run_conflict"):
            store.commit(conflicting)
    finally:
        manager.close()


def test_postgres_skill_review_rejects_ineligible_observation_without_mutation(postgres_storage_config) -> None:
    """Reject a workspace mismatch before changing the observation.

    Postgres must retain the same tenant-boundary validation as SQLite.
    """
    manager, repository, store = _database(postgres_storage_config)
    try:
        _observation(repository, "mem-1", workspace_id="other-workspace")

        with pytest.raises(SkillReviewCommitRejected, match="workspace_mismatch"):
            store.commit(_request("mem-1"))

        record = repository.get_memory("mem-1")
        assert record is not None
        assert record.status == "active"
    finally:
        manager.close()


def test_postgres_skill_review_rolls_back_all_dispositions_on_later_failure(postgres_storage_config) -> None:
    """Roll back earlier updates when a later Postgres update fails.

    The audit row and every disposition must remain absent after the transaction aborts.
    """
    manager, repository, store = _database(postgres_storage_config)
    try:
        _observation(repository, "mem-1")
        _observation(repository, "mem-2")
        row: tuple[object, ...] | None = None
        with manager.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    CREATE FUNCTION fail_skill_review_second() RETURNS trigger AS $$
                    BEGIN
                        IF NEW.id = 'mem-2' THEN
                            RAISE EXCEPTION 'forced failure';
                        END IF;
                        RETURN NEW;
                    END;
                    $$ LANGUAGE plpgsql
                    """
                )
                cursor.execute(
                    """
                    CREATE TRIGGER fail_skill_review_second
                    BEFORE UPDATE OF status ON memories
                    FOR EACH ROW EXECUTE FUNCTION fail_skill_review_second()
                    """
                )
            connection.commit()

        with pytest.raises(psycopg.Error, match="forced failure"):
            store.commit(_request("mem-1", "mem-2"))

        first = repository.get_memory("mem-1")
        second = repository.get_memory("mem-2")
        assert first is not None
        assert second is not None
        assert first.status == "active"
        assert second.status == "active"
        with manager.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM skill_review_commits")
                row = cursor.fetchone()
        assert row is not None
        assert row[0] == 0
    finally:
        manager.close()
