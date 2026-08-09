"""Atomic SQLite persistence for skill-review disposition batches."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Protocol, cast

from mcp_memory.application.skill_review_contract import (
    SkillReviewCommitDisposition,
    SkillReviewCommitRequest,
    SkillReviewCommitResponse,
    Resolution,
)
from mcp_memory.core.ports.memory import MemoryRecord
from mcp_memory.utils.db import DatabaseManager


class SkillReviewCommitRejected(ValueError):
    """The request cannot change one or more requested observations."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class SkillReviewCommitConflict(ValueError):
    """A review run identifier was reused for different request content."""


class SkillObservationRepository(Protocol):
    def get_memory(self, memory_id: str) -> MemoryRecord | None: ...


class SQLiteSkillReviewCommitStore:
    def __init__(self, db_manager: DatabaseManager, repository: SkillObservationRepository) -> None:
        self._db = db_manager
        self._repository = repository

    def commit(self, request: SkillReviewCommitRequest) -> SkillReviewCommitResponse:
        conn = self._db.get_connection()
        request_digest = request.digest()
        with conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT request_digest, response_json FROM skill_review_commits WHERE review_run_id = ?",
                (request.review_run_id,),
            ).fetchone()
            if existing is not None:
                if existing["request_digest"] != request_digest:
                    raise SkillReviewCommitConflict("review_run_conflict")
                stored = json.loads(existing["response_json"])
                return _response_from_dict(stored, idempotent=True)

            records = [self._eligible_record(request, item.memory_id) for item in request.dispositions]
            outcomes = tuple(
                SkillReviewCommitDisposition(item.memory_id, item.resolution, _status_for(item.resolution))
                for item in request.dispositions
            )
            for item, record in zip(request.dispositions, records, strict=True):
                metadata = dict(record.metadata)
                metadata.update(
                    {
                        "review_status": item.resolution,
                        "resolution_note": item.note,
                        "review_run_id": request.review_run_id,
                        "review_evidence": request.evidence.as_dict(),
                    }
                )
                conn.execute(
                    "UPDATE memories SET status = ?, archived_at = ?, updated_at = ?, metadata = ? WHERE id = ? AND status = 'active'",
                    (
                        _status_for(item.resolution),
                        _archived_at(item.resolution),
                        _now(),
                        json.dumps(metadata, sort_keys=True),
                        record.id,
                    ),
                )
            response = SkillReviewCommitResponse(outcomes, request.review_run_id, idempotent=False)
            conn.execute(
                "INSERT INTO skill_review_commits (review_run_id, request_digest, request_json, response_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (
                    request.review_run_id,
                    request_digest,
                    json.dumps(request.as_dict(), sort_keys=True),
                    json.dumps(response.as_dict(), sort_keys=True),
                    _now(),
                ),
            )
            return response

    def _eligible_record(self, request: SkillReviewCommitRequest, memory_id: str) -> MemoryRecord:
        record = self._repository.get_memory(memory_id)
        if record is None:
            raise SkillReviewCommitRejected("memory_not_found")
        if record.type != "observation" or "skill-observation" not in record.tags:
            raise SkillReviewCommitRejected("not_skill_observation")
        if record.status != "active":
            raise SkillReviewCommitRejected("observation_not_active")
        if request.workspace_id not in record.workspace_ids:
            raise SkillReviewCommitRejected("workspace_mismatch")
        if record.metadata.get("skill") not in request.evidence.skills:
            raise SkillReviewCommitRejected("skill_mismatch")
        return record


def _status_for(resolution: str) -> str:
    return "archived" if resolution == "actioned" else "stale"


def _archived_at(resolution: str) -> str | None:
    return _now() if resolution == "actioned" else None


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _response_from_dict(value: object, *, idempotent: bool) -> SkillReviewCommitResponse:
    if not isinstance(value, dict) or value.get("status") != "committed":
        raise ValueError("stored_skill_review_response_invalid")
    raw_dispositions = value.get("dispositions")
    if not isinstance(raw_dispositions, list):
        raise ValueError("stored_skill_review_response_invalid")
    dispositions = tuple(
        SkillReviewCommitDisposition(
            str(item["memory_id"]),
            cast(Resolution, str(item["resolution"])),
            str(item["status"]),
        )
        for item in raw_dispositions
        if isinstance(item, dict)
    )
    if len(dispositions) != len(raw_dispositions):
        raise ValueError("stored_skill_review_response_invalid")
    return SkillReviewCommitResponse(
        dispositions=dispositions,
        review_run_id=str(value["review_run_id"]),
        idempotent=idempotent,
    )


__all__ = [
    "SQLiteSkillReviewCommitStore",
    "SkillReviewCommitConflict",
    "SkillReviewCommitRejected",
]
