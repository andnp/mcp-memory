"""Versioned wire types for skill-review disposition commits."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Mapping
from uuid import UUID

PROTOCOL_VERSION = 2
LEDGER_PROTOCOL_VERSION = 1
LEDGER_ID = "skill-observation-ledger-v1"
LEDGER_PAGE_SIZE_MAX = 100
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LEDGER_TOKEN = re.compile(r"^([0-9a-f]{64}):(0|[1-9][0-9]*)$")
Resolution = Literal["actioned", "verified", "deferred"]
ReviewOutcome = Literal["done", "bad"]


@dataclass(frozen=True)
class SkillReviewEvidence:
    skills: tuple[str, ...]
    source_hashes: Mapping[str, str]
    deployment_status: Literal["passed"]
    deployment_receipt_hash: str
    ledger_snapshot_id: str

    def as_dict(self) -> dict[str, object]:
        return {
            "skills": list(self.skills),
            "source_hashes": dict(sorted(self.source_hashes.items())),
            "deployment_status": self.deployment_status,
            "deployment_receipt_hash": self.deployment_receipt_hash,
            "ledger_snapshot_id": self.ledger_snapshot_id,
        }


@dataclass(frozen=True)
class SkillReviewDisposition:
    memory_id: str
    outcome: ReviewOutcome
    note: str

    def as_dict(self) -> dict[str, str]:
        return {
            "memory_id": self.memory_id,
            "outcome": self.outcome,
            "note": self.note,
        }


@dataclass(frozen=True)
class SkillReviewCommitRequest:
    review_run_id: str
    workspace_id: str | None
    evidence: SkillReviewEvidence
    dispositions: tuple[SkillReviewDisposition, ...]
    protocol_version: int = PROTOCOL_VERSION

    def as_dict(self) -> dict[str, object]:
        return {
            "protocol_version": self.protocol_version,
            "review_run_id": self.review_run_id,
            "workspace_id": self.workspace_id,
            "evidence": self.evidence.as_dict(),
            "dispositions": [item.as_dict() for item in self.dispositions],
        }

    def digest(self) -> str:
        encoded = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class SkillReviewCommitDisposition:
    memory_id: str
    outcome: ReviewOutcome
    status: str

    def as_dict(self) -> dict[str, str]:
        return {
            "memory_id": self.memory_id,
            "outcome": self.outcome,
            "status": self.status,
        }


@dataclass(frozen=True)
class SkillReviewCommitResponse:
    dispositions: tuple[SkillReviewCommitDisposition, ...]
    review_run_id: str
    idempotent: bool
    protocol_version: int = PROTOCOL_VERSION
    status: Literal["committed"] = "committed"

    def as_dict(self) -> dict[str, object]:
        return {
            "protocol_version": self.protocol_version,
            "status": self.status,
            "review_run_id": self.review_run_id,
            "idempotent": self.idempotent,
            "dispositions": [item.as_dict() for item in self.dispositions],
        }


@dataclass(frozen=True)
class SkillReviewLedgerRequest:
    workspace_id: str | None
    page_size: int
    page_token: str | None = None
    protocol_version: int = LEDGER_PROTOCOL_VERSION


def parse_skill_review_ledger_request(arguments: Mapping[str, object]) -> SkillReviewLedgerRequest:
    """Validate one read-only ledger page request."""
    version = arguments.get("protocol_version")
    if isinstance(version, bool) or version != LEDGER_PROTOCOL_VERSION:
        raise ValueError("unsupported_ledger_protocol_version")
    workspace_id_value = arguments.get("workspace_id")
    if workspace_id_value is None:
        workspace_id = None
    else:
        workspace_id = _required_text(arguments, "workspace_id")
    page_size_value = arguments.get("page_size", LEDGER_PAGE_SIZE_MAX)
    if isinstance(page_size_value, bool) or not isinstance(page_size_value, int):
        raise ValueError("page_size must be an integer")
    if not 1 <= page_size_value <= LEDGER_PAGE_SIZE_MAX:
        raise ValueError(f"page_size must be between 1 and {LEDGER_PAGE_SIZE_MAX}")
    page_token = arguments.get("page_token")
    if page_token is not None:
        if not isinstance(page_token, str) or _LEDGER_TOKEN.fullmatch(page_token) is None:
            raise ValueError("page_token is invalid")
    return SkillReviewLedgerRequest(workspace_id, page_size_value, page_token)


def ledger_token_parts(page_token: str | None) -> tuple[str | None, int]:
    """Decode a snapshot-bound page cursor into its identity and offset."""
    if page_token is None:
        return None, 0
    match = _LEDGER_TOKEN.fullmatch(page_token)
    if match is None:
        raise ValueError("page_token is invalid")
    return match.group(1), int(match.group(2))


def skill_review_ledger_entry(record: object) -> dict[str, object]:
    """Build the bounded summary projection used by the review agent."""
    metadata = getattr(record, "metadata", {})
    return {
        "memory_id": str(getattr(record, "id")),
        "memory_ref": getattr(record, "memory_ref"),
        "title": str(getattr(record, "title")),
        "summary": str(getattr(record, "summary") or ""),
        "skill": str(metadata.get("skill", "")) if isinstance(metadata, Mapping) else "",
        "created_at": str(getattr(record, "created_at")),
        "updated_at": str(getattr(record, "updated_at")),
    }


def skill_review_ledger_snapshot_id(entries: Sequence[Mapping[str, object]]) -> str:
    """Hash the complete ordered ledger projection for page consistency."""
    encoded = json.dumps(list(entries), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def parse_skill_review_commit_request(arguments: Mapping[str, object]) -> SkillReviewCommitRequest:
    """Validate one protocol request before any storage mutation is attempted."""
    version = arguments.get("protocol_version")
    if isinstance(version, bool) or version != PROTOCOL_VERSION:
        raise ValueError("unsupported_protocol_version")
    review_run_id = _required_text(arguments, "review_run_id")
    try:
        if str(UUID(review_run_id)) != review_run_id:
            raise ValueError
    except ValueError as error:
        raise ValueError("review_run_id must be a UUID") from error
    workspace_id_value = arguments.get("workspace_id")
    if workspace_id_value is None:
        workspace_id = None
    else:
        workspace_id = _required_text(arguments, "workspace_id")
    evidence_value = arguments.get("evidence")
    if not isinstance(evidence_value, Mapping):
        raise ValueError("evidence must be an object")
    evidence = _parse_evidence(evidence_value)
    dispositions_value = arguments.get("dispositions")
    if not isinstance(dispositions_value, list) or not dispositions_value:
        raise ValueError("dispositions must be a non-empty array")
    dispositions = tuple(_parse_disposition(item) for item in dispositions_value)
    memory_ids = [item.memory_id for item in dispositions]
    if len(memory_ids) != len(set(memory_ids)):
        raise ValueError("dispositions must not contain duplicate memory_id values")
    return SkillReviewCommitRequest(
        review_run_id=review_run_id,
        workspace_id=workspace_id,
        evidence=evidence,
        dispositions=dispositions,
    )


def _parse_evidence(value: Mapping[str, object]) -> SkillReviewEvidence:
    skills_value = value.get("skills")
    if not isinstance(skills_value, list) or not skills_value or not all(
        isinstance(skill, str) and skill.strip() for skill in skills_value
    ):
        raise ValueError("evidence.skills must be a non-empty array of strings")
    skills = tuple(dict.fromkeys(str(skill).strip() for skill in skills_value))
    source_hashes_value = value.get("source_hashes")
    if not isinstance(source_hashes_value, Mapping):
        raise ValueError("evidence.source_hashes must be an object")
    source_hashes = {str(key): str(hash_value) for key, hash_value in source_hashes_value.items()}
    if set(source_hashes) != set(skills) or not all(_SHA256.fullmatch(item) for item in source_hashes.values()):
        raise ValueError("evidence.source_hashes must contain one sha256 hash per skill")
    if value.get("deployment_status") != "passed":
        raise ValueError("evidence.deployment_status must be passed")
    deployment_receipt_hash = _required_text(value, "deployment_receipt_hash")
    if not _SHA256.fullmatch(deployment_receipt_hash):
        raise ValueError("evidence.deployment_receipt_hash must be a sha256 hash")
    ledger_snapshot_id = _required_text(value, "ledger_snapshot_id")
    if not _SHA256.fullmatch(ledger_snapshot_id):
        raise ValueError("evidence.ledger_snapshot_id must be a sha256 hash")
    return SkillReviewEvidence(
        skills=skills,
        source_hashes=source_hashes,
        deployment_status="passed",
        deployment_receipt_hash=deployment_receipt_hash,
        ledger_snapshot_id=ledger_snapshot_id,
    )


def _parse_disposition(value: object) -> SkillReviewDisposition:
    if not isinstance(value, Mapping):
        raise ValueError("each disposition must be an object")
    memory_id = _required_text(value, "memory_id")
    outcome = value.get("outcome")
    if outcome not in {"done", "bad"}:
        raise ValueError("outcome must be done or bad")
    return SkillReviewDisposition(memory_id, outcome, _required_text(value, "note"))


def _required_text(value: Mapping[str, object], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return item.strip()


__all__ = [
    "LEDGER_ID",
    "LEDGER_PAGE_SIZE_MAX",
    "LEDGER_PROTOCOL_VERSION",
    "PROTOCOL_VERSION",
    "ReviewOutcome",
    "Resolution",
    "SkillReviewCommitDisposition",
    "SkillReviewCommitRequest",
    "SkillReviewCommitResponse",
    "SkillReviewDisposition",
    "SkillReviewEvidence",
    "SkillReviewLedgerRequest",
    "ledger_token_parts",
    "parse_skill_review_commit_request",
    "parse_skill_review_ledger_request",
    "skill_review_ledger_entry",
    "skill_review_ledger_snapshot_id",
]
