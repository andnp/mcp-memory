"""Versioned wire types for skill-review disposition commits."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Literal, Mapping
from uuid import UUID


PROTOCOL_VERSION = 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
Resolution = Literal["actioned", "verified", "deferred"]


@dataclass(frozen=True)
class SkillReviewEvidence:
    skills: tuple[str, ...]
    source_hashes: Mapping[str, str]
    deployment_status: Literal["passed"]
    deployment_receipt_hash: str

    def as_dict(self) -> dict[str, object]:
        return {
            "skills": list(self.skills),
            "source_hashes": dict(sorted(self.source_hashes.items())),
            "deployment_status": self.deployment_status,
            "deployment_receipt_hash": self.deployment_receipt_hash,
        }


@dataclass(frozen=True)
class SkillReviewDisposition:
    memory_id: str
    resolution: Resolution
    note: str

    def as_dict(self) -> dict[str, str]:
        return {
            "memory_id": self.memory_id,
            "resolution": self.resolution,
            "note": self.note,
        }


@dataclass(frozen=True)
class SkillReviewCommitRequest:
    review_run_id: str
    workspace_id: str
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
    resolution: Resolution
    status: str

    def as_dict(self) -> dict[str, str]:
        return {
            "memory_id": self.memory_id,
            "resolution": self.resolution,
            "status": self.status,
        }


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
    return SkillReviewEvidence(
        skills=skills,
        source_hashes=source_hashes,
        deployment_status="passed",
        deployment_receipt_hash=deployment_receipt_hash,
    )


def _parse_disposition(value: object) -> SkillReviewDisposition:
    if not isinstance(value, Mapping):
        raise ValueError("each disposition must be an object")
    memory_id = _required_text(value, "memory_id")
    resolution = value.get("resolution")
    if resolution not in {"actioned", "verified", "deferred"}:
        raise ValueError("resolution must be actioned, verified, or deferred")
    return SkillReviewDisposition(memory_id, resolution, _required_text(value, "note"))


def _required_text(value: Mapping[str, object], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return item.strip()


__all__ = [
    "PROTOCOL_VERSION",
    "Resolution",
    "SkillReviewCommitDisposition",
    "SkillReviewCommitRequest",
    "SkillReviewDisposition",
    "SkillReviewEvidence",
    "parse_skill_review_commit_request",
]
