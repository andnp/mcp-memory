import hashlib
from typing import cast

import pytest

from mcp_memory.application.skill_review_contract import parse_skill_review_commit_request


pytestmark = pytest.mark.small


def _request(**overrides: object) -> dict[str, object]:
    """Build a valid protocol request with deterministic evidence."""
    value: dict[str, object] = {
        "protocol_version": 1,
        "review_run_id": "123e4567-e89b-12d3-a456-426614174000",
        "workspace_id": "skill-manager-workspace",
        "evidence": {
            "skills": ["delegate-code-review"],
            "source_hashes": {"delegate-code-review": "a" * 64},
            "deployment_status": "passed",
            "deployment_receipt_hash": "b" * 64,
        },
        "dispositions": [
            {"memory_id": "mem-123", "resolution": "verified", "note": "Already applied."}
        ],
    }
    value.update(overrides)
    return value


def test_parse_skill_review_request_preserves_versioned_wire_shape() -> None:
    """Parse valid evidence and produce a stable digestable request."""
    request = parse_skill_review_commit_request(_request())

    assert request.as_dict()["protocol_version"] == 1
    assert request.evidence.skills == ("delegate-code-review",)
    assert request.dispositions[0].resolution == "verified"
    assert len(request.digest()) == hashlib.sha256().digest_size * 2


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("protocol_version", 2, "unsupported_protocol_version"),
        ("review_run_id", "not-a-uuid", "review_run_id must be a UUID"),
        ("dispositions", [], "dispositions must be a non-empty array"),
    ],
)
def test_parse_skill_review_request_rejects_invalid_top_level_values(
    field: str, value: object, message: str
) -> None:
    """Reject malformed top-level requests before they reach persistence."""
    with pytest.raises(ValueError, match=message):
        parse_skill_review_commit_request(_request(**{field: value}))


def test_parse_skill_review_request_rejects_bad_evidence_hashes() -> None:
    """Require source and deployment receipt hashes to be complete SHA-256 values."""
    evidence = dict(cast(dict[str, object], _request()["evidence"]))
    evidence["source_hashes"] = {"delegate-code-review": "bad"}

    with pytest.raises(ValueError, match="source_hashes"):
        parse_skill_review_commit_request(_request(evidence=evidence))


def test_parse_skill_review_request_rejects_duplicate_dispositions() -> None:
    """Reject duplicate memory identifiers so one batch has one clear outcome."""
    disposition = {"memory_id": "mem-123", "resolution": "verified", "note": "Again."}

    with pytest.raises(ValueError, match="duplicate"):
        parse_skill_review_commit_request(_request(dispositions=[disposition, disposition]))
