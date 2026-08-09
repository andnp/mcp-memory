from __future__ import annotations

from dataclasses import replace

import pytest

from mcp_memory.core.curation_quality_policy import QualityOutcome
from mcp_memory.core.ingress_evidence import IngressQualityEvidence
from mcp_memory.core.ports.ingress import IngressQualityEvidenceConflictError
from mcp_memory.storage.ingress_quality_store import SQLiteIngressQualityEvidenceStore
from mcp_memory.utils.db import DatabaseManager

pytestmark = pytest.mark.small


def _evidence() -> IngressQualityEvidence:
    return IngressQualityEvidence(
        action_id="action-1",
        disposition=QualityOutcome.PRODUCTIVE,
        evaluated_at="2026-08-08T12:00:04+00:00",
        evaluator="quality-check-1",
        query_provenance={"query": "memory", "search_epoch": 12},
    )


def test_sqlite_quality_store_round_trips_and_replays_immutable_evidence(
    db_manager: DatabaseManager,
) -> None:
    """SQLite restores provenance and returns the first exact action evaluation."""
    store = SQLiteIngressQualityEvidenceStore(db_manager)
    evidence = _evidence()

    assert store.save(evidence) == evidence
    assert store.save(evidence) == evidence
    assert store.get(evidence.action_id) == evidence


def test_sqlite_quality_store_rejects_action_identity_conflicts(db_manager: DatabaseManager) -> None:
    """SQLite does not replace quality evidence when the same action is reevaluated differently."""
    store = SQLiteIngressQualityEvidenceStore(db_manager)
    evidence = _evidence()
    store.save(evidence)

    with pytest.raises(IngressQualityEvidenceConflictError):
        store.save(replace(evidence, disposition=QualityOutcome.REGRESSED))

    assert store.get(evidence.action_id) == evidence
