from __future__ import annotations

import sqlite3
from dataclasses import replace
from typing import Any, cast

import pytest

from mcp_memory.core.curation_quality_policy import QualityOutcome
from mcp_memory.core.ingress_evidence import IngressQualityEvidence
from mcp_memory.core.ports.ingress import IngressQualityEvidenceConflictError
from mcp_memory.storage.postgres_ingress_quality_store import PostgresIngressQualityEvidenceStore

pytestmark = pytest.mark.small


class FakeCursor:
    def __init__(self, connection: FakeConnection) -> None:
        self._cursor = connection.database.cursor()

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._cursor.close()
        return False

    def execute(self, query: str, params: tuple[object, ...] | None = None) -> None:
        self._cursor.execute(query.replace("%s::jsonb", "?").replace("%s", "?"), tuple(params or ()))

    def fetchone(self) -> tuple[object, ...] | None:
        return self._cursor.fetchone()


class FakeConnection:
    def __init__(self) -> None:
        self.database = sqlite3.connect(":memory:")
        self.database.execute(
            """
            CREATE TABLE ingress_quality_evidence (
                action_id TEXT PRIMARY KEY,
                disposition TEXT NOT NULL,
                reason TEXT,
                evaluated_at TEXT NOT NULL,
                evaluator TEXT,
                query_provenance_json TEXT NOT NULL DEFAULT '{}'
            )
            """
        )

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)

    def commit(self) -> None:
        self.database.commit()

    def rollback(self) -> None:
        self.database.rollback()


class FakeLease:
    def __init__(self, connection: FakeConnection) -> None:
        self.connection = connection

    def __enter__(self) -> FakeConnection:
        return self.connection

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class FakeSessionManager:
    def __init__(self) -> None:
        self.connection = FakeConnection()

    def open_connection(self) -> FakeLease:
        return FakeLease(self.connection)


def _evidence() -> IngressQualityEvidence:
    return IngressQualityEvidence(
        action_id="action-1",
        disposition=QualityOutcome.UNOBSERVED,
        reason="query was not trusted",
        evaluated_at="2026-08-08T12:00:04+00:00",
        evaluator="quality-check-1",
        query_provenance={"query": "memory", "search_epoch": 12},
    )


def test_postgres_quality_store_round_trips_and_replays_immutable_evidence() -> None:
    """The Postgres repository contract survives native JSONB-like hydration."""
    manager = cast(Any, FakeSessionManager())
    store = PostgresIngressQualityEvidenceStore(manager)
    evidence = _evidence()

    assert store.save(evidence) == evidence
    assert store.save(evidence) == evidence
    assert store.get(evidence.action_id) == evidence


def test_postgres_quality_store_rejects_action_identity_conflicts() -> None:
    """A second disposition for an action fails without overwriting the first."""
    manager = cast(Any, FakeSessionManager())
    store = PostgresIngressQualityEvidenceStore(manager)
    evidence = _evidence()
    store.save(evidence)

    with pytest.raises(IngressQualityEvidenceConflictError):
        store.save(replace(evidence, evaluator="different-evaluator"))

    assert store.get(evidence.action_id) == evidence
