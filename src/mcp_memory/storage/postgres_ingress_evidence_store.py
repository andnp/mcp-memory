"""Postgres persistence for ingress replay evidence."""

from __future__ import annotations

import json
from collections.abc import Mapping
from contextlib import contextmanager
from enum import Enum
from typing import Any, Iterator, cast

from mcp_memory.core.ingress_evidence import (
    IngressActionReceipt,
    IngressBatchEvidence,
    IngressReceiptStatus,
    IngressSourceSnapshot,
    SourceCoverage,
    SourceCoverageOutcome,
)
from mcp_memory.storage.session import DbConnectionLike, SessionManager


class _PostgresIngressEvidenceRepository:
    def __init__(self, session_manager: SessionManager[DbConnectionLike] | None) -> None:
        self._sessions = session_manager

    @contextmanager
    def _open_connection(self) -> Iterator[DbConnectionLike]:
        if self._sessions is None:
            raise RuntimeError("ingress_evidence_unavailable")
        with self._sessions.open_connection() as connection:
            yield connection


class PostgresIngressBatchEvidenceRepository(_PostgresIngressEvidenceRepository):
    def save(self, evidence: IngressBatchEvidence) -> IngressBatchEvidence:
        with self._open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO ingress_batch_evidence (
                        batch_id, task_id, execution_epoch, batch_sequence,
                        claimed_entry_ids_json, source_fingerprint, source_entries_json,
                        provider_route, execution_mode, policy_version, schema_version,
                        claimed_at, finalized_at, grouping_strategy, grouping_fallback_reason
                    ) VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s::jsonb, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (batch_id) DO UPDATE SET
                        task_id = EXCLUDED.task_id, execution_epoch = EXCLUDED.execution_epoch,
                        batch_sequence = EXCLUDED.batch_sequence,
                        claimed_entry_ids_json = EXCLUDED.claimed_entry_ids_json,
                        source_fingerprint = EXCLUDED.source_fingerprint,
                        source_entries_json = EXCLUDED.source_entries_json,
                        provider_route = EXCLUDED.provider_route, execution_mode = EXCLUDED.execution_mode,
                        policy_version = EXCLUDED.policy_version, schema_version = EXCLUDED.schema_version,
                        claimed_at = EXCLUDED.claimed_at, finalized_at = EXCLUDED.finalized_at,
                        grouping_strategy = EXCLUDED.grouping_strategy,
                        grouping_fallback_reason = EXCLUDED.grouping_fallback_reason
                    """,
                    _batch_values(evidence),
                )
            connection.commit()
        return evidence

    def get(self, batch_id: str) -> IngressBatchEvidence | None:
        return self._fetch(
            _BATCH_SELECT + " WHERE batch_id = %s", (batch_id,)
        )

    def list_for_execution(self, task_id: str, execution_epoch: int) -> list[IngressBatchEvidence]:
        with self._open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    _BATCH_SELECT
                    + " WHERE task_id = %s AND execution_epoch = %s ORDER BY batch_sequence, batch_id",
                    (task_id, execution_epoch),
                )
                return [_batch_from_row(row) for row in cursor.fetchall()]

    def _fetch(self, query: str, params: tuple[object, ...]) -> IngressBatchEvidence | None:
        with self._open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, params)
                row = cursor.fetchone()
                return None if row is None else _batch_from_row(row)


class PostgresIngressActionReceiptRepository(_PostgresIngressEvidenceRepository):
    def save(self, receipt: IngressActionReceipt) -> IngressActionReceipt:
        with self._open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO ingress_action_receipts (
                        action_id, batch_id, operation, entry_ids_json, target_ids_json,
                        canonical_payload_digest, status, mutation_evidence_id, created_at,
                        terminalized_at, error_code, before_revision_tokens_json,
                        after_revision_tokens_json
                    ) VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb)
                    ON CONFLICT (action_id) DO UPDATE SET
                        batch_id = EXCLUDED.batch_id, operation = EXCLUDED.operation,
                        entry_ids_json = EXCLUDED.entry_ids_json, target_ids_json = EXCLUDED.target_ids_json,
                        canonical_payload_digest = EXCLUDED.canonical_payload_digest,
                        status = EXCLUDED.status, mutation_evidence_id = EXCLUDED.mutation_evidence_id,
                        created_at = EXCLUDED.created_at, terminalized_at = EXCLUDED.terminalized_at,
                        error_code = EXCLUDED.error_code,
                        before_revision_tokens_json = EXCLUDED.before_revision_tokens_json,
                        after_revision_tokens_json = EXCLUDED.after_revision_tokens_json
                    """,
                    _receipt_values(receipt),
                )
            connection.commit()
        return receipt

    def get(self, action_id: str) -> IngressActionReceipt | None:
        return self._fetch(_RECEIPT_SELECT + " WHERE action_id = %s", (action_id,))

    def list_for_batch(self, batch_id: str) -> list[IngressActionReceipt]:
        with self._open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    _RECEIPT_SELECT + " WHERE batch_id = %s ORDER BY action_id", (batch_id,)
                )
                return [_receipt_from_row(row) for row in cursor.fetchall()]

    def _fetch(self, query: str, params: tuple[object, ...]) -> IngressActionReceipt | None:
        with self._open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, params)
                row = cursor.fetchone()
                return None if row is None else _receipt_from_row(row)


class PostgresSourceCoverageRepository(_PostgresIngressEvidenceRepository):
    def save(self, coverage: SourceCoverage) -> SourceCoverage:
        with self._open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO ingress_source_coverage (entry_id, action_id, outcome, reason)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (entry_id) DO UPDATE SET
                        action_id = EXCLUDED.action_id, outcome = EXCLUDED.outcome,
                        reason = EXCLUDED.reason
                    """,
                    (coverage.entry_id, coverage.action_id, coverage.outcome.value, coverage.reason),
                )
            connection.commit()
        return coverage

    def get(self, entry_id: str) -> SourceCoverage | None:
        return self._fetch(_COVERAGE_SELECT + " WHERE entry_id = %s", (entry_id,))

    def list_for_entries(self, entry_ids: tuple[str, ...]) -> list[SourceCoverage]:
        if not entry_ids:
            return []
        placeholders = ", ".join(["%s"] * len(entry_ids))
        with self._open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    _COVERAGE_SELECT + f" WHERE entry_id IN ({placeholders}) ORDER BY entry_id",
                    entry_ids,
                )
                return [_coverage_from_row(row) for row in cursor.fetchall()]

    def _fetch(self, query: str, params: tuple[object, ...]) -> SourceCoverage | None:
        with self._open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, params)
                row = cursor.fetchone()
                return None if row is None else _coverage_from_row(row)


PostgresIngressBatchEvidenceStore = PostgresIngressBatchEvidenceRepository
PostgresIngressActionReceiptStore = PostgresIngressActionReceiptRepository
PostgresSourceCoverageStore = PostgresSourceCoverageRepository

_BATCH_SELECT = """SELECT batch_id, task_id, execution_epoch, batch_sequence,
    claimed_entry_ids_json, source_fingerprint, source_entries_json, provider_route,
    execution_mode, policy_version, schema_version, claimed_at, finalized_at,
    grouping_strategy, grouping_fallback_reason FROM ingress_batch_evidence"""
_RECEIPT_SELECT = """SELECT action_id, batch_id, operation, entry_ids_json, target_ids_json,
    canonical_payload_digest, status, mutation_evidence_id, created_at, terminalized_at,
    error_code, before_revision_tokens_json, after_revision_tokens_json
    FROM ingress_action_receipts"""
_COVERAGE_SELECT = "SELECT entry_id, action_id, outcome, reason FROM ingress_source_coverage"


def _batch_values(evidence: IngressBatchEvidence) -> tuple[object, ...]:
    return (
        evidence.batch_id, evidence.task_id, evidence.execution_epoch, evidence.batch_sequence,
        _json_text(evidence.claimed_entry_ids), evidence.source_fingerprint,
        _json_text(tuple(_source_entry_value(item) for item in evidence.source_entries)),
        evidence.provider_route, evidence.execution_mode,
        evidence.policy_version, evidence.schema_version, evidence.claimed_at,
        evidence.finalized_at, evidence.grouping_strategy, evidence.grouping_fallback_reason,
    )


def _receipt_values(receipt: IngressActionReceipt) -> tuple[object, ...]:
    return (
        receipt.action_id, receipt.batch_id, receipt.operation, _json_text(receipt.entry_ids),
        _json_text(receipt.target_ids), receipt.canonical_payload_digest, receipt.status.value,
        receipt.mutation_evidence_id, receipt.created_at, receipt.terminalized_at,
        receipt.error_code, _json_text(receipt.before_revision_tokens),
        _json_text(receipt.after_revision_tokens),
    )


def _source_entry_value(entry: IngressSourceSnapshot) -> dict[str, object]:
    return {
        "entry_id": entry.entry_id,
        "workspace_ids": entry.workspace_ids,
        "timestamp": entry.timestamp,
        "content_digest": entry.content_digest,
        "snapshot": entry.snapshot,
    }


def _batch_from_row(row: tuple[object, ...]) -> IngressBatchEvidence:
    source_entries = _json_value(row[6], [])
    return IngressBatchEvidence(
        batch_id=cast(str, row[0]), task_id=cast(str, row[1]), execution_epoch=cast(int, row[2]),
        batch_sequence=cast(int, row[3]), claimed_entry_ids=tuple(_json_value(row[4], [])),
        source_fingerprint=cast(str, row[5]),
        source_entries=tuple(
            IngressSourceSnapshot(
                entry_id=item["entry_id"], workspace_ids=tuple(item["workspace_ids"]),
                timestamp=item["timestamp"], content_digest=item["content_digest"],
                snapshot=item.get("snapshot", {}),
            ) for item in source_entries
        ),
        provider_route=cast(str, row[7]), execution_mode=cast(str, row[8]),
        policy_version=cast(str, row[9]), schema_version=cast(str, row[10]),
        claimed_at=cast(str, row[11]), finalized_at=cast(str | None, row[12]),
        grouping_strategy=cast(str | None, row[13]), grouping_fallback_reason=cast(str | None, row[14]),
    )


def _receipt_from_row(row: tuple[object, ...]) -> IngressActionReceipt:
    return IngressActionReceipt(
        action_id=cast(str, row[0]), batch_id=cast(str, row[1]), operation=cast(str, row[2]),
        entry_ids=tuple(_json_value(row[3], [])), target_ids=tuple(_json_value(row[4], [])),
        canonical_payload_digest=cast(str, row[5]), status=IngressReceiptStatus(cast(str, row[6])),
        mutation_evidence_id=cast(str | None, row[7]), created_at=cast(str, row[8]),
        terminalized_at=cast(str | None, row[9]), error_code=cast(str | None, row[10]),
        before_revision_tokens=_json_value(row[11], {}), after_revision_tokens=_json_value(row[12], {}),
    )


def _coverage_from_row(row: tuple[object, ...]) -> SourceCoverage:
    return SourceCoverage(
        entry_id=cast(str, row[0]), action_id=cast(str | None, row[1]),
        outcome=SourceCoverageOutcome(cast(str, row[2])), reason=cast(str | None, row[3]),
    )


def _json_text(value: object) -> str:
    return json.dumps(_json_compatible(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _json_compatible(value: object) -> object:
    if isinstance(value, Enum):
        return _json_compatible(value.value)
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (tuple, list)):
        return [_json_compatible(item) for item in value]
    return value


def _json_value(value: object, default: object) -> Any:
    if isinstance(value, str):
        value = json.loads(value)
    return value if isinstance(value, (dict, list)) else default
