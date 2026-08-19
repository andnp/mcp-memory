"""Atomic Postgres execution for one create or append ingress mutation."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast
from uuid import uuid4

from mcp_memory.core.curation_identity import record_snapshot, record_token
from mcp_memory.core.ingress_evidence import (
    IngressActionReceipt,
    IngressReceiptStatus,
    SourceCoverage,
)
from mcp_memory.core.ingress_identity import action_identity, canonical_payload_digest
from mcp_memory.core.journal import RECOVERABLE_RETENTION_SECONDS
from mcp_memory.core.ports.ingress import SourceCoverageAssignmentConflictError
from mcp_memory.core.ports.memory import MemoryRecord
from mcp_memory.storage.ingress_mutation_uow import (
    IngressDomainTransaction,
    IngressMutationInjectedFailure,
    IngressMutationResult,
    check_digest,
    coverage_for_action,
    fail_stage,
    identifier,
    json_text,
    now_text,
    prepare_append,
    prepare_create,
)
from mcp_memory.storage.ingress_mutation_uow import (
    journal_entry_ids as _to_journal_entry_ids,
)
from mcp_memory.storage.session import CursorLike, DbConnectionLike, SessionManager

__all__ = [
    "IngressDomainTransaction",
    "IngressMutationInjectedFailure",
    "IngressMutationResult",
    "PostgresIngressMutationStore",
]


class PostgresIngressMutationStore:
    """Apply one ingress create or append with one Postgres commit."""

    def __init__(
        self,
        session_manager: SessionManager[DbConnectionLike],
        *,
        embedding_model: str = "default",
        fault_stage: str | None = None,
        fault_injector: Callable[[str], None] | None = None,
    ) -> None:
        if not embedding_model.strip():
            raise ValueError("embedding_model must be non-empty")
        self._sessions = session_manager
        self._embedding_model = embedding_model.strip()
        self._fault_stage = fault_stage
        self._fault_injector = fault_injector

    def execute(
        self,
        *,
        batch_id: str,
        operation: str,
        entry_ids: Sequence[str],
        target_ids: Sequence[str],
        payload: object,
        apply: Callable[[IngressDomainTransaction], IngressMutationResult],
        source_coverage: Sequence[SourceCoverage] | None = None,
        mutation_evidence_id: str | None = None,
        journal_task_id: str | None = None,
    ) -> IngressActionReceipt:
        normalized_batch = identifier(batch_id, "batch_id")
        normalized_operation = identifier(operation, "operation")
        if normalized_operation not in {"create", "append"}:
            raise ValueError("ingress operation must be create or append")
        identity = action_identity(normalized_operation, entry_ids, target_ids)
        digest = canonical_payload_digest(payload)
        normalized_journal_task = (
            None if journal_task_id is None else identifier(journal_task_id, "journal_task_id")
        )
        entry_journal_ids = (
            () if normalized_journal_task is None else _to_journal_entry_ids(identity.entry_ids)
        )
        normalized_coverage = coverage_for_action(
            source_coverage,
            entry_ids=identity.entry_ids,
            action_id=identity.action_id,
            operation=normalized_operation,
        )
        provisional = IngressActionReceipt(
            action_id=identity.action_id,
            batch_id=normalized_batch,
            operation=identity.operation,
            entry_ids=identity.entry_ids,
            target_ids=identity.target_ids,
            canonical_payload_digest=digest,
            status=IngressReceiptStatus.UNOBSERVED,
            mutation_evidence_id=mutation_evidence_id,
            created_at=now_text(),
        )

        with self._sessions.open_connection() as connection:
            try:
                with connection.cursor() as cursor:
                    inserted = _reserve_receipt(cursor, provisional)
                    if not inserted:
                        existing = _receipt_for_action(cursor, identity.action_id, lock=True)
                        if existing is None:
                            raise RuntimeError("ingress action receipt disappeared after conflict")
                        check_digest(existing, digest)
                        connection.rollback()
                        return existing

                    _lock_targets(cursor, identity.target_ids)
                    self._fail_stage("action_reserved")
                    before = _read_records(cursor, identity.target_ids)
                    domain = _PostgresIngressDomainTransaction(cursor)
                    result = apply(domain)
                    if not isinstance(result, IngressMutationResult):
                        raise TypeError("ingress callback must return IngressMutationResult")
                    if result.operation.strip() != normalized_operation:
                        raise ValueError("ingress operation does not match mutation result")
                    self._fail_stage("after_domain_mutation")

                    affected_ids = _canonical_ids([*identity.target_ids, *domain.touched_ids, *result.affected_ids])
                    after = _read_records(cursor, affected_ids)
                    event_id = str(uuid4())
                    self._write_repair_intents(cursor, before, after)
                    self._fail_stage("repair_intent")
                    self._write_history(
                        cursor,
                        event_id=event_id,
                        action_id=identity.action_id,
                        operation=normalized_operation,
                        before=before,
                        after=after,
                    )
                    self._fail_stage("history")

                    receipt = IngressActionReceipt(
                        action_id=identity.action_id,
                        batch_id=normalized_batch,
                        operation=normalized_operation,
                        entry_ids=identity.entry_ids,
                        target_ids=identity.target_ids,
                        canonical_payload_digest=digest,
                        status=IngressReceiptStatus.APPLIED_UNVERIFIED,
                        mutation_evidence_id=mutation_evidence_id or event_id,
                        created_at=provisional.created_at,
                        terminalized_at=now_text(),
                        before_revision_tokens={key: record_token(value) for key, value in before.items()},
                        after_revision_tokens={key: record_token(value) for key, value in after.items()},
                    )
                    self._fail_stage("before_receipt_coverage_commit")
                    cursor.execute(
                        """
                        UPDATE ingress_action_receipts SET
                            status = %s, mutation_evidence_id = %s,
                            before_revision_tokens_json = %s::jsonb,
                            after_revision_tokens_json = %s::jsonb,
                            terminalized_at = %s, error_code = %s
                        WHERE action_id = %s
                        """,
                        (
                            receipt.status.value,
                            receipt.mutation_evidence_id,
                            json_text(receipt.before_revision_tokens),
                            json_text(receipt.after_revision_tokens),
                            receipt.terminalized_at,
                            receipt.error_code,
                            receipt.action_id,
                        ),
                    )
                    for coverage in normalized_coverage:
                        _assign_coverage(cursor, coverage)
                    if normalized_journal_task is not None:
                        _move_claimed_journal_entries(
                            cursor,
                            entry_journal_ids,
                            task_id=normalized_journal_task,
                        )
                        self._fail_stage("journal_reconcile")
                connection.commit()
                return receipt
            except Exception:
                connection.rollback()
                raise

    def _write_repair_intents(
        self,
        cursor: CursorLike,
        before: Mapping[str, MemoryRecord],
        after: Mapping[str, MemoryRecord],
    ) -> None:
        now = time.time()
        for memory_id, record in after.items():
            previous = before.get(memory_id)
            if previous is not None and record_token(previous) == record_token(record):
                continue
            cursor.execute(
                """
                INSERT INTO embedding_repair_queue (
                    id, memory_id, workspace_id, model_name, memory_updated_at,
                    status, attempt_count, available_at, created_at, updated_at,
                    claimed_at, completed_at, lease_owner, lease_expires_at, last_error
                ) VALUES (%s, %s, %s, %s, %s, 'pending', 0, %s, %s, %s,
                          NULL, NULL, NULL, NULL, NULL)
                ON CONFLICT (memory_id, model_name, memory_updated_at) DO NOTHING
                """,
                (
                    str(uuid4()),
                    memory_id,
                    record.workspace_ids[0] if record.workspace_ids else None,
                    self._embedding_model,
                    record.updated_at,
                    now,
                    now,
                    now,
                ),
            )

    def _write_history(
        self,
        cursor: CursorLike,
        *,
        event_id: str,
        action_id: str,
        operation: str,
        before: Mapping[str, MemoryRecord],
        after: Mapping[str, MemoryRecord],
    ) -> None:
        cursor.execute(
            """
            INSERT INTO memory_mutation_events (
                id, operation, actor_kind, family, action_id, schema_version, status, created_at
            ) VALUES (%s, %s, 'system', 'ingress', %s, 1, 'applied', %s)
            """,
            (event_id, operation, action_id, now_text()),
        )
        for memory_id in _canonical_ids([*before, *after]):
            previous = before.get(memory_id)
            current = after.get(memory_id)
            if previous is not None and current is not None and record_token(previous) == record_token(current):
                continue
            cursor.execute(
                """
                INSERT INTO memory_record_revisions (
                    event_id, memory_id, role, before_exists, before_snapshot,
                    after_exists, after_snapshot, before_token, after_token
                ) VALUES (%s, %s, 'target', %s, %s::jsonb, %s, %s::jsonb, %s, %s)
                """,
                (
                    event_id,
                    memory_id,
                    previous is not None,
                    None if previous is None else json_text(record_snapshot(previous)),
                    current is not None,
                    None if current is None else json_text(record_snapshot(current)),
                    None if previous is None else record_token(previous),
                    None if current is None else record_token(current),
                ),
            )

    def _fail_stage(self, stage: str) -> None:
        fail_stage(stage, fault_stage=self._fault_stage, fault_injector=self._fault_injector)


class _PostgresIngressDomainTransaction:
    def __init__(self, cursor: CursorLike) -> None:
        self._cursor = cursor
        self.touched_ids: list[str] = []

    def get_memory(self, memory_id: str) -> MemoryRecord | None:
        normalized_id = identifier(memory_id, "memory_id")
        row = _select_memory_row(self._cursor, normalized_id)
        return None if row is None else _hydrate_record(self._cursor, row)

    def create_memory(self, **values: object) -> MemoryRecord:
        prepared = prepare_create(values)
        self._cursor.execute(
            """
            INSERT INTO memories (
                id, title, content, summary, type, status, created_at, updated_at,
                read_count, access_score, last_accessed_at, last_surfaced_at, metadata, archived_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 0, 0.0, NULL, NULL, %s::jsonb, NULL)
            """,
            (
                prepared.memory_id,
                prepared.title,
                prepared.content,
                prepared.summary,
                prepared.memory_type,
                prepared.status,
                prepared.created_at,
                prepared.created_at,
                json_text(prepared.metadata),
            ),
        )
        self._replace_mappings(prepared.memory_id, prepared.workspace_ids, prepared.tags)
        self._replace_search_document(
            prepared.memory_id, prepared.title, prepared.summary, prepared.content, prepared.tags
        )
        self.touched_ids.append(prepared.memory_id)
        record = self.get_memory(prepared.memory_id)
        assert record is not None
        return record

    def append_memory(
        self,
        memory_id: str,
        content: str,
        *,
        summary: str | object | None = None,
        tags: Sequence[str] | None = None,
        workspace_ids: Sequence[str] | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> MemoryRecord:
        normalized_id = identifier(memory_id, "memory_id")
        existing = self.get_memory(normalized_id)
        if existing is None:
            raise ValueError(f"memory {normalized_id!r} was not found")
        prepared = prepare_append(
            existing, content, summary=summary, tags=tags, workspace_ids=workspace_ids, metadata=metadata
        )
        updated_at = now_text()
        self._cursor.execute(
            """
            UPDATE memories
            SET content = %s, summary = %s, metadata = %s::jsonb, updated_at = %s
            WHERE id = %s
            """,
            (prepared.content, prepared.summary, json_text(prepared.metadata), updated_at, normalized_id),
        )
        self._replace_mappings(normalized_id, prepared.workspace_ids, prepared.tags)
        self._replace_search_document(
            normalized_id, existing.title, prepared.summary or "", prepared.content, prepared.tags
        )
        self.touched_ids.append(normalized_id)
        record = self.get_memory(normalized_id)
        assert record is not None
        return record

    def _replace_mappings(self, memory_id: str, workspace_ids: list[str], tags: list[str]) -> None:
        self._cursor.execute("DELETE FROM memory_workspaces WHERE memory_id = %s", (memory_id,))
        for workspace_id in workspace_ids:
            self._cursor.execute(
                "INSERT INTO memory_workspaces (memory_id, workspace_id) VALUES (%s, %s)",
                (memory_id, workspace_id),
            )
        self._cursor.execute("DELETE FROM memory_tags WHERE memory_id = %s", (memory_id,))
        for tag in tags:
            self._cursor.execute(
                "INSERT INTO tags (name) VALUES (%s) ON CONFLICT (name) DO NOTHING", (tag,)
            )
            self._cursor.execute("SELECT id FROM tags WHERE name = %s", (tag,))
            row = self._cursor.fetchone()
            if row is not None:
                self._cursor.execute(
                    "INSERT INTO memory_tags (memory_id, tag_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                    (memory_id, row[0]),
                )

    def _replace_search_document(
        self, memory_id: str, title: str, summary: str, content: str, tags: list[str]
    ) -> None:
        tags_text = " ".join(tags)
        self._cursor.execute(
            """
            INSERT INTO memory_search_documents (
                memory_id, title, summary, content, tags_text, search_document
            ) VALUES (
                %s, %s, %s, %s, %s,
                setweight(to_tsvector('simple', COALESCE(%s, '')), 'A')
                || setweight(to_tsvector('simple', COALESCE(%s, '')), 'A')
                || setweight(to_tsvector('simple', COALESCE(%s, '')), 'B')
                || setweight(to_tsvector('simple', COALESCE(%s, '')), 'C')
            )
            ON CONFLICT (memory_id) DO UPDATE SET
                title = EXCLUDED.title, summary = EXCLUDED.summary,
                content = EXCLUDED.content, tags_text = EXCLUDED.tags_text,
                search_document = EXCLUDED.search_document
            """,
            (memory_id, title, summary, content, tags_text, title, summary, tags_text, content),
        )


def _reserve_receipt(cursor: CursorLike, receipt: IngressActionReceipt) -> bool:
    cursor.execute(
        """
        INSERT INTO ingress_action_receipts (
            action_id, batch_id, operation, entry_ids_json, target_ids_json,
            canonical_payload_digest, status, mutation_evidence_id, created_at,
            terminalized_at, error_code, before_revision_tokens_json,
            after_revision_tokens_json
        ) VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb)
        ON CONFLICT (action_id) DO NOTHING
        RETURNING action_id
        """,
        _receipt_values(receipt),
    )
    return cursor.fetchone() is not None


def _assign_coverage(cursor: CursorLike, coverage: SourceCoverage) -> None:
    cursor.execute(
        """
        INSERT INTO ingress_source_coverage (entry_id, action_id, outcome, reason)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (entry_id) DO NOTHING
        """,
        (coverage.entry_id, coverage.action_id, coverage.outcome.value, coverage.reason),
    )
    cursor.execute(
        "SELECT outcome, action_id FROM ingress_source_coverage WHERE entry_id = %s FOR UPDATE",
        (coverage.entry_id,),
    )
    row = cursor.fetchone()
    if row is None:
        raise RuntimeError("source coverage assignment was not persisted")
    from mcp_memory.core.ingress_evidence import SourceCoverageOutcome

    stored = (None if row[1] is None else str(row[1]), SourceCoverageOutcome(str(row[0])))
    requested = (coverage.action_id, coverage.outcome)
    if stored != requested:
        raise SourceCoverageAssignmentConflictError(
            coverage.entry_id, stored[0], stored[1], requested[0], requested[1]
        )


def _move_claimed_journal_entries(
    cursor: CursorLike,
    entry_ids: Sequence[int],
    *,
    task_id: str,
) -> None:
    if not entry_ids:
        return
    placeholders = ",".join("%s" for _ in entry_ids)
    cursor.execute(
        f"""
        UPDATE system1_journal
        SET status = %s, claim_task_id = NULL, claimed_at = NULL,
            recoverable_until = %s
        WHERE id IN ({placeholders}) AND status = %s AND claim_task_id = %s
        """,
        ("recoverable", time.time() + RECOVERABLE_RETENTION_SECONDS, *entry_ids, "claimed", task_id),
    )


def _receipt_for_action(cursor: CursorLike, action_id: str, *, lock: bool = False) -> IngressActionReceipt | None:
    query = _RECEIPT_SELECT + " WHERE action_id = %s"
    if lock:
        query += " FOR UPDATE"
    cursor.execute(query, (action_id,))
    row = cursor.fetchone()
    if row is None:
        return None
    return IngressActionReceipt(
        action_id=str(row[0]),
        batch_id=str(row[1]),
        operation=str(row[2]),
        entry_ids=tuple(str(value) for value in _json_array(row[3])),
        target_ids=tuple(str(value) for value in _json_array(row[4])),
        canonical_payload_digest=str(row[5]),
        status=IngressReceiptStatus(str(row[6])),
        mutation_evidence_id=None if row[7] is None else str(row[7]),
        created_at=str(row[8]),
        terminalized_at=None if row[9] is None else str(row[9]),
        error_code=None if row[10] is None else str(row[10]),
        before_revision_tokens={str(key): str(value) for key, value in _json_object(row[11]).items()},
        after_revision_tokens={str(key): str(value) for key, value in _json_object(row[12]).items()},
    )


def _receipt_values(receipt: IngressActionReceipt) -> tuple[object, ...]:
    return (
        receipt.action_id,
        receipt.batch_id,
        receipt.operation,
        json_text(receipt.entry_ids),
        json_text(receipt.target_ids),
        receipt.canonical_payload_digest,
        receipt.status.value,
        receipt.mutation_evidence_id,
        receipt.created_at,
        receipt.terminalized_at,
        receipt.error_code,
        json_text(receipt.before_revision_tokens),
        json_text(receipt.after_revision_tokens),
    )


def _lock_targets(cursor: CursorLike, target_ids: Sequence[str]) -> None:
    for memory_id in target_ids:
        cursor.execute("SELECT id FROM memories WHERE id = %s FOR UPDATE", (memory_id,))
        if cursor.fetchone() is None:
            raise ValueError(f"memory {memory_id!r} was not found")


def _read_records(cursor: CursorLike, memory_ids: Sequence[str]) -> dict[str, MemoryRecord]:
    records: dict[str, MemoryRecord] = {}
    for memory_id in memory_ids:
        row = _select_memory_row(cursor, memory_id)
        if row is not None:
            records[memory_id] = _hydrate_record(cursor, row)
    return records


def _select_memory_row(cursor: CursorLike, memory_id: str) -> tuple[object, ...] | None:
    cursor.execute(
        """
        SELECT id, title, content, summary, type, status, created_at, updated_at,
               read_count, access_score, last_accessed_at, last_surfaced_at,
               metadata, memory_ref, archived_at
        FROM memories WHERE id = %s
        """,
        (memory_id,),
    )
    return cursor.fetchone()


def _hydrate_record(cursor: CursorLike, row: tuple[object, ...]) -> MemoryRecord:
    memory_id = str(row[0])
    cursor.execute("SELECT workspace_id FROM memory_workspaces WHERE memory_id = %s ORDER BY workspace_id", (memory_id,))
    workspace_rows = cursor.fetchall()
    cursor.execute(
        "SELECT tags.name FROM tags JOIN memory_tags ON memory_tags.tag_id = tags.id "
        "WHERE memory_tags.memory_id = %s ORDER BY tags.name",
        (memory_id,),
    )
    tag_rows = cursor.fetchall()
    metadata = _json_object(row[12])
    return MemoryRecord(
        id=memory_id,
        title=str(row[1]),
        content=str(row[2]),
        summary=None if row[3] is None else str(row[3]),
        type=str(row[4]),
        status=str(row[5]),
        created_at=str(row[6]),
        updated_at=str(row[7]),
        read_count=int(cast(Any, row[8]) or 0),
        access_score=float(cast(Any, row[9]) or 0),
        last_accessed_at=None if row[10] is None else str(row[10]),
        last_surfaced_at=None if row[11] is None else str(row[11]),
        metadata={str(key): value for key, value in metadata.items()},
        workspace_ids=[str(item[0]) for item in workspace_rows],
        tags=[str(item[0]) for item in tag_rows],
        memory_ref=None if row[13] is None else int(cast(Any, row[13])),
        archived_at=None if row[14] is None else str(row[14]),
    )


def _canonical_ids(values: Sequence[str]) -> tuple[str, ...]:
    normalized = {identifier(value, "identity") for value in values}
    return tuple(sorted(normalized, key=lambda value: value.encode("utf-8")))


def _json_value(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, str):
        return json.loads(value)
    return value


def _json_array(value: object) -> list[object]:
    loaded = _json_value(value)
    return loaded if isinstance(loaded, list) else []


def _json_object(value: object) -> dict[str, object]:
    loaded = _json_value(value)
    return loaded if isinstance(loaded, dict) else {}


_RECEIPT_SELECT = """SELECT action_id, batch_id, operation, entry_ids_json, target_ids_json,
    canonical_payload_digest, status, mutation_evidence_id, created_at, terminalized_at,
    error_code, before_revision_tokens_json, after_revision_tokens_json
    FROM ingress_action_receipts"""
