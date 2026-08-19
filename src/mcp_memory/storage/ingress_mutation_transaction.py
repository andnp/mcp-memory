"""Atomic SQLite execution for one create or append ingress mutation."""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Callable, Mapping, Sequence
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
from mcp_memory.utils.db import DatabaseManager

__all__ = [
    "IngressDomainTransaction",
    "IngressMutationInjectedFailure",
    "IngressMutationResult",
    "SQLiteIngressMutationStore",
]


class SQLiteIngressMutationStore:
    """Apply one ingress create or append with one SQLite commit.

    The callback must use only the supplied transaction surface.  The public
    ingress repositories intentionally are not used here because they commit
    independently and would break the atomicity boundary.
    """

    def __init__(
        self,
        db_manager: DatabaseManager,
        *,
        embedding_model: str = "default",
        fault_stage: str | None = None,
        fault_injector: Callable[[str], None] | None = None,
    ) -> None:
        if not embedding_model.strip():
            raise ValueError("embedding_model must be non-empty")
        self._db = db_manager
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

        connection = self._db.open_connection()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = _receipt_for_action(connection, identity.action_id)
            if existing is not None:
                check_digest(existing, digest)
                connection.rollback()
                return existing

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
            connection.execute(
                """
                INSERT INTO ingress_action_receipts (
                    action_id, batch_id, operation, entry_ids_json, target_ids_json,
                    canonical_payload_digest, status, mutation_evidence_id,
                    before_revision_tokens_json, after_revision_tokens_json,
                    created_at, terminalized_at, error_code
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                _receipt_values(provisional),
            )
            self._fail_stage("action_reserved")

            before = _read_records(connection, identity.target_ids)
            domain = _SQLiteIngressDomainTransaction(connection)
            result = apply(domain)
            if not isinstance(result, IngressMutationResult):
                raise TypeError("ingress callback must return IngressMutationResult")
            if result.operation.strip() != normalized_operation:
                raise ValueError("ingress operation does not match mutation result")
            self._fail_stage("after_domain_mutation")

            affected_ids = _canonical_ids([*identity.target_ids, *domain.touched_ids, *result.affected_ids])
            after = _read_records(connection, affected_ids)
            event_id = uuid4()
            self._write_repair_intents(connection, before, after)
            self._fail_stage("repair_intent")
            self._write_history(
                connection,
                event_id=str(event_id),
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
                mutation_evidence_id=mutation_evidence_id or str(event_id),
                created_at=provisional.created_at,
                terminalized_at=now_text(),
                before_revision_tokens={key: record_token(value) for key, value in before.items()},
                after_revision_tokens={key: record_token(value) for key, value in after.items()},
            )
            self._fail_stage("before_receipt_coverage_commit")
            connection.execute(
                """
                UPDATE ingress_action_receipts SET
                    status = ?, mutation_evidence_id = ?, before_revision_tokens_json = ?,
                    after_revision_tokens_json = ?, terminalized_at = ?, error_code = ?
                WHERE action_id = ?
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
                _assign_coverage(connection, coverage)
            if normalized_journal_task is not None:
                _move_claimed_journal_entries(
                    connection,
                    entry_journal_ids,
                    task_id=normalized_journal_task,
                )
                self._fail_stage("journal_reconcile")
            connection.commit()
            return receipt
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _write_repair_intents(
        self,
        connection: sqlite3.Connection,
        before: Mapping[str, MemoryRecord],
        after: Mapping[str, MemoryRecord],
    ) -> None:
        now = time.time()
        for memory_id, record in after.items():
            previous = before.get(memory_id)
            if previous is not None and record_token(previous) == record_token(record):
                continue
            connection.execute(
                """
                INSERT OR IGNORE INTO embedding_repair_queue (
                    id, memory_id, workspace_id, model_name, memory_updated_at,
                    status, attempt_count, available_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'pending', 0, ?, ?, ?)
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
        connection: sqlite3.Connection,
        *,
        event_id: str,
        action_id: str,
        operation: str,
        before: Mapping[str, MemoryRecord],
        after: Mapping[str, MemoryRecord],
    ) -> None:
        connection.execute(
            """
            INSERT INTO memory_mutation_events (
                id, operation, actor_kind, family, action_id, schema_version, status, created_at
            ) VALUES (?, ?, 'system', 'ingress', ?, 1, 'applied', ?)
            """,
            (event_id, operation, action_id, now_text()),
        )
        for memory_id in _canonical_ids([*before, *after]):
            previous = before.get(memory_id)
            current = after.get(memory_id)
            if previous is not None and current is not None and record_token(previous) == record_token(current):
                continue
            connection.execute(
                """
                INSERT INTO memory_record_revisions (
                    event_id, memory_id, role, before_exists, before_snapshot,
                    after_exists, after_snapshot, before_token, after_token
                ) VALUES (?, ?, 'target', ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    memory_id,
                    int(previous is not None),
                    None if previous is None else json_text(record_snapshot(previous)),
                    int(current is not None),
                    None if current is None else json_text(record_snapshot(current)),
                    None if previous is None else record_token(previous),
                    None if current is None else record_token(current),
                ),
            )

    def _fail_stage(self, stage: str) -> None:
        fail_stage(stage, fault_stage=self._fault_stage, fault_injector=self._fault_injector)


class _SQLiteIngressDomainTransaction:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection
        self.touched_ids: list[str] = []

    def get_memory(self, memory_id: str) -> MemoryRecord | None:
        normalized_id = identifier(memory_id, "memory_id")
        row = self._connection.execute("SELECT * FROM memories WHERE id = ?", (normalized_id,)).fetchone()
        return None if row is None else _hydrate_record(self._connection, row)

    def create_memory(self, **values: object) -> MemoryRecord:
        prepared = prepare_create(values)
        next_ref = self._connection.execute("SELECT COALESCE(MAX(memory_ref), 0) + 1 FROM memories").fetchone()[0]
        self._connection.execute(
            """
            INSERT INTO memories (
                id, memory_ref, title, content, summary, type, status, created_at, updated_at,
                archived_at, read_count, access_score, last_accessed_at, last_surfaced_at, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, 0, 0.0, NULL, NULL, ?)
            """,
            (
                prepared.memory_id,
                int(next_ref),
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
        self._replace_fts(prepared.memory_id, prepared.title, prepared.summary, prepared.content, prepared.tags)
        self.touched_ids.append(prepared.memory_id)
        return self.get_memory(prepared.memory_id)  # type: ignore[return-value]

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
        self._connection.execute(
            "UPDATE memories SET content = ?, summary = ?, metadata = ?, updated_at = ? WHERE id = ?",
            (prepared.content, prepared.summary, json_text(prepared.metadata), updated_at, normalized_id),
        )
        self._replace_mappings(normalized_id, prepared.workspace_ids, prepared.tags)
        self._replace_fts(normalized_id, existing.title, prepared.summary or "", prepared.content, prepared.tags)
        self.touched_ids.append(normalized_id)
        return self.get_memory(normalized_id)  # type: ignore[return-value]

    def _replace_mappings(self, memory_id: str, workspace_ids: list[str], tags: list[str]) -> None:
        self._connection.execute("DELETE FROM memory_workspaces WHERE memory_id = ?", (memory_id,))
        self._connection.executemany(
            "INSERT INTO memory_workspaces (memory_id, workspace_id) VALUES (?, ?)",
            [(memory_id, workspace_id) for workspace_id in workspace_ids],
        )
        self._connection.execute("DELETE FROM memory_tags WHERE memory_id = ?", (memory_id,))
        for tag in tags:
            self._connection.execute("INSERT OR IGNORE INTO tags (name) VALUES (?)", (tag,))
            tag_id = self._connection.execute("SELECT id FROM tags WHERE name = ?", (tag,)).fetchone()[0]
            self._connection.execute("INSERT INTO memory_tags (memory_id, tag_id) VALUES (?, ?)", (memory_id, tag_id))

    def _replace_fts(self, memory_id: str, title: str, summary: str, content: str, tags: list[str]) -> None:
        self._connection.execute("DELETE FROM memories_fts WHERE memory_id = ?", (memory_id,))
        self._connection.execute(
            "INSERT INTO memories_fts(memory_id, title, summary, content, tags) VALUES (?, ?, ?, ?, ?)",
            (memory_id, title, summary, content, " ".join(tags)),
        )


def _assign_coverage(connection: sqlite3.Connection, coverage: SourceCoverage) -> None:
    connection.execute(
        "INSERT OR IGNORE INTO ingress_source_coverage (entry_id, outcome, action_id, reason) VALUES (?, ?, ?, ?)",
        (coverage.entry_id, coverage.outcome.value, coverage.action_id, coverage.reason),
    )
    row = connection.execute("SELECT outcome, action_id FROM ingress_source_coverage WHERE entry_id = ?", (coverage.entry_id,)).fetchone()
    assert row is not None
    from mcp_memory.core.ingress_evidence import SourceCoverageOutcome

    stored = (str(row[1]) if row[1] is not None else None, SourceCoverageOutcome(str(row[0])))
    requested = (coverage.action_id, coverage.outcome)
    if stored != requested:
        raise SourceCoverageAssignmentConflictError(coverage.entry_id, stored[0], stored[1], requested[0], requested[1])


def _move_claimed_journal_entries(
    connection: sqlite3.Connection,
    entry_ids: Sequence[int],
    *,
    task_id: str,
) -> None:
    if not entry_ids:
        return
    placeholders = ",".join("?" for _ in entry_ids)
    connection.execute(
        f"""
        UPDATE system1_journal
        SET status = 'recoverable', claim_task_id = NULL, claimed_at = NULL,
            recoverable_until = ?
        WHERE id IN ({placeholders}) AND status = 'claimed' AND claim_task_id = ?
        """,
        [time.time() + RECOVERABLE_RETENTION_SECONDS, *entry_ids, task_id],
    )


def _receipt_for_action(connection: sqlite3.Connection, action_id: str) -> IngressActionReceipt | None:
    row = connection.execute("SELECT * FROM ingress_action_receipts WHERE action_id = ?", (action_id,)).fetchone()
    if row is None:
        return None
    return IngressActionReceipt(
        action_id=row["action_id"], batch_id=row["batch_id"], operation=row["operation"],
        entry_ids=tuple(json.loads(row["entry_ids_json"])), target_ids=tuple(json.loads(row["target_ids_json"])),
        canonical_payload_digest=row["canonical_payload_digest"], status=IngressReceiptStatus(row["status"]),
        mutation_evidence_id=row["mutation_evidence_id"], created_at=row["created_at"],
        terminalized_at=row["terminalized_at"], error_code=row["error_code"],
        before_revision_tokens=dict(json.loads(row["before_revision_tokens_json"])),
        after_revision_tokens=dict(json.loads(row["after_revision_tokens_json"])),
    )


def _receipt_values(receipt: IngressActionReceipt) -> tuple[object, ...]:
    return (
        receipt.action_id, receipt.batch_id, receipt.operation, json_text(receipt.entry_ids),
        json_text(receipt.target_ids), receipt.canonical_payload_digest, receipt.status.value,
        receipt.mutation_evidence_id, json_text(receipt.before_revision_tokens),
        json_text(receipt.after_revision_tokens), receipt.created_at, receipt.terminalized_at,
        receipt.error_code,
    )


def _read_records(connection: sqlite3.Connection, memory_ids: Sequence[str]) -> dict[str, MemoryRecord]:
    return {
        memory_id: record
        for memory_id in memory_ids
        if (record := _record_by_id(connection, memory_id)) is not None
    }


def _record_by_id(connection: sqlite3.Connection, memory_id: str) -> MemoryRecord | None:
    row = connection.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
    return None if row is None else _hydrate_record(connection, row)


def _hydrate_record(connection: sqlite3.Connection, row: sqlite3.Row) -> MemoryRecord:
    workspaces = connection.execute("SELECT workspace_id FROM memory_workspaces WHERE memory_id = ? ORDER BY workspace_id", (row["id"],)).fetchall()
    tags = connection.execute(
        "SELECT tags.name FROM tags JOIN memory_tags ON memory_tags.tag_id = tags.id WHERE memory_tags.memory_id = ? ORDER BY tags.name",
        (row["id"],),
    ).fetchall()
    return MemoryRecord(
        id=str(row["id"]), title=str(row["title"]), content=str(row["content"]), summary=row["summary"],
        type=str(row["type"]), status=str(row["status"]), created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]), read_count=int(row["read_count"] or 0),
        access_score=float(row["access_score"] or 0), last_accessed_at=row["last_accessed_at"],
        last_surfaced_at=row["last_surfaced_at"], metadata=json.loads(row["metadata"] or "{}"),
        workspace_ids=[str(item[0]) for item in workspaces], tags=[str(item[0]) for item in tags],
        memory_ref=row["memory_ref"], archived_at=row["archived_at"],
    )


def _canonical_ids(values: Sequence[str]) -> tuple[str, ...]:
    normalized = {identifier(value, "identity") for value in values}
    return tuple(sorted(normalized, key=lambda value: value.encode("utf-8")))
