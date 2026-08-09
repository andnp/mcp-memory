"""Atomic SQLite execution for one create or append ingress mutation."""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Protocol
from unicodedata import normalize
from uuid import uuid4

from mcp_memory.core.curation_identity import record_snapshot, record_token
from mcp_memory.core.ingress_evidence import (
    IngressActionReceipt,
    IngressReceiptStatus,
    SourceCoverage,
    SourceCoverageOutcome,
)
from mcp_memory.core.ingress_identity import action_identity, canonical_payload_digest
from mcp_memory.core.ports.ingress import (
    IngressActionReceiptIdentityConflictError,
    SourceCoverageAssignmentConflictError,
)
from mcp_memory.core.ports.memory import MemoryRecord
from mcp_memory.utils.db import DatabaseManager


class IngressMutationInjectedFailure(RuntimeError):
    """Failure raised by a test-only transaction stage hook."""


class IngressMutationResult:
    """Normalized result returned by an ingress domain callback."""

    def __init__(self, operation: str, affected_ids: Sequence[str] = ()) -> None:
        self.operation = operation
        self.affected_ids = tuple(str(value) for value in affected_ids)


class IngressDomainTransaction(Protocol):
    """Connection-scoped create/append domain surface."""

    touched_ids: list[str]

    def get_memory(self, memory_id: str) -> MemoryRecord | None: ...

    def create_memory(self, **values: object) -> MemoryRecord: ...

    def append_memory(
        self,
        memory_id: str,
        content: str,
        *,
        summary: str | None | object = None,
        tags: Sequence[str] | None = None,
        workspace_ids: Sequence[str] | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> MemoryRecord: ...


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
    ) -> IngressActionReceipt:
        normalized_batch = _identifier(batch_id, "batch_id")
        normalized_operation = _identifier(operation, "operation")
        if normalized_operation not in {"create", "append"}:
            raise ValueError("ingress operation must be create or append")
        identity = action_identity(normalized_operation, entry_ids, target_ids)
        digest = canonical_payload_digest(payload)
        normalized_coverage = _coverage_for_action(
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
                _check_digest(existing, digest)
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
                created_at=_now_text(),
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
                terminalized_at=_now_text(),
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
                    _json_text(receipt.before_revision_tokens),
                    _json_text(receipt.after_revision_tokens),
                    receipt.terminalized_at,
                    receipt.error_code,
                    receipt.action_id,
                ),
            )
            for coverage in normalized_coverage:
                _assign_coverage(connection, coverage)
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
            (event_id, operation, action_id, _now_text()),
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
                    None if previous is None else _json_text(record_snapshot(previous)),
                    int(current is not None),
                    None if current is None else _json_text(record_snapshot(current)),
                    None if previous is None else record_token(previous),
                    None if current is None else record_token(current),
                ),
            )

    def _fail_stage(self, stage: str) -> None:
        if self._fault_injector is not None:
            self._fault_injector(stage)
        normalized = None if self._fault_stage is None else self._fault_stage.replace("-", "_")
        aliases = {
            "action_reserved": {"action_reserved", "reserve"},
            "after_domain_mutation": {"domain", "after_domain_mutation"},
            "repair_intent": {"repair", "repair_intent"},
            "history": {"history"},
            "before_receipt_coverage_commit": {
                "receipt", "coverage", "before_receipt_coverage_commit"
            },
        }
        if normalized in aliases.get(stage, {stage}):
            raise IngressMutationInjectedFailure(f"injected ingress transaction failure at {stage}")


class _SQLiteIngressDomainTransaction:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection
        self.touched_ids: list[str] = []

    def get_memory(self, memory_id: str) -> MemoryRecord | None:
        normalized_id = _identifier(memory_id, "memory_id")
        row = self._connection.execute("SELECT * FROM memories WHERE id = ?", (normalized_id,)).fetchone()
        return None if row is None else _hydrate_record(self._connection, row)

    def create_memory(self, **values: object) -> MemoryRecord:
        memory_id = _identifier(str(values.get("memory_id", uuid4())), "memory_id")
        title = _required_text(values.get("title"), "title")
        content = _required_text(values.get("content"), "content")
        memory_type = str(values.get("memory_type", values.get("type", "observation")))
        status = str(values.get("status", "active"))
        workspace_ids = _values(values.get("workspace_ids", ()))
        if not workspace_ids:
            raise ValueError("workspace_ids must contain at least one non-empty value")
        tags = _values(values.get("tags", ()), normalize_tags=True)
        created_at = str(values.get("created_at", _now_text()))
        summary = str(values.get("summary") or _build_summary(title, content, memory_type))
        metadata = values.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("metadata must be a mapping")
        next_ref = self._connection.execute("SELECT COALESCE(MAX(memory_ref), 0) + 1 FROM memories").fetchone()[0]
        self._connection.execute(
            """
            INSERT INTO memories (
                id, memory_ref, title, content, summary, type, status, created_at, updated_at,
                archived_at, read_count, access_score, last_accessed_at, last_surfaced_at, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, 0, 0.0, NULL, NULL, ?)
            """,
            (memory_id, int(next_ref), title, content, summary, memory_type, status, created_at, created_at, _json_text(dict(metadata))),
        )
        self._replace_mappings(memory_id, workspace_ids, tags)
        self._replace_fts(memory_id, title, summary, content, tags)
        self.touched_ids.append(memory_id)
        return self.get_memory(memory_id)  # type: ignore[return-value]

    def append_memory(
        self,
        memory_id: str,
        content: str,
        *,
        summary: str | None | object = None,
        tags: Sequence[str] | None = None,
        workspace_ids: Sequence[str] | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> MemoryRecord:
        normalized_id = _identifier(memory_id, "memory_id")
        existing = self.get_memory(normalized_id)
        if existing is None:
            raise ValueError(f"memory {normalized_id!r} was not found")
        addition = _required_text(content, "content")
        merged_content = existing.content if addition in existing.content else f"{existing.content.rstrip()}\n\n{addition}"
        merged_tags = _values(tags if tags is not None else existing.tags, normalize_tags=True)
        merged_workspaces = _values(workspace_ids if workspace_ids is not None else existing.workspace_ids)
        merged_metadata = dict(existing.metadata)
        if metadata is not None:
            merged_metadata.update(metadata)
        resolved_summary = existing.summary if summary is None else str(summary)
        updated_at = _now_text()
        self._connection.execute(
            "UPDATE memories SET content = ?, summary = ?, metadata = ?, updated_at = ? WHERE id = ?",
            (merged_content.strip(), resolved_summary, _json_text(merged_metadata), updated_at, normalized_id),
        )
        self._replace_mappings(normalized_id, merged_workspaces, merged_tags)
        self._replace_fts(normalized_id, existing.title, resolved_summary or "", merged_content.strip(), merged_tags)
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


def _coverage_for_action(
    coverage: Sequence[SourceCoverage] | None,
    *,
    entry_ids: Sequence[str],
    action_id: str,
    operation: str,
) -> tuple[SourceCoverage, ...]:
    if coverage is None:
        outcome = SourceCoverageOutcome.CREATED if operation == "create" else SourceCoverageOutcome.APPENDED
        return tuple(SourceCoverage(entry_id, outcome, action_id) for entry_id in entry_ids)
    normalized = tuple(coverage)
    if _canonical_ids([item.entry_id for item in normalized]) != _canonical_ids(entry_ids):
        raise ValueError("source coverage must contain exactly the action entry IDs")
    result: list[SourceCoverage] = []
    for item in normalized:
        if item.action_id not in {None, action_id}:
            raise ValueError("source coverage action ID does not match the mutation")
        if item.outcome is SourceCoverageOutcome.UNOBSERVED:
            raise ValueError("create and append require terminal handled source coverage")
        result.append(SourceCoverage(item.entry_id, item.outcome, action_id, item.reason))
    return tuple(sorted(result, key=lambda value: value.entry_id.encode("utf-8")))


def _assign_coverage(connection: sqlite3.Connection, coverage: SourceCoverage) -> None:
    connection.execute(
        "INSERT OR IGNORE INTO ingress_source_coverage (entry_id, outcome, action_id, reason) VALUES (?, ?, ?, ?)",
        (coverage.entry_id, coverage.outcome.value, coverage.action_id, coverage.reason),
    )
    row = connection.execute("SELECT outcome, action_id FROM ingress_source_coverage WHERE entry_id = ?", (coverage.entry_id,)).fetchone()
    assert row is not None
    stored = (str(row[1]) if row[1] is not None else None, SourceCoverageOutcome(str(row[0])))
    requested = (coverage.action_id, coverage.outcome)
    if stored != requested:
        raise SourceCoverageAssignmentConflictError(coverage.entry_id, stored[0], stored[1], requested[0], requested[1])


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


def _check_digest(receipt: IngressActionReceipt, requested_digest: str) -> None:
    if receipt.canonical_payload_digest != requested_digest:
        raise IngressActionReceiptIdentityConflictError(
            receipt.action_id, receipt.canonical_payload_digest, requested_digest
        )


def _receipt_values(receipt: IngressActionReceipt) -> tuple[object, ...]:
    return (
        receipt.action_id, receipt.batch_id, receipt.operation, _json_text(receipt.entry_ids),
        _json_text(receipt.target_ids), receipt.canonical_payload_digest, receipt.status.value,
        receipt.mutation_evidence_id, _json_text(receipt.before_revision_tokens),
        _json_text(receipt.after_revision_tokens), receipt.created_at, receipt.terminalized_at,
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
    normalized = {_identifier(value, "identity") for value in values}
    return tuple(sorted(normalized, key=lambda value: value.encode("utf-8")))


def _identifier(value: object, name: str) -> str:
    normalized = normalize("NFC", str(value)).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty")
    return normalized


def _required_text(value: object, name: str) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty")
    return text


def _values(value: object, *, normalize_tags: bool = False) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("collection value expected")
    result = set()
    for item in value:
        text = str(item).strip()
        if normalize_tags:
            text = text.lower().replace("_", "-")
        if text:
            result.add(text)
    return sorted(result, key=lambda item: item.encode("utf-8"))


def _json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _now_text() -> str:
    return datetime.now(UTC).isoformat()


def _build_summary(title: str, content: str, memory_type: str) -> str:
    from mcp_memory.core.summaries import build_deterministic_summary

    return build_deterministic_summary(title=title, content=content, memory_type=memory_type)
