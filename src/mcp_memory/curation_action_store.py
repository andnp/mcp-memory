"""Transaction-scoped SQLite execution for one curation action.

This module is intentionally an execution primitive, not a planner or a
production curation entry point.  ``execute_action`` owns one SQLite
transaction and exposes only ``CurationTransaction`` to its callback.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol, cast
from uuid import UUID, uuid4

from mcp_memory.core.curation_identity import (
    action_intent_token,
    canonical_token,
    graph_token,
    link_token,
    record_snapshot,
    record_token,
)
from mcp_memory.core.curation_models import CurationVerificationDescriptor
from mcp_memory.curation_store import (
    CurationActionReceipt,
    CurationReceiptState,
    CurationRunState,
    SQLiteCurationStore,
)
from mcp_memory.mutation_history import (
    MutationActorKind,
    MutationEventStatus,
    ProtectionMode,
    RevisionRole,
    is_protection_active,
)
from mcp_memory.core.ports.memory import MemoryLink, MemoryRecord

from mcp_memory.utils.db import DatabaseManager


logger = logging.getLogger(__name__)


class CurationActionError(RuntimeError):
    """Base class for action-transaction failures."""


class CurationActionStaleError(CurationActionError):
    """The action preconditions no longer describe the authoritative state."""


class CurationActionTransientError(CurationActionError):
    """The backend failed in a way that is safe to retry with the same action."""


class CurationActionFatalError(CurationActionError):
    """The action or transaction callback violated a non-retryable contract."""


class CurationActionInjectedFailure(CurationActionFatalError):
    """Failure raised by the test-only transaction stage hook."""


@dataclass(frozen=True)
class MutationResult:
    """The compact normalized result returned by a transaction callback."""

    operation: str
    affected_ids: tuple[str, ...] = ()
    verification_descriptor: CurationVerificationDescriptor | None = None

    def __init__(
        self,
        operation: str,
        affected_ids: Sequence[str | UUID] = (),
        verification_descriptor: CurationVerificationDescriptor | None = None,
    ) -> None:
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "affected_ids", tuple(str(value) for value in affected_ids))
        object.__setattr__(self, "verification_descriptor", verification_descriptor)


class CurationTransaction(Protocol):
    """The callback-only, transaction-scoped domain mutation surface."""

    def get_memory(self, memory_id: str | UUID) -> MemoryRecord | None: ...

    def get_links(
        self,
        memory_id: str | UUID,
        *,
        direction: str = "outgoing",
        link_type: str | None = None,
    ) -> list[MemoryLink]: ...

    def get_protections(self, memory_id: str | UUID) -> list[object]: ...

    def update_memory(self, memory_id: str | UUID, **changes: object) -> MemoryRecord: ...

    def create_memory(self, **values: object) -> MemoryRecord: ...

    def delete_memory(self, memory_id: str | UUID) -> MemoryRecord: ...

    def add_link(
        self,
        source_id: str | UUID,
        target_id: str | UUID,
        link_type: str,
        context: str = "",
    ) -> MemoryLink: ...

    def remove_link(self, source_id: str | UUID, target_id: str | UUID, link_type: str) -> bool: ...

    def set_lineage(self, memory_id: str | UUID, lineage: Mapping[str, object]) -> MemoryRecord: ...


class CurationActionStore(Protocol):
    def execute_action(
        self,
        *,
        run_id: UUID,
        action_id: UUID,
        target_ids: Sequence[str],
        expected_tokens: Mapping[str, str],
        apply: Callable[[CurationTransaction], MutationResult],
        preconditions: Any | None = None,
        operation: str | None = None,
        payload: Any | None = None,
    ) -> CurationActionReceipt: ...


@dataclass
class _MutationResultData:
    operation: str
    affected_ids: list[str] = field(default_factory=list)
    verification_descriptor: CurationVerificationDescriptor | None = None


_SUMMARY_UNSET = object()
_VALID_MEMORY_TYPES = frozenset({"journal", "plan", "fact", "observation", "reflection"})
_VALID_MEMORY_STATUSES = frozenset({"active", "stale", "degraded", "archived"})
_DESTRUCTIVE_OPERATIONS = frozenset({"archive_memory", "delete_memory", "merge_memories", "split_memory"})
class SQLiteCurationActionStore:
    """Apply one action atomically using a dedicated SQLite connection.

    ``fault_stage`` and ``fault_injector`` exist to make rollback boundaries
    directly testable.  They are intentionally inert unless supplied by a
    caller and are not used by runtime wiring.
    """

    def __init__(
        self,
        db_manager: DatabaseManager,
        *,
        embedding_model: str = "default",
        model_name: str | None = None,
        read_cache: Any | None = None,
        fault_stage: str | None = None,
        fault_injector: Callable[[str], None] | None = None,
    ) -> None:
        if not embedding_model.strip() and model_name is None:
            raise ValueError("embedding_model must be non-empty")
        self._db = db_manager
        self._embedding_model = (model_name or embedding_model).strip()
        if not self._embedding_model:
            raise ValueError("embedding_model must be non-empty")
        self._fault_stage = fault_stage
        self._fault_injector = fault_injector
        self._read_cache = read_cache

    def execute_action(
        self,
        *,
        run_id: UUID,
        action_id: UUID,
        target_ids: Sequence[str],
        expected_tokens: Mapping[str, str],
        apply: Callable[[CurationTransaction], MutationResult],
        preconditions: Any | None = None,
        operation: str | None = None,
        payload: Any | None = None,
        verification_descriptor: CurationVerificationDescriptor | None = None,
        actor_kind: MutationActorKind | str = MutationActorKind.MAINTENANCE,
        restores_event_id: UUID | None = None,
        idempotency_key: str | None = None,
    ) -> CurationActionReceipt:
        normalized_targets = _canonical_target_ids(target_ids)
        if not normalized_targets:
            raise CurationActionFatalError("at least one target ID is required")
        normalized_tokens = {_canonical_id(key): str(value) for key, value in expected_tokens.items()}
        if any(not value for value in normalized_tokens.values()):
            raise CurationActionFatalError("revision tokens must be non-empty")
        if operation is None:
            raise CurationActionFatalError("action operation is required")
        normalized_operation = operation.strip()
        if not normalized_operation:
            raise CurationActionFatalError("action operation must be non-empty")

        # The fast path avoids invoking the callback on every replay.  It is
        # repeated inside BEGIN IMMEDIATE to close the duplicate-execution race.
        existing = SQLiteCurationStore(self._db).get_receipt(run_id, action_id)
        if existing is not None:
            _check_replay_identity(
                existing,
                run_id=run_id,
                action_id=action_id,
                operation=normalized_operation,
                target_ids=normalized_targets,
                expected_tokens=normalized_tokens,
                preconditions=preconditions,
                payload=payload,
                verification_descriptor=verification_descriptor,
            )
            return existing

        connection = self._db.open_connection()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = self._receipt_on(connection, run_id, action_id)
            if existing is not None:
                _check_replay_identity(
                    existing,
                    run_id=run_id,
                    action_id=action_id,
                    operation=normalized_operation,
                    target_ids=normalized_targets,
                    expected_tokens=normalized_tokens,
                    preconditions=preconditions,
                    payload=payload,
                    verification_descriptor=verification_descriptor,
                )
                connection.rollback()
                return existing

            run = connection.execute(
                "SELECT state, plan_id, policy_version FROM curation_runs WHERE run_id = ?",
                (str(run_id),),
            ).fetchone()
            if run is None and restores_event_id is not None:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO curation_runs (
                        run_id, frontier_key, context_fingerprint, policy_version,
                        schema_version, state, budget_usage_json, created_at
                    ) VALUES (?, ?, ?, '1', 1, ?, '{}', ?)
                    """,
                    (
                        str(run_id),
                        f"restore:{restores_event_id}",
                        idempotency_key or str(action_id),
                        str(CurationRunState.EXECUTING),
                        _now_text(),
                    ),
                )
                run = connection.execute(
                    "SELECT state, plan_id, policy_version FROM curation_runs WHERE run_id = ?",
                    (str(run_id),),
                ).fetchone()
            if run is None and restores_event_id is None:
                raise CurationActionFatalError(f"curation run {run_id} was not found")
            if run is not None and str(run[0]) == str(CurationRunState.TERMINAL) and restores_event_id is None:
                raise CurationActionFatalError(f"curation run {run_id} is terminal")

            self._lock_targets(connection, normalized_targets)
            before_records = self._read_records(connection, normalized_targets)
            self._check_tokens(connection, before_records, normalized_targets, normalized_tokens)
            self._check_preconditions(connection, normalized_targets, preconditions)
            self._check_protections(connection, normalized_targets, None)
            before_links = self._read_related_links(connection, set(normalized_targets))

            transaction = _SQLiteCurationTransaction(
                connection,
                target_ids=set(normalized_targets),
                stage_hook=self._fail_stage,
            )
            raw_result = apply(transaction)
            result = _normalize_result(raw_result)
            if not result.operation.strip():
                raise CurationActionFatalError("mutation result operation must be non-empty")
            result_operation = result.operation.strip()
            if normalized_operation is not None and result_operation != normalized_operation:
                raise CurationActionFatalError("action operation does not match mutation result")
            self._check_protections(connection, normalized_targets, result.operation)
            self._fail_stage("after_domain_mutation")

            after_ids = _canonical_target_ids([*normalized_targets, *transaction.touched_ids])
            after_records = self._read_records(connection, after_ids)
            after_links = self._read_related_links(connection, set(after_ids))
            event_id = uuid4()
            before_token = _state_token(before_records, normalized_targets)
            after_token = _state_token(after_records, after_ids)

            self._write_repair_intents(connection, before_records, after_records)
            self._fail_stage("repair_intent")
            self._write_history(
                connection,
                event_id=event_id,
                run_id=run_id,
                action_id=action_id,
                plan_id=None if run is None or run[1] is None else UUID(str(run[1])),
                policy_version=None if run is None or run[2] is None else str(run[2]),
                operation=result.operation.strip(),
                actor_kind=actor_kind,
                restores_event_id=restores_event_id,
                idempotency_key=idempotency_key,
                before_records=before_records,
                after_records=after_records,
                before_links=before_links,
                after_links=after_links,
            )
            self._fail_stage("history")

            affected_ids = _canonical_target_ids(result.affected_ids or transaction.touched_ids or normalized_targets)
            receipt = CurationActionReceipt(
                run_id=run_id,
                action_id=action_id,
                operation=result_operation,
                affected_ids=[UUID(value) for value in affected_ids],
                status=CurationReceiptState.APPLIED_UNVERIFIED,
                before_token=before_token,
                after_token=after_token,
                mutation_event_id=event_id,
                intent_hash=_action_intent_hash(
                    operation=result_operation,
                    target_ids=normalized_targets,
                    expected_tokens=normalized_tokens,
                    preconditions=preconditions,
                    payload=payload,
                ),
                verification_descriptor=result.verification_descriptor or verification_descriptor,
                applied_at=datetime.now(UTC),
            )
            self._insert_receipt(connection, receipt)
            self._fail_stage("receipt_preparation")
            connection.commit()
            self._invalidate_derivative_caches(after_ids)
            return receipt
        except CurationActionError:
            connection.rollback()
            raise
        except sqlite3.OperationalError as exc:
            connection.rollback()
            if _is_retryable_sqlite_error(exc):
                raise CurationActionTransientError(
                    f"SQLite action transaction was busy for {run_id}/{action_id}"
                ) from exc
            raise CurationActionFatalError(
                f"SQLite action transaction failed for {run_id}/{action_id}"
            ) from exc
        except Exception as exc:
            connection.rollback()
            raise CurationActionFatalError(
                f"curation action failed for {run_id}/{action_id}"
            ) from exc
        finally:
            connection.close()

    def _invalidate_derivative_caches(self, memory_ids: Sequence[str]) -> None:
        if self._read_cache is None:
            return
        try:
            self._read_cache.invalidate_for_mutation(list(memory_ids))
        except Exception:
            # The cache is derivative; a cache failure must not turn a
            # committed authoritative mutation into a retryable action.
            logger.warning("Unable to invalidate shared read cache after curation mutation", exc_info=True)

    def _lock_targets(self, connection: sqlite3.Connection, target_ids: Sequence[str]) -> None:
        for memory_id in target_ids:
            row = connection.execute("SELECT id FROM memories WHERE id = ?", (memory_id,)).fetchone()
            if row is None:
                raise CurationActionStaleError(f"target memory {memory_id!r} is missing")

    def _read_records(self, connection: sqlite3.Connection, memory_ids: Sequence[str]) -> dict[str, MemoryRecord]:
        records: dict[str, MemoryRecord] = {}
        for memory_id in memory_ids:
            row = connection.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
            if row is not None:
                records[memory_id] = _hydrate_record(connection, row)
        return records

    def _check_tokens(
        self,
        connection: sqlite3.Connection,
        records: Mapping[str, MemoryRecord],
        target_ids: Sequence[str],
        expected_tokens: Mapping[str, str],
    ) -> None:
        for memory_id in target_ids:
            expected = expected_tokens.get(memory_id) or expected_tokens.get(f"record:{memory_id}")
            if expected is not None:
                record = records.get(memory_id)
                if record is None or record_token(record) != expected:
                    raise CurationActionStaleError(f"record revision token is stale for {memory_id!r}")
            graph_expected = expected_tokens.get(f"graph:{memory_id}")
            if graph_expected is not None:
                actual = graph_token(memory_id, [_link_mapping(link) for link in _links_for_memory(connection, memory_id)])
                if actual != graph_expected:
                    raise CurationActionStaleError(f"graph revision token is stale for {memory_id!r}")
        for key, expected in expected_tokens.items():
            parts = _link_token_key(key)
            if parts is None:
                continue
            source_id, target_id, link_type = parts
            row = connection.execute(
                "SELECT context FROM links WHERE source_id = ? AND target_id = ? AND type = ?",
                (source_id, target_id, _normalize_link_type(link_type)),
            ).fetchone()
            actual = link_token(
                source_id,
                target_id,
                link_type,
                None if row is None else str(row[0] or ""),
                exists=row is not None,
            )
            if actual != expected:
                raise CurationActionStaleError(f"link revision token is stale for {source_id}:{target_id}:{link_type}")

    def _check_preconditions(self, connection: sqlite3.Connection, target_ids: Sequence[str], preconditions: Any) -> None:
        if preconditions is None:
            return
        required_statuses = cast(Mapping[object, object], _field(preconditions, "required_statuses", {}))
        for memory_id, expected_status in required_statuses.items():
            normalized_id = _canonical_id(memory_id)
            if normalized_id not in target_ids:
                raise CurationActionFatalError("status precondition references an unlocked target")
            row = connection.execute("SELECT status FROM memories WHERE id = ?", (normalized_id,)).fetchone()
            if row is None or str(row[0]) != str(getattr(expected_status, "value", expected_status)):
                raise CurationActionStaleError(f"status precondition is stale for {normalized_id!r}")
        for name, should_exist in (("required_links", True), ("absent_links", False)):
            assertions = cast(Sequence[object], _field(preconditions, name, []))
            for assertion in assertions:
                source_id = _canonical_id(_field(assertion, "source_id"))
                target_id = _canonical_id(_field(assertion, "target_id"))
                if source_id not in target_ids or target_id not in target_ids:
                    raise CurationActionFatalError("link precondition references an unlocked target")
                link_type = _normalize_link_type(str(_field(assertion, "link_type")))
                context = _field(assertion, "context", None)
                if not should_exist and context is not None:
                    raise CurationActionFatalError(
                        "absent-link precondition must omit relationship context"
                    )
                row = connection.execute(
                    "SELECT context FROM links WHERE source_id = ? AND target_id = ? AND type = ?",
                    (source_id, target_id, link_type),
                ).fetchone()
                present = row is not None and (
                    not should_exist or context is None or str(row[0] or "") == str(context)
                )
                if present != should_exist:
                    raise CurationActionStaleError("link precondition is stale")

    def _check_protections(self, connection: sqlite3.Connection, target_ids: Sequence[str], operation: str | None) -> None:
        destructive = operation is not None and operation in _DESTRUCTIVE_OPERATIONS
        now = datetime.now(UTC)
        placeholders = ",".join("?" for _ in target_ids)
        rows = connection.execute(
            f"SELECT memory_id, mode, expires_at FROM memory_protections WHERE memory_id IN ({placeholders})",
            list(target_ids),
        ).fetchall()
        for row in rows:
            if not is_protection_active(row[2], now=now):
                continue
            mode = str(row[1])
            if mode == ProtectionMode.NO_AUTONOMOUS_MUTATION.value:
                raise CurationActionFatalError(f"autonomous mutation is protected for {row[0]!r}")
            if mode == ProtectionMode.MANUAL_REVIEW_REQUIRED.value and operation in {
                "normalize_memory",
                "create_link",
            }:
                raise CurationActionFatalError(f"manual review is required for {row[0]!r}")
            if destructive and mode in {
                ProtectionMode.NO_AUTONOMOUS_DESTRUCTIVE_CHANGE.value,
                ProtectionMode.PINNED_ACTIVE.value,
            }:
                raise CurationActionFatalError(f"destructive mutation is protected for {row[0]!r}")

    def _write_repair_intents(
        self,
        connection: sqlite3.Connection,
        before_records: Mapping[str, MemoryRecord],
        after_records: Mapping[str, MemoryRecord],
    ) -> None:
        now = time.time()
        for memory_id, after in after_records.items():
            before = before_records.get(memory_id)
            if before is not None and _semantic_token(before) == _semantic_token(after):
                continue
            connection.execute(
                """
                INSERT OR IGNORE INTO embedding_repair_queue (
                    id, memory_id, workspace_id, model_name, memory_updated_at,
                    status, attempt_count, available_at, created_at, updated_at,
                    claimed_at, completed_at, lease_owner, lease_expires_at, last_error
                ) VALUES (?, ?, ?, ?, ?, 'pending', 0, ?, ?, ?, NULL, NULL, NULL, NULL, NULL)
                """,
                (
                    str(uuid4()),
                    memory_id,
                    after.workspace_ids[0] if after.workspace_ids else None,
                    self._embedding_model,
                    after.updated_at,
                    now,
                    now,
                    now,
                ),
            )

    def _write_history(
        self,
        connection: sqlite3.Connection,
        *,
        event_id: UUID,
        run_id: UUID,
        action_id: UUID,
        plan_id: UUID | None,
        policy_version: str | None,
        operation: str,
        actor_kind: MutationActorKind | str,
        restores_event_id: UUID | None,
        idempotency_key: str | None,
        before_records: Mapping[str, MemoryRecord],
        after_records: Mapping[str, MemoryRecord],
        before_links: Mapping[tuple[str, str, str], MemoryLink],
        after_links: Mapping[tuple[str, str, str], MemoryLink],
    ) -> None:
        created_at = datetime.now(UTC).isoformat()
        connection.execute(
            """
            INSERT INTO memory_mutation_events (
                id, operation, actor_kind, family, curation_run_id, plan_id,
                action_id, policy_version, schema_version, status, restores_event_id,
                idempotency_key, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?)
            """,
            (
                str(event_id),
                operation,
                str(actor_kind),
                "curation",
                str(run_id),
                None if plan_id is None else str(plan_id),
                str(action_id),
                policy_version,
                str(MutationEventStatus.APPLIED),
                None if restores_event_id is None else str(restores_event_id),
                idempotency_key or f"{run_id}:{action_id}",
                created_at,
            ),
        )
        for memory_id in sorted(set(before_records) | set(after_records), key=lambda value: value.encode("utf-8")):
            before = before_records.get(memory_id)
            after = after_records.get(memory_id)
            if before is not None and after is not None and _semantic_token(before) == _semantic_token(after):
                continue
            connection.execute(
                """
                INSERT INTO memory_record_revisions (
                    event_id, memory_id, role, before_exists, before_snapshot,
                    after_exists, after_snapshot, before_token, after_token
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(event_id),
                    memory_id,
                    str(RevisionRole.TARGET),
                    int(before is not None),
                    None if before is None else json.dumps(_snapshot(before), sort_keys=True, ensure_ascii=False),
                    int(after is not None),
                    None if after is None else json.dumps(_snapshot(after), sort_keys=True, ensure_ascii=False),
                    None if before is None else _semantic_token(before),
                    None if after is None else _semantic_token(after),
                ),
            )
        for key in sorted(
            set(before_links) | set(after_links),
            key=lambda value: (value[0].encode("utf-8"), value[1].encode("utf-8"), value[2].encode("utf-8")),
        ):
            before = before_links.get(key)
            after = after_links.get(key)
            if before is not None and after is not None and before.context == after.context:
                continue
            source_id, target_id, link_type = key
            connection.execute(
                """
                INSERT INTO memory_link_revisions (
                    event_id, source_id, target_id, link_type, context,
                    before_exists, after_exists
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(event_id),
                    source_id,
                    target_id,
                    link_type,
                    None if after is None else after.context,
                    int(before is not None),
                    int(after is not None),
                ),
            )

    def _insert_receipt(self, connection: sqlite3.Connection, receipt: CurationActionReceipt) -> None:
        connection.execute(
            """
            INSERT INTO curation_action_receipts (
                run_id, action_id, operation, affected_ids_json, status,
                before_token, after_token, mutation_event_id, intent_hash,
                verification_descriptor_json, error_code, applied_at, verified_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(receipt.run_id),
                str(receipt.action_id),
                receipt.operation,
                json.dumps([str(value) for value in receipt.affected_ids]),
                str(receipt.status),
                receipt.before_token,
                receipt.after_token,
                str(receipt.mutation_event_id) if receipt.mutation_event_id is not None else None,
                receipt.intent_hash,
                None
                if receipt.verification_descriptor is None
                else json.dumps(
                    receipt.verification_descriptor.model_dump(mode="json"),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                receipt.error_code,
                receipt.applied_at.isoformat() if receipt.applied_at is not None else None,
                receipt.verified_at.isoformat() if receipt.verified_at is not None else None,
            ),
        )

    def _receipt_on(
        self,
        connection: sqlite3.Connection,
        run_id: UUID,
        action_id: UUID,
    ) -> CurationActionReceipt | None:
        row = connection.execute(
            "SELECT * FROM curation_action_receipts WHERE run_id = ? AND action_id = ?",
            (str(run_id), str(action_id)),
        ).fetchone()
        if row is None:
            return None
        return SQLiteCurationStore(self._db).get_receipt(run_id, action_id)

    def _fail_stage(self, stage: str) -> None:
        if self._fault_injector is not None:
            self._fault_injector(stage)
        aliases = {
            "domain_mutation": {"domain", "after_domain", "domain_mutation", "after_domain_mutation"},
            "projection_update": {"projection", "after_projection", "projection_update", "after_projection_update"},
            "repair_intent": {"repair", "after_repair", "repair_intent", "after_repair_intent"},
            "history": {"history", "after_history"},
            "receipt_preparation": {
                "receipt",
                "before_receipt",
                "receipt_preparation",
                "after_receipt_preparation",
            },
        }
        normalized_stage = None if self._fault_stage is None else self._fault_stage.replace("-", "_")
        if normalized_stage in aliases.get(stage, {stage}):
            raise CurationActionInjectedFailure(f"injected curation transaction failure at {stage}")

    def _read_related_links(
        self,
        connection: sqlite3.Connection,
        memory_ids: set[str],
    ) -> dict[tuple[str, str, str], MemoryLink]:
        if not memory_ids:
            return {}
        placeholders = ",".join("?" for _ in memory_ids)
        values = sorted(memory_ids, key=lambda value: value.encode("utf-8"))
        rows = connection.execute(
            f"SELECT source_id, target_id, type, context FROM links WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})",
            [*values, *values],
        ).fetchall()
        return {
            (str(row[0]), str(row[1]), str(row[2])): MemoryLink(
                source_id=str(row[0]), target_id=str(row[1]), link_type=str(row[2]), context=str(row[3] or "")
            )
            for row in rows
        }


class _SQLiteCurationTransaction:
    def __init__(
        self,
        connection: sqlite3.Connection,
        *,
        target_ids: set[str],
        stage_hook: Callable[[str], None],
    ) -> None:
        self._connection = connection
        self._target_ids = target_ids
        self._stage_hook = stage_hook
        self.touched_ids: list[str] = []
        self._new_ids: set[str] = set()

    def get_memory(self, memory_id: str | UUID) -> MemoryRecord | None:
        normalized_id = self._require_access(memory_id)
        row = self._connection.execute("SELECT * FROM memories WHERE id = ?", (normalized_id,)).fetchone()
        return None if row is None else _hydrate_record(self._connection, row)

    def get_links(
        self,
        memory_id: str | UUID,
        *,
        direction: str = "outgoing",
        link_type: str | None = None,
    ) -> list[MemoryLink]:
        normalized_id = self._require_access(memory_id)
        column = "target_id" if direction == "incoming" else "source_id"
        params: list[object] = [normalized_id]
        query = f"SELECT source_id, target_id, type, context FROM links WHERE {column} = ?"
        if link_type is not None:
            query += " AND type = ?"
            params.append(_normalize_link_type(link_type))
        query += " ORDER BY source_id ASC, target_id ASC, type ASC"
        return [
            MemoryLink(str(row[0]), str(row[1]), str(row[2]), str(row[3] or ""))
            for row in self._connection.execute(query, params).fetchall()
        ]

    def get_protections(self, memory_id: str | UUID) -> list[object]:
        normalized_id = self._require_access(memory_id)
        return list(
            self._connection.execute(
                "SELECT memory_id, mode, reason, actor_id, created_at, expires_at FROM memory_protections WHERE memory_id = ? ORDER BY mode",
                (normalized_id,),
            ).fetchall()
        )

    def update_memory(self, memory_id: str | UUID, **changes: object) -> MemoryRecord:
        normalized_id = self._require_access(memory_id)
        existing = self.get_memory(normalized_id)
        if existing is None:
            raise CurationActionStaleError(f"target memory {normalized_id!r} is missing")
        title = _text(changes.get("title"), existing.title)
        content = _text(changes.get("content"), existing.content)
        if not title or not content:
            raise ValueError("title and content must be non-empty")
        memory_type = str(changes.get("memory_type", changes.get("type", existing.type)))
        status = str(changes.get("status", existing.status))
        if memory_type not in _VALID_MEMORY_TYPES:
            raise ValueError(f"invalid memory_type: {memory_type!r}")
        if status not in _VALID_MEMORY_STATUSES:
            raise ValueError(f"invalid status: {status!r}")
        summary_value = changes.get("summary", _SUMMARY_UNSET)
        summary = (
            _build_summary(title, content, memory_type)
            if summary_value is _SUMMARY_UNSET and ("title" in changes or "content" in changes or "type" in changes or "memory_type" in changes)
            else None if summary_value is None else str(summary_value) if summary_value is not _SUMMARY_UNSET else existing.summary
        )
        workspace_ids = _values(changes.get("workspace_ids"), existing.workspace_ids)
        tags = _values(changes.get("tags"), existing.tags)
        metadata = changes.get("metadata", existing.metadata)
        if not isinstance(metadata, Mapping):
            raise ValueError("metadata must be a mapping")
        columns = ["title = ?", "content = ?", "summary = ?", "type = ?", "status = ?", "metadata = ?", "updated_at = ?"]
        values: list[object] = [title, content, summary, memory_type, status, json.dumps(dict(metadata), sort_keys=True), _now_text()]
        if "access_score" in changes:
            columns.append("access_score = ?")
            values.append(changes["access_score"])
        if "last_accessed_at" in changes:
            columns.append("last_accessed_at = ?")
            values.append(changes["last_accessed_at"])
        if "last_surfaced_at" in changes:
            columns.append("last_surfaced_at = ?")
            values.append(changes["last_surfaced_at"])
        values.append(normalized_id)
        self._connection.execute(f"UPDATE memories SET {', '.join(columns)} WHERE id = ?", values)
        self._replace_workspaces(normalized_id, workspace_ids)
        self._replace_tags(normalized_id, tags)
        self.touched_ids.append(normalized_id)
        self._stage_hook("domain_mutation")
        self._replace_fts(normalized_id, title, summary or "", content, tags)
        self._stage_hook("projection_update")
        return self.get_memory(normalized_id)  # type: ignore[return-value]

    def create_memory(self, **values: object) -> MemoryRecord:
        memory_id = _canonical_id(values.get("memory_id", uuid4()))
        if self._connection.execute("SELECT 1 FROM memories WHERE id = ?", (memory_id,)).fetchone() is not None:
            raise CurationActionFatalError(f"memory {memory_id!r} already exists")
        title = str(values.get("title", "")).strip()
        content = str(values.get("content", "")).strip()
        if not title or not content:
            raise ValueError("title and content must be non-empty")
        memory_type = str(values.get("memory_type", values.get("type", "journal")))
        status = str(values.get("status", "active"))
        workspace_ids = _values(values.get("workspace_ids", []), [])
        if not workspace_ids:
            raise ValueError("workspace_ids must contain at least one non-empty value")
        tags = _values(values.get("tags", []), [])
        summary = values.get("summary") or _build_summary(title, content, memory_type)
        created_at = str(values.get("created_at", _now_text()))
        updated_at = str(values.get("updated_at", created_at))
        metadata = cast(Mapping[str, object], values.get("metadata", {}))
        next_ref_row = self._connection.execute(
            "SELECT COALESCE(MAX(memory_ref), 0) + 1 FROM memories"
        ).fetchone()
        memory_ref = int(next_ref_row[0]) if next_ref_row is not None else 1
        self._connection.execute(
            "INSERT INTO memories (id, memory_ref, title, content, summary, type, status, created_at, updated_at, metadata) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (memory_id, memory_ref, title, content, str(summary), memory_type, status, created_at, updated_at, json.dumps(dict(metadata), sort_keys=True)),
        )
        self._replace_workspaces(memory_id, workspace_ids)
        self._replace_tags(memory_id, tags)
        self.touched_ids.append(memory_id)
        self._new_ids.add(memory_id)
        self._stage_hook("domain_mutation")
        self._replace_fts(memory_id, title, str(summary), content, tags)
        self._stage_hook("projection_update")
        record = self.get_memory(memory_id)
        assert record is not None
        return record

    def delete_memory(self, memory_id: str | UUID) -> MemoryRecord:
        normalized_id = self._require_access(memory_id)
        existing = self.get_memory(normalized_id)
        if existing is None:
            raise CurationActionStaleError(f"target memory {normalized_id!r} is missing")
        self._connection.execute("DELETE FROM memories_fts WHERE memory_id = ?", (normalized_id,))
        self._connection.execute("DELETE FROM memories WHERE id = ?", (normalized_id,))
        self.touched_ids.append(normalized_id)
        self._stage_hook("domain_mutation")
        return existing

    def add_link(self, source_id: str | UUID, target_id: str | UUID, link_type: str, context: str = "") -> MemoryLink:
        source = self._require_access(source_id)
        target = self._require_endpoint(target_id)
        normalized_type = _normalize_link_type(link_type)
        normalized_context = context.strip()
        self._connection.execute(
            "INSERT INTO links (source_id, target_id, type, context) VALUES (?, ?, ?, ?) ON CONFLICT(source_id, target_id, type) DO UPDATE SET context = excluded.context",
            (source, target, normalized_type, normalized_context),
        )
        self.touched_ids.extend([source, target])
        self._stage_hook("domain_mutation")
        return MemoryLink(source, target, normalized_type, normalized_context)

    def remove_link(self, source_id: str | UUID, target_id: str | UUID, link_type: str) -> bool:
        source = self._require_access(source_id)
        target = self._require_endpoint(target_id)
        cursor = self._connection.execute(
            "DELETE FROM links WHERE source_id = ? AND target_id = ? AND type = ?",
            (source, target, _normalize_link_type(link_type)),
        )
        self.touched_ids.extend([source, target])
        self._stage_hook("domain_mutation")
        return cursor.rowcount > 0

    def set_lineage(self, memory_id: str | UUID, lineage: Mapping[str, object]) -> MemoryRecord:
        record = self.get_memory(memory_id)
        if record is None:
            raise CurationActionStaleError(f"target memory {memory_id!r} is missing")
        metadata = dict(record.metadata)
        metadata["lineage"] = dict(lineage)
        return self.update_memory(memory_id, metadata=metadata)

    def _require_access(self, memory_id: str | UUID) -> str:
        normalized_id = _canonical_id(memory_id)
        if normalized_id not in self._target_ids and normalized_id not in self._new_ids:
            if self._connection.execute("SELECT 1 FROM memories WHERE id = ?", (normalized_id,)).fetchone() is not None:
                raise CurationActionFatalError(f"callback discovered unlocked target {normalized_id!r}")
        return normalized_id

    def _require_endpoint(self, memory_id: str | UUID) -> str:
        normalized_id = _canonical_id(memory_id)
        if normalized_id in self._target_ids or normalized_id in self._new_ids:
            return normalized_id
        if self._connection.execute("SELECT 1 FROM memories WHERE id = ?", (normalized_id,)).fetchone() is not None:
            raise CurationActionFatalError(f"callback discovered unlocked link endpoint {normalized_id!r}")
        raise CurationActionFatalError(f"link endpoint {normalized_id!r} does not exist")

    def _replace_workspaces(self, memory_id: str, workspace_ids: list[str]) -> None:
        self._connection.execute("DELETE FROM memory_workspaces WHERE memory_id = ?", (memory_id,))
        self._connection.executemany(
            "INSERT INTO memory_workspaces (memory_id, workspace_id) VALUES (?, ?)",
            [(memory_id, value) for value in workspace_ids],
        )

    def _replace_tags(self, memory_id: str, tags: list[str]) -> None:
        self._connection.execute("DELETE FROM memory_tags WHERE memory_id = ?", (memory_id,))
        for tag in tags:
            self._connection.execute("INSERT OR IGNORE INTO tags (name) VALUES (?)", (tag,))
            row = self._connection.execute("SELECT id FROM tags WHERE name = ?", (tag,)).fetchone()
            assert row is not None
            self._connection.execute("INSERT INTO memory_tags (memory_id, tag_id) VALUES (?, ?)", (memory_id, row[0]))

    def _replace_fts(self, memory_id: str, title: str, summary: str, content: str, tags: list[str]) -> None:
        self._connection.execute("DELETE FROM memories_fts WHERE memory_id = ?", (memory_id,))
        self._connection.execute(
            "INSERT INTO memories_fts(memory_id, title, summary, content, tags) VALUES (?, ?, ?, ?, ?)",
            (memory_id, title, summary, content, " ".join(tags)),
        )


def _normalize_result(result: MutationResult | Mapping[str, object] | object) -> _MutationResultData:
    if isinstance(result, MutationResult):
        return _MutationResultData(result.operation, list(result.affected_ids), result.verification_descriptor)
    operation = _field(result, "operation", None)
    affected_ids = _field(result, "affected_ids", [])
    if not isinstance(operation, str) or not isinstance(affected_ids, Sequence) or isinstance(affected_ids, (str, bytes)):
        raise CurationActionFatalError("callback must return MutationResult")
    descriptor = _field(result, "verification_descriptor", None)
    if descriptor is not None and not isinstance(descriptor, CurationVerificationDescriptor):
        raise CurationActionFatalError("mutation result verification descriptor is invalid")
    return _MutationResultData(operation, [str(value) for value in affected_ids], descriptor)


def _canonical_target_ids(values: Sequence[str | UUID]) -> list[str]:
    normalized = {_canonical_id(value) for value in values}
    if "" in normalized:
        raise CurationActionFatalError("target IDs must be non-empty")
    return sorted(normalized, key=lambda value: value.encode("utf-8"))


def _canonical_id(value: object) -> str:
    if value is None:
        raise CurationActionFatalError("identity must be non-null")
    return unicodedata.normalize("NFC", str(value)).strip()


def _field(value: object, name: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _text(value: object, fallback: str) -> str:
    return fallback if value is None else str(value).strip()


def _values(value: object, fallback: Sequence[str]) -> list[str]:
    source = fallback if value is None else value
    if not isinstance(source, Sequence) or isinstance(source, (str, bytes)):
        raise ValueError("collection value expected")
    return sorted({str(item).strip() for item in source if str(item).strip()}, key=lambda item: item.encode("utf-8"))


def _normalize_link_type(link_type: str) -> str:
    normalized = re.sub(r"[\s-]+", "_", link_type.strip()).upper()
    if not normalized:
        raise ValueError("link_type must be non-empty")
    return normalized


def _link_token_key(key: str) -> tuple[str, str, str] | None:
    raw = key.removeprefix("link:") if key.startswith("link:") else key
    parts = raw.split(":")
    if len(parts) != 3 or any(not part for part in parts):
        return None
    return parts[0], parts[1], parts[2]


def _now_text() -> str:
    return datetime.now(UTC).isoformat()


def _build_summary(title: str, content: str, memory_type: str) -> str:
    # Keep the transaction primitive independent from the public repository
    # while retaining its established summary shape.
    from mcp_memory.core.summaries import build_deterministic_summary

    return build_deterministic_summary(title=title, content=content, memory_type=memory_type)


def _hydrate_record(connection: sqlite3.Connection, row: sqlite3.Row) -> MemoryRecord:
    workspace_rows = connection.execute(
        "SELECT workspace_id FROM memory_workspaces WHERE memory_id = ? ORDER BY workspace_id ASC", (row["id"],)
    ).fetchall()
    tag_rows = connection.execute(
        "SELECT tags.name FROM tags JOIN memory_tags ON memory_tags.tag_id = tags.id WHERE memory_tags.memory_id = ? ORDER BY tags.name ASC",
        (row["id"],),
    ).fetchall()
    return MemoryRecord(
        id=str(row["id"]), memory_ref=row["memory_ref"], title=str(row["title"]), content=str(row["content"]), summary=row["summary"],
        type=str(row["type"]), status=str(row["status"]), created_at=str(row["created_at"]), updated_at=str(row["updated_at"]),
        read_count=int(row["read_count"] or 0), access_score=float(row["access_score"] or 0),
        last_accessed_at=row["last_accessed_at"], last_surfaced_at=row["last_surfaced_at"],
        metadata=json.loads(row["metadata"] or "{}"), workspace_ids=[str(item[0]) for item in workspace_rows],
        tags=[str(item[0]) for item in tag_rows],
    )


def _links_for_memory(connection: sqlite3.Connection, memory_id: str) -> list[MemoryLink]:
    rows = connection.execute(
        "SELECT source_id, target_id, type, context FROM links WHERE source_id = ? OR target_id = ?",
        (memory_id, memory_id),
    ).fetchall()
    return [MemoryLink(str(row[0]), str(row[1]), str(row[2]), str(row[3] or "")) for row in rows]


def _link_mapping(link: MemoryLink) -> dict[str, str | None]:
    return {"source_id": link.source_id, "target_id": link.target_id, "type": link.link_type, "context": link.context}


def _snapshot(record: MemoryRecord) -> dict[str, Any]:
    return record_snapshot(record)


def _semantic_token(record: MemoryRecord) -> str:
    return record_token(record)


def _state_token(records: Mapping[str, MemoryRecord], ids: Sequence[str]) -> str | None:
    if not records:
        return None
    return canonical_token({"targets": list(ids), "records": {key: _snapshot(records[key]) for key in sorted(records, key=lambda value: value.encode("utf-8"))}})


def _is_retryable_sqlite_error(error: sqlite3.OperationalError) -> bool:
    message = str(error).lower()
    return "locked" in message or "busy" in message


def _check_replay_identity(
    receipt: CurationActionReceipt,
    *,
    run_id: UUID,
    action_id: UUID,
    operation: str | None,
    target_ids: Sequence[str],
    expected_tokens: Mapping[str, str],
    preconditions: Any | None,
    payload: Any | None,
    verification_descriptor: CurationVerificationDescriptor | None,
) -> None:
    if receipt.intent_hash is None:
        raise CurationActionFatalError(
            f"curation action identity collision: legacy-unverifiable receipt for {run_id}/{action_id}"
        )
    if operation is None:
        raise CurationActionFatalError(
            f"curation action identity collision: operation is required for {run_id}/{action_id}"
        )
    candidate = _action_intent_hash(
        operation=operation,
        target_ids=target_ids,
        expected_tokens=expected_tokens,
        preconditions=preconditions,
        payload=payload,
    )
    if candidate != receipt.intent_hash:
        raise CurationActionFatalError(f"curation action identity collision for {run_id}/{action_id}")
    if (
        receipt.verification_descriptor is not None
        and verification_descriptor is not None
        and receipt.verification_descriptor != verification_descriptor
    ):
        raise CurationActionFatalError(f"curation action identity collision for {run_id}/{action_id}")


def _action_intent_hash(
    *,
    operation: str,
    target_ids: Sequence[str],
    expected_tokens: Mapping[str, str],
    preconditions: Any | None,
    payload: Any | None,
) -> str:
    try:
        return action_intent_token(
            operation=operation,
            target_ids=target_ids,
            expected_tokens=expected_tokens,
            preconditions=preconditions,
            payload=payload,
        )
    except (TypeError, ValueError) as exc:
        raise CurationActionFatalError("action intent is not canonically representable") from exc



__all__ = [
    "CurationActionError",
    "CurationActionFatalError",
    "CurationActionInjectedFailure",
    "CurationActionStaleError",
    "CurationActionStore",
    "CurationActionTransientError",
    "CurationTransaction",
    "MutationReceipt",
    "MutationResult",
    "SQLiteCurationActionStore",
]


MutationReceipt = CurationActionReceipt
