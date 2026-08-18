"""Transaction-scoped Postgres execution for one curation action.

This is the Postgres counterpart to ``SQLiteCurationActionStore``.  It is an
execution primitive only: callers provide an already validated callback and
the store owns the transaction, projections, history, repair intent, and
receipt boundary.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, cast
from uuid import UUID, uuid4

from mcp_memory.core.curation_identity import graph_token, link_token, record_token
from mcp_memory.core.curation_models import CurationVerificationDescriptor
from mcp_memory.core.ports.memory import MemoryLink, MemoryRecord
from mcp_memory.curation_action_store import (
    CurationActionError,
    CurationActionFatalError,
    CurationActionInjectedFailure,
    CurationActionStaleError,
    CurationActionTransientError,
    CurationTransaction,
    MutationResult,
    _action_intent_hash,
    _build_summary,
    _canonical_id,
    _canonical_target_ids,
    _check_replay_identity,
    _field,
    _link_mapping,
    _local_memory_ids,
    _normalize_link_type,
    _normalize_result,
    _semantic_token,
    _snapshot,
    _state_token,
    _text,
    _values,
)
from mcp_memory.curation_store import (
    CurationActionReceipt,
    CurationReceiptState,
    CurationRunState,
)
from mcp_memory.mutation_history import (
    MutationActorKind,
    MutationEventStatus,
    RevisionRole,
)
from mcp_memory.storage.postgres_curation_store import _receipt_from_row
from mcp_memory.storage.session import CursorLike, DbConnectionLike, SessionManager

logger = logging.getLogger(__name__)


_SUMMARY_UNSET = object()
_VALID_MEMORY_TYPES = frozenset({"journal", "plan", "fact", "observation", "reflection"})
_VALID_MEMORY_STATUSES = frozenset({"active", "stale", "degraded", "archived"})
class PostgresCurationActionStore:
    """Apply one action atomically on one Postgres connection."""

    def __init__(
        self,
        session_manager: SessionManager[DbConnectionLike],
        *,
        embedding_model: str = "default",
        model_name: str | None = None,
        read_cache: Any | None = None,
        fault_stage: str | None = None,
        fault_injector: Callable[[str], None] | None = None,
    ) -> None:
        if not embedding_model.strip() and model_name is None:
            raise ValueError("embedding_model must be non-empty")
        self._sessions = session_manager
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

        # The fast path is only an optimization.  The receipt is checked again
        # while holding the action transaction's locks to close the race.
        existing = self._receipt(run_id, action_id)
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

        with self._sessions.open_connection() as connection:
            try:
                with connection.cursor() as cursor:
                    existing = self._receipt_on(cursor, run_id, action_id)
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

                    cursor.execute(
                        "SELECT state, task_id, plan_id, policy_version FROM curation_runs WHERE run_id = %s",
                        (str(run_id),),
                    )
                    run = cursor.fetchone()
                    if run is None and restores_event_id is not None:
                        cursor.execute(
                            """
                            INSERT INTO curation_runs (
                                run_id, frontier_key, context_fingerprint, policy_version,
                                schema_version, state, budget_usage_json, created_at
                            ) VALUES (%s, %s, %s, '1', 1, %s, '{}'::jsonb, %s)
                            ON CONFLICT (run_id) DO NOTHING
                            """,
                            (
                                str(run_id),
                                f"restore:{restores_event_id}",
                                idempotency_key or str(action_id),
                                str(CurationRunState.EXECUTING),
                                _now_text(),
                            ),
                        )
                        cursor.execute(
                            "SELECT state, task_id, plan_id, policy_version FROM curation_runs WHERE run_id = %s",
                            (str(run_id),),
                        )
                        run = cursor.fetchone()
                    if run is None and restores_event_id is None:
                        raise CurationActionFatalError(f"curation run {run_id} was not found")
                    if run is not None and str(run[0]) == str(CurationRunState.TERMINAL) and restores_event_id is None:
                        raise CurationActionFatalError(f"curation run {run_id} is terminal")

                    # Never replace this loop with an ORDER BY.  The order is
                    # defined by canonical UTF-8 bytes, not DB collation.
                    self._lock_targets(cursor, normalized_targets)
                    existing = self._receipt_on(cursor, run_id, action_id)
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

                    before_records = self._read_records(cursor, normalized_targets)
                    self._check_tokens(cursor, before_records, normalized_targets, normalized_tokens)
                    self._check_preconditions(cursor, normalized_targets, preconditions)
                    before_links = self._read_related_links(cursor, set(normalized_targets))

                    transaction = _PostgresCurationTransaction(
                        cursor,
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
                    self._fail_stage("after_domain_mutation")

                    after_ids = _canonical_target_ids([*normalized_targets, *transaction.touched_ids])
                    after_records = self._read_records(cursor, after_ids)
                    after_links = self._read_related_links(cursor, set(after_ids))
                    event_id = uuid4()
                    before_token = _state_token(before_records, normalized_targets)
                    after_token = _state_token(after_records, after_ids)

                    self._write_repair_intents(cursor, before_records, after_records)
                    self._fail_stage("repair_intent")
                    self._write_history(
                        cursor,
                        event_id=event_id,
                        run_id=run_id,
                        action_id=action_id,
                        task_id=None if run is None or run[1] is None else UUID(str(run[1])),
                        plan_id=None if run is None or run[2] is None else UUID(str(run[2])),
                        policy_version=None if run is None or run[3] is None else str(run[3]),
                        operation=result_operation,
                        actor_kind=actor_kind,
                        restores_event_id=restores_event_id,
                        idempotency_key=idempotency_key,
                        before_records=before_records,
                        after_records=after_records,
                        before_links=before_links,
                        after_links=after_links,
                    )
                    self._fail_stage("history")

                    affected_ids = _local_memory_ids(
                        result.affected_ids or transaction.touched_ids or normalized_targets
                    )
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
                    self._insert_receipt(cursor, receipt)
                    self._fail_stage("receipt_preparation")
                connection.commit()
                self._invalidate_derivative_caches(after_ids)
                return receipt
            except CurationActionError:
                connection.rollback()
                raise
            except Exception as exc:
                connection.rollback()
                if _is_retryable_postgres_error(exc):
                    raise CurationActionTransientError(
                        f"Postgres action transaction was retryable for {run_id}/{action_id}"
                    ) from exc
                raise CurationActionFatalError(f"curation action failed for {run_id}/{action_id}") from exc

    def _invalidate_derivative_caches(self, memory_ids: Sequence[str]) -> None:
        if self._read_cache is None:
            return
        try:
            self._read_cache.invalidate_for_mutation(list(memory_ids))
        except Exception:
            # The cache is derivative; a cache failure must not turn a
            # committed authoritative mutation into a retryable action.
            logger.warning("Unable to invalidate shared read cache after curation mutation", exc_info=True)

    def _receipt(self, run_id: UUID, action_id: UUID) -> CurationActionReceipt | None:
        with self._sessions.open_connection() as connection:
            with connection.cursor() as cursor:
                return self._receipt_on(cursor, run_id, action_id)

    def _receipt_on(self, cursor: CursorLike, run_id: UUID, action_id: UUID) -> CurationActionReceipt | None:
        cursor.execute(
            "SELECT run_id, action_id, operation, affected_ids_json, status, before_token, after_token, "
            "mutation_event_id, intent_hash, verification_descriptor_json, error_code, applied_at, verified_at "
            "FROM curation_action_receipts WHERE run_id = %s AND action_id = %s",
            (str(run_id), str(action_id)),
        )
        row = cursor.fetchone()
        return None if row is None else _receipt_from_row(row)

    def _lock_targets(self, cursor: CursorLike, target_ids: Sequence[str]) -> None:
        for memory_id in target_ids:
            if memory_id.startswith("ext:"):
                continue
            cursor.execute("SELECT id FROM memories WHERE id = %s FOR UPDATE", (memory_id,))
            if cursor.fetchone() is None:
                raise CurationActionStaleError(f"target memory {memory_id!r} is missing")

    def _read_records(self, cursor: CursorLike, memory_ids: Sequence[str]) -> dict[str, MemoryRecord]:
        if not memory_ids:
            return {}
        cursor.execute(
            "SELECT id, title, content, summary, type, status, created_at, updated_at, read_count, access_score, "
            "last_accessed_at, last_surfaced_at, metadata, memory_ref FROM memories WHERE id = ANY(%s::text[])",
            (list(memory_ids),),
        )
        return {str(row[0]): _hydrate_record(cursor, row) for row in cursor.fetchall()}

    def _check_tokens(
        self,
        cursor: CursorLike,
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
                actual = graph_token(memory_id, [_link_mapping(link) for link in _links_for_memory(cursor, memory_id)])
                if actual != graph_expected:
                    raise CurationActionStaleError(f"graph revision token is stale for {memory_id!r}")
        for key, expected in expected_tokens.items():
            parts = _link_token_key(key)
            if parts is None:
                continue
            source_id, target_id, link_type = parts
            cursor.execute(
                "SELECT context FROM links WHERE source_id = %s AND target_id = %s AND type = %s",
                (source_id, target_id, _normalize_link_type(link_type)),
            )
            row = cursor.fetchone()
            actual = link_token(
                source_id,
                target_id,
                link_type,
                None if row is None else str(row[0] or ""),
                exists=row is not None,
            )
            if actual != expected:
                raise CurationActionStaleError(f"link revision token is stale for {source_id}:{target_id}:{link_type}")

    def _check_preconditions(self, cursor: CursorLike, target_ids: Sequence[str], preconditions: Any) -> None:
        if preconditions is None:
            return
        required_statuses = cast(Mapping[object, object], _field(preconditions, "required_statuses", {}))
        for memory_id, expected_status in required_statuses.items():
            normalized_id = _canonical_id(memory_id)
            if normalized_id not in target_ids:
                raise CurationActionFatalError("status precondition references an unlocked target")
            cursor.execute("SELECT status FROM memories WHERE id = %s", (normalized_id,))
            row = cursor.fetchone()
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
                cursor.execute(
                    "SELECT context FROM links WHERE source_id = %s AND target_id = %s AND type = %s",
                    (source_id, target_id, link_type),
                )
                row = cursor.fetchone()
                present = row is not None and (
                    not should_exist or context is None or str(row[0] or "") == str(context)
                )
                if present != should_exist:
                    raise CurationActionStaleError("link precondition is stale")

    def _write_repair_intents(
        self,
        cursor: CursorLike,
        before_records: Mapping[str, MemoryRecord],
        after_records: Mapping[str, MemoryRecord],
    ) -> None:
        now = time.time()
        for memory_id, after in after_records.items():
            before = before_records.get(memory_id)
            if before is not None and _semantic_token(before) == _semantic_token(after):
                continue
            cursor.execute(
                """
                INSERT INTO embedding_repair_queue (
                    id, memory_id, workspace_id, model_name, memory_updated_at,
                    status, attempt_count, available_at, created_at, updated_at,
                    claimed_at, completed_at, lease_owner, lease_expires_at, last_error
                ) VALUES (%s, %s, %s, %s, %s, 'pending', 0, %s, %s, %s, NULL, NULL, NULL, NULL, NULL)
                ON CONFLICT (memory_id, model_name, memory_updated_at) DO NOTHING
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
        cursor: CursorLike,
        *,
        event_id: UUID,
        run_id: UUID,
        action_id: UUID,
        task_id: UUID | None,
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
        cursor.execute(
            """
            INSERT INTO memory_mutation_events (
                id, operation, actor_kind, family, task_id, curation_run_id, plan_id,
                action_id, policy_version, schema_version, status, restores_event_id,
                idempotency_key, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 1, %s, %s, %s, %s)
            """,
            (
                str(event_id),
                operation,
                str(actor_kind),
                "curation",
                None if task_id is None else str(task_id),
                str(run_id),
                None if plan_id is None else str(plan_id),
                str(action_id),
                policy_version,
                str(MutationEventStatus.APPLIED),
                None if restores_event_id is None else str(restores_event_id),
                idempotency_key or f"{run_id}:{action_id}",
                _now_text(),
            ),
        )
        for memory_id in sorted(set(before_records) | set(after_records), key=lambda value: value.encode("utf-8")):
            before = before_records.get(memory_id)
            after = after_records.get(memory_id)
            if before is not None and after is not None and _semantic_token(before) == _semantic_token(after):
                continue
            cursor.execute(
                """
                INSERT INTO memory_record_revisions (
                    event_id, memory_id, role, before_exists, before_snapshot,
                    after_exists, after_snapshot, before_token, after_token
                ) VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s::jsonb, %s, %s)
                """,
                (
                    str(event_id),
                    memory_id,
                    str(RevisionRole.TARGET),
                    before is not None,
                    None if before is None else json.dumps(_snapshot(before), ensure_ascii=False, sort_keys=True),
                    after is not None,
                    None if after is None else json.dumps(_snapshot(after), ensure_ascii=False, sort_keys=True),
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
            cursor.execute(
                """
                INSERT INTO memory_link_revisions (
                    event_id, source_id, target_id, link_type, context,
                    before_exists, after_exists
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    str(event_id),
                    source_id,
                    target_id,
                    link_type,
                    None if after is None else after.context,
                    before is not None,
                    after is not None,
                ),
            )

    def _insert_receipt(self, cursor: CursorLike, receipt: CurationActionReceipt) -> None:
        cursor.execute(
            """
            INSERT INTO curation_action_receipts (
                run_id, action_id, operation, affected_ids_json, status,
                before_token, after_token, mutation_event_id, intent_hash,
                verification_descriptor_json, error_code, applied_at, verified_at
            ) VALUES (%s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s)
            """,
            (
                str(receipt.run_id),
                str(receipt.action_id),
                receipt.operation,
                json.dumps([str(value) for value in receipt.affected_ids]),
                str(receipt.status),
                receipt.before_token,
                receipt.after_token,
                None if receipt.mutation_event_id is None else str(receipt.mutation_event_id),
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
                None if receipt.applied_at is None else receipt.applied_at.isoformat(),
                None if receipt.verified_at is None else receipt.verified_at.isoformat(),
            ),
        )

    def _fail_stage(self, stage: str) -> None:
        if self._fault_injector is not None:
            self._fault_injector(stage)
        aliases = {
            "domain_mutation": {"domain", "after_domain", "domain_mutation", "after_domain_mutation"},
            "projection_update": {"projection", "after_projection", "projection_update", "after_projection_update"},
            "repair_intent": {"repair", "after_repair", "repair_intent", "after_repair_intent"},
            "history": {"history", "after_history"},
            "receipt_preparation": {"receipt", "before_receipt", "receipt_preparation", "after_receipt_preparation"},
        }
        normalized_stage = None if self._fault_stage is None else self._fault_stage.replace("-", "_")
        if normalized_stage in aliases.get(stage, {stage}):
            raise CurationActionInjectedFailure(f"injected curation transaction failure at {stage}")

    def _read_related_links(
        self, cursor: CursorLike, memory_ids: set[str]
    ) -> dict[tuple[str, str, str], MemoryLink]:
        if not memory_ids:
            return {}
        values = sorted(memory_ids, key=lambda value: value.encode("utf-8"))
        cursor.execute(
            "SELECT source_id, target_id, type, context FROM links "
            "WHERE source_id = ANY(%s::text[]) OR target_id = ANY(%s::text[])",
            (values, values),
        )
        return {
            (str(row[0]), str(row[1]), str(row[2])): MemoryLink(
                source_id=str(row[0]), target_id=str(row[1]), link_type=str(row[2]), context=str(row[3] or "")
            )
            for row in cursor.fetchall()
        }


class _PostgresCurationTransaction:
    def __init__(self, cursor: CursorLike, *, target_ids: set[str], stage_hook: Callable[[str], None]) -> None:
        self._cursor = cursor
        self._target_ids = target_ids
        self._stage_hook = stage_hook
        self.touched_ids: list[str] = []
        self._new_ids: set[str] = set()

    def get_memory(self, memory_id: str | UUID) -> MemoryRecord | None:
        normalized_id = self._require_access(memory_id)
        row = _select_memory_row(self._cursor, normalized_id)
        return None if row is None else _hydrate_record(self._cursor, row)

    def get_links(
        self, memory_id: str | UUID, *, direction: str = "outgoing", link_type: str | None = None
    ) -> list[MemoryLink]:
        normalized_id = self._require_access(memory_id)
        column = "target_id" if direction == "incoming" else "source_id"
        query = f"SELECT source_id, target_id, type, context FROM links WHERE {column} = %s"
        params: list[object] = [normalized_id]
        if link_type is not None:
            query += " AND type = %s"
            params.append(_normalize_link_type(link_type))
        query += " ORDER BY source_id ASC, target_id ASC, type ASC"
        self._cursor.execute(query, tuple(params))
        return [
            MemoryLink(str(row[0]), str(row[1]), str(row[2]), str(row[3] or ""))
            for row in self._cursor.fetchall()
        ]

    def get_protections(self, memory_id: str | UUID) -> list[object]:
        normalized_id = self._require_access(memory_id)
        self._cursor.execute(
            "SELECT memory_id, mode, reason, actor_id, created_at, expires_at "
            "FROM memory_protections WHERE memory_id = %s ORDER BY mode",
            (normalized_id,),
        )
        return list(self._cursor.fetchall())

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
            if summary_value is _SUMMARY_UNSET
            and ("title" in changes or "content" in changes or "type" in changes or "memory_type" in changes)
            else None
            if summary_value is None
            else str(summary_value)
            if summary_value is not _SUMMARY_UNSET
            else existing.summary
        )
        workspace_ids = _values(changes.get("workspace_ids"), existing.workspace_ids)
        tags = _values(changes.get("tags"), existing.tags)
        metadata = changes.get("metadata", existing.metadata)
        if not isinstance(metadata, Mapping):
            raise ValueError("metadata must be a mapping")
        columns = [
            "title = %s",
            "content = %s",
            "summary = %s",
            "type = %s",
            "status = %s",
            "metadata = %s::jsonb",
            "updated_at = %s",
        ]
        values: list[object] = [
            title,
            content,
            summary,
            memory_type,
            status,
            json.dumps(dict(metadata), sort_keys=True),
            _now_text(),
        ]
        for column_name in ("access_score", "last_accessed_at", "last_surfaced_at"):
            if column_name in changes:
                columns.append(f"{column_name} = %s")
                values.append(changes[column_name])
        values.extend([normalized_id, existing.updated_at])
        self._cursor.execute(
            f"UPDATE memories SET {', '.join(columns)} WHERE id = %s AND updated_at = %s",
            tuple(values),
        )
        if int(getattr(self._cursor, "rowcount", 0) or 0) != 1:
            raise CurationActionStaleError(f"record revision changed for {normalized_id!r}")
        self._replace_workspaces(normalized_id, workspace_ids)
        self._replace_tags(normalized_id, tags)
        self.touched_ids.append(normalized_id)
        self._stage_hook("domain_mutation")
        self._replace_search_document(normalized_id, title, summary, content, tags)
        self._stage_hook("projection_update")
        updated = self.get_memory(normalized_id)
        assert updated is not None
        return updated

    def create_memory(self, **values: object) -> MemoryRecord:
        memory_id = _canonical_id(values.get("memory_id", uuid4()))
        title = str(values.get("title", "")).strip()
        content = str(values.get("content", "")).strip()
        if not title or not content:
            raise ValueError("title and content must be non-empty")
        memory_type = str(values.get("memory_type", values.get("type", "journal")))
        status = str(values.get("status", "active"))
        if memory_type not in _VALID_MEMORY_TYPES or status not in _VALID_MEMORY_STATUSES:
            raise ValueError("invalid memory type or status")
        workspace_ids = _values(values.get("workspace_ids", []), [])
        if not workspace_ids:
            raise ValueError("workspace_ids must contain at least one non-empty value")
        tags = _values(values.get("tags", []), [])
        summary = values.get("summary") or _build_summary(title, content, memory_type)
        created_at = str(values.get("created_at", _now_text()))
        updated_at = str(values.get("updated_at", created_at))
        metadata = values.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("metadata must be a mapping")
        self._cursor.execute(
            """
            INSERT INTO memories (
                id, title, content, summary, type, status, created_at, updated_at,
                read_count, access_score, last_accessed_at, last_surfaced_at, metadata
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 0, 0.0, NULL, NULL, %s::jsonb)
            """,
            (memory_id, title, content, str(summary), memory_type, status, created_at, updated_at, json.dumps(dict(metadata), sort_keys=True)),
        )
        self._replace_workspaces(memory_id, workspace_ids)
        self._replace_tags(memory_id, tags)
        self.touched_ids.append(memory_id)
        self._new_ids.add(memory_id)
        self._stage_hook("domain_mutation")
        self._replace_search_document(memory_id, title, str(summary), content, tags)
        self._stage_hook("projection_update")
        record = self.get_memory(memory_id)
        assert record is not None
        return record

    def delete_memory(self, memory_id: str | UUID) -> MemoryRecord:
        normalized_id = self._require_access(memory_id)
        existing = self.get_memory(normalized_id)
        if existing is None:
            raise CurationActionStaleError(f"target memory {normalized_id!r} is missing")
        self._cursor.execute("DELETE FROM links WHERE source_id = %s OR target_id = %s", (normalized_id, normalized_id))
        self._cursor.execute("DELETE FROM memory_search_documents WHERE memory_id = %s", (normalized_id,))
        self._cursor.execute("DELETE FROM memories WHERE id = %s AND updated_at = %s", (normalized_id, existing.updated_at))
        if int(getattr(self._cursor, "rowcount", 0) or 0) != 1:
            raise CurationActionStaleError(f"record revision changed for {normalized_id!r}")
        self.touched_ids.append(normalized_id)
        self._stage_hook("domain_mutation")
        return existing

    def add_link(self, source_id: str | UUID, target_id: str | UUID, link_type: str, context: str = "") -> MemoryLink:
        source = self._require_access(source_id)
        target = self._require_endpoint(target_id)
        normalized_type = _normalize_link_type(link_type)
        normalized_context = context.strip()
        self._cursor.execute(
            """
            INSERT INTO links (source_id, target_id, type, context) VALUES (%s, %s, %s, %s)
            ON CONFLICT (source_id, target_id, type) DO UPDATE SET context = EXCLUDED.context
            """,
            (source, target, normalized_type, normalized_context),
        )
        self.touched_ids.extend([source, target])
        self._stage_hook("domain_mutation")
        return MemoryLink(source, target, normalized_type, normalized_context)

    def remove_link(self, source_id: str | UUID, target_id: str | UUID, link_type: str) -> bool:
        source = self._require_access(source_id)
        target = self._require_endpoint(target_id)
        self._cursor.execute(
            "DELETE FROM links WHERE source_id = %s AND target_id = %s AND type = %s",
            (source, target, _normalize_link_type(link_type)),
        )
        self.touched_ids.extend([source, target])
        self._stage_hook("domain_mutation")
        return int(getattr(self._cursor, "rowcount", 0) or 0) > 0

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
            self._cursor.execute("SELECT 1 FROM memories WHERE id = %s", (normalized_id,))
            if self._cursor.fetchone() is not None:
                raise CurationActionFatalError(f"callback discovered unlocked target {normalized_id!r}")
        return normalized_id

    def _require_endpoint(self, memory_id: str | UUID) -> str:
        normalized_id = _canonical_id(memory_id)
        if normalized_id.startswith("ext:"):
            return normalized_id
        if normalized_id in self._target_ids or normalized_id in self._new_ids:
            return normalized_id
        self._cursor.execute("SELECT 1 FROM memories WHERE id = %s", (normalized_id,))
        if self._cursor.fetchone() is not None:
            raise CurationActionFatalError(f"callback discovered unlocked link endpoint {normalized_id!r}")
        raise CurationActionFatalError(f"link endpoint {normalized_id!r} does not exist")

    def _replace_workspaces(self, memory_id: str, workspace_ids: list[str]) -> None:
        self._cursor.execute("DELETE FROM memory_workspaces WHERE memory_id = %s", (memory_id,))
        for workspace_id in workspace_ids:
            self._cursor.execute(
                "INSERT INTO memory_workspaces (memory_id, workspace_id) VALUES (%s, %s)",
                (memory_id, workspace_id),
            )

    def _replace_tags(self, memory_id: str, tags: list[str]) -> None:
        self._cursor.execute("DELETE FROM memory_tags WHERE memory_id = %s", (memory_id,))
        for tag in tags:
            self._cursor.execute("INSERT INTO tags (name) VALUES (%s) ON CONFLICT (name) DO NOTHING", (tag,))
            self._cursor.execute("SELECT id FROM tags WHERE name = %s", (tag,))
            row = self._cursor.fetchone()
            if row is not None:
                self._cursor.execute(
                    "INSERT INTO memory_tags (memory_id, tag_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                    (memory_id, row[0]),
                )

    def _replace_search_document(
        self, memory_id: str, title: str, summary: str | None, content: str, tags: list[str]
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
            (memory_id, title, summary or "", content, tags_text, title, summary or "", tags_text, content),
        )


def _link_token_key(key: str) -> tuple[str, str, str] | None:
    raw = key.removeprefix("link:") if key.startswith("link:") else key
    parts = raw.split(":")
    if len(parts) != 3 or any(not part for part in parts):
        return None
    return parts[0], parts[1], parts[2]


def _select_memory_row(cursor: CursorLike, memory_id: str) -> tuple[object, ...] | None:
    cursor.execute(
        "SELECT id, title, content, summary, type, status, created_at, updated_at, read_count, access_score, "
        "last_accessed_at, last_surfaced_at, metadata, memory_ref FROM memories WHERE id = %s",
        (memory_id,),
    )
    return cursor.fetchone()


def _hydrate_record(cursor: CursorLike, row: tuple[object, ...]) -> MemoryRecord:
    memory_id = str(row[0])
    cursor.execute("SELECT workspace_id FROM memory_workspaces WHERE memory_id = %s ORDER BY workspace_id ASC", (memory_id,))
    workspace_rows = cursor.fetchall()
    cursor.execute(
        "SELECT tags.name FROM tags JOIN memory_tags ON memory_tags.tag_id = tags.id "
        "WHERE memory_tags.memory_id = %s ORDER BY tags.name ASC",
        (memory_id,),
    )
    tag_rows = cursor.fetchall()
    metadata = row[12]
    if isinstance(metadata, str):
        loaded = json.loads(metadata or "{}")
        metadata = loaded if isinstance(loaded, dict) else {}
    elif not isinstance(metadata, Mapping):
        metadata = {}
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
        memory_ref=int(cast(Any, row[13])) if row[13] is not None else None,
        workspace_ids=[str(item[0]) for item in workspace_rows],
        tags=[str(item[0]) for item in tag_rows],
    )


def _links_for_memory(cursor: CursorLike, memory_id: str) -> list[MemoryLink]:
    cursor.execute(
        "SELECT source_id, target_id, type, context FROM links WHERE source_id = %s OR target_id = %s",
        (memory_id, memory_id),
    )
    return [MemoryLink(str(row[0]), str(row[1]), str(row[2]), str(row[3] or "")) for row in cursor.fetchall()]


def _now_text() -> str:
    return datetime.now(UTC).isoformat()


def _is_retryable_postgres_error(error: Exception) -> bool:
    sqlstate = getattr(error, "sqlstate", None) or getattr(error, "pgcode", None)
    return sqlstate in {
        "40001",  # serialization_failure
        "40P01",  # deadlock_detected
        "55P03",  # lock_not_available
        "57014",  # query_canceled / statement timeout
        "08000",
        "08001",
        "08003",
        "08004",
        "08006",
    }


__all__ = ["PostgresCurationActionStore"]
