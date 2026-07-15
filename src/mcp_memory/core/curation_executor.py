"""Provider-free executors for accepted curation operations.

This module deliberately contains no production orchestration.  It turns a
typed action into the callback expected by a backend action transaction; the
transaction primitive remains responsible for atomic history, receipts, and
idempotency.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import NoReturn
from uuid import UUID

from mcp_memory.core.curation_models import NormalizeMemoryAction
from mcp_memory.core.curation_policy import PolicyDecision, evaluate_curation_action
from mcp_memory.curation_action_store import (
    CurationActionFatalError,
    CurationActionStore,
    CurationTransaction,
    MutationResult,
)
from mcp_memory.curation_store import CurationActionReceipt
from mcp_memory.mutation_history import ProtectionMode


class CurationPolicyRejection(CurationActionFatalError):
    """A pure policy or protection check rejected an action before mutation."""

    def __init__(self, decision: PolicyDecision) -> None:
        self.decision = decision
        codes = ", ".join(code.value for code in decision.rejection_codes)
        super().__init__(f"{decision.operation} rejected by policy: {codes}")


class CurationExecutor:
    """Execute individually accepted curation actions through an action store."""

    def __init__(self, action_store: CurationActionStore) -> None:
        self._action_store = action_store

    def execute_normalize(
        self,
        action: NormalizeMemoryAction,
        *,
        run_id: UUID,
        memory_type: str,
        protections: Iterable[ProtectionMode | str] = (),
        expected_token: str | None = None,
    ) -> CurationActionReceipt:
        """Apply title/summary/tags only and return the compact transaction receipt.

        The record token comes from the typed action preconditions.  The
        optional ``expected_token`` is a convenience for callers that keep
        preconditions separately; supplying both values must agree.
        """
        normalized_protections = _normalize_protections(protections)
        decision = evaluate_curation_action(
            action,
            memory_types={action.target_id: memory_type},
            protections_by_memory={action.target_id: normalized_protections},
        )
        if not decision.authorized:
            raise CurationPolicyRejection(decision)

        record_token = action.preconditions.record_tokens.get(action.target_id)
        if record_token is not None and expected_token is not None and record_token != expected_token:
            _reject_missing_or_conflicting_token("expected record token conflicts with action preconditions")
        token = expected_token or record_token
        if not token:
            _reject_missing_or_conflicting_token("normalize_memory requires an expected record token")

        changes: dict[str, object] = {}
        if action.title is not None:
            changes["title"] = action.title
        if action.summary is not None:
            changes["summary"] = action.summary
        if action.tags is not None:
            changes["tags"] = action.tags

        def apply(transaction: CurationTransaction) -> MutationResult:
            # Do not add content here: normalize is intentionally metadata-only.
            transaction.update_memory(str(action.target_id), **changes)
            return MutationResult(action.operation, [action.target_id])

        return self._action_store.execute_action(
            run_id=run_id,
            action_id=action.action_id,
            target_ids=[str(action.target_id)],
            expected_tokens={str(action.target_id): token},
            preconditions=action.preconditions,
            apply=apply,
        )


def execute_normalize_memory(
    action_store: CurationActionStore,
    action: NormalizeMemoryAction,
    *,
    run_id: UUID,
    memory_type: str,
    protections: Iterable[ProtectionMode | str] = (),
    expected_token: str | None = None,
) -> CurationActionReceipt:
    """Functional entry point for backend-neutral normalize execution."""
    return CurationExecutor(action_store).execute_normalize(
        action,
        run_id=run_id,
        memory_type=memory_type,
        protections=protections,
        expected_token=expected_token,
    )


def _normalize_protections(protections: Iterable[ProtectionMode | str]) -> frozenset[ProtectionMode]:
    try:
        return frozenset(
            mode if isinstance(mode, ProtectionMode) else ProtectionMode(str(mode)) for mode in protections
        )
    except ValueError as exc:
        raise CurationActionFatalError("unknown memory protection mode") from exc


def _reject_missing_or_conflicting_token(message: str) -> NoReturn:
    raise CurationActionFatalError(message)


__all__ = [
    "CurationExecutor",
    "CurationPolicyRejection",
    "execute_normalize_memory",
]
