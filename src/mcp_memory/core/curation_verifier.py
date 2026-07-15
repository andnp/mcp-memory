"""Authoritative postcondition verification for low-risk curation actions."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from mcp_memory.core.curation_identity import canonical_token, record_snapshot
from mcp_memory.core.curation_models import CreateLinkAction, NormalizeMemoryAction
from mcp_memory.curation_store import (
    CurationActionReceipt,
    CurationReceiptState,
    CurationRepository,
)
from mcp_memory.relational.repository import MemoryLink, RelationalMemoryReadContext
from mcp_memory.relational.search import MaintenanceReadRepositoryLike


class CurationVerificationError(RuntimeError):
    """The receipt could not be verified against the authoritative store."""


class _PostconditionMismatch(CurationVerificationError):
    def __init__(self, error_code: str) -> None:
        self.error_code = error_code
        super().__init__(error_code)


LowRiskCurationAction = NormalizeMemoryAction | CreateLinkAction


class CurationVerifier:
    """Verify committed low-risk actions using fresh, non-instrumenting reads.

    Planner rationale and evidence are intentionally not consulted.  The typed
    action supplies only the exact postcondition shape; the current record and
    graph state come from a fresh maintenance read.
    """

    def __init__(
        self,
        curation_store: CurationRepository,
        maintenance_reads: MaintenanceReadRepositoryLike,
    ) -> None:
        self._curation_store = curation_store
        self._maintenance_reads = maintenance_reads

    def verify(
        self,
        receipt: CurationActionReceipt,
        action: LowRiskCurationAction,
    ) -> CurationActionReceipt:
        """Advance an applied receipt after checking its authoritative postcondition."""
        current = self._curation_store.get_receipt(receipt.run_id, receipt.action_id)
        if current is None:
            raise CurationVerificationError("curation receipt was not found")
        if current.status is not CurationReceiptState.APPLIED_UNVERIFIED:
            return current
        if current.action_id != action.action_id or current.run_id != receipt.run_id:
            raise CurationVerificationError("receipt identity does not match action")
        if current.operation != action.operation:
            raise CurationVerificationError("receipt operation does not match action")

        expected_ids = _expected_ids(action)
        if _receipt_ids(current) != expected_ids:
            return self._finish(
                current,
                CurationReceiptState.VERIFICATION_FAILED,
                "affected_ids_mismatch",
            )

        try:
            contexts = self._fresh_contexts(expected_ids)
        except _PostconditionMismatch as mismatch:
            return self._finish(
                current,
                CurationReceiptState.VERIFICATION_FAILED,
                mismatch.error_code,
            )
        actual_after_token = _state_token(
            {memory_id: context.record for memory_id, context in contexts.items()},
            expected_ids,
        )
        if actual_after_token != current.after_token:
            return self._finish(
                current,
                CurationReceiptState.VERIFICATION_FAILED,
                "after_token_mismatch",
            )

        if isinstance(action, CreateLinkAction) and not _has_exact_edge(
            contexts[str(action.source_id)],
            source_id=str(action.source_id),
            target_id=str(action.target_id),
            link_type=action.link_type,
            edge_context=action.context or "",
        ):
            return self._finish(
                current,
                CurationReceiptState.VERIFICATION_FAILED,
                "edge_missing",
            )

        return self._finish(current, CurationReceiptState.VERIFIED, None)

    def verify_receipt(
        self,
        receipt: CurationActionReceipt,
        action: LowRiskCurationAction,
    ) -> CurationActionReceipt:
        """Descriptive alias for :meth:`verify`."""
        return self.verify(receipt, action)

    def _fresh_contexts(self, memory_ids: Sequence[str]) -> dict[str, RelationalMemoryReadContext]:
        contexts: dict[str, RelationalMemoryReadContext] = {}
        for memory_id in memory_ids:
            context = self._maintenance_reads.peek_memory(memory_id)
            if context is None or context.record.id != memory_id:
                raise _PostconditionMismatch("record_missing")
            contexts[memory_id] = context
        return contexts

    def _finish(
        self,
        current: CurationActionReceipt,
        status: CurationReceiptState,
        error_code: str | None,
    ) -> CurationActionReceipt:
        candidate = current.model_copy(
            update={
                "status": status,
                "error_code": error_code,
                "verified_at": datetime.now(UTC),
            }
        )
        transitioned = self._curation_store.transition_receipt(
            current.run_id,
            current.action_id,
            CurationReceiptState.APPLIED_UNVERIFIED,
            candidate,
        )
        if transitioned is not None:
            return transitioned
        latest = self._curation_store.get_receipt(current.run_id, current.action_id)
        if latest is None:
            raise CurationVerificationError("curation receipt disappeared during verification")
        return latest


def verify_curation_receipt(
    curation_store: CurationRepository,
    maintenance_reads: MaintenanceReadRepositoryLike,
    receipt: CurationActionReceipt,
    action: LowRiskCurationAction,
) -> CurationActionReceipt:
    """Functional entry point for authoritative low-risk receipt verification."""
    return CurationVerifier(curation_store, maintenance_reads).verify(receipt, action)


def _expected_ids(action: LowRiskCurationAction) -> list[str]:
    if isinstance(action, NormalizeMemoryAction):
        return [str(action.target_id)]
    return sorted(
        {str(action.source_id), str(action.target_id)},
        key=lambda value: value.encode("utf-8"),
    )


def _receipt_ids(receipt: CurationActionReceipt) -> list[str]:
    return sorted({str(value) for value in receipt.affected_ids}, key=lambda value: value.encode("utf-8"))


def _state_token(records: Mapping[str, Any], ids: Sequence[str]) -> str | None:
    if not records:
        return None
    return canonical_token(
        {
            "targets": list(ids),
            "records": {
                key: record_snapshot(records[key])
                for key in sorted(records, key=lambda value: value.encode("utf-8"))
            },
        }
    )


def _has_exact_edge(
    read_context: RelationalMemoryReadContext,
    *,
    source_id: str,
    target_id: str,
    link_type: str,
    edge_context: str,
) -> bool:
    normalized_type = _normalize_link_type(link_type)
    normalized_context = edge_context.strip()
    return any(
        _edge_matches(
            edge,
            source_id=source_id,
            target_id=target_id,
            link_type=normalized_type,
            context=normalized_context,
        )
        for edge in read_context.relationships.get("outgoing", [])
    )


def _edge_matches(
    edge: MemoryLink,
    *,
    source_id: str,
    target_id: str,
    link_type: str,
    context: str,
) -> bool:
    return (
        edge.source_id == source_id
        and edge.target_id == target_id
        and edge.link_type == link_type
        and edge.context == context
    )


def _normalize_link_type(link_type: str) -> str:
    normalized = re.sub(r"[\s-]+", "_", link_type.strip()).upper()
    if not normalized:
        raise CurationVerificationError("link_type must be non-empty")
    return normalized


__all__ = [
    "CurationVerificationError",
    "CurationVerifier",
    "verify_curation_receipt",
]
