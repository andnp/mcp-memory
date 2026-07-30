"""Authoritative postcondition verification for low-risk curation actions."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from mcp_memory.core.curation_identity import canonical_token, record_snapshot
from mcp_memory.core.curation_models import (
    ArchiveMemoryAction,
    ClaimManifest,
    CurationVerificationDescriptor,
    CreateLinkAction,
    MergeMemoriesAction,
    NormalizeMemoryAction,
    RemoveLinkAction,
    RewriteMemoryAction,
    SplitMemoryAction,
)
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


VerifiedCurationAction = (
    NormalizeMemoryAction
    | RewriteMemoryAction
    | CreateLinkAction
    | RemoveLinkAction
    | MergeMemoriesAction
    | SplitMemoryAction
    | ArchiveMemoryAction
)

LowRiskCurationAction = NormalizeMemoryAction | CreateLinkAction

_MERGE_LINK_CONTEXT = "Merged into canonical memory by provider-free curation."
_SPLIT_LINK_TYPE = "DEPENDS_ON"
_SPLIT_LINK_CONTEXT = "Derived from a provider-free memory split."


class CurationVerifier:
    """Verify committed typed actions using fresh, non-instrumenting reads.

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
        action: VerifiedCurationAction,
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

        if isinstance(action, NormalizeMemoryAction):
            return self._verify_record_action(current, [str(action.target_id)])
        if isinstance(action, RewriteMemoryAction):
            return self._verify_record_action(current, [str(action.target_id)])
        if isinstance(action, ArchiveMemoryAction):
            return self._verify_record_action(current, [str(action.target_id)])
        if isinstance(action, CreateLinkAction):
            expected_ids = sorted(
                {str(action.source_id), str(action.target_id)},
                key=lambda value: value.encode("utf-8"),
            )
            return self._verify_create_link(current, action, expected_ids)
        if isinstance(action, RemoveLinkAction):
            expected_ids = sorted(
                {str(action.source_id), str(action.target_id)},
                key=lambda value: value.encode("utf-8"),
            )
            return self._verify_remove_link(current, action, expected_ids)
        if isinstance(action, MergeMemoriesAction):
            expected_ids = _merge_ids(action.canonical_id, action.source_ids)
            return self._verify_merge(current, action, expected_ids)
        if isinstance(action, SplitMemoryAction):
            return self._verify_split(current, action)
        raise CurationVerificationError("unsupported curation action")

    def verify_descriptor(
        self,
        receipt: CurationActionReceipt,
        descriptor: CurationVerificationDescriptor,
    ) -> CurationActionReceipt:
        """Verify a persisted, content-free recovery postcondition."""
        current = self._curation_store.get_receipt(receipt.run_id, receipt.action_id)
        if current is None:
            raise CurationVerificationError("curation receipt was not found")
        if current.status is not CurationReceiptState.APPLIED_UNVERIFIED:
            return current
        if current.run_id != receipt.run_id or current.action_id != receipt.action_id:
            raise CurationVerificationError("receipt identity does not match descriptor")
        if current.verification_descriptor != descriptor:
            return self._finish(current, CurationReceiptState.VERIFICATION_FAILED, "descriptor_mismatch")
        if descriptor.schema_version != 1 or descriptor.operation != current.operation:
            return self._finish(current, CurationReceiptState.VERIFICATION_FAILED, "descriptor_mismatch")

        expected_ids = [str(value) for value in descriptor.target_ids]
        if descriptor.operation == "split_memory":
            expected_ids = sorted(
                {str(descriptor.target_id), *(str(value) for value in descriptor.child_ids)},
                key=lambda value: value.encode("utf-8"),
            )
        else:
            expected_ids = sorted(set(expected_ids), key=lambda value: value.encode("utf-8"))
        if _receipt_ids(current) != expected_ids:
            return self._finish(current, CurationReceiptState.VERIFICATION_FAILED, "affected_ids_mismatch")

        if descriptor.operation in {"normalize_memory", "rewrite_memory", "archive_memory"}:
            return self._verify_record_action(current, expected_ids, descriptor.target_status)
        if descriptor.operation in {"create_link", "remove_link"}:
            assert descriptor.source_id is not None
            assert descriptor.target_id is not None
            assert descriptor.link_type is not None
            try:
                contexts = self._fresh_contexts(expected_ids)
            except _PostconditionMismatch as mismatch:
                return self._finish(current, CurationReceiptState.VERIFICATION_FAILED, mismatch.error_code)
            if _state_token({key: value.record for key, value in contexts.items()}, expected_ids) != current.after_token:
                return self._finish(current, CurationReceiptState.VERIFICATION_FAILED, "after_token_mismatch")
            edge_present = (
                _has_exact_edge(
                    contexts[str(descriptor.source_id)],
                    source_id=str(descriptor.source_id),
                    target_id=str(descriptor.target_id),
                    link_type=descriptor.link_type,
                    edge_context=descriptor.context or "",
                )
                if descriptor.exists
                else _has_any_edge(
                    contexts[str(descriptor.source_id)],
                    source_id=str(descriptor.source_id),
                    target_id=str(descriptor.target_id),
                    link_type=descriptor.link_type,
                )
            )
            if edge_present is not descriptor.exists:
                return self._finish(
                    current,
                    CurationReceiptState.VERIFICATION_FAILED,
                    "edge_missing" if descriptor.exists else "edge_still_present",
                )
            return self._finish(current, CurationReceiptState.VERIFIED, None)
        if descriptor.operation == "merge_memories":
            assert descriptor.canonical_id is not None
            action = MergeMemoriesAction(
                action_id=current.action_id,
                canonical_id=descriptor.canonical_id,
                source_ids=descriptor.source_ids,
                confidence=1.0,
                rationale="descriptor recovery",
                content="",
                claim_manifest=ClaimManifest(),
            )
            return self._verify_merge(current, action, expected_ids)
        if descriptor.operation == "split_memory":
            return self._verify_split_descriptor(current, descriptor, expected_ids)
        return self._finish(current, CurationReceiptState.VERIFICATION_FAILED, "unsupported_descriptor")

    def _verify_split_descriptor(
        self,
        current: CurationActionReceipt,
        descriptor: CurationVerificationDescriptor,
        expected_ids: Sequence[str],
    ) -> CurationActionReceipt:
        original = self._maintenance_reads.peek_memory(str(descriptor.target_id))
        if original is None or original.record.id != str(descriptor.target_id):
            return self._finish(current, CurationReceiptState.VERIFICATION_FAILED, "record_missing")
        try:
            contexts = self._fresh_contexts(expected_ids)
        except _PostconditionMismatch as mismatch:
            return self._finish(current, CurationReceiptState.VERIFICATION_FAILED, mismatch.error_code)
        if _state_token({key: value.record for key, value in contexts.items()}, expected_ids) != current.after_token:
            return self._finish(current, CurationReceiptState.VERIFICATION_FAILED, "after_token_mismatch")
        if original.record.metadata.get("split_child_memory_ids") != list(descriptor.child_ids):
            return self._finish(current, CurationReceiptState.VERIFICATION_FAILED, "split_children_missing")
        if original.record.metadata.get("split_child_count") != descriptor.child_count:
            return self._finish(current, CurationReceiptState.VERIFICATION_FAILED, "split_child_count_missing")
        for index, child_id in enumerate((str(value) for value in descriptor.child_ids), start=1):
            metadata = contexts[child_id].record.metadata
            if metadata.get("split_from_memory_id") != str(descriptor.target_id):
                return self._finish(current, CurationReceiptState.VERIFICATION_FAILED, "split_lineage_missing")
            if metadata.get("split_group_id") != descriptor.split_group_id:
                return self._finish(current, CurationReceiptState.VERIFICATION_FAILED, "split_group_missing")
            if metadata.get("split_part_index") != index or metadata.get("split_part_count") != descriptor.child_count:
                return self._finish(current, CurationReceiptState.VERIFICATION_FAILED, "split_part_metadata_missing")
            sibling_ids = [
                candidate for candidate in (str(value) for value in descriptor.child_ids) if candidate != child_id
            ]
            if (
                metadata.get("split_child_memory_ids") != [str(value) for value in descriptor.child_ids]
                or metadata.get("split_sibling_memory_ids") != sibling_ids
            ):
                return self._finish(current, CurationReceiptState.VERIFICATION_FAILED, "split_child_metadata_missing")
            if not _has_exact_edge(
                contexts[child_id],
                source_id=child_id,
                target_id=str(descriptor.target_id),
                link_type=_SPLIT_LINK_TYPE,
                edge_context=_SPLIT_LINK_CONTEXT,
            ):
                return self._finish(current, CurationReceiptState.VERIFICATION_FAILED, "split_edge_missing")
        return self._finish(current, CurationReceiptState.VERIFIED, None)

    def _verify_record_action(
        self,
        current: CurationActionReceipt,
        expected_ids: Sequence[str],
        target_status: str | None = None,
    ) -> CurationActionReceipt:
        try:
            contexts = self._fresh_contexts(expected_ids)
        except _PostconditionMismatch as mismatch:
            return self._finish(
                current,
                CurationReceiptState.VERIFICATION_FAILED,
                mismatch.error_code,
            )
        if _receipt_ids(current) != expected_ids:
            return self._finish(current, CurationReceiptState.VERIFICATION_FAILED, "affected_ids_mismatch")
        if target_status is not None and any(
            context.record.status != target_status for context in contexts.values()
        ):
            return self._finish(current, CurationReceiptState.VERIFICATION_FAILED, "status_mismatch")
        actual_after_token = _state_token(
            {memory_id: context.record for memory_id, context in contexts.items()},
            expected_ids,
        )
        if actual_after_token != current.after_token:
            return self._finish(current, CurationReceiptState.VERIFICATION_FAILED, "after_token_mismatch")
        return self._finish(current, CurationReceiptState.VERIFIED, None)

    def _verify_create_link(
        self,
        current: CurationActionReceipt,
        action: CreateLinkAction,
        expected_ids: Sequence[str],
    ) -> CurationActionReceipt:
        try:
            contexts = self._fresh_contexts(expected_ids)
        except _PostconditionMismatch as mismatch:
            return self._finish(current, CurationReceiptState.VERIFICATION_FAILED, mismatch.error_code)
        if _receipt_ids(current) != expected_ids:
            return self._finish(current, CurationReceiptState.VERIFICATION_FAILED, "affected_ids_mismatch")
        actual_after_token = _state_token(
            {memory_id: context.record for memory_id, context in contexts.items()},
            expected_ids,
        )
        if actual_after_token != current.after_token:
            return self._finish(current, CurationReceiptState.VERIFICATION_FAILED, "after_token_mismatch")
        if not _has_exact_edge(
            contexts[str(action.source_id)],
            source_id=str(action.source_id),
            target_id=str(action.target_id),
            link_type=action.link_type,
            edge_context=action.context or "",
        ):
            return self._finish(current, CurationReceiptState.VERIFICATION_FAILED, "edge_missing")
        return self._finish(current, CurationReceiptState.VERIFIED, None)

    def _verify_remove_link(
        self,
        current: CurationActionReceipt,
        action: RemoveLinkAction,
        expected_ids: Sequence[str],
    ) -> CurationActionReceipt:
        try:
            contexts = self._fresh_contexts(expected_ids)
        except _PostconditionMismatch as mismatch:
            return self._finish(current, CurationReceiptState.VERIFICATION_FAILED, mismatch.error_code)
        if _receipt_ids(current) != expected_ids:
            return self._finish(current, CurationReceiptState.VERIFICATION_FAILED, "affected_ids_mismatch")
        actual_after_token = _state_token(
            {memory_id: context.record for memory_id, context in contexts.items()},
            expected_ids,
        )
        if actual_after_token != current.after_token:
            return self._finish(current, CurationReceiptState.VERIFICATION_FAILED, "after_token_mismatch")
        if _has_any_edge(
            contexts[str(action.source_id)],
            source_id=str(action.source_id),
            target_id=str(action.target_id),
            link_type=action.link_type,
        ):
            return self._finish(current, CurationReceiptState.VERIFICATION_FAILED, "edge_still_present")
        return self._finish(current, CurationReceiptState.VERIFIED, None)

    def _verify_merge(
        self,
        current: CurationActionReceipt,
        action: MergeMemoriesAction,
        expected_ids: Sequence[str],
    ) -> CurationActionReceipt:
        try:
            contexts = self._fresh_contexts(expected_ids)
        except _PostconditionMismatch as mismatch:
            return self._finish(current, CurationReceiptState.VERIFICATION_FAILED, mismatch.error_code)
        if _receipt_ids(current) != expected_ids:
            return self._finish(current, CurationReceiptState.VERIFICATION_FAILED, "affected_ids_mismatch")
        actual_after_token = _state_token(
            {memory_id: context.record for memory_id, context in contexts.items()},
            expected_ids,
        )
        if actual_after_token != current.after_token:
            return self._finish(current, CurationReceiptState.VERIFICATION_FAILED, "after_token_mismatch")

        canonical_context = contexts[str(action.canonical_id)]
        merged_source_ids = sorted(
            {str(source_id) for source_id in action.source_ids if str(source_id) != str(action.canonical_id)},
            key=lambda value: value.encode("utf-8"),
        )
        merged_metadata = canonical_context.record.metadata.get("merged_source_ids")
        if not isinstance(merged_metadata, list):
            return self._finish(current, CurationReceiptState.VERIFICATION_FAILED, "merge_lineage_missing")
        merged_metadata_ids = {str(value) for value in merged_metadata}
        if not set(merged_source_ids).issubset(merged_metadata_ids):
            return self._finish(current, CurationReceiptState.VERIFICATION_FAILED, "merge_lineage_missing")
        for source_id in merged_source_ids:
            source_context = contexts[str(source_id)]
            if source_context.record.status != "archived":
                return self._finish(current, CurationReceiptState.VERIFICATION_FAILED, "source_not_archived")
            if not _has_exact_edge(
                canonical_context,
                source_id=str(action.canonical_id),
                target_id=str(source_id),
                link_type="SUPERSEDES",
                edge_context=_MERGE_LINK_CONTEXT,
            ):
                return self._finish(current, CurationReceiptState.VERIFICATION_FAILED, "merge_edge_missing")
        return self._finish(current, CurationReceiptState.VERIFIED, None)

    def _verify_split(
        self,
        current: CurationActionReceipt,
        action: SplitMemoryAction,
    ) -> CurationActionReceipt:
        original = self._maintenance_reads.peek_memory(str(action.target_id))
        if original is None or original.record.id != str(action.target_id):
            return self._finish(current, CurationReceiptState.VERIFICATION_FAILED, "record_missing")
        child_ids = _split_child_ids(original)
        expected_ids = sorted(
            [str(action.target_id), *child_ids],
            key=lambda value: value.encode("utf-8"),
        )
        try:
            contexts = self._fresh_contexts(expected_ids)
        except _PostconditionMismatch as mismatch:
            return self._finish(current, CurationReceiptState.VERIFICATION_FAILED, mismatch.error_code)
        if _receipt_ids(current) != expected_ids:
            return self._finish(current, CurationReceiptState.VERIFICATION_FAILED, "affected_ids_mismatch")
        actual_after_token = _state_token(
            {memory_id: context.record for memory_id, context in contexts.items()},
            expected_ids,
        )
        if actual_after_token != current.after_token:
            return self._finish(current, CurationReceiptState.VERIFICATION_FAILED, "after_token_mismatch")

        original_context = contexts[str(action.target_id)]
        if original_context.record.metadata.get("split_child_memory_ids") != child_ids:
            return self._finish(current, CurationReceiptState.VERIFICATION_FAILED, "split_children_missing")
        if original_context.record.metadata.get("split_child_count") != len(child_ids):
            return self._finish(current, CurationReceiptState.VERIFICATION_FAILED, "split_child_count_missing")
        for index, child_id in enumerate(child_ids, start=1):
            child_context = contexts[child_id]
            metadata = child_context.record.metadata
            if metadata.get("split_from_memory_id") != str(action.target_id):
                return self._finish(current, CurationReceiptState.VERIFICATION_FAILED, "split_lineage_missing")
            if metadata.get("split_group_id") != str(action.action_id):
                return self._finish(current, CurationReceiptState.VERIFICATION_FAILED, "split_group_missing")
            if metadata.get("split_part_index") != index or metadata.get("split_part_count") != len(child_ids):
                return self._finish(current, CurationReceiptState.VERIFICATION_FAILED, "split_part_metadata_missing")
            sibling_ids = [candidate for candidate in child_ids if candidate != child_id]
            if metadata.get("split_child_memory_ids") != child_ids or metadata.get("split_sibling_memory_ids") != sibling_ids:
                return self._finish(current, CurationReceiptState.VERIFICATION_FAILED, "split_child_metadata_missing")
            if not _has_exact_edge(
                child_context,
                source_id=child_id,
                target_id=str(action.target_id),
                link_type=_SPLIT_LINK_TYPE,
                edge_context=_SPLIT_LINK_CONTEXT,
            ):
                return self._finish(current, CurationReceiptState.VERIFICATION_FAILED, "split_edge_missing")
        return self._finish(current, CurationReceiptState.VERIFIED, None)

    def verify_receipt(
        self,
        receipt: CurationActionReceipt,
        action: VerifiedCurationAction,
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
    action: VerifiedCurationAction,
) -> CurationActionReceipt:
    """Functional entry point for authoritative typed receipt verification."""
    return CurationVerifier(curation_store, maintenance_reads).verify(receipt, action)


def _merge_ids(canonical_id: Any, source_ids: Sequence[Any]) -> list[str]:
    values = [str(canonical_id), *[str(source_id) for source_id in source_ids if str(source_id) != str(canonical_id)]]
    return sorted(dict.fromkeys(values), key=lambda value: value.encode("utf-8"))


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


def _has_any_edge(
    read_context: RelationalMemoryReadContext,
    *,
    source_id: str,
    target_id: str,
    link_type: str,
) -> bool:
    normalized_type = _normalize_link_type(link_type)
    return any(
        edge.source_id == source_id and edge.target_id == target_id and edge.link_type == normalized_type
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


def _split_child_ids(read_context: RelationalMemoryReadContext) -> list[str]:
    raw_ids = read_context.record.metadata.get("split_child_memory_ids")
    if not isinstance(raw_ids, list) or not raw_ids:
        raise _PostconditionMismatch("split_children_missing")
    child_ids = [str(value) for value in raw_ids]
    if len(child_ids) != len({*child_ids}):
        raise _PostconditionMismatch("split_children_missing")
    return child_ids


__all__ = [
    "CurationVerificationError",
    "CurationVerifier",
    "verify_curation_receipt",
]
