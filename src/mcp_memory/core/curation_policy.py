"""Pure quality-policy decisions for curation actions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping
from uuid import UUID

from .curation_models import (
    ArchiveMemoryAction,
    ClaimManifest,
    CreateLinkAction,
    CurationAction,
    MergeMemoriesAction,
    NormalizeMemoryAction,
    RemoveLinkAction,
    RewriteMemoryAction,
    SplitMemoryAction,
)


class OperationRisk(StrEnum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    DISABLED = "disabled"


class PolicyMode(StrEnum):
    ENABLED = "enabled"
    SHADOW = "shadow"
    DISABLED = "disabled"


class RejectionCode(StrEnum):
    DELETE_NOT_SUPPORTED = "delete_not_supported"
    OPERATION_NOT_SUPPORTED = "operation_not_supported"
    EVIDENCE_REQUIRED = "evidence_required"
    CLAIM_MANIFEST_INCOMPLETE = "claim_manifest_incomplete"
    CONTRADICTORY_FACTS = "contradictory_facts"
    UNKNOWN_MEMORY_TYPE = "unknown_memory_type"
    LINK_TYPE_NOT_CANONICAL = "link_type_not_canonical"
    EMPTY_CONTENT = "empty_content"


@dataclass(frozen=True)
class PolicyDecision:
    operation: str
    risk: OperationRisk
    mode: PolicyMode
    rejection_codes: tuple[RejectionCode, ...] = ()

    @property
    def authorized(self) -> bool:
        return self.mode == PolicyMode.ENABLED and not self.rejection_codes


_SHADOW_OPERATIONS = {
    "rewrite_memory",
    "remove_link",
    "merge_memories",
    "split_memory",
    "archive_memory",
}


def classify_operation_risk(operation: str) -> OperationRisk:
    """Return the conservative risk class without inspecting storage."""
    if operation == "normalize_memory":
        return OperationRisk.LOW
    if operation == "create_link":
        return OperationRisk.MODERATE
    if operation in _SHADOW_OPERATIONS:
        return OperationRisk.HIGH
    if operation == "delete_memory":
        return OperationRisk.DISABLED
    return OperationRisk.DISABLED


def policy_mode(operation: str) -> PolicyMode:
    if operation in {"normalize_memory", "create_link"}:
        return PolicyMode.ENABLED
    if operation in _SHADOW_OPERATIONS:
        return PolicyMode.SHADOW
    return PolicyMode.DISABLED


def validate_claim_manifest(manifest: ClaimManifest, *, requires_mapping: bool = False) -> tuple[RejectionCode, ...]:
    """Check the manifest's deterministic preservation surface, not its truth."""
    has_disposition = bool(
        manifest.preserved_claims
        or manifest.transformed_claims
        or manifest.omitted_material
        or manifest.unresolved_tensions
    )
    if not has_disposition:
        return (RejectionCode.CLAIM_MANIFEST_INCOMPLETE,)
    if requires_mapping and (
        not manifest.source_mapping
        or any(not item.output.strip() or not item.source_memory_ids for item in manifest.source_mapping)
    ):
        return (RejectionCode.CLAIM_MANIFEST_INCOMPLETE,)
    if any(not claim.strip() for claim in manifest.preserved_claims + manifest.transformed_claims + manifest.unresolved_tensions):
        return (RejectionCode.CLAIM_MANIFEST_INCOMPLETE,)
    if any(not item.material.strip() for item in manifest.omitted_material):
        return (RejectionCode.CLAIM_MANIFEST_INCOMPLETE,)
    return ()


def evaluate_curation_action(
    action: CurationAction | object,
    *,
    memory_types: Mapping[UUID, str] | None = None,
    contradictory_memory_ids: set[UUID] | frozenset[UUID] = frozenset(),
) -> PolicyDecision:
    """Evaluate an already-typed action using only caller-supplied facts."""
    operation = action.get("operation", "") if isinstance(action, Mapping) else getattr(action, "operation", "")
    risk = classify_operation_risk(operation)
    mode = policy_mode(operation)
    codes: list[RejectionCode] = []
    if operation == "delete_memory":
        codes.append(RejectionCode.DELETE_NOT_SUPPORTED)
    elif not isinstance(
        action,
        (NormalizeMemoryAction, CreateLinkAction, RemoveLinkAction, RewriteMemoryAction, MergeMemoriesAction, SplitMemoryAction, ArchiveMemoryAction),
    ):
        codes.append(RejectionCode.OPERATION_NOT_SUPPORTED)

    if isinstance(action, (RewriteMemoryAction, MergeMemoriesAction, SplitMemoryAction, ArchiveMemoryAction)):
        if not action.evidence:
            codes.append(RejectionCode.EVIDENCE_REQUIRED)
        codes.extend(validate_claim_manifest(action.claim_manifest, requires_mapping=isinstance(action, (RewriteMemoryAction, MergeMemoriesAction, SplitMemoryAction))))
        if isinstance(action, (RewriteMemoryAction, MergeMemoriesAction)) and not action.content.strip():
            codes.append(RejectionCode.EMPTY_CONTENT)
    if isinstance(action, (RemoveLinkAction, CreateLinkAction)):
        if not action.evidence:
            codes.append(RejectionCode.EVIDENCE_REQUIRED)
        if not action.link_type or action.link_type != action.link_type.upper():
            codes.append(RejectionCode.LINK_TYPE_NOT_CANONICAL)

    types = memory_types or {}
    affected_ids = _affected_ids(action)
    if any(types.get(memory_id) not in {"journal", "observation", "fact", "reflection", "plan"} for memory_id in affected_ids):
        codes.append(RejectionCode.UNKNOWN_MEMORY_TYPE)
    if isinstance(action, MergeMemoriesAction):
        fact_ids = {memory_id for memory_id in affected_ids if types.get(memory_id) == "fact"}
        if len(fact_ids) > 1 and (fact_ids & set(contradictory_memory_ids) or not action.claim_manifest.unresolved_tensions):
            codes.append(RejectionCode.CONTRADICTORY_FACTS)
    return PolicyDecision(operation=operation, risk=risk, mode=mode, rejection_codes=tuple(dict.fromkeys(codes)))


def _affected_ids(action: object) -> set[UUID]:
    if isinstance(action, MergeMemoriesAction):
        return set(action.source_ids) | {action.canonical_id}
    if isinstance(action, (CreateLinkAction, RemoveLinkAction)):
        return {action.source_id, action.target_id}
    target_id = getattr(action, "target_id", None)
    if isinstance(target_id, UUID):
        return {target_id}
    return set()
