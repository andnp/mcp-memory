"""Pure evidence records for the ingress execution and mutation boundaries."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from mcp_memory.core.curation_quality_policy import QualityOutcome


class IngressReceiptStatus(StrEnum):
    """Terminal state of an ingress mutation receipt."""

    APPLIED_UNVERIFIED = "applied_unverified"
    NO_OP = "no_op"
    STALE = "stale"
    FAILED = "failed"
    UNOBSERVED = "unobserved"


class SourceCoverageOutcome(StrEnum):
    """Terminal disposition for one claimed source entry."""

    CREATED = "created"
    APPENDED = "appended"
    MATCHED_EXISTING = "matched_existing"
    IGNORED = "ignored"
    NO_MUTATION = "no_mutation"
    UNOBSERVED = "unobserved"


@dataclass(frozen=True, slots=True)
class IngressQualityEvidence:
    """Observed quality disposition for one stable ingress action."""

    action_id: str
    disposition: QualityOutcome
    evaluated_at: str
    reason: str | None = None
    evaluator: str | None = None
    query_provenance: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.disposition in {QualityOutcome.UNOBSERVED, QualityOutcome.UNVERIFIED} and not self.reason:
            raise ValueError(f"{self.disposition.value} quality evidence requires a reason")


@dataclass(frozen=True, slots=True)
class IngressSourceSnapshot:
    """Replay-relevant metadata captured for one claimed source entry."""

    entry_id: str
    workspace_ids: tuple[str, ...]
    timestamp: str
    content_digest: str
    snapshot: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class IngressBatchEvidence:
    """Evidence for one claimed ingress batch and its execution context."""

    batch_id: str
    task_id: str
    execution_epoch: int
    batch_sequence: int
    claimed_entry_ids: tuple[str, ...]
    source_fingerprint: str
    source_entries: tuple[IngressSourceSnapshot, ...]
    provider_route: str
    execution_mode: str
    policy_version: str
    schema_version: str
    claimed_at: str
    finalized_at: str | None = None
    grouping_strategy: str | None = None
    grouping_fallback_reason: str | None = None


@dataclass(frozen=True, slots=True)
class IngressActionReceipt:
    """Receipt state for one logical ingress mutation or explicit no-op."""

    action_id: str
    batch_id: str
    operation: str
    entry_ids: tuple[str, ...]
    target_ids: tuple[str, ...]
    canonical_payload_digest: str
    status: IngressReceiptStatus
    mutation_evidence_id: str | None
    created_at: str
    terminalized_at: str | None = None
    error_code: str | None = None
    before_revision_tokens: Mapping[str, str] = field(default_factory=dict)
    after_revision_tokens: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SourceCoverage:
    """Terminal coverage outcome for one claimed source entry."""

    entry_id: str
    outcome: SourceCoverageOutcome
    action_id: str | None = None
    reason: str | None = None
