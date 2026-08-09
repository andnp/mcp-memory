"""Backend-neutral persistence contracts for ingress evidence."""

from __future__ import annotations

from typing import Protocol

from mcp_memory.core.ingress_evidence import (
    IngressActionReceipt,
    IngressBatchEvidence,
    SourceCoverage,
    SourceCoverageOutcome,
)


class IngressActionReceiptIdentityConflictError(ValueError):
    """Raised when an action ID is reused with a different payload digest."""

    def __init__(self, action_id: str, stored_digest: str, requested_digest: str) -> None:
        self.action_id = action_id
        self.stored_digest = stored_digest
        self.requested_digest = requested_digest
        super().__init__(
            f"ingress action receipt payload collision for {action_id!r}: "
            f"stored digest {stored_digest!r}, requested digest {requested_digest!r}"
        )


class SourceCoverageAssignmentConflictError(ValueError):
    """Raised when a source entry has a different terminal assignment."""

    def __init__(
        self,
        entry_id: str,
        stored_action_id: str | None,
        stored_outcome: SourceCoverageOutcome,
        requested_action_id: str | None,
        requested_outcome: SourceCoverageOutcome,
    ) -> None:
        self.entry_id = entry_id
        self.stored_action_id = stored_action_id
        self.stored_outcome = stored_outcome
        self.requested_action_id = requested_action_id
        self.requested_outcome = requested_outcome
        super().__init__(
            f"source coverage assignment conflict for {entry_id!r}: "
            f"stored ({stored_action_id!r}, {stored_outcome.value!r}), "
            f"requested ({requested_action_id!r}, {requested_outcome.value!r})"
        )


class IngressBatchEvidenceRepository(Protocol):
    def save(self, evidence: IngressBatchEvidence) -> IngressBatchEvidence: ...

    def get(self, batch_id: str) -> IngressBatchEvidence | None: ...

    def list_for_execution(self, task_id: str, execution_epoch: int) -> list[IngressBatchEvidence]: ...


class IngressActionReceiptRepository(Protocol):
    def reserve(self, receipt: IngressActionReceipt) -> IngressActionReceipt: ...

    def save(self, receipt: IngressActionReceipt) -> IngressActionReceipt: ...

    def get(self, action_id: str) -> IngressActionReceipt | None: ...

    def list_for_batch(self, batch_id: str) -> list[IngressActionReceipt]: ...


class SourceCoverageRepository(Protocol):
    def assign_terminal(self, coverage: SourceCoverage) -> SourceCoverage: ...

    def save(self, coverage: SourceCoverage) -> SourceCoverage: ...

    def get(self, entry_id: str) -> SourceCoverage | None: ...

    def list_for_entries(self, entry_ids: tuple[str, ...]) -> list[SourceCoverage]: ...
