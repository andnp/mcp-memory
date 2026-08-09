"""Backend-neutral persistence contracts for ingress evidence."""

from __future__ import annotations

from typing import Protocol

from mcp_memory.core.ingress_evidence import (
    IngressActionReceipt,
    IngressBatchEvidence,
    SourceCoverage,
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
    def save(self, coverage: SourceCoverage) -> SourceCoverage: ...

    def get(self, entry_id: str) -> SourceCoverage | None: ...

    def list_for_entries(self, entry_ids: tuple[str, ...]) -> list[SourceCoverage]: ...
