"""Backend-neutral persistence contracts for ingress evidence."""

from __future__ import annotations

from typing import Protocol

from mcp_memory.core.ingress_evidence import (
    IngressActionReceipt,
    IngressBatchEvidence,
    SourceCoverage,
)


class IngressBatchEvidenceRepository(Protocol):
    def save(self, evidence: IngressBatchEvidence) -> IngressBatchEvidence: ...

    def get(self, batch_id: str) -> IngressBatchEvidence | None: ...

    def list_for_execution(self, task_id: str, execution_epoch: int) -> list[IngressBatchEvidence]: ...


class IngressActionReceiptRepository(Protocol):
    def save(self, receipt: IngressActionReceipt) -> IngressActionReceipt: ...

    def get(self, action_id: str) -> IngressActionReceipt | None: ...

    def list_for_batch(self, batch_id: str) -> list[IngressActionReceipt]: ...


class SourceCoverageRepository(Protocol):
    def save(self, coverage: SourceCoverage) -> SourceCoverage: ...

    def get(self, entry_id: str) -> SourceCoverage | None: ...

    def list_for_entries(self, entry_ids: tuple[str, ...]) -> list[SourceCoverage]: ...
