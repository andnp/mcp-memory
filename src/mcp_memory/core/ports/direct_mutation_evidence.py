"""Backend-neutral persistence contract for direct mutation evidence."""

from __future__ import annotations

from typing import Protocol

from mcp_memory.core.direct_mutation_evidence import DirectMutationEvidence


class DirectMutationEvidenceRepository(Protocol):
    def append(self, evidence: DirectMutationEvidence) -> DirectMutationEvidence: ...

    def save(self, evidence: DirectMutationEvidence) -> DirectMutationEvidence: ...

    def get(self, evidence_id: str) -> DirectMutationEvidence | None: ...

    def get_by_idempotency_key(self, idempotency_key: str) -> DirectMutationEvidence | None: ...

    def list_for_execution(
        self,
        task_id: str,
        execution_epoch: int,
    ) -> list[DirectMutationEvidence]: ...
