from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest

from mcp_memory.context import ApplicationContext
from mcp_memory.core.journal import JournalEntry
from mcp_memory.core.ports.tasks import TaskRecord
from mcp_memory.core.task_handlers import ingest as ingest_module
from mcp_memory.core.task_handlers.ingest import _persist_ingress_batch_evidence, process_ingest_batch

pytestmark = pytest.mark.small


class _EvidenceStore:
    def __init__(self, events: list[str] | None = None) -> None:
        self.saved = []
        self.events = events

    def save(self, evidence) -> None:
        self.saved.append(evidence)
        if self.events is not None:
            self.events.append("evidence")

    def get(self, batch_id: str):
        return next((item for item in self.saved if item.batch_id == batch_id), None)

    def list_for_execution(self, task_id: str, execution_epoch: int):
        return [
            item
            for item in self.saved
            if item.task_id == task_id and item.execution_epoch == execution_epoch
        ]


class _Journal:
    def __init__(self, events: list[str] | None = None) -> None:
        self.recovered = []
        self.events = events

    def release_claims(self, task_id: str) -> list[int]:
        if self.events is not None:
            self.events.append("release")
        return []

    def move_claims_to_recoverable(self, task_id: str) -> None:
        self.recovered.append(task_id)


def _task() -> SimpleNamespace:
    return SimpleNamespace(id="task-1", execution_epoch=4, data={})


def _entries() -> list[JournalEntry]:
    return [
        JournalEntry(2, "second thought", "workspace-b", 2.0, "claimed"),
        JournalEntry(1, "first thought", None, 1.0, "claimed"),
    ]


def test_persist_ingress_batch_evidence_captures_deterministic_source() -> None:
    """A claimed batch stores normalized replay evidence before processing."""
    store = _EvidenceStore()
    ctx = SimpleNamespace(
        config=SimpleNamespace(ingress_evidence_mode="shadow"),
        ingress_batch_evidence=store,
        journal=_Journal(),
    )

    _persist_ingress_batch_evidence(
        cast(ApplicationContext, ctx),
        cast(TaskRecord, _task()),
        SimpleNamespace(_budget_key="route-a"),
        workspace_id="workspace-a",
        grouping_strategy="fifo",
        entries=_entries(),
        batch_sequence=None,
    )

    evidence = store.saved[0]
    assert evidence.batch_sequence == 1
    assert evidence.claimed_entry_ids == ("1", "2")
    assert [entry.entry_id for entry in evidence.source_entries] == ["1", "2"]
    assert evidence.source_entries[0].workspace_ids == ("workspace-a",)
    assert evidence.source_entries[0].snapshot == {"content": "first thought"}
    assert evidence.provider_route == "route-a"
    assert evidence.execution_mode == "provider_analysis"
    assert evidence.source_fingerprint
    assert evidence.batch_id


@pytest.mark.asyncio
async def test_process_ingest_batch_persists_evidence_before_analysis(monkeypatch) -> None:
    """Batch evidence is durable before provider analysis or fallback work starts."""
    events: list[str] = []
    store = _EvidenceStore(events)
    ctx = SimpleNamespace(
        config=SimpleNamespace(ingress_evidence_mode="shadow"),
        ingress_batch_evidence=store,
        journal=_Journal(events),
    )

    async def analyze(*args) -> tuple[list[dict[str, object]], int]:
        events.append("analysis")
        return [], 0

    monkeypatch.setattr(ingest_module, "_build_ingest_groups", lambda *args, **kwargs: [_entries()])
    monkeypatch.setattr(ingest_module, "_fallback_ingest_entries", lambda *args: ([], [], 0, []))
    await process_ingest_batch(
        cast(ApplicationContext, ctx),
        cast(TaskRecord, _task()),
        SimpleNamespace(_budget_key="route-a"),
        workspace_id="workspace-a",
        grouping_strategy="fifo",
        analyze_ingest_actions=analyze,
        entries=_entries(),
    )

    assert events == ["evidence", "analysis", "release"]


def test_shadow_ingress_evidence_failure_preserves_legacy_processing() -> None:
    """Shadow evidence failures do not interrupt the existing ingest path."""
    ctx = SimpleNamespace(
        config=SimpleNamespace(ingress_evidence_mode="shadow"),
        ingress_batch_evidence=None,
        journal=_Journal(),
    )

    _persist_ingress_batch_evidence(
        cast(ApplicationContext, ctx),
        cast(TaskRecord, _task()),
        None,
        workspace_id="workspace-a",
        grouping_strategy="fifo",
        entries=_entries(),
        batch_sequence=2,
    )

    assert ctx.journal.recovered == []


def test_enforce_ingress_evidence_failure_recovers_claims_and_raises() -> None:
    """Enforce mode keeps claims recoverable when evidence admission fails."""
    ctx = SimpleNamespace(
        config=SimpleNamespace(ingress_evidence_mode="enforce"),
        ingress_batch_evidence=None,
        journal=_Journal(),
    )

    with pytest.raises(RuntimeError, match="ingress_evidence_unavailable"):
        _persist_ingress_batch_evidence(
            cast(ApplicationContext, ctx),
            cast(TaskRecord, _task()),
            None,
            workspace_id="workspace-a",
            grouping_strategy="fifo",
            entries=_entries(),
            batch_sequence=2,
        )

    assert ctx.journal.recovered == ["task-1"]


def test_agentic_claim_persists_evidence_with_claim_metadata(monkeypatch) -> None:
    """Agentic claims store the normalized batch snapshot and execution metadata."""
    store = _EvidenceStore()
    task = _task()
    task.data = {"batch_sequence": 3, "policy_version": "policy-2", "schema_version": "schema-4"}
    journal = SimpleNamespace(
        claim_pending=lambda **kwargs: _entries(),
        count_by_status=lambda **kwargs: {"pending": 0},
    )
    queue = SimpleNamespace(get_task=lambda task_id: task)
    ctx = SimpleNamespace(
        config=SimpleNamespace(ingress_evidence_mode="shadow"),
        ingress_batch_evidence=store,
        journal=journal,
        task_queue=queue,
        workspace_id="workspace-a",
        embedder=None,
        vector_store=None,
    )
    monkeypatch.setattr(ingest_module, "build_ingest_groups", lambda *args, **kwargs: [])
    monkeypatch.setattr(ingest_module, "_record_ingest_tool_invocation", lambda *args, **kwargs: None)

    payload = ingest_module.build_next_ingest_batch_payload(cast(ApplicationContext, ctx), {"task_id": "task-1"})

    evidence = store.saved[0]
    assert payload["claimed_entry_ids"] == [2, 1]
    assert evidence.execution_epoch == 4
    assert evidence.batch_sequence == 3
    assert evidence.grouping_strategy == "lexical-seeded"
    assert evidence.grouping_fallback_reason is None
    assert evidence.provider_route == "agentic_mcp"
    assert evidence.execution_mode == "agentic_mcp"
    assert evidence.policy_version == "policy-2"
    assert evidence.schema_version == "schema-4"


def test_agentic_claim_retry_does_not_save_duplicate_evidence(monkeypatch) -> None:
    """Retrying an identical claim reuses the stable batch evidence identity."""
    store = _EvidenceStore()
    task = _task()
    journal = SimpleNamespace(
        claim_pending=lambda **kwargs: _entries(),
        count_by_status=lambda **kwargs: {"pending": 0},
    )
    ctx = SimpleNamespace(
        config=SimpleNamespace(ingress_evidence_mode="shadow"),
        ingress_batch_evidence=store,
        journal=journal,
        task_queue=SimpleNamespace(get_task=lambda task_id: task),
        workspace_id="workspace-a",
        embedder=None,
        vector_store=None,
    )
    monkeypatch.setattr(ingest_module, "build_ingest_groups", lambda *args, **kwargs: [])
    monkeypatch.setattr(ingest_module, "_record_ingest_tool_invocation", lambda *args, **kwargs: None)

    for _ in range(2):
        ingest_module.build_next_ingest_batch_payload(cast(ApplicationContext, ctx), {"task_id": "task-1"})

    assert len(store.saved) == 1


@pytest.mark.asyncio
async def test_deterministic_enforce_mode_blocks_before_mutation() -> None:
    """Deterministic ingress stays blocked until atomic mutation coordination exists."""
    store = _EvidenceStore()
    ctx = SimpleNamespace(
        config=SimpleNamespace(ingress_evidence_mode="enforce"),
        ingress_batch_evidence=store,
        journal=_Journal(),
        embedder=None,
        vector_store=None,
    )

    with pytest.raises(RuntimeError, match="atomic_boundary_required"):
        await process_ingest_batch(
            cast(ApplicationContext, ctx),
            cast(TaskRecord, _task()),
            None,
            workspace_id="workspace-a",
            grouping_strategy="fifo",
            analyze_ingest_actions=pytest.fail,
            entries=_entries(),
        )

    assert ctx.journal.recovered == ["task-1"]
