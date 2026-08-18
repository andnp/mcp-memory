from __future__ import annotations

import json
from uuid import uuid4

import pytest

from mcp_memory.context import ApplicationContext
from mcp_memory.core.curation_direct_mcp import _direct_quality_run
from mcp_memory.core.curator_evidence import (
    begin_tool_call,
    current_curator_execution,
    finalize_curator_execution,
    finish_tool_call,
    reset_curator_execution,
)
from mcp_memory.core.direct_mutation_evidence import DirectMutationEvidence
from mcp_memory.core.ports.tasks import TaskRecord
from mcp_memory.curation_action_store import SQLiteCurationActionStore
from mcp_memory.curation_store import CurationRunState, SQLiteCurationStore
from mcp_memory.mcp import transport
from mcp_memory.mcp.curator_action_adapter import execute_curator_mutation
from mcp_memory.mcp.internal_mutation_services import (
    internal_create_memory_link_service,
    internal_update_memory_record_service,
)
from mcp_memory.relational.repository import RelationalMemoryRepository
from mcp_memory.storage.direct_mutation_evidence_store import SQLiteDirectMutationEvidenceStore


def _task() -> TaskRecord:
    return TaskRecord(
        id="adapter-task",
        task_name="memory_curator",
        data={},
        workspace_id="workspace",
        status="running",
        priority=100,
        retries_count=0,
        max_retries=3,
        created_at=0.0,
        updated_at=0.0,
        available_at=0.0,
        claimed_at=None,
        started_at=None,
        completed_at=None,
        last_error=None,
        execution_epoch=0,
    )


def test_curator_action_replay_returns_one_canonical_receipt(db_manager) -> None:
    """Replay the same direct action without a second domain mutation.

    The receipt identity is reused while the transaction callback runs once.
    """
    repository = RelationalMemoryRepository(db_manager)
    record = repository.create_memory(
        title="Adapter target",
        content="This record exercises the curator action adapter.",
        workspace_ids=["workspace"],
        summary="Adapter target summary.",
        memory_type="fact",
    )
    assert record is not None
    task = _task()
    run = _direct_quality_run(task)
    curation = SQLiteCurationStore(db_manager)
    curation.create_run(run)
    planning = run.model_copy(update={"state": CurationRunState.PLANNING})
    curation.transition_run(run.run_id, CurationRunState.CREATED, planning)
    executing = planning.model_copy(update={"state": CurationRunState.EXECUTING})
    curation.transition_run(run.run_id, CurationRunState.PLANNING, executing)
    ctx = ApplicationContext(
        repository=repository,
        curation_action_store=SQLiteCurationActionStore(db_manager),
        curation=curation,
    )
    arguments = {
        "memory_id": record.id,
        "summary": "A specific adapter conclusion.",
        "task_id": task.id,
    }
    evidence = DirectMutationEvidence.start(
        task_id=task.id,
        execution_epoch=task.execution_epoch,
        session_id=None,
        call_id="fixed-call-id",
        sequence=1,
        tool_name="internal_update_memory_record",
        arguments=arguments,
    )
    reset_curator_execution(task.id, execution_epoch=task.execution_epoch, run_id=run.run_id)
    try:
        first = execute_curator_mutation(
            ctx,
            "internal_update_memory_record",
            arguments,
            internal_update_memory_record_service,
            evidence,
        )
        replay = execute_curator_mutation(
            ctx,
            "internal_update_memory_record",
            arguments,
            internal_update_memory_record_service,
            evidence,
        )
    finally:
        finalize_curator_execution()

    assert replay["curation_receipt"] == first["curation_receipt"]
    connection = db_manager.get_connection()
    assert connection.execute("SELECT COUNT(*) FROM memory_mutation_events").fetchone()[0] == 1
    assert connection.execute("SELECT COUNT(*) FROM memory_record_revisions").fetchone()[0] == 1
    refreshed = repository.get_memory(record.id)
    assert refreshed is not None
    assert refreshed.summary == "A specific adapter conclusion."


@pytest.mark.asyncio
async def test_dispatch_replay_reuses_direct_action_identity(db_manager) -> None:
    """Retrying one dispatch reuses its receipt without reapplying the write."""
    repository = RelationalMemoryRepository(db_manager)
    record = repository.create_memory(
        title="Dispatch target",
        content="This record exercises dispatch-level replay.",
        workspace_ids=["workspace"],
        summary="Dispatch target summary.",
        memory_type="fact",
    )
    assert record is not None
    task = _task()
    run = _direct_quality_run(task)
    curation = SQLiteCurationStore(db_manager)
    curation.create_run(run)
    planning = run.model_copy(update={"state": CurationRunState.PLANNING})
    curation.transition_run(run.run_id, CurationRunState.CREATED, planning)
    executing = planning.model_copy(update={"state": CurationRunState.EXECUTING})
    curation.transition_run(run.run_id, CurationRunState.PLANNING, executing)
    evidence_store = SQLiteDirectMutationEvidenceStore(db_manager)
    ctx = ApplicationContext(
        repository=repository,
        curation_action_store=SQLiteCurationActionStore(db_manager),
        curation=curation,
        direct_mutation_evidence=evidence_store,
    )
    arguments = {
        "memory_id": record.id,
        "summary": "A replayable dispatch conclusion.",
        "task_id": task.id,
    }

    async def dispatch_once() -> dict:
        reset_curator_execution(task.id, execution_epoch=task.execution_epoch, run_id=run.run_id)
        try:
            response = await transport.dispatch_internal_memory_tool(
                ctx,
                "internal_update_memory_record",
                arguments,
            )
        finally:
            finalize_curator_execution()
        return json.loads(response[0].text)

    first = await dispatch_once()
    replay = await dispatch_once()

    assert replay["curation_receipt"] == first["curation_receipt"]
    assert len(evidence_store.list_for_execution(task.id, task.execution_epoch)) == 1
    connection = db_manager.get_connection()
    assert connection.execute("SELECT COUNT(*) FROM memory_mutation_events").fetchone()[0] == 1


def test_direct_evidence_identity_includes_arguments_and_sequence() -> None:
    """Distinct direct calls do not collapse when arguments or sequence changes."""
    common = {
        "task_id": "task-1",
        "execution_epoch": 2,
        "session_id": "session-1",
        "call_id": "dispatch-call",
        "tool_name": "internal_update_memory_record",
    }
    first = DirectMutationEvidence.start(
        **common,
        sequence=1,
        arguments={"memory_id": "memory-1", "summary": "first"},
    )
    changed_arguments = DirectMutationEvidence.start(
        **common,
        sequence=1,
        arguments={"memory_id": "memory-1", "summary": "second"},
    )
    changed_sequence = DirectMutationEvidence.start(
        **common,
        sequence=2,
        arguments={"memory_id": "memory-1", "summary": "first"},
    )

    assert first.evidence_id != changed_arguments.evidence_id
    assert first.evidence_id != changed_sequence.evidence_id


def test_curator_adapter_preserves_external_link_targets(db_manager) -> None:
    """External link targets remain service-owned instead of transaction targets."""
    repository = RelationalMemoryRepository(db_manager)
    source = repository.create_memory(
        title="External source",
        content="This record links to an external document.",
        workspace_ids=["workspace"],
        summary="External source summary.",
        memory_type="fact",
    )
    assert source is not None
    task = _task()
    run = _direct_quality_run(task)
    curation = SQLiteCurationStore(db_manager)
    curation.create_run(run)
    planning = run.model_copy(update={"state": CurationRunState.PLANNING})
    curation.transition_run(run.run_id, CurationRunState.CREATED, planning)
    executing = planning.model_copy(update={"state": CurationRunState.EXECUTING})
    curation.transition_run(run.run_id, CurationRunState.PLANNING, executing)
    ctx = ApplicationContext(
        repository=repository,
        curation_action_store=SQLiteCurationActionStore(db_manager),
        curation=curation,
    )
    arguments = {
        "source_id": source.id,
        "target_id": "ext:docs/plan.md",
        "link_type": "REFERENCES",
        "context": "The source references the external plan.",
    }
    evidence = DirectMutationEvidence.start(
        task_id=task.id,
        execution_epoch=task.execution_epoch,
        session_id=None,
        call_id="external-link-call",
        sequence=1,
        tool_name="internal_create_memory_link",
        arguments=arguments,
    )
    reset_curator_execution(task.id, execution_epoch=task.execution_epoch, run_id=run.run_id)
    try:
        result = execute_curator_mutation(
            ctx,
            "internal_create_memory_link",
            arguments,
            internal_create_memory_link_service,
            evidence,
        )
    finally:
        finalize_curator_execution()

    assert result["link"]["target_id"] == "ext:docs/plan.md"
    assert repository.get_links(source.id, direction="outgoing")[0].target_id == "ext:docs/plan.md"


def test_curator_adapter_uses_isolated_transaction_context(db_manager) -> None:
    """Transaction services receive an isolated context without changing the caller."""
    repository = RelationalMemoryRepository(db_manager)
    record = repository.create_memory(
        title="Context target",
        content="This record exercises transaction context isolation.",
        workspace_ids=["workspace"],
        summary="Context target summary.",
        memory_type="fact",
    )
    assert record is not None
    task = _task()
    run = _direct_quality_run(task)
    curation = SQLiteCurationStore(db_manager)
    curation.create_run(run)
    planning = run.model_copy(update={"state": CurationRunState.PLANNING})
    curation.transition_run(run.run_id, CurationRunState.CREATED, planning)
    executing = planning.model_copy(update={"state": CurationRunState.EXECUTING})
    curation.transition_run(run.run_id, CurationRunState.PLANNING, executing)
    task_queue = object()
    ctx = ApplicationContext(
        repository=repository,
        task_queue=task_queue,
        curation_action_store=SQLiteCurationActionStore(db_manager),
        curation=curation,
    )
    observed: list[ApplicationContext] = []

    def service(service_ctx: ApplicationContext, arguments: dict) -> dict:
        observed.append(service_ctx)
        return internal_update_memory_record_service(service_ctx, arguments)

    arguments = {"memory_id": record.id, "summary": "Isolated context conclusion."}
    evidence = DirectMutationEvidence.start(
        task_id=task.id,
        execution_epoch=task.execution_epoch,
        session_id=None,
        call_id="isolated-context-call",
        sequence=1,
        tool_name="internal_update_memory_record",
        arguments=arguments,
    )
    reset_curator_execution(task.id, execution_epoch=task.execution_epoch, run_id=run.run_id)
    try:
        execute_curator_mutation(ctx, "internal_update_memory_record", arguments, service, evidence)
    finally:
        finalize_curator_execution()

    assert observed
    assert observed[0] is not ctx
    assert observed[0].task_queue is None
    assert ctx.repository is repository
    assert ctx.task_queue is task_queue


@pytest.mark.asyncio
async def test_non_curator_dispatch_allows_unpersisted_task_id() -> None:
    """Non-curator dispatches keep working for ad-hoc task identifiers."""
    class MissingTaskQueue:
        def get_task(self, task_id: str):
            raise ValueError(f"Task {task_id} was not found")

    ctx = ApplicationContext(task_queue=MissingTaskQueue())

    def service(_ctx: ApplicationContext, _arguments: dict) -> dict:
        return {"status": "ok"}

    result = await transport._dispatch_tool(
        ctx,
        "internal_peek_record",
        {"task_id": "adhoc-task"},
        service_resolver=lambda: {"internal_peek_record": service},
        on_success=transport._record_internal_tool_call,
    )

    assert json.loads(result[0].text) == {"status": "ok"}


@pytest.mark.asyncio
async def test_curator_dispatch_requires_epoch_for_missing_task() -> None:
    """Curator dispatches reject a task switch without authoritative identity."""
    class MissingTaskQueue:
        def get_task(self, task_id: str):
            raise ValueError(f"Task {task_id} was not found")

    reset_curator_execution("curator-task", execution_epoch=3)
    try:
        with pytest.raises(ValueError, match="execution_epoch_unavailable"):
            await transport._dispatch_tool(
                ApplicationContext(task_queue=MissingTaskQueue()),
                "internal_peek_record",
                {"task_id": "missing-task"},
                service_resolver=lambda: {"internal_peek_record": lambda _ctx, _args: {"status": "ok"}},
                on_success=transport._record_internal_tool_call,
            )
    finally:
        finalize_curator_execution()


def test_switching_curator_tasks_clears_run_and_sequence() -> None:
    """A task switch starts a fresh identity sequence without the old run."""
    run_id = uuid4()
    reset_curator_execution("first-task", execution_epoch=4, run_id=run_id)
    first_token = begin_tool_call()
    finish_tool_call(first_token)

    switched_token = begin_tool_call(task_id="second-task")
    switched = current_curator_execution()
    finish_tool_call(switched_token)
    finalize_curator_execution()

    assert switched is not None
    assert switched.run_id is None
    assert switched.sequence == 1
    assert switched.execution_epoch == 0
