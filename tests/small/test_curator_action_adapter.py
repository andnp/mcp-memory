from __future__ import annotations

from mcp_memory.context import ApplicationContext
from mcp_memory.core.curation_direct_mcp import _direct_quality_run
from mcp_memory.core.curator_evidence import finalize_curator_execution, reset_curator_execution
from mcp_memory.core.direct_mutation_evidence import DirectMutationEvidence
from mcp_memory.core.ports.tasks import TaskRecord
from mcp_memory.curation_action_store import SQLiteCurationActionStore
from mcp_memory.curation_store import CurationRunState, SQLiteCurationStore
from mcp_memory.mcp.curator_action_adapter import execute_curator_mutation
from mcp_memory.mcp.internal_mutation_services import internal_update_memory_record_service
from mcp_memory.relational.repository import RelationalMemoryRepository


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
