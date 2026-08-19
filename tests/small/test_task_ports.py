from mcp_memory.core.ports.tasks import (
    TaskCancellationPort,
    TaskDataMutationPort,
    TaskLifecyclePort,
    TaskProcessSupervisionPort,
    TaskReportingPort,
    TaskSubmissionPort,
)
from mcp_memory.core.tasks import SQLiteTaskQueue


def test_sqlite_task_queue_satisfies_each_task_consumer_port(db_manager) -> None:
    """
    The concrete queue remains compatible with every consumer capability.

    Each assertion exercises the structural contract used by one task
    consumer category without requiring the aggregate protocol at the call
    site.
    """
    queue = SQLiteTaskQueue(db_manager)

    assert isinstance(queue, TaskSubmissionPort)
    assert isinstance(queue, TaskLifecyclePort)
    assert isinstance(queue, TaskProcessSupervisionPort)
    assert isinstance(queue, TaskCancellationPort)
    assert isinstance(queue, TaskDataMutationPort)
    assert isinstance(queue, TaskReportingPort)
