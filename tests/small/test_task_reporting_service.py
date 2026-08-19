import pytest

from mcp_memory.management.models import AgentRunHistoryListPayload, AgentRunHistoryPayload
from mcp_memory.utils.sql_portable_runner import SQLiteManagementQueryAdapter
from mcp_memory.management.task_reporting_service import (
    TaskReportingService,
    TaskReportingServiceDependencies,
)

pytestmark = pytest.mark.small


def test_list_recent_agent_runs_passes_explicit_dependencies_and_arguments(monkeypatch) -> None:
    expected_runs = [
        AgentRunHistoryPayload(
            task_name="memory-curator",
            status="completed",
            started_at=1.0,
            completed_at=2.0,
            duration_seconds=1.0,
        )
    ]
    seen: dict[str, object] = {}

    query_adapter = SQLiteManagementQueryAdapter()

    def fake_build_recent_agent_runs(db_manager, workspace_id, *, limit, detail_level, query_adapter):
        seen.update(
            db_manager=db_manager,
            workspace_id=workspace_id,
            limit=limit,
            detail_level=detail_level,
            query_adapter=query_adapter,
        )
        return expected_runs

    monkeypatch.setattr(
        "mcp_memory.management.task_reporting_service.build_recent_agent_runs",
        fake_build_recent_agent_runs,
    )
    db_manager = object()
    service = TaskReportingService(
        TaskReportingServiceDependencies(
            db_manager=db_manager,
            workspace_id="workspace-a",
            query_adapter=query_adapter,
        )
    )

    payload = service.list_recent_agent_runs(limit=7, detail_level="full")

    assert isinstance(payload, AgentRunHistoryListPayload)
    assert payload.runs == expected_runs
    assert seen == {
        "db_manager": db_manager,
        "workspace_id": "workspace-a",
        "limit": 7,
        "detail_level": "full",
        "query_adapter": query_adapter,
    }


def test_get_task_sampling_summary_uses_recent_runs(monkeypatch) -> None:
    expected_runs = [
        AgentRunHistoryPayload(
            task_name="memory-curator",
            status="completed",
            started_at=1.0,
            completed_at=2.0,
            duration_seconds=1.0,
        )
    ]
    expected_summary = object()
    seen: dict[str, object] = {}
    service = TaskReportingService(
        TaskReportingServiceDependencies(db_manager=object(), workspace_id=None)
    )

    def fake_list_recent_agent_runs(*, limit, detail_level="compact"):
        seen.update(limit=limit, detail_level=detail_level)
        return AgentRunHistoryListPayload(runs=expected_runs)

    def fake_build_task_sampling_summary(runs):
        seen["runs"] = runs
        return expected_summary

    monkeypatch.setattr(service, "list_recent_agent_runs", fake_list_recent_agent_runs)
    monkeypatch.setattr(
        "mcp_memory.management.task_reporting_service.build_task_sampling_summary",
        fake_build_task_sampling_summary,
    )

    assert service.get_task_sampling_summary(limit=9) is expected_summary
    assert seen == {"limit": 9, "detail_level": "compact", "runs": expected_runs}
