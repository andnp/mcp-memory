from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mcp_memory.management.agent_run_reporting import build_recent_agent_runs
from mcp_memory.management.models import (
    AgentRunHistoryListPayload,
    TaskSamplingSummaryPayload,
)
from mcp_memory.management.task_sampling_summary import build_task_sampling_summary


@dataclass(frozen=True)
class TaskReportingServiceDependencies:
    db_manager: Any
    workspace_id: str | None


class TaskReportingService:
    def __init__(self, dependencies: TaskReportingServiceDependencies) -> None:
        self._dependencies = dependencies

    def list_recent_agent_runs(
        self,
        *,
        limit: int = 20,
        detail_level: str = "compact",
    ) -> AgentRunHistoryListPayload:
        dependencies = self._dependencies
        return AgentRunHistoryListPayload(
            runs=build_recent_agent_runs(
                dependencies.db_manager,
                dependencies.workspace_id,
                limit=limit,
                detail_level=detail_level,
            )
        )

    def get_task_sampling_summary(self, *, limit: int = 50) -> TaskSamplingSummaryPayload:
        return build_task_sampling_summary(self.list_recent_agent_runs(limit=limit).runs)
