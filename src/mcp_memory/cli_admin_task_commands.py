from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import click


type AdminTaskListAction = Callable[[str | None, str | None, int, bool], None]
type AdminTaskRecentRunsAction = Callable[[str | None, int, bool], None]
type AdminTaskSamplingSummaryAction = Callable[[str | None, int, bool], None]
type AdminTaskCancelAction = Callable[[str, str | None, str, bool], None]
type AdminTaskShowAction = Callable[[str, str | None, bool], None]


@dataclass(frozen=True)
class AdminTaskCommandFamily:
    group: click.Group
    list_command: click.Command
    recent_runs_command: click.Command
    sampling_summary_command: click.Command
    cancel_command: click.Command
    show_command: click.Command


def build_admin_task_command_family(
    workspace_root_option,
    *,
    list_tasks: AdminTaskListAction,
    list_recent_task_runs: AdminTaskRecentRunsAction,
    show_task_sampling_summary: AdminTaskSamplingSummaryAction,
    cancel_task: AdminTaskCancelAction,
    show_task: AdminTaskShowAction,
) -> AdminTaskCommandFamily:
    @click.group(name="task")
    def admin_task_group() -> None:
        """Canonical background task operator commands."""

    @admin_task_group.command(name="list")
    @workspace_root_option
    @click.option("--status", help="Filter by task status")
    @click.option("--limit", default=20, show_default=True, type=int, help="Maximum number of task rows to print")
    @click.option("--json", "json_output", is_flag=True, help="Print JSON instead of a table")
    def admin_list_tasks_command(workspace_root: str | None, status: str | None, limit: int, json_output: bool) -> None:
        """List queued or running tasks."""
        list_tasks(workspace_root, status, limit, json_output)

    @admin_task_group.command(name="recent-runs")
    @workspace_root_option
    @click.option("--limit", default=20, show_default=True, type=int, help="Maximum number of recent runs to print")
    @click.option("--json", "json_output", is_flag=True, help="Print JSON instead of a table")
    def admin_recent_task_runs_command(workspace_root: str | None, limit: int, json_output: bool) -> None:
        """Show recent completed background task runs and their sampling metadata."""
        list_recent_task_runs(workspace_root, limit, json_output)

    @admin_task_group.command(name="sampling-summary")
    @workspace_root_option
    @click.option("--limit", default=50, show_default=True, type=int, help="Maximum number of recent runs to summarize")
    @click.option("--json", "json_output", is_flag=True, help="Print JSON instead of tables")
    def admin_task_sampling_summary_command(workspace_root: str | None, limit: int, json_output: bool) -> None:
        """Summarize recent selection and ingest grouping strategy usage."""
        show_task_sampling_summary(workspace_root, limit, json_output)

    @admin_task_group.command(name="cancel")
    @workspace_root_option
    @click.argument("task_id")
    @click.option("--reason", default="cancelled_by_user", show_default=True, help="Cancellation reason")
    @click.option("--json", "json_output", is_flag=True, help="Print JSON instead of human-readable output")
    def admin_cancel_task_command(task_id: str, workspace_root: str | None, reason: str, json_output: bool) -> None:
        """Cancel a pending or running task."""
        cancel_task(task_id, workspace_root, reason, json_output)

    @admin_task_group.command(name="show")
    @workspace_root_option
    @click.argument("task_id")
    @click.option("--json", "json_output", is_flag=True, help="Print JSON instead of human-readable output")
    def admin_show_task_command(task_id: str, workspace_root: str | None, json_output: bool) -> None:
        """Show one task and its persisted run result details."""
        show_task(task_id, workspace_root, json_output)

    return AdminTaskCommandFamily(
        group=admin_task_group,
        list_command=admin_list_tasks_command,
        recent_runs_command=admin_recent_task_runs_command,
        sampling_summary_command=admin_task_sampling_summary_command,
        cancel_command=admin_cancel_task_command,
        show_command=admin_show_task_command,
    )