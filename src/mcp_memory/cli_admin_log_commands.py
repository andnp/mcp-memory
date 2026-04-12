from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import click


type AdminLogListAction = Callable[
    [
        str | None,
        str,
        str | None,
        int,
        str | None,
        str | None,
        str | None,
        str | None,
        float | None,
        float | None,
        bool,
    ],
    None,
]
type AdminLogSummaryAction = Callable[
    [
        str | None,
        str,
        str | None,
        str | None,
        str | None,
        str | None,
        str | None,
        float | None,
        float | None,
        bool,
    ],
    None,
]
type AdminLogPruneAction = Callable[[str | None, str, str | None, int | None, int | None, bool], None]


@dataclass(frozen=True)
class AdminLogCommandFamily:
    group: click.Group
    list_command: click.Command
    summary_command: click.Command
    prune_command: click.Command


def build_admin_log_command_family(
    workspace_root_option,
    *,
    show_logs: AdminLogListAction,
    summarize_logs: AdminLogSummaryAction,
    prune_logs: AdminLogPruneAction,
) -> AdminLogCommandFamily:
    @click.group(name="log")
    def admin_log_group() -> None:
        """Canonical structured runtime log commands."""

    @admin_log_group.command(name="list")
    @workspace_root_option
    @click.option(
        "--scope",
        type=click.Choice(["global", "workspace"]),
        default="global",
        show_default=True,
        help="Read logs across the shared runtime or only the active workspace context.",
    )
    @click.option("--workspace-id", help="Explicit workspace ID override for log filtering")
    @click.option("--limit", default=20, show_default=True, type=int, help="Maximum number of log rows to print")
    @click.option(
        "--level",
        type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False),
        help="Filter by log level.",
    )
    @click.option("--logger", "logger_name", help="Filter by logger name")
    @click.option("--source", help="Filter by log source (for example: daemon, stdio)")
    @click.option("--query", help="Case-insensitive text filter across message, logger, and source")
    @click.option("--after", type=float, help="Only include logs at or after this UNIX timestamp")
    @click.option("--before", type=float, help="Only include logs at or before this UNIX timestamp")
    @click.option("--json", "json_output", is_flag=True, help="Print JSON instead of a table")
    def admin_logs(
        workspace_root: str | None,
        scope: str,
        workspace_id: str | None,
        limit: int,
        level: str | None,
        logger_name: str | None,
        source: str | None,
        query: str | None,
        after: float | None,
        before: float | None,
        json_output: bool,
    ) -> None:
        """Print recent structured runtime logs from SQLite."""
        show_logs(workspace_root, scope, workspace_id, limit, level, logger_name, source, query, after, before, json_output)

    @admin_log_group.command(name="summary")
    @workspace_root_option
    @click.option(
        "--scope",
        type=click.Choice(["global", "workspace"]),
        default="global",
        show_default=True,
        help="Summarize logs across the shared runtime or only the active workspace context.",
    )
    @click.option("--workspace-id", help="Explicit workspace ID override for log filtering")
    @click.option(
        "--level",
        type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False),
        help="Filter by log level.",
    )
    @click.option("--logger", "logger_name", help="Filter by logger name")
    @click.option("--source", help="Filter by log source (for example: daemon, stdio)")
    @click.option("--query", help="Case-insensitive text filter across message, logger, and source")
    @click.option("--after", type=float, help="Only include logs at or after this UNIX timestamp")
    @click.option("--before", type=float, help="Only include logs at or before this UNIX timestamp")
    @click.option("--json", "json_output", is_flag=True, help="Print JSON instead of tables")
    def admin_log_summary(
        workspace_root: str | None,
        scope: str,
        workspace_id: str | None,
        level: str | None,
        logger_name: str | None,
        source: str | None,
        query: str | None,
        after: float | None,
        before: float | None,
        json_output: bool,
    ) -> None:
        """Print aggregated runtime log counts."""
        summarize_logs(workspace_root, scope, workspace_id, level, logger_name, source, query, after, before, json_output)

    @admin_log_group.command(name="prune")
    @workspace_root_option
    @click.option(
        "--scope",
        type=click.Choice(["global", "workspace"]),
        default="global",
        show_default=True,
        help="Prune logs across the shared runtime or only the active workspace context.",
    )
    @click.option("--workspace-id", help="Explicit workspace ID override for log pruning")
    @click.option("--max-runtime-logs", type=int, help="Keep at most this many recent runtime logs")
    @click.option("--max-log-age-days", type=int, help="Delete runtime logs older than this many days")
    @click.option("--json", "json_output", is_flag=True, help="Print JSON instead of human-readable output")
    def admin_log_prune(
        workspace_root: str | None,
        scope: str,
        workspace_id: str | None,
        max_runtime_logs: int | None,
        max_log_age_days: int | None,
        json_output: bool,
    ) -> None:
        """Prune runtime logs using explicit or configured retention limits."""
        prune_logs(workspace_root, scope, workspace_id, max_runtime_logs, max_log_age_days, json_output)

    return AdminLogCommandFamily(
        group=admin_log_group,
        list_command=admin_logs,
        summary_command=admin_log_summary,
        prune_command=admin_log_prune,
    )