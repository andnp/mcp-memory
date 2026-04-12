from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import click


type AdminOverviewAction = Callable[[str | None, bool, float, bool], None]
type AdminHealthAction = Callable[[str | None, bool], None]
type AdminMonitorAction = Callable[[str | None, float], None]
type AdminQualityCleanupAction = Callable[[str | None, int, int, int, bool], None]


@dataclass(frozen=True)
class AdminOperatorCommandCluster:
    overview_command: click.Command
    health_command: click.Command
    monitor_command: click.Command
    quality_cleanup_command: click.Command


def build_admin_operator_command_cluster(
    workspace_root_option,
    *,
    show_overview: AdminOverviewAction,
    show_health: AdminHealthAction,
    open_monitor: AdminMonitorAction,
    show_quality_cleanup: AdminQualityCleanupAction,
) -> AdminOperatorCommandCluster:
    @click.command(name="overview")
    @workspace_root_option
    @click.option("--watch", is_flag=True, help="Refresh the stats view continuously")
    @click.option(
        "--interval",
        default=2.0,
        show_default=True,
        type=click.FloatRange(min=0.1),
        help="Seconds between watch refreshes",
    )
    @click.option("--verbose", is_flag=True, help="Show detailed recent agent status lines")
    def admin_overview_command(workspace_root: str | None, watch: bool, interval: float, verbose: bool) -> None:
        """Print background task and memory statistics."""
        show_overview(workspace_root, watch, interval, verbose)

    @click.command(name="health")
    @workspace_root_option
    @click.option("--json", "json_output", is_flag=True, help="Print JSON instead of human-readable output")
    def admin_health_command(workspace_root: str | None, json_output: bool) -> None:
        """Print an AI-friendly operator health snapshot."""
        show_health(workspace_root, json_output)

    @click.command(name="monitor")
    @workspace_root_option
    @click.option(
        "--interval",
        default=2.0,
        show_default=True,
        type=click.FloatRange(min=0.1),
        help="Seconds between automatic refreshes",
    )
    def admin_monitor_command(workspace_root: str | None, interval: float) -> None:
        """Open the live operations TUI."""
        open_monitor(workspace_root, interval)

    @click.command(name="quality-cleanup")
    @workspace_root_option
    @click.option("--window-hours", default=24, show_default=True, type=int, help="Analytics window to inspect")
    @click.option("--bucket-minutes", default=60, show_default=True, type=int, help="Bucket size used for supporting analytics")
    @click.option("--limit", default=20, show_default=True, type=int, help="Maximum number of cleanup candidates to show")
    @click.option("--json", "json_output", is_flag=True, help="Print JSON instead of a table")
    def admin_quality_cleanup_command(
        workspace_root: str | None,
        window_hours: int,
        bucket_minutes: int,
        limit: int,
        json_output: bool,
    ) -> None:
        """Show prioritized memory-quality cleanup candidates."""
        show_quality_cleanup(workspace_root, window_hours, bucket_minutes, limit, json_output)

    return AdminOperatorCommandCluster(
        overview_command=admin_overview_command,
        health_command=admin_health_command,
        monitor_command=admin_monitor_command,
        quality_cleanup_command=admin_quality_cleanup_command,
    )