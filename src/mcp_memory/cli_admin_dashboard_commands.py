from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import click


type AdminDashboardOpenAction = Callable[[str | None], None]
type AdminDashboardBuildAction = Callable[[], None]


@dataclass(frozen=True)
class AdminDashboardCommandFamily:
    group: click.Group
    open_command: click.Command
    build_command: click.Command


def build_admin_dashboard_command_family(
    workspace_root_option,
    *,
    open_dashboard: AdminDashboardOpenAction,
    build_dashboard_frontend: AdminDashboardBuildAction,
) -> AdminDashboardCommandFamily:
    @click.group(name="dashboard")
    def admin_dashboard_group() -> None:
        """Canonical operator dashboard commands."""

    @admin_dashboard_group.command(name="open")
    @workspace_root_option
    def admin_dashboard_open_command(workspace_root: str | None) -> None:
        """Ensure the daemon is running and open the operator dashboard."""
        open_dashboard(workspace_root)

    @admin_dashboard_group.command(name="build")
    def admin_dashboard_build_command() -> None:
        """Build the operator dashboard frontend bundle."""
        build_dashboard_frontend()

    return AdminDashboardCommandFamily(
        group=admin_dashboard_group,
        open_command=admin_dashboard_open_command,
        build_command=admin_dashboard_build_command,
    )
