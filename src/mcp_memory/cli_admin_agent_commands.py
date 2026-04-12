from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import click


type AdminAgentRunAction = Callable[[str | None, str | None, bool, bool], None]


@dataclass(frozen=True)
class AdminAgentCommandFamily:
    group: click.Group
    run_command: click.Command


def build_admin_agent_command_family(
    workspace_root_option,
    *,
    agent_names: tuple[str, ...],
    run_agent: AdminAgentRunAction,
) -> AdminAgentCommandFamily:
    @click.group(name="agent")
    def admin_agent_group() -> None:
        """Canonical background agent operator commands."""

    @admin_agent_group.command(name="run")
    @workspace_root_option
    @click.argument("agent_name", required=False, type=click.Choice(agent_names))
    @click.option("--all", "run_all", is_flag=True, help="Trigger all background agents")
    @click.option("--force", is_flag=True, help="Enqueue new tasks even if matching tasks are already open")
    def admin_run_agent_command(
        agent_name: str | None,
        workspace_root: str | None,
        run_all: bool,
        force: bool,
    ) -> None:
        """Trigger one or all background agents for the active workspace."""
        run_agent(agent_name, workspace_root, run_all, force)

    return AdminAgentCommandFamily(
        group=admin_agent_group,
        run_command=admin_run_agent_command,
    )