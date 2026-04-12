from __future__ import annotations

from collections.abc import Callable

import click


type HookRunnerCommandAction = Callable[[str | None], None]


def build_hook_runner_command(*, hook_runner_command_action: HookRunnerCommandAction) -> click.Command:
    @click.command(name="hook-runner", hidden=True)
    @click.option("--workspace-root", help="Override the target workspace root")
    def hook_runner(workspace_root: str | None) -> None:
        """Forward VS Code hook payloads into the global daemon."""
        hook_runner_command_action(workspace_root)

    return hook_runner