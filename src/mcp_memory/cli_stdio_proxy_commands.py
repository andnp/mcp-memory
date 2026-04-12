from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import click


type StdioProxyCommandAction = Callable[[bool, str | None], None]


@dataclass(frozen=True)
class StdioProxyCommandFamily:
    run: click.Command
    internal_run: click.Command


def build_stdio_proxy_command_family(
    workspace_root_option,
    *,
    run_stdio_proxy: StdioProxyCommandAction,
    run_internal_stdio_proxy: StdioProxyCommandAction,
) -> StdioProxyCommandFamily:
    @click.command(name="run")
    @workspace_root_option
    @click.pass_context
    def run(ctx: click.Context, workspace_root: str | None) -> None:
        """Run the MCP stdio proxy, auto-starting the global daemon when needed."""
        run_stdio_proxy(bool(ctx.obj.get("debug", False)), workspace_root)

    @click.command(name="internal-run", hidden=True)
    @workspace_root_option
    @click.pass_context
    def internal_run(ctx: click.Context, workspace_root: str | None) -> None:
        """Run the internal maintenance MCP stdio proxy for trusted tool-using agents."""
        run_internal_stdio_proxy(bool(ctx.obj.get("debug", False)), workspace_root)

    return StdioProxyCommandFamily(
        run=run,
        internal_run=internal_run,
    )