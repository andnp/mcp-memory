from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import click


type DaemonStartAction = Callable[[bool, str | None, str, int | None, bool], None]
type DaemonCommandAction = Callable[[str | None], None]


@dataclass(frozen=True)
class DaemonCommandFamily:
    group: click.Group
    start: click.Command
    status: click.Command
    stop: click.Command
    restart: click.Command


def build_daemon_command_family(
    workspace_root_option,
    *,
    start_daemon: DaemonStartAction,
    print_daemon_status: DaemonCommandAction,
    stop_daemon_command: DaemonCommandAction,
    restart_daemon_command: DaemonCommandAction,
) -> DaemonCommandFamily:
    @click.group(name="daemon", invoke_without_command=True)
    @workspace_root_option
    @click.option("--host", default="127.0.0.1", show_default=True, help="Daemon bind host")
    @click.option("--port", default=None, show_default="configured", type=int, help="Daemon bind port (use 0 for an ephemeral port)")
    @click.option("--internal-preflight-done", is_flag=True, hidden=True, help="Internal: skip the reclaim guard (set by the auto-start spawner).")
    @click.pass_context
    def daemon_group(ctx: click.Context, workspace_root: str | None, host: str, port: int | None, internal_preflight_done: bool) -> None:
        """Run and manage the global daemon."""
        if ctx.invoked_subcommand is not None:
            return
        start_daemon(bool(ctx.obj.get("debug", False)), workspace_root, host, port, internal_preflight_done)

    @daemon_group.command(name="start")
    @workspace_root_option
    @click.option("--host", default="127.0.0.1", show_default=True, help="Daemon bind host")
    @click.option("--port", default=None, show_default="configured", type=int, help="Daemon bind port (use 0 for an ephemeral port)")
    @click.option("--internal-preflight-done", is_flag=True, hidden=True, help="Internal: skip the reclaim guard (set by the auto-start spawner).")
    @click.pass_context
    def daemon_start(ctx: click.Context, workspace_root: str | None, host: str, port: int | None, internal_preflight_done: bool) -> None:
        """Start the global daemon."""
        start_daemon(bool(ctx.obj.get("debug", False)), workspace_root, host, port, internal_preflight_done)

    @daemon_group.command(name="status")
    @workspace_root_option
    def daemon_status(workspace_root: str | None) -> None:
        """Print the current global daemon status."""
        print_daemon_status(workspace_root)

    @daemon_group.command(name="stop")
    @workspace_root_option
    def daemon_stop(workspace_root: str | None) -> None:
        """Stop the global daemon if it is running."""
        stop_daemon_command(workspace_root)

    @daemon_group.command(name="restart")
    @workspace_root_option
    def daemon_restart(workspace_root: str | None) -> None:
        """Restart the global daemon and print the active transport endpoint."""
        restart_daemon_command(workspace_root)

    return DaemonCommandFamily(
        group=daemon_group,
        start=daemon_start,
        status=daemon_status,
        stop=daemon_stop,
        restart=daemon_restart,
    )