from __future__ import annotations

import asyncio

import click

from mcp_memory.cli_stdio_proxy_commands import build_stdio_proxy_command_family


workspace_root_option = click.option("--workspace-root", help="Override the active workspace root")


def _run_stdio_proxy(
    debug_enabled: bool,
    workspace_root: str | None,
    source: str,
    server_name: str,
    tool_path_prefix: str,
) -> None:
    from mcp_memory.runtime_logging import configure_workspace_logging
    from mcp_memory.server import MCPServer

    configure_workspace_logging(
        debug_enabled,
        workspace_root_override=workspace_root,
        console_output=False,
        source=source,
    )
    server = MCPServer(
        workspace_root=workspace_root,
        server_name=server_name,
        tool_path_prefix=tool_path_prefix,
    )
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        return
    except Exception as exc:
        click.echo(f"mcp-memory: {exc}", err=True)
        raise click.exceptions.Exit(1) from exc


def _run_command_action(debug_enabled: bool, workspace_root: str | None) -> None:
    _run_stdio_proxy(
        debug_enabled,
        workspace_root,
        "stdio",
        "mcp-memory",
        "/internal/tools",
    )


def _internal_run_command_action(debug_enabled: bool, workspace_root: str | None) -> None:
    _run_stdio_proxy(
        debug_enabled,
        workspace_root,
        "internal-stdio",
        "mcp-memory-internal",
        "/internal/maintenance/tools",
    )


@click.group()
@click.option("--debug", is_flag=True, help="Enable debug logging")
@click.pass_context
def main(ctx: click.Context, debug: bool) -> None:
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug


_stdio_proxy_command_family = build_stdio_proxy_command_family(
    workspace_root_option,
    run_stdio_proxy=_run_command_action,
    run_internal_stdio_proxy=_internal_run_command_action,
)
main.add_command(_stdio_proxy_command_family.run)
main.add_command(_stdio_proxy_command_family.internal_run)
