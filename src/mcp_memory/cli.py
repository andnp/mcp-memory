from __future__ import annotations

import asyncio
import logging
from pathlib import Path
import socket
import sys

import click
from rich.console import Console
from rich.logging import RichHandler
import uvicorn

from mcp_memory.daemon import create_daemon_app, ensure_daemon_started
from mcp_memory.mcp.runtime import create_runtime
from mcp_memory.relational.importer import import_markdown_memory
from mcp_memory.server import MCPServer


console = Console()


@click.group()
@click.option("--debug", is_flag=True, help="Enable debug logging")
def main(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, console=console)],
    )


@main.command()
@click.option("--workspace-root", help="Override the active workspace root")
def run(workspace_root: str | None) -> None:
    """Run the MCP stdio proxy, auto-starting the workspace daemon when needed."""
    server = MCPServer(workspace_root=workspace_root)
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        return
    except Exception as exc:
        console.print(f"[red]Error:[/] {exc}")
        sys.exit(1)


@main.command()
@click.option("--workspace-root", help="Override the active workspace root")
@click.option("--host", default="127.0.0.1", show_default=True, help="Daemon bind host")
@click.option("--port", default=0, show_default=True, type=int, help="Daemon bind port")
def daemon(workspace_root: str | None, host: str, port: int) -> None:
    """Run the workspace daemon backend."""
    daemon_port = port
    if daemon_port == 0:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((host, 0))
            daemon_port = int(sock.getsockname()[1])
    app = create_daemon_app(workspace_root_override=workspace_root, host=host, port=daemon_port)
    uvicorn.run(app, host=host, port=daemon_port, log_level="info")


@main.command(name="dashboard")
@click.option("--workspace-root", help="Override the active workspace root")
def dashboard(workspace_root: str | None) -> None:
    """Ensure the daemon is running and print the dashboard URL."""
    try:
        metadata = ensure_daemon_started(workspace_root, None)
    except Exception as exc:
        console.print(f"[red]Error:[/] {exc}")
        sys.exit(1)
    console.print(f"[green]Dashboard ready:[/] {metadata.base_url}/")


@main.command(name="import-markdown")
@click.argument("file_path", type=click.Path(exists=True, dir_okay=False, path_type=str))
@click.option("--workspace-root", help="Override the active workspace root")
@click.option(
    "--workspace-id",
    "workspace_ids",
    multiple=True,
    help="Attach the imported memory to explicit workspace IDs (defaults to the active workspace)",
)
def import_markdown(file_path: str, workspace_root: str | None, workspace_ids: tuple[str, ...]) -> None:
    """Import one markdown memory file into the relational store."""
    runtime = create_runtime(workspace_root_override=workspace_root)
    try:
        if runtime.repository is None:
            raise RuntimeError("repository_not_initialized")

        resolved_workspace_ids = [workspace_id.strip() for workspace_id in workspace_ids if workspace_id.strip()]
        if not resolved_workspace_ids and runtime.workspace_id is not None:
            resolved_workspace_ids = [runtime.workspace_id]

        imported = import_markdown_memory(runtime.repository, Path(file_path), resolved_workspace_ids)
        console.print(f"[green]Imported memory:[/] {imported.id} — {imported.title}")
    except Exception as exc:
        console.print(f"[red]Error:[/] {exc}")
        sys.exit(1)
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
