import asyncio
import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler
import uvicorn

from mcp_memory.management import create_management_app
from mcp_memory.server import MCPServer

console = Console()


@click.group()
@click.option("--debug", is_flag=True, help="Enable debug logging")
def main(debug: bool):
    """MCP Memory Server CLI."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, console=console)],
    )


@main.command()
@click.option("--project", help="Override workspace root path (legacy option name)")
def run(project: str | None):
    """Run the MCP server over stdio."""
    server = MCPServer(project_override=project)
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        console.print(f"[red]Error:[/] {e}")
        sys.exit(1)


@main.command()
def stats():
    """Get memory bank statistics (Mockup)."""
    console.print("[yellow]Stats tool coming soon...[/]")


@main.command(name="dashboard")
@click.option("--host", default="127.0.0.1", show_default=True, help="Host interface to bind")
@click.option("--port", default=8765, show_default=True, type=int, help="Port to bind")
@click.option("--project", help="Override workspace root path (legacy option name)")
def dashboard(host: str, port: int, project: str | None):
    """Run the read-only management API and operations dashboard."""
    app = create_management_app(project_override=project)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
