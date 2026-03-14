import asyncio
import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler

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
@click.option("--project", help="Override project path")
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


if __name__ == "__main__":
    main()
