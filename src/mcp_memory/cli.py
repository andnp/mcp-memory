from __future__ import annotations

import asyncio
from datetime import datetime
import logging
import socket
import sys
from types import SimpleNamespace

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
import uvicorn

from mcp_memory.core.task_handlers import TRIGGERABLE_BACKGROUND_TASK_NAMES
from mcp_memory.daemon import create_daemon_app, ensure_daemon_started
from mcp_memory.embeddings import describe_embedder
from mcp_memory.management.service import ManagementService
from mcp_memory.mcp.runtime import create_runtime
from mcp_memory.relational.importer import (
    import_markdown_memory_paths,
    record_markdown_memory_paths_as_thoughts,
)
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


@main.command(name="internal-run", hidden=True)
@click.option("--workspace-root", help="Override the active workspace root")
def internal_run(workspace_root: str | None) -> None:
    """Run the internal maintenance MCP stdio proxy for trusted tool-using agents."""
    server = MCPServer(
        workspace_root=workspace_root,
        server_name="mcp-memory-internal",
        tool_path_prefix="/internal/maintenance/tools",
    )
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


@main.command(name="prefetch-model")
@click.option("--workspace-root", help="Override the active workspace root")
def prefetch_model(workspace_root: str | None) -> None:
    """Download and cache the configured local embedding model in the foreground."""
    runtime = create_runtime(workspace_root_override=workspace_root)
    try:
        embedder = runtime.embedder
        if embedder is None:
            raise RuntimeError("embedder_not_initialized")
        cache_model = getattr(embedder, "cache_model", None)
        if not callable(cache_model):
            raise RuntimeError("embedder_cache_not_supported")
        cached = bool(cache_model())
        status = describe_embedder(embedder)
        if not cached:
            raise RuntimeError("embedding_model_cache_failed")
        console.print(f"[green]Embedding model cached:[/] {status.model_name if status is not None else 'unknown'}")
        if status is not None:
            console.print(f"backend={status.backend} cached={status.model_cached}")
    except Exception as exc:
        console.print(f"[red]Error:[/] {exc}")
        sys.exit(1)
    finally:
        runtime.close()


@main.group(name="agents")
def agents() -> None:
    """Trigger and inspect background agents."""


@agents.command(name="run")
@click.argument("agent_name", type=click.Choice(TRIGGERABLE_BACKGROUND_TASK_NAMES))
@click.option("--workspace-root", help="Override the active workspace root")
@click.option("--force", is_flag=True, help="Enqueue a new task even if one is already open")
def run_agent(agent_name: str, workspace_root: str | None, force: bool) -> None:
    """Trigger one background agent for the active workspace."""
    ensure_daemon_started(workspace_root, None)
    runtime = create_runtime(workspace_root_override=workspace_root)
    try:
        payload = _build_management_service(runtime).enqueue_background_task(agent_name, force=force)
        console.print(f"[green]{payload['status']}[/]: {agent_name}")
        console.print(f"task_id={payload['task']['id']} status={payload['task']['status']}")
    except Exception as exc:
        console.print(f"[red]Error:[/] {exc}")
        sys.exit(1)
    finally:
        runtime.close()


@agents.command(name="run-all")
@click.option("--workspace-root", help="Override the active workspace root")
@click.option("--force", is_flag=True, help="Enqueue new tasks even if matching tasks are already open")
def run_all_agents(workspace_root: str | None, force: bool) -> None:
    """Trigger all background agents for the active workspace."""
    ensure_daemon_started(workspace_root, None)
    runtime = create_runtime(workspace_root_override=workspace_root)
    try:
        service = _build_management_service(runtime)
        results = [service.enqueue_background_task(task_name, force=force) for task_name in TRIGGERABLE_BACKGROUND_TASK_NAMES]
        created_count = sum(1 for result in results if result["created"])
        console.print(f"[green]Agents queued:[/] {created_count}/{len(results)} newly created")
        for result in results:
            console.print(f"- {result['task']['task_name']}: {result['status']}")
    except Exception as exc:
        console.print(f"[red]Error:[/] {exc}")
        sys.exit(1)
    finally:
        runtime.close()


@main.command(name="stats")
@click.option("--workspace-root", help="Override the active workspace root")
def stats(workspace_root: str | None) -> None:
    """Print background task and memory statistics."""
    ensure_daemon_started(workspace_root, None)
    runtime = create_runtime(workspace_root_override=workspace_root)
    try:
        service = _build_management_service(runtime)
        health = service.get_health()
        overview = service.get_overview()

        console.print(f"[bold]Workspace:[/] {health.workspace_id}")
        console.print(f"[bold]Database:[/] {health.db_path}")

        metrics = Table(title="Memory Metrics")
        metrics.add_column("Metric")
        metrics.add_column("Value", justify="right")
        metrics.add_row("Total memories", str(overview.memory_metrics.total_memories))
        metrics.add_row("Total memory lines", str(overview.memory_metrics.total_memory_lines))
        metrics.add_row("Total summary lines", str(overview.memory_metrics.total_summary_lines))
        metrics.add_row("Total lines compressed", str(overview.memory_metrics.total_lines_compressed))
        metrics.add_row("Thought buffer entries", str(overview.memory_metrics.thought_buffer_entries))
        metrics.add_row("Thought buffer lines", str(overview.memory_metrics.thought_buffer_lines))
        console.print(metrics)

        agent_table = Table(title="Background Agents")
        agent_table.add_column("Agent", no_wrap=True)
        agent_table.add_column("Running", justify="right")
        agent_table.add_column("Age")
        agent_table.add_column("Next")
        agent_table.add_column("Last status")
        agent_table.add_column("Runs", justify="right")
        agent_table.add_column("Failures", justify="right")
        agent_table.add_column("Avg duration", justify="right")
        agent_table.add_column("Compressed", justify="right")
        agent_table.add_column("Last result")
        for agent in overview.agent_runs:
            agent_table.add_row(
                agent.task_name,
                str(agent.running_count),
                _format_age(agent.seconds_since_last_completion),
                _format_age(agent.seconds_until_next_run),
                agent.last_status or "never",
                str(agent.total_runs),
                str(agent.failed_runs),
                f"{agent.avg_duration_seconds:.2f}s",
                str(agent.total_lines_compressed),
                agent.last_result_summary or "-",
            )
        console.print(agent_table)

        console.print("[bold]Agent Details[/]")
        for agent in overview.agent_runs:
            console.print(
                "- "
                f"{agent.task_name}: "
                f"running={agent.running_count} "
                f"next={_format_age(agent.seconds_until_next_run)} "
                f"last_status={agent.last_status or 'never'} "
                f"last_result={agent.last_result_summary or '-'}"
            )

        console.print("[bold]Recent Agent Runs[/]")
        for run in overview.recent_agent_runs:
            console.print(
                "- "
                f"{run.task_name}: "
                f"status={run.status} "
                f"duration={run.duration_seconds:.2f}s "
                f"result={run.result_summary or '-'} "
                f"error={run.error_text or '-'}"
            )
    except Exception as exc:
        console.print(f"[red]Error:[/] {exc}")
        sys.exit(1)
    finally:
        runtime.close()


@main.command(name="import-markdown")
@click.argument("file_paths", nargs=-1, type=str)
@click.option("--workspace-root", help="Override the active workspace root")
@click.option(
    "--workspace-id",
    "workspace_ids",
    multiple=True,
    help="Attach the imported memory to explicit workspace IDs (defaults to the active workspace)",
)
@click.option(
    "--thought",
    is_flag=True,
    help="Import as a thought into the thought buffer instead of directly into storage (for intelligent processing by agents)",
)
def import_markdown(
    file_paths: tuple[str, ...],
    workspace_root: str | None,
    workspace_ids: tuple[str, ...],
    thought: bool,
) -> None:
    """Import one or more markdown memory files into the relational store or thought buffer."""
    if not file_paths:
        raise click.UsageError("Provide at least one markdown file path or glob pattern.")

    runtime = create_runtime(workspace_root_override=workspace_root)
    try:
        resolved_workspace_id = None
        if workspace_ids:
            # When --thought is used with multiple workspace IDs, use the first one
            resolved_workspace_id = next(
                (wid.strip() for wid in workspace_ids if wid.strip()), 
                None
            )
        elif runtime.workspace_id is not None:
            resolved_workspace_id = runtime.workspace_id

        if thought:
            # Import as thoughts into the journal buffer
            if runtime.journal is None:
                raise RuntimeError("journal_not_initialized")

            recorded = record_markdown_memory_paths_as_thoughts(
                runtime.journal,
                list(file_paths),
                workspace_id=resolved_workspace_id,
            )
            for entry_id, file_name in recorded:
                console.print(f"[green]Recorded thought:[/] {file_name} (entry {entry_id})")
            console.print(f"[green]Recorded total:[/] {len(recorded)} thoughts → buffer")
        else:
            # Import directly to relational storage (existing behavior)
            if runtime.repository is None:
                raise RuntimeError("repository_not_initialized")

            resolved_workspace_ids = [workspace_id.strip() for workspace_id in workspace_ids if workspace_id.strip()]
            if not resolved_workspace_ids and runtime.workspace_id is not None:
                resolved_workspace_ids = [runtime.workspace_id]

            imported_records = import_markdown_memory_paths(
                runtime.repository,
                list(file_paths),
                resolved_workspace_ids,
            )
            for imported in imported_records:
                console.print(f"[green]Imported memory:[/] {imported.id} — {imported.title}")
            console.print(f"[green]Imported total:[/] {len(imported_records)}")
    except Exception as exc:
        console.print(f"[red]Error:[/] {exc}")
        sys.exit(1)
    finally:
        runtime.close()


def _build_management_service(runtime) -> ManagementService:
    return ManagementService(runtime, SimpleNamespace(has_runtime=True, client_count=1))


def _format_timestamp(value: float | None) -> str:
    if value is None:
        return "never"
    return datetime.fromtimestamp(value).isoformat(timespec="seconds")
def _format_age(value: float | None) -> str:
    if value is None:
        return "never"
    if value < 60:
        return f"{int(value)}s"
    if value < 3600:
        return f"{int(value // 60)}m"
    if value < 86400:
        return f"{int(value // 3600)}h"
    return f"{int(value // 86400)}d"


if __name__ == "__main__":
    main()
