from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any, TypeVar
import webbrowser

import click
from rich.console import Console
from rich.table import Table
import uvicorn

from mcp_memory.cli_admin_agent_commands import build_admin_agent_command_family
from mcp_memory.cli_admin_conversation_commands import build_admin_conversation_command_family
from mcp_memory.cli_admin_dashboard_commands import build_admin_dashboard_command_family
from mcp_memory.cli_admin_log_commands import build_admin_log_command_family
from mcp_memory.cli_admin_operator_commands import build_admin_operator_command_cluster
from mcp_memory.cli_admin_search_commands import build_admin_search_command_family
from mcp_memory.cli_admin_task_commands import build_admin_task_command_family
from mcp_memory.cli_admin_utility_commands import build_admin_utility_command_cluster
from mcp_memory.cli_daemon_commands import build_daemon_command_family
from mcp_memory.cli_hook_runner_commands import build_hook_runner_command
from mcp_memory.cli_memory import build_memory_group
from mcp_memory.cli_stdio_proxy_commands import build_stdio_proxy_command_family
from mcp_memory.config import load_config, resolve_memory_path
from mcp_memory.core.journal_operations import RecordThoughtOperation
from mcp_memory.core.maintenance_idle import resume_paused_recurring_maintenance
from mcp_memory.core.task_handlers import TRIGGERABLE_BACKGROUND_TASK_NAMES
from mcp_memory.daemon import DaemonStopResult, create_daemon_app, ensure_daemon_started, inspect_daemon, stop_daemon
from mcp_memory.daemon_process import find_free_port
from mcp_memory.cli_tui import run_monitor_tui
from mcp_memory.embeddings import describe_embedder
from mcp_memory.installer import install_integrations, load_hook_payload, safe_forward_hook_event
from mcp_memory.management.task_sampling_summary import build_task_sampling_summary
from mcp_memory.management.scope_policy import ScopePolicyKind, resolve_workspace_id_for_policy
from mcp_memory.management.service import ManagementService
from mcp_memory.management.frontend_build import ensure_dashboard_frontend_built
from mcp_memory.mcp.runtime import create_runtime, resolve_global_daemon_bootstrap_spec
from mcp_memory.mcp.services import search_memory_records_service
from mcp_memory.relational.importer import (
    import_markdown_memory_paths,
    record_markdown_memory_paths_as_thoughts,
)
from mcp_memory.runtime_logging import configure_cli_logging, configure_workspace_logging
from mcp_memory.server import MCPServer
from mcp_memory.storage.sqlite_to_postgres_migration import migrate_sqlite_to_postgres

console = Console()
workspace_root_option = click.option("--workspace-root", help="Override the active workspace root")
_T = TypeVar("_T")


def _exit_cli_error(exc: Exception) -> None:
    console.print(f"[red]Error:[/] {exc}")
    sys.exit(1)


def _run_or_exit(action: Callable[[], None]) -> None:
    try:
        action()
    except Exception as exc:
        _exit_cli_error(exc)


def _run_stdio_proxy(
    debug_enabled: bool,
    workspace_root: str | None,
    source: str,
    server_name: str,
    tool_path_prefix: str,
) -> None:
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
        sys.exit(1)


def _start_daemon(debug_enabled: bool, workspace_root: str | None, host: str, port: int | None) -> None:
    configure_workspace_logging(
        debug_enabled,
        workspace_root_override=workspace_root,
        console_output=True,
        source="daemon",
    )
    resolved_port = port
    if resolved_port is None:
        resolved_port = resolve_global_daemon_bootstrap_spec(workspace_root_override=workspace_root).config.daemon.port
    if resolved_port == 0:
        resolved_port = find_free_port()
    app = create_daemon_app(
        workspace_root_override=workspace_root,
        host=host,
        port=resolved_port,
        enable_idle_shutdown=True,
    )
    config = uvicorn.Config(app, host=host, port=resolved_port, log_level="warning", access_log=False)
    server = uvicorn.Server(config)
    try:
        asyncio.run(server.serve())
    except KeyboardInterrupt:
        return


def _print_daemon_status(workspace_root: str | None) -> None:
    metadata, healthy = inspect_daemon(workspace_root, None)
    if metadata is None:
        console.print("[yellow]Global daemon not registered[/]")
        return
    console.print("[bold]Daemon scope:[/] global")
    console.print(f"[bold]Status:[/] {'running' if healthy else 'stale'}")
    console.print(f"[bold]PID:[/] {metadata.pid}")
    console.print(f"[bold]Transport:[/] {metadata.transport}")
    console.print(f"[bold]Endpoint:[/] {metadata.transport_endpoint}")
    if metadata.version is not None:
        console.print(f"[bold]Version:[/] {metadata.version}")
    if metadata.binary_path is not None:
        console.print(f"[bold]Binary:[/] {metadata.binary_path}")
    console.print(f"[bold]Started:[/] {_format_timestamp(metadata.started_at)}")


def _stop_daemon_command(workspace_root: str | None) -> None:
    stop_result = stop_daemon(workspace_root, None)
    if stop_result is None:
        console.print("[yellow]No daemon metadata found.[/]")
        return
    metadata = stop_result.metadata if isinstance(stop_result, DaemonStopResult) else stop_result
    console.print(f"[green]Daemon stopped:[/] pid={metadata.pid} scope=global")
    if isinstance(stop_result, DaemonStopResult):
        signals = ",".join(stop_result.signal_sequence) if stop_result.signal_sequence else "none"
        console.print(
            "reason="
            f"{stop_result.stop_reason} "
            f"signals={signals} "
            f"escalated={stop_result.escalated_to_sigkill} "
            f"pgid={stop_result.process_group_id or '-'} "
            f"stale_socket_removed={stop_result.stale_socket_removed}"
        )


def _restart_daemon_command(workspace_root: str | None) -> None:
    stop_result = stop_daemon(workspace_root, None)
    metadata = ensure_daemon_started(workspace_root, None)
    if metadata is None:
        return
    if isinstance(stop_result, DaemonStopResult):
        signals = ",".join(stop_result.signal_sequence) if stop_result.signal_sequence else "none"
        console.print(
            f"[dim]Previous daemon stop:[/] reason={stop_result.stop_reason} signals={signals} escalated={stop_result.escalated_to_sigkill}"
        )
    console.print(f"[green]Daemon restarted:[/] {metadata.transport_endpoint} (pid={metadata.pid})")


def _print_dashboard_url(workspace_root: str | None, *, open_browser: bool = False) -> None:
    metadata = ensure_daemon_started(workspace_root, None)
    if metadata is None:
        return
    dashboard_url = f"{metadata.base_url}/dashboard"
    console.print(f"[green]Dashboard ready:[/] {dashboard_url}")
    if open_browser:
        opened = webbrowser.open(dashboard_url)
        console.print(f"browser_opened={opened}")


def _build_dashboard_frontend() -> None:
    result = ensure_dashboard_frontend_built(static_root=Path(__file__).with_name("management") / "static")
    if result.status == "up_to_date":
        console.print(f"[green]Dashboard frontend up to date:[/] {result.dist_index_path}")
        return
    if result.status == "built":
        console.print(f"[green]Dashboard frontend built:[/] {result.dist_index_path}")
        return
    message = f" message={result.message}" if result.message else ""
    returncode = f" returncode={result.returncode}" if result.returncode is not None else ""
    console.print(
        f"[red]Dashboard frontend build failed:[/] status={result.status}{returncode}{message}"
    )
    raise click.ClickException(f"dashboard_frontend_build_failed:{result.status}")


def _resolve_stash_content(text_parts: tuple[str, ...]) -> str:
    inline_text = " ".join(part for part in text_parts if part.strip()).strip()
    if inline_text:
        return inline_text

    stdin_text = click.get_text_stream("stdin").read().strip()
    if stdin_text:
        return stdin_text

    raise click.UsageError("Provide stash text as arguments or via stdin.")


def _with_runtime(workspace_root: str | None, action: Callable[[Any], _T]) -> _T:
    runtime = create_runtime(workspace_root_override=workspace_root)
    try:
        return action(runtime)
    finally:
        runtime.close()


def _with_management_service(
    workspace_root: str | None,
    action: Callable[[ManagementService], _T],
    *,
    workspace_id: str | None | object = ...,
) -> _T:
    return _with_runtime(
        workspace_root,
        lambda runtime: action(_build_management_service(runtime, workspace_id=workspace_id)),
    )


def _stash_thought(workspace_root: str | None, content: str) -> None:
    def _run(runtime) -> None:
        if runtime.journal is None:
            raise RuntimeError("journal_not_initialized")
        payload = RecordThoughtOperation(runtime.journal, runtime.task_queue, runtime.workspace_id).execute(content)
        click.echo(f"Thought stashed successfully (ID: {payload['entry']['id']})")

    _with_runtime(workspace_root, _run)


def _prefetch_embedding_model(workspace_root: str | None) -> None:
    def _run(runtime) -> None:
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

    _with_runtime(workspace_root, _run)


def _cancel_task(task_id: str, workspace_root: str | None, reason: str, json_output: bool) -> None:
    payload = _with_management_service(
        workspace_root,
        lambda service: service.cancel_task(task_id, cancelled_by="cli", reason=reason),
    )
    if json_output:
        click.echo(json.dumps(payload, sort_keys=True))
        return
    console.print(f"[green]{payload['status']}[/]: {task_id}")
    console.print(f"signal_sent={payload['signal_sent']} status={payload['task']['status']}")


def _show_task(task_id: str, workspace_root: str | None, json_output: bool) -> None:
    payload = _with_management_service(
        workspace_root,
        lambda service: service.get_task_detail(task_id),
        workspace_id=None,
    )
    if json_output:
        click.echo(json.dumps(payload.model_dump(), sort_keys=True))
        return
    _render_task_detail(payload)


def _list_tasks(workspace_root: str | None, status: str | None, limit: int, json_output: bool) -> None:
    def _run(service: ManagementService) -> None:
        payload = service.list_tasks(
            status=status,
            limit=limit,
        )
        if json_output:
            click.echo(json.dumps(payload.model_dump(), sort_keys=True))
            return
        _render_task_table(payload)

    _with_management_service(workspace_root, _run, workspace_id=None)


def _show_search_health(workspace_root: str | None, json_output: bool) -> None:
    def _run(service: ManagementService) -> None:
        payload = service.get_health()
        if json_output:
            click.echo(json.dumps(payload.search.model_dump(), sort_keys=True))
            return
        _render_search_health_table(payload.search)

    _with_management_service(workspace_root, _run)


def _repair_search_index(workspace_root: str | None, json_output: bool) -> None:
    payload = _with_management_service(workspace_root, lambda service: service.repair_search_index())
    if json_output:
        click.echo(json.dumps(payload, sort_keys=True))
        return
    if not payload["rebuilt"]:
        console.print(f"[yellow]Search repair skipped:[/] {payload['reason']}")
        return
    console.print(f"[green]Search index rebuilt:[/] {payload['records_indexed']} records")


def _show_search_debug(
    workspace_root: str | None,
    query: str,
    limit: int,
    json_output: bool,
) -> None:
    payload = _with_runtime(
        workspace_root,
        lambda runtime: search_memory_records_service(
            runtime,
            {"query": query, "limit": limit, "debug": True},
            caller_kind="operator",
        ),
    )
    if payload.get("status") != "ok":
        raise RuntimeError(str(payload.get("error") or "search_debug_failed"))

    results = payload.get("results")
    result_count = len(results) if isinstance(results, list) else 0
    search_diagnostics = payload.get("search_diagnostics")
    timing_ms = payload.get("timing_ms")
    cli_payload = dict(payload)
    cli_payload["query"] = query
    cli_payload["result_count"] = result_count
    cli_payload["semantic_candidate_strategy"] = (
        search_diagnostics.get("semantic_candidate_strategy")
        if isinstance(search_diagnostics, dict)
        else None
    )
    cli_payload["total_timing_ms"] = (
        timing_ms.get("total")
        if isinstance(timing_ms, dict)
        else None
    )

    if json_output:
        click.echo(json.dumps(cli_payload, sort_keys=True))
        return

    console.print(f"[bold]Query:[/] {query}")
    console.print(f"[bold]Result count:[/] {result_count}")
    console.print(
        "[bold]Semantic candidate strategy:[/] "
        f"{cli_payload['semantic_candidate_strategy'] or '-'}"
    )
    total_timing_ms = cli_payload["total_timing_ms"]
    total_timing_label = "-"
    if isinstance(total_timing_ms, int | float):
        total_timing_label = f"{total_timing_ms:.3f}ms"
    console.print(f"[bold]Total timing:[/] {total_timing_label}")

    timing_breakdown = timing_ms if isinstance(timing_ms, dict) else {}
    timing_rows = [
        (name, value)
        for name, value in timing_breakdown.items()
        if name != "total" and isinstance(value, int | float)
    ]
    timing_rows.sort(key=lambda item: item[1], reverse=True)

    timing_table = Table(title="Timing Breakdown")
    timing_table.add_column("Component")
    timing_table.add_column("Time", justify="right")
    if not timing_rows:
        timing_table.add_row("-", "-")
    for name, value in timing_rows:
        timing_table.add_row(name, f"{value:.3f}ms")
    console.print(timing_table)

    if isinstance(results, list):
        _render_search_debug_results(results)


def _render_search_debug_results(results: list[object]) -> None:
    table = Table(title="Ranked Search Results")
    table.add_column("#", justify="right", no_wrap=True)
    table.add_column("Score", justify="right", no_wrap=True)
    table.add_column("Title", overflow="fold")
    table.add_column("Summary", overflow="fold")
    table.add_column("Evidence", overflow="fold")
    if not results:
        table.add_row("-", "-", "No results", "-", "-")
        console.print(table)
        return

    for index, item in enumerate(results, start=1):
        if not isinstance(item, dict):
            continue
        score = item.get("score")
        score_text = f"{float(score):.3f}" if isinstance(score, int | float) else "-"
        table.add_row(
            str(index),
            score_text,
            str(item.get("title") or item.get("memory_id") or "-"),
            str(item.get("summary") or "-"),
            _format_ranking_evidence(item.get("ranking_debug")),
        )
    console.print(table)


def _format_ranking_evidence(ranking_debug: object) -> str:
    if not isinstance(ranking_debug, dict):
        return "-"

    evidence: list[str] = []
    if ranking_debug.get("matched_by_keyword"):
        evidence.append("keyword")
    if ranking_debug.get("matched_by_semantic"):
        evidence.append("semantic")
    if ranking_debug.get("workspace_match"):
        multiplier = ranking_debug.get("workspace_multiplier")
        if isinstance(multiplier, int | float):
            evidence.append(f"workspace×{float(multiplier):.2f}")
        else:
            evidence.append("workspace")
    if ranking_debug.get("expanded_by_graph"):
        link_type = ranking_debug.get("graph_link_type")
        seed_id = ranking_debug.get("graph_seed_id")
        graph_label = "graph" if not isinstance(link_type, str) else f"graph:{link_type}"
        if isinstance(seed_id, str) and seed_id:
            graph_label = f"{graph_label}@{seed_id[:8]}"
        evidence.append(graph_label)

    numeric_bits: list[str] = []
    for key, label in (
        ("semantic_score", "sem"),
        ("keyword_token_coverage", "kw"),
        ("graph_support_bonus", "graph"),
        ("access_bonus", "access"),
        ("authority_multiplier", "authority"),
        ("final_score", "final"),
    ):
        value = ranking_debug.get(key)
        if isinstance(value, int | float):
            numeric_bits.append(f"{label}={float(value):.2f}")

    bits = evidence + numeric_bits
    return ", ".join(bits) if bits else "-"


def _enqueue_agent(agent_name: str, workspace_root: str | None, force: bool) -> None:
    ensure_daemon_started(workspace_root, None)
    payload = _with_management_service(
        workspace_root,
        lambda service: service.enqueue_background_task(agent_name, force=force),
    )
    console.print(f"[green]{payload['status']}[/]: {agent_name}")
    console.print(f"task_id={payload['task']['id']} status={payload['task']['status']}")


def _enqueue_all_agents(workspace_root: str | None, force: bool) -> None:
    ensure_daemon_started(workspace_root, None)
    results = _with_management_service(
        workspace_root,
        lambda service: [service.enqueue_background_task(task_name, force=force) for task_name in TRIGGERABLE_BACKGROUND_TASK_NAMES],
    )
    created_count = sum(1 for result in results if result["created"])
    console.print(f"[green]Agents queued:[/] {created_count}/{len(results)} newly created")
    for result in results:
        console.print(f"- {result['task']['task_name']}: {result['status']}")


def _show_stats_command(workspace_root: str | None, watch: bool, interval: float, verbose: bool) -> None:
    _with_runtime(
        workspace_root,
        lambda runtime: _show_stats(runtime, watch=watch, interval_seconds=interval, verbose=verbose),
    )


def _show_operator_health_snapshot(workspace_root: str | None, json_output: bool) -> None:
    def _run(service: ManagementService) -> None:
        payload = service.get_operator_health_snapshot()
        if json_output:
            click.echo(json.dumps(payload.model_dump(), sort_keys=True))
            return
        _render_operator_health_snapshot(payload)

    _with_management_service(workspace_root, _run, workspace_id=None)


def _import_markdown_files(
    file_paths: tuple[str, ...],
    workspace_root: str | None,
    workspace_ids: tuple[str, ...],
    thought: bool,
) -> None:
    if not file_paths:
        raise click.UsageError("Provide at least one markdown file path or glob pattern.")

    def _run(runtime) -> None:
        resolved_workspace_id = None
        if workspace_ids:
            resolved_workspace_id = next((wid.strip() for wid in workspace_ids if wid.strip()), None)
        elif runtime.workspace_id is not None:
            resolved_workspace_id = runtime.workspace_id

        if thought:
            if runtime.journal is None:
                raise RuntimeError("journal_not_initialized")

            recorded = record_markdown_memory_paths_as_thoughts(
                runtime.journal,
                list(file_paths),
                workspace_id=resolved_workspace_id,
            )
            if runtime.task_queue is not None:
                resume_paused_recurring_maintenance(runtime.task_queue)
            for entry_id, file_name in recorded:
                console.print(f"[green]Recorded thought:[/] {file_name} (entry {entry_id})")
            console.print(f"[green]Recorded total:[/] {len(recorded)} thoughts → buffer")
            return

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

    _with_runtime(workspace_root, _run)


def _resolve_default_sqlite_db_path() -> Path:
    config = load_config()
    return resolve_memory_path(config) / "indices" / "memory.db"


@contextmanager
def _suppress_config_bootstrap_logging(enabled: bool):
    if not enabled:
        yield
        return
    config_logger = logging.getLogger("mcp_memory.config")
    previous_disabled = config_logger.disabled
    config_logger.disabled = True
    try:
        yield
    finally:
        config_logger.disabled = previous_disabled


@contextmanager
def _suppress_all_logging(enabled: bool):
    if not enabled:
        yield
        return
    previous_disable_level = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        yield
    finally:
        logging.disable(previous_disable_level)


def _migrate_sqlite_to_postgres_command(
    sqlite_path: Path | None,
    postgres_dsn: str | None,
    dry_run: bool,
    allow_non_empty_target: bool,
    json_output: bool,
) -> None:
    with _suppress_all_logging(json_output):
        with _suppress_config_bootstrap_logging(json_output):
            config = load_config()
            default_sqlite_path = _resolve_default_sqlite_db_path() if sqlite_path is None else sqlite_path
        resolved_sqlite_path = default_sqlite_path.expanduser().resolve()
        resolved_postgres_dsn = postgres_dsn.strip() if isinstance(postgres_dsn, str) and postgres_dsn.strip() else config.storage.postgres.dsn.strip()
        if not resolved_postgres_dsn:
            raise click.UsageError("Provide --postgres-dsn or set storage.postgres.dsn in config.toml.")

        summary = migrate_sqlite_to_postgres(
            resolved_sqlite_path,
            replace(config.storage.postgres, dsn=resolved_postgres_dsn),
            dry_run=dry_run,
            allow_non_empty_target=allow_non_empty_target,
        )
    if json_output:
        click.echo(json.dumps(summary.to_dict(), sort_keys=True))
        return

    mode_label = "Dry run" if dry_run else "Migration complete"
    console.print(f"[green]{mode_label}:[/] SQLite → Postgres")
    console.print(f"sqlite_path={summary.sqlite_path}")
    console.print(f"target_existing_memories={summary.target_existing_memories}")
    console.print(
        "counts "
        f"memories={summary.memories} "
        f"workspace_mappings={summary.workspace_mappings} "
        f"tag_names={summary.tag_names} "
        f"tag_mappings={summary.tag_mappings} "
        f"links={summary.links} "
        f"skipped_links={summary.skipped_links}"
    )
    if allow_non_empty_target:
        console.print("[yellow]Non-empty target mode enabled[/]")


def _render_logs_table(payload) -> None:
    table = Table(title="Runtime Logs")
    table.add_column("Time", no_wrap=True)
    table.add_column("Level", no_wrap=True)
    table.add_column("Source", no_wrap=True)
    table.add_column("Logger", no_wrap=True)
    table.add_column("Message")
    if not payload.logs:
        table.add_row("-", "-", "-", "-", "No logs found")
    for entry in payload.logs:
        table.add_row(
            _format_timestamp(entry.created_at),
            entry.level,
            entry.source,
            entry.logger_name,
            entry.message,
        )
    console.print(table)


def _render_log_summary(payload) -> None:
    console.print(f"[bold]Matching logs:[/] {payload.total}")

    by_level = Table(title="By Level")
    by_level.add_column("Level")
    by_level.add_column("Count", justify="right")
    if not payload.by_level:
        by_level.add_row("-", "0")
    for level_name, count in sorted(payload.by_level.items()):
        by_level.add_row(level_name, str(count))
    console.print(by_level)

    by_source = Table(title="By Source")
    by_source.add_column("Source")
    by_source.add_column("Count", justify="right")
    if not payload.by_source:
        by_source.add_row("-", "0")
    for source_name, count in sorted(payload.by_source.items()):
        by_source.add_row(source_name, str(count))
    console.print(by_source)


def _show_logs(
    workspace_root: str | None,
    scope: str,
    workspace_id: str | None,
    limit: int,
    level: str | None,
    logger_name: str | None,
    source: str | None,
    query: str | None,
    after: float | None,
    before: float | None,
    json_output: bool,
) -> None:
    def _run(service: ManagementService) -> None:
        effective_workspace_id = _resolve_operator_workspace_filter(service, scope=scope, workspace_id=workspace_id)
        payload = service.list_logs(
            workspace_id=effective_workspace_id,
            level=None if level is None else level.upper(),
            logger_name=logger_name,
            source=source,
            query=query,
            after=after,
            before=before,
            limit=limit,
        )
        if json_output:
            click.echo(json.dumps(payload.model_dump(), sort_keys=True))
            return
        _render_logs_table(payload)

    _with_management_service(workspace_root, _run, workspace_id=None)


def _summarize_logs(
    workspace_root: str | None,
    scope: str,
    workspace_id: str | None,
    level: str | None,
    logger_name: str | None,
    source: str | None,
    query: str | None,
    after: float | None,
    before: float | None,
    json_output: bool,
) -> None:
    def _run(service: ManagementService) -> None:
        effective_workspace_id = _resolve_operator_workspace_filter(service, scope=scope, workspace_id=workspace_id)
        payload = service.summarize_logs(
            workspace_id=effective_workspace_id,
            level=None if level is None else level.upper(),
            logger_name=logger_name,
            source=source,
            query=query,
            after=after,
            before=before,
        )
        if json_output:
            click.echo(json.dumps(payload.model_dump(), sort_keys=True))
            return
        _render_log_summary(payload)

    _with_management_service(workspace_root, _run, workspace_id=None)


def _prune_logs(
    workspace_root: str | None,
    scope: str,
    workspace_id: str | None,
    max_runtime_logs: int | None,
    max_log_age_days: int | None,
    json_output: bool,
) -> None:
    def _run(service: ManagementService) -> None:
        effective_workspace_id = _resolve_operator_workspace_filter(service, scope=scope, workspace_id=workspace_id)
        payload = service.prune_logs(
            workspace_id=effective_workspace_id,
            max_runtime_logs=max_runtime_logs,
            max_log_age_days=max_log_age_days,
        )
        if json_output:
            click.echo(json.dumps(payload.model_dump(), sort_keys=True))
            return
        console.print(f"[green]Pruned logs:[/] {payload.deleted}")
        console.print(
            f"policy max_runtime_logs={payload.max_runtime_logs} max_log_age_days={payload.max_log_age_days}"
        )

    _with_management_service(workspace_root, _run, workspace_id=None)


def _resolve_operator_workspace_filter(
    service: ManagementService,
    *,
    scope: str,
    workspace_id: str | None,
) -> str | None:
    explicit_workspace_id = None
    if workspace_id is not None and workspace_id.strip():
        explicit_workspace_id = workspace_id.strip()
    return resolve_workspace_id_for_policy(
        ScopePolicyKind.GLOBAL_DEFAULT_FILTERABLE,
        scope=scope,
        workspace_id=explicit_workspace_id,
        current_workspace_id=service.workspace_id,
    )


def _render_memory_metrics_table(overview, journal_counts: dict[str, int]) -> None:
    metrics = Table(title="Memory Metrics")
    metrics.add_column("Metric")
    metrics.add_column("Value", justify="right")
    metrics.add_row("Total memories", str(overview.memory_metrics.total_memories))
    metrics.add_row("Total memory lines", str(overview.memory_metrics.total_memory_lines))
    metrics.add_row("Total summary lines", str(overview.memory_metrics.total_summary_lines))
    metrics.add_row("Total lines compressed", str(overview.memory_metrics.total_lines_compressed))
    metrics.add_row("Pending journal entries", str(journal_counts.get("pending", 0)))
    metrics.add_row("Processed journal entries", str(journal_counts.get("processed", 0)))
    metrics.add_row("Archived journal entries", str(journal_counts.get("archived", 0)))
    metrics.add_row("Pending journal lines", str(overview.memory_metrics.thought_buffer_lines))
    console.print(metrics)


def _render_search_health_table(search_health) -> None:
    table = Table(title="Search Health")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Semantic enabled", str(search_health.semantic_enabled))
    table.add_row("Available", str(search_health.available))
    table.add_row("Degraded", str(search_health.degraded))
    table.add_row("Fallback count", str(search_health.fallback_count))
    table.add_row("Rebuild count", str(search_health.rebuild_count))
    table.add_row("Last integrity check", search_health.last_integrity_check_at or "-")
    table.add_row("Last failure", search_health.last_failure_at or "-")
    table.add_row("Last recovery", search_health.last_recovery_at or "-")
    table.add_row("Last error", search_health.last_error or search_health.integrity_check_error or "-")
    console.print(table)


def _render_embedding_repair_table(search_health) -> None:
    table = Table(title="Embedding Repair Backlog")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Background repair enabled", str(search_health.background_repair_enabled))
    table.add_row("Queued repairs", str(search_health.queued_repair_backlog_count))
    table.add_row("Running repairs", str(search_health.running_repair_count))
    table.add_row(
        "Oldest queued age",
        "-" if search_health.oldest_queued_repair_age_seconds is None else _format_age(search_health.oldest_queued_repair_age_seconds),
    )
    table.add_row("Repair waits", str(search_health.repair_wait_count))
    console.print(table)


def _render_execution_attempt_health_table(execution_attempts) -> None:
    table = Table(title="Execution Attempts")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Stale threshold", f"{execution_attempts.stale_after_seconds:.0f}s")
    table.add_row("Running tasks", str(execution_attempts.running_task_count))
    table.add_row("Running attempts", str(execution_attempts.running_attempt_count))
    table.add_row("Fresh attempts", str(execution_attempts.fresh_attempt_count))
    table.add_row("Stale attempts", str(execution_attempts.stale_attempt_count))
    table.add_row("Missing attempts", str(execution_attempts.missing_attempt_count))
    table.add_row("Live subprocesses", str(execution_attempts.live_subprocess_count))
    table.add_row("Dead subprocesses", str(execution_attempts.dead_subprocess_count))
    console.print(table)


def _render_cache_health_table(cache_health) -> None:
    table = Table(title="Cache")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Enabled", str(cache_health.enabled))
    table.add_row("Mode", cache_health.mode or "-")
    table.add_row("State", cache_health.state)
    table.add_row("Path", cache_health.path or "-")
    metrics = cache_health.metrics
    table.add_row("Cached search responses", str(metrics.cached_search_result_count))
    table.add_row("Cached read records", str(metrics.cached_read_record_count))
    table.add_row("Cached projections", str(metrics.cached_projection_count))
    table.add_row("Search requests", str(metrics.search_requests))
    table.add_row(
        "Fresh exact hits",
        f"{metrics.fresh_exact_search_hits} ({metrics.fresh_exact_search_hit_rate:.1%})",
    )
    table.add_row(
        "Stale exact fallbacks",
        f"{metrics.stale_exact_search_fallbacks} ({metrics.stale_exact_search_fallback_rate:.1%})",
    )
    table.add_row(
        "Projection fallbacks",
        f"{metrics.projection_fallbacks} ({metrics.projection_fallback_rate:.1%})",
    )
    table.add_row("Read requests", str(metrics.read_requests))
    table.add_row(
        "Validated read hits",
        f"{metrics.validated_read_hits} ({metrics.validated_read_hit_rate:.1%})",
    )
    table.add_row(
        "Validation mismatches",
        f"{metrics.read_validation_mismatches} ({metrics.read_validation_mismatch_rate:.1%})",
    )
    table.add_row(
        "Validation failures",
        f"{metrics.read_validation_failures} ({metrics.read_validation_failure_rate:.1%})",
    )
    table.add_row("Warmed projection rows", str(metrics.warmed_projection_rows))
    recent = metrics.recent
    table.add_row("Recent window", f"{recent.window_minutes}m")
    table.add_row("Recent search requests", str(recent.search_requests))
    table.add_row(
        "Recent fresh exact hits",
        f"{recent.fresh_exact_search_hits} ({recent.fresh_exact_search_hit_rate:.1%})",
    )
    table.add_row(
        "Recent stale exact fallbacks",
        f"{recent.stale_exact_search_fallbacks} ({recent.stale_exact_search_fallback_rate:.1%})",
    )
    table.add_row(
        "Recent projection fallbacks",
        f"{recent.projection_fallbacks} ({recent.projection_fallback_rate:.1%})",
    )
    table.add_row("Recent read requests", str(recent.read_requests))
    table.add_row(
        "Recent validated read hits",
        f"{recent.validated_read_hits} ({recent.validated_read_hit_rate:.1%})",
    )
    table.add_row(
        "Recent validation mismatches",
        f"{recent.read_validation_mismatches} ({recent.read_validation_mismatch_rate:.1%})",
    )
    table.add_row(
        "Recent validation failures",
        f"{recent.read_validation_failures} ({recent.read_validation_failure_rate:.1%})",
    )
    table.add_row("Recent warmed projection rows", str(recent.warmed_projection_rows))
    console.print(table)


def _render_agent_table(overview) -> None:
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


def _render_provider_usage_table(overview) -> None:
    provider_table = Table(title="AI Provider Usage")
    provider_table.add_column("Task")
    provider_table.add_column("Provider", no_wrap=True)
    provider_table.add_column("Model")
    provider_table.add_column("Calls (1h)", justify="right")
    provider_table.add_column("Calls (24h)", justify="right")
    provider_table.add_column("Failures (1h)", justify="right")
    provider_table.add_column("Failures (24h)", justify="right")
    provider_table.add_column("Avg Duration (1h)", justify="right")
    provider_table.add_column("Avg Duration (24h)", justify="right")
    if not overview.provider_usage:
        provider_table.add_row("-", "-", "-", "0", "0", "0", "0", "0.00s", "0.00s")
    for usage in overview.provider_usage:
        provider_table.add_row(
            usage.task_name or "-",
            usage.provider_key,
            usage.model_name,
            str(usage.calls_last_hour),
            str(usage.calls_last_day),
            str(usage.failures_last_hour),
            str(usage.failures_last_day),
            f"{usage.avg_duration_last_hour:.2f}s",
            f"{usage.avg_duration_last_day:.2f}s",
        )
    console.print(provider_table)


def _render_top_reads_table(overview) -> None:
    top_reads_table = Table(title="Top Read Memories")
    top_reads_table.add_column("Reads", justify="right", no_wrap=True)
    top_reads_table.add_column("Type", no_wrap=True)
    top_reads_table.add_column("Status", no_wrap=True)
    top_reads_table.add_column("Title")
    if not overview.top_read_memories:
        top_reads_table.add_row("0", "-", "-", "No memories have been read yet")
    for record in overview.top_read_memories:
        top_reads_table.add_row(
            str(record.read_count),
            record.type,
            record.status,
            record.title,
        )
    console.print(top_reads_table)


def _render_queue_diagnostics_table(overview) -> None:
    queue_table = Table(title="Next Pending Tasks")
    queue_table.add_column("Task", no_wrap=True)
    queue_table.add_column("State", no_wrap=True)
    queue_table.add_column("Priority", justify="right", no_wrap=True)
    queue_table.add_column("Age", no_wrap=True)
    queue_table.add_column("Ready / Overdue", no_wrap=True)
    queue_table.add_column("Trigger")
    queue_table.add_column("Workspace")
    if not overview.queue_diagnostics:
        queue_table.add_row("-", "-", "-", "-", "No pending tasks", "-", "-")
    for item in overview.queue_diagnostics:
        ready_text = (
            f"overdue {_format_age(item.overdue_seconds)}"
            if item.pending_state == "runnable"
            else f"ready in {_format_age(item.ready_in_seconds)}"
        )
        queue_table.add_row(
            item.task_name,
            item.pending_state,
            str(item.priority),
            _format_age(item.age_seconds),
            ready_text,
            item.trigger or "-",
            item.workspace_id or "global",
        )
    console.print(queue_table)


def _print_agent_details(overview) -> None:
    console.print("[bold]Agent Details[/]")
    for agent in overview.agent_runs:
        metadata_text = _format_run_result_metadata(agent.last_result_metadata)
        console.print(
            "- "
            f"{agent.task_name}: "
            f"running={agent.running_count} "
            f"next={_format_age(agent.seconds_until_next_run)} "
            f"last_status={agent.last_status or 'never'} "
            f"last_result={agent.last_result_summary or '-'} "
            f"metadata={metadata_text or '-'}"
        )


def _print_recent_agent_runs(overview) -> None:
    console.print("[bold]Recent Agent Runs[/]")
    for run in overview.recent_agent_runs:
        metadata_text = _format_run_result_metadata(run.result_metadata)
        console.print(
            "- "
            f"{run.task_name}: "
            f"status={run.status} "
            f"duration={run.duration_seconds:.2f}s "
            f"result={run.result_summary or '-'} "
            f"metadata={metadata_text or '-'} "
            f"error={run.error_text or '-'}"
        )


def _format_run_result_metadata(metadata) -> str | None:
    parts: list[str] = []
    if getattr(metadata, "strategy_used", None) is not None:
        parts.append(f"strategy={metadata.strategy_used}")
    if getattr(metadata, "strategy_fallback_reason", None) is not None:
        parts.append(f"strategy_fallback={metadata.strategy_fallback_reason}")
    if getattr(metadata, "strategy_selection_mode", None) is not None:
        parts.append(f"selector={metadata.strategy_selection_mode}")
    if getattr(metadata, "strategy_selection_reason", None) is not None:
        parts.append(f"selector_reason={metadata.strategy_selection_reason}")
    selector_scores = _format_strategy_selection_scores(getattr(metadata, "strategy_selection_scores", {}), limit=2)
    if selector_scores is not None:
        parts.append(f"selector_scores={selector_scores}")
    if getattr(metadata, "candidate_count", None) is not None:
        parts.append(f"candidate_count={metadata.candidate_count}")
    sampled_memory_ids = getattr(metadata, "sampled_memory_ids", [])
    if sampled_memory_ids:
        parts.append(f"sampled={len(sampled_memory_ids)}")
    if getattr(metadata, "grouping_strategy_used", None) is not None:
        parts.append(f"grouping={metadata.grouping_strategy_used}")
    if getattr(metadata, "grouping_fallback_reason", None) is not None:
        parts.append(f"grouping_fallback={metadata.grouping_fallback_reason}")
    if getattr(metadata, "group_count", None) is not None:
        parts.append(f"group_count={metadata.group_count}")
    return ", ".join(parts) if parts else None


def _format_strategy_selection_scores(scores: dict[str, float], *, limit: int = 2) -> str | None:
    if not scores:
        return None
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    shown = [f"{name}:{value:.2f}" for name, value in ranked[: max(limit, 1)]]
    remaining = len(ranked) - len(shown)
    if remaining > 0:
        shown.append(f"+{remaining} more")
    return ", ".join(shown)


def _format_ingest_audit_hint(ingest_audit) -> str | None:
    if ingest_audit is None:
        return None
    has_signal = any(
        [
            getattr(ingest_audit, "claimed_count", 0) > 0,
            getattr(ingest_audit, "handled_count", 0) > 0,
            getattr(ingest_audit, "released_count", 0) > 0,
            getattr(ingest_audit, "touched_count", 0) > 0,
            getattr(ingest_audit, "mutations", None) is not None,
            bool(getattr(ingest_audit, "entry_dispositions", [])),
        ]
    )
    if not has_signal:
        return None
    parts = [
        f"handled={ingest_audit.handled_count}",
        f"released={ingest_audit.released_count}",
        f"mut={ingest_audit.mutations if ingest_audit.mutations is not None else 0}",
        f"touched={ingest_audit.touched_count}",
    ]
    if getattr(ingest_audit, "provider_reported_mutations", None) is not None:
        parts.append(f"provider_mut={ingest_audit.provider_reported_mutations}")
    return "\n".join(parts)


def _render_recent_agent_runs_table(payload) -> None:
    table = Table(title="Recent Agent Runs")
    table.add_column("Task ID", no_wrap=True)
    table.add_column("Task", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Duration", justify="right", no_wrap=True)
    table.add_column("Completed", no_wrap=True)
    table.add_column("Selection")
    table.add_column("Grouping")
    table.add_column("Ingest", overflow="fold")
    table.add_column("Result")
    if not payload.runs:
        table.add_row("-", "-", "-", "-", "-", "-", "-", "No recent runs")
    for run in payload.runs:
        selection = []
        grouping = []
        if run.result_metadata.requested_strategy is not None:
            selection.append(f"requested={run.result_metadata.requested_strategy}")
        if run.result_metadata.strategy_used is not None:
            selection.append(f"used={run.result_metadata.strategy_used}")
        if run.result_metadata.strategy_fallback_reason is not None:
            selection.append(f"fallback={run.result_metadata.strategy_fallback_reason}")
        if run.result_metadata.strategy_selection_mode is not None:
            selection.append(f"selector={run.result_metadata.strategy_selection_mode}")
        if run.result_metadata.strategy_selection_reason is not None:
            selection.append(f"why={run.result_metadata.strategy_selection_reason}")
        selector_scores = _format_strategy_selection_scores(run.result_metadata.strategy_selection_scores, limit=2)
        if selector_scores is not None:
            selection.append(f"scores={selector_scores}")
        if run.result_metadata.candidate_count is not None:
            selection.append(f"candidates={run.result_metadata.candidate_count}")
        if run.result_metadata.requested_grouping_strategy is not None:
            grouping.append(f"requested={run.result_metadata.requested_grouping_strategy}")
        if run.result_metadata.grouping_strategy_used is not None:
            grouping.append(f"used={run.result_metadata.grouping_strategy_used}")
        if run.result_metadata.grouping_fallback_reason is not None:
            grouping.append(f"fallback={run.result_metadata.grouping_fallback_reason}")
        if run.result_metadata.group_count is not None:
            grouping.append(f"groups={run.result_metadata.group_count}")
        table.add_row(
            run.task_id or "-",
            run.task_name,
            run.status,
            f"{run.duration_seconds:.2f}s",
            _format_timestamp(run.completed_at),
            " ".join(selection) or "-",
            " ".join(grouping) or "-",
            _format_ingest_audit_hint(run.ingest_audit) or "-",
            run.result_summary or "-",
        )
    console.print(table)
    ingest_hints = [
        (run.task_id or run.task_name, _format_ingest_audit_hint(run.ingest_audit))
        for run in payload.runs
    ]
    ingest_hints = [(label, hint) for label, hint in ingest_hints if hint is not None]
    if ingest_hints:
        console.print("[bold]Ingest audit hints[/]")
        for label, hint in ingest_hints:
            console.print(f"- {label}: {hint.replace(chr(10), ' ')}")


def _render_task_detail(payload) -> None:
    task = payload.task
    console.print(f"[bold]Task ID:[/] {task['id']}")
    console.print(f"[bold]Task:[/] {task['task_name']}")
    console.print(f"[bold]Status:[/] {task['status']}")
    console.print(f"[bold]Workspace:[/] {task['workspace_id'] or '-'}")
    console.print(f"[bold]Updated:[/] {_format_timestamp(task.get('updated_at'))}")
    if not payload.runs:
        console.print("[yellow]No persisted task runs found.[/]")
        return

    _render_recent_agent_runs_table(type("_Payload", (), {"runs": payload.runs})())
    for index, run in enumerate(payload.runs, start=1):
        console.print(f"\n[bold]Stored run {index}[/]")
        console.print(
            f"status={run.status} completed={_format_timestamp(run.completed_at)} duration={run.duration_seconds:.2f}s error={run.error_text or '-'}"
        )
        metadata_text = _format_run_result_metadata(run.result_metadata)
        if metadata_text is not None:
            console.print(f"metadata={metadata_text}")
        ingest_hint = _format_ingest_audit_hint(run.ingest_audit)
        if ingest_hint is not None:
            console.print(f"ingest:\n{ingest_hint}")
        if run.ingest_audit.entry_dispositions:
            dispositions = Table(title=f"Entry Dispositions ({len(run.ingest_audit.entry_dispositions)})")
            dispositions.add_column("Entry", justify="right", no_wrap=True)
            dispositions.add_column("Disposition", no_wrap=True)
            dispositions.add_column("Finalization", no_wrap=True)
            dispositions.add_column("Memory", no_wrap=True)
            dispositions.add_column("Reason")
            for disposition in run.ingest_audit.entry_dispositions:
                dispositions.add_row(
                    str(disposition.entry_id),
                    disposition.disposition,
                    disposition.finalization_status or "-",
                    disposition.memory_id or "-",
                    disposition.reason or "-",
                )
            console.print(dispositions)
        if run.result is not None:
            console.print("[bold]Stored result[/]")
            console.print_json(json.dumps(run.result, sort_keys=True))


def _render_quality_cleanup_candidates(payload) -> None:
    table = Table(title="Quality Cleanup Candidates")
    table.add_column("Priority", justify="right", no_wrap=True)
    table.add_column("Signals", justify="right", no_wrap=True)
    table.add_column("Type", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Conversion", no_wrap=True)
    table.add_column("Title")
    table.add_column("Criteria", overflow="fold")
    table.add_column("Suggested actions", overflow="fold")
    if not payload.candidates:
        table.add_row("-", "0", "-", "-", "-", "No cleanup candidates", "-", "-")
        console.print(table)
        return

    for candidate in payload.candidates:
        conversion_text = (
            "-"
            if candidate.conversion_rate is None
            else f"{candidate.conversion_rate:.2f} ({candidate.converted_search_count or 0}/{candidate.search_count or 0})"
        )
        table.add_row(
            str(candidate.priority_score),
            str(len(candidate.criteria)),
            candidate.memory_type,
            candidate.status,
            conversion_text,
            candidate.title,
            ", ".join(criterion.label for criterion in candidate.criteria) or "-",
            ", ".join(recommendation.label for recommendation in candidate.recommendations) or "-",
        )
    console.print(table)


def _build_sampling_summary(payload):
    return build_task_sampling_summary(payload.runs)


def _render_sampling_summary(summary) -> None:
    selection_table = Table(title="Selection Strategy Usage")
    selection_table.add_column("Strategy")
    selection_table.add_column("Runs", justify="right")
    selection_table.add_column("Fallbacks", justify="right")
    selection_table.add_column("Tasks")
    if not summary.selection:
        selection_table.add_row("-", "0", "0", "No strategy metadata recorded")
    for row in summary.selection:
        tasks_text = ", ".join(row.tasks) if row.tasks else "-"
        selection_table.add_row(row.name, str(row.runs), str(row.fallbacks), tasks_text)
    console.print(selection_table)

    grouping_table = Table(title="Ingest Grouping Strategy Usage")
    grouping_table.add_column("Grouping")
    grouping_table.add_column("Runs", justify="right")
    grouping_table.add_column("Fallbacks", justify="right")
    grouping_table.add_column("Tasks")
    if not summary.grouping:
        grouping_table.add_row("-", "0", "0", "No grouping metadata recorded")
    for row in summary.grouping:
        tasks_text = ", ".join(row.tasks) if row.tasks else "-"
        grouping_table.add_row(row.name, str(row.runs), str(row.fallbacks), tasks_text)
    console.print(grouping_table)

    utility_table = Table(title="Selection Strategy Utility by Task")
    utility_table.add_column("Task", no_wrap=True)
    utility_table.add_column("Strategy", no_wrap=True)
    utility_table.add_column("Runs", justify="right")
    utility_table.add_column("Fallbacks", justify="right")
    utility_table.add_column("Mutation runs", justify="right")
    utility_table.add_column("Total mutations", justify="right")
    utility_table.add_column("Tool calls", justify="right")
    utility_table.add_column("Avg candidates", justify="right")
    utility_table.add_column("Mutation rate", justify="right")
    utility_table.add_column("Mut/run", justify="right")
    utility_table.add_column("Mut/tool", justify="right")
    utility_table.add_column("No-op", justify="right")
    if not summary.selection_utility:
        utility_table.add_row("-", "-", "0", "0", "0", "0", "0", "-", "0.0%", "0.00", "-", "0 (0.0%)")
    for row in summary.selection_utility:
        utility_table.add_row(
            row.task_name,
            row.strategy_used,
            str(row.runs),
            str(row.fallback_count),
            str(row.mutation_runs),
            str(row.total_mutations),
            str(row.total_tool_calls),
            "-" if row.average_candidate_count is None else f"{row.average_candidate_count:.2f}",
            f"{row.mutation_rate:.1%}",
            f"{row.mutations_per_run:.2f}",
            "-" if row.mutations_per_tool_call is None else f"{row.mutations_per_tool_call:.2f}",
            f"{row.no_op_runs} ({row.no_op_rate:.1%})",
        )
    console.print(utility_table)

    selector_table = Table(title="Selector Behavior by Recent Runs")
    selector_table.add_column("Task", no_wrap=True)
    selector_table.add_column("Selector mode", no_wrap=True)
    selector_table.add_column("Strategy", no_wrap=True)
    selector_table.add_column("Reason family", no_wrap=True)
    selector_table.add_column("Runs", justify="right")
    selector_table.add_column("Fallbacks", justify="right")
    selector_table.add_column("Mutation runs", justify="right")
    selector_table.add_column("Total mutations", justify="right")
    selector_table.add_column("Tool calls", justify="right")
    selector_table.add_column("Avg candidates", justify="right")
    selector_table.add_column("No-op", justify="right")
    if not summary.selector_behavior:
        selector_table.add_row("-", "-", "-", "-", "0", "0", "0", "0", "0", "-", "0 (0.0%)")
    for row in summary.selector_behavior:
        selector_table.add_row(
            row.task_name,
            row.strategy_selection_mode,
            row.strategy_used,
            row.reason_family,
            str(row.runs),
            str(row.fallback_count),
            str(row.mutation_runs),
            str(row.total_mutations),
            str(row.total_tool_calls),
            "-" if row.average_candidate_count is None else f"{row.average_candidate_count:.2f}",
            f"{row.no_op_runs} ({row.no_op_rate:.1%})",
        )
    console.print(selector_table)


def _render_stats_snapshot(
    health,
    overview,
    journal_counts: dict[str, int],
    *,
    verbose: bool,
    watch: bool,
    interval_seconds: float,
) -> None:
    console.print("[bold]Stats:[/] global")
    console.print(f"[bold]Database:[/] {health.db_path}")
    if watch:
        console.print(f"[dim]Watching every {interval_seconds:.1f}s — press Ctrl+C to stop[/]")
        console.print(f"[dim]Refreshed:[/] {_format_timestamp(time.time())}")

    _render_search_health_table(health.search)
    _render_embedding_repair_table(health.search)
    _render_cache_health_table(overview.cache)
    _render_execution_attempt_health_table(overview.execution_attempts)
    _render_memory_metrics_table(overview, journal_counts)
    _render_queue_diagnostics_table(overview)
    _render_agent_table(overview)
    _render_provider_usage_table(overview)
    _render_top_reads_table(overview)
    if verbose:
        _print_agent_details(overview)
        _print_recent_agent_runs(overview)


def _show_stats(
    runtime,
    *,
    watch: bool,
    interval_seconds: float,
    verbose: bool,
) -> None:
    global_service = _build_management_service(runtime, workspace_id=None)

    try:
        while True:
            health = global_service.get_health()
            overview = global_service.get_overview()
            journal_counts = {} if runtime.journal is None else runtime.journal.count_by_status()

            if watch and console.is_terminal:
                console.clear()
            _render_stats_snapshot(
                health,
                overview,
                journal_counts,
                verbose=verbose,
                watch=watch,
                interval_seconds=interval_seconds,
            )

            if not watch:
                return
            time.sleep(interval_seconds)
    except KeyboardInterrupt:
        if watch:
            console.print("\n[dim]Stopped stats watch.[/]")


def _render_task_table(payload) -> None:
    table = Table(title="Tasks")
    table.add_column("Task ID")
    table.add_column("Task")
    table.add_column("Status", no_wrap=True)
    table.add_column("Selection", no_wrap=True)
    table.add_column("Grouping", no_wrap=True)
    table.add_column("Workspace")
    table.add_column("PID", justify="right")
    table.add_column("Request")
    table.add_column("Updated", no_wrap=True)
    table.add_column("Error")
    if not payload.tasks:
        table.add_row("-", "-", "-", "-", "-", "-", "-", "-", "No tasks found")
    for task in payload.tasks:
        table.add_row(
            task["id"],
            task["task_name"],
            task["status"],
            task.get("strategy") or "-",
            task.get("grouping_strategy") or "-",
            task["workspace_id"] or "-",
            "-" if task.get("subprocess_pid") is None else str(task["subprocess_pid"]),
            task.get("active_request_id") or "-",
            _format_timestamp(task.get("updated_at")),
            task.get("last_error") or task.get("cancellation_reason") or "-",
        )
    console.print(table)


def _render_ai_conversation_table(payload) -> None:
    table = Table(title="AI Conversations")
    table.add_column("Request")
    table.add_column("Attempt", justify="right")
    table.add_column("Task")
    table.add_column("PID", justify="right")
    table.add_column("Status", no_wrap=True)
    table.add_column("Duration", justify="right")
    table.add_column("Last Update", no_wrap=True)
    if not payload.conversations:
        table.add_row("-", "-", "-", "-", "-", "-", "-")
    for conversation in payload.conversations:
        completion_label = _format_timestamp(conversation.completed_at)
        if conversation.status == "running":
            completion_label = f"heartbeat {completion_label}"
        table.add_row(
            conversation.request_id,
            str(conversation.attempt),
            f"{conversation.task_name or '-'}\n{conversation.workspace_id or 'global'}",
            "-" if conversation.subprocess_pid is None else str(conversation.subprocess_pid),
            conversation.status,
            f"{conversation.duration_seconds:.2f}s",
            completion_label,
        )
    console.print(table)


def _render_operator_health_snapshot(payload) -> None:
    console.print(f"[bold]Operator Health Snapshot:[/] status={payload.status}")
    console.print(f"[bold]Generated:[/] {_format_timestamp(payload.generated_at)}")
    console.print(f"[bold]Alerts:[/] {', '.join(payload.alerts) if payload.alerts else 'none'}")

    runtime_table = Table(title="Runtime")
    runtime_table.add_column("Metric")
    runtime_table.add_column("Value")
    runtime_table.add_row("Runtime active", str(payload.health.runtime_active))
    runtime_table.add_row("Client count", str(payload.health.client_count))
    runtime_table.add_row("Task queue enabled", str(payload.health.task_queue_enabled))
    runtime_table.add_row("DB path", payload.health.db_path or "-")
    console.print(runtime_table)

    _render_cache_health_table(payload.health.cache)
    _render_search_health_table(payload.health.search)
    _render_execution_attempt_health_table(payload.health.execution_attempts)

    log_table = Table(title=f"Recent Logs ({payload.logs.window_minutes}m)")
    log_table.add_column("Metric")
    log_table.add_column("Value")
    log_table.add_row("Total", str(payload.logs.total))
    log_table.add_row(
        "By level",
        ", ".join(f"{level}={count}" for level, count in sorted(payload.logs.by_level.items())) or "none",
    )
    log_table.add_row("Recent errors", str(len(payload.logs.recent_errors)))
    console.print(log_table)
    if payload.logs.recent_errors:
        _render_logs_table(type("_Payload", (), {"logs": payload.logs.recent_errors})())

    warning_table = Table(title=f"Recent Warnings ({payload.warnings.window_minutes}m)")
    warning_table.add_column("Metric")
    warning_table.add_column("Value")
    warning_table.add_row("Total", str(payload.warnings.total))
    warning_table.add_row("Rows included", str(len(payload.warnings.recent)))
    console.print(warning_table)
    if payload.warnings.recent:
        _render_logs_table(type("_Payload", (), {"logs": payload.warnings.recent})())

    task_table = Table(title="Recent Task Runs")
    task_table.add_column("Metric")
    task_table.add_column("Value")
    task_table.add_row(
        "Recent statuses",
        ", ".join(f"{status}={count}" for status, count in sorted(payload.tasks.recent_status_counts.items())) or "none",
    )
    task_table.add_row("Runs included", str(len(payload.tasks.recent)))
    task_table.add_row("Recent failures", str(payload.tasks.recent_failure_count))
    task_table.add_row("Recent retries", str(payload.tasks.recent_retry_count))
    console.print(task_table)
    if payload.tasks.recent:
        _render_recent_agent_runs_table(type("_Payload", (), {"runs": payload.tasks.recent})())
    if payload.tasks.recent_failures:
        console.print("[bold]Recent failed task runs[/]")
        _render_recent_agent_runs_table(type("_Payload", (), {"runs": payload.tasks.recent_failures})())
    if payload.tasks.recent_retries:
        console.print("[bold]Recent retried task runs[/]")
        _render_recent_agent_runs_table(type("_Payload", (), {"runs": payload.tasks.recent_retries})())

    provider_policy_table = Table(title="Provider Policy Churn")
    provider_policy_table.add_column("Metric")
    provider_policy_table.add_column("Value")
    provider_policy_stats = {stat.key: stat.value for stat in payload.provider_policy.stats}
    provider_policy_table.add_row(
        "Route exhaustion",
        str(int(provider_policy_stats.get("provider_policy_route_exhaustion_count", 0.0))),
    )
    provider_policy_table.add_row(
        "Legacy fallback denied",
        str(int(provider_policy_stats.get("provider_policy_legacy_fallback_denied_count", 0.0))),
    )
    provider_policy_table.add_row(
        "Admission skips",
        str(int(provider_policy_stats.get("provider_policy_admission_skip_count", 0.0))),
    )
    provider_policy_table.add_row(
        "Suppressed duplicates",
        str(int(provider_policy_stats.get("provider_policy_warning_suppressed_count", 0.0))),
    )
    console.print(provider_policy_table)
    if payload.provider_policy.by_task:
        by_task_table = Table(title="Provider Policy by Task")
        by_task_table.add_column("Task")
        by_task_table.add_column("Route Exhaust", justify="right")
        by_task_table.add_column("Fallback Denied", justify="right")
        by_task_table.add_column("Admission Skips", justify="right")
        by_task_table.add_column("Top Skip Provider")
        by_task_table.add_column("Top Reason")
        for row in payload.provider_policy.by_task:
            by_task_table.add_row(
                row.task_name,
                str(row.route_exhaustion_count),
                str(row.legacy_fallback_denied_count),
                str(row.admission_skip_count),
                row.top_skip_provider_key or "-",
                row.top_skip_reason_code or "-",
            )
        console.print(by_task_table)

    conversation_table = Table(title=f"Recent Conversations ({payload.conversations.window_hours}h)")
    conversation_table.add_column("Metric")
    conversation_table.add_column("Value")
    conversation_table.add_row("Total", str(payload.conversations.total))
    conversation_table.add_row(
        "By status",
        ", ".join(
            f"{status}={count}" for status, count in sorted(payload.conversations.by_status.items())
        ) or "none",
    )
    conversation_table.add_row("Rows included", str(len(payload.conversations.recent)))
    console.print(conversation_table)
    if payload.conversations.recent:
        _render_ai_conversation_table(type("_Payload", (), {"conversations": payload.conversations.recent})())

    memory_table = Table(title="Recent Memory Activity")
    memory_table.add_column("Metric")
    memory_table.add_column("Value")
    memory_table.add_row("Updated last 15m", str(payload.memory_activity.updated_last_15_minutes))
    memory_table.add_row("Updated last 1h", str(payload.memory_activity.updated_last_hour))
    memory_table.add_row("Updated last 24h", str(payload.memory_activity.updated_last_day))
    memory_table.add_row("Rows included", str(len(payload.memory_activity.recent)))
    console.print(memory_table)
    if payload.memory_activity.recent:
        recent_memories = Table(title="Recent Memory Edits")
        recent_memories.add_column("Updated", no_wrap=True)
        recent_memories.add_column("Type", no_wrap=True)
        recent_memories.add_column("Status", no_wrap=True)
        recent_memories.add_column("Title")
        for record in payload.memory_activity.recent:
            recent_memories.add_row(record.updated_at, record.type, record.status, record.title)
        console.print(recent_memories)

    latency_table = Table(title=f"Memory Tool Latency ({payload.tool_latency.window_minutes}m)")
    latency_table.add_column("Tool")
    latency_table.add_column("Count", justify="right")
    latency_table.add_column("Avg", justify="right")
    latency_table.add_column("P95", justify="right")
    latency_table.add_column("Max", justify="right")
    latency_table.add_column("Slow", justify="right")
    if not payload.tool_latency.by_event_kind:
        latency_table.add_row("-", "0", "-", "-", "-", "0")
    for metric in payload.tool_latency.by_event_kind:
        latency_table.add_row(
            metric.event_kind,
            str(metric.count),
            f"{metric.avg_duration_ms:.1f}ms",
            f"{metric.p95_duration_ms:.1f}ms",
            f"{metric.max_duration_ms:.1f}ms",
            str(metric.slow_count),
        )
    console.print(latency_table)


@click.group()
@click.option("--debug", is_flag=True, help="Enable debug logging")
@click.pass_context
def main(ctx: click.Context, debug: bool) -> None:
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug
    configure_cli_logging(debug)


def _memory_stash_command_action(workspace_root: str | None, text: tuple[str, ...]) -> None:
    _run_or_exit(lambda: _stash_thought(workspace_root, _resolve_stash_content(text)))


def _memory_import_markdown_command_action(
    file_paths: tuple[str, ...],
    workspace_root: str | None,
    workspace_ids: tuple[str, ...],
    thought: bool,
) -> None:
    _run_or_exit(lambda: _import_markdown_files(file_paths, workspace_root, workspace_ids, thought))


memory_group = build_memory_group(
    workspace_root_option,
    _memory_stash_command_action,
    import_markdown_command_action=_memory_import_markdown_command_action,
)
import_markdown = memory_group.commands["import-markdown"]
main.add_command(memory_group)


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


_stdio_proxy_command_family = build_stdio_proxy_command_family(
    workspace_root_option,
    run_stdio_proxy=_run_command_action,
    run_internal_stdio_proxy=_internal_run_command_action,
)
run = _stdio_proxy_command_family.run
internal_run = _stdio_proxy_command_family.internal_run
main.add_command(run)
main.add_command(internal_run)


def _daemon_start_command_action(
    debug_enabled: bool,
    workspace_root: str | None,
    host: str,
    port: int | None,
) -> None:
    _start_daemon(debug_enabled, workspace_root, host, port)


def _daemon_status_command_action(workspace_root: str | None) -> None:
    _print_daemon_status(workspace_root)


def _daemon_stop_command_action(workspace_root: str | None) -> None:
    _run_or_exit(lambda: _stop_daemon_command(workspace_root))


def _daemon_restart_command_action(workspace_root: str | None) -> None:
    _run_or_exit(lambda: _restart_daemon_command(workspace_root))


_daemon_command_family = build_daemon_command_family(
    workspace_root_option,
    start_daemon=_daemon_start_command_action,
    print_daemon_status=_daemon_status_command_action,
    stop_daemon_command=_daemon_stop_command_action,
    restart_daemon_command=_daemon_restart_command_action,
)
daemon_group = _daemon_command_family.group
daemon_start = _daemon_command_family.start
daemon_status = _daemon_command_family.status
daemon_stop = _daemon_command_family.stop
daemon_restart = _daemon_command_family.restart
main.add_command(daemon_group)


@main.group(name="admin")
def admin_group() -> None:
    """Canonical operator commands."""


def _admin_agent_run_command_action(
    agent_name: str | None,
    workspace_root: str | None,
    run_all: bool,
    force: bool,
) -> None:
    if run_all:
        if agent_name is not None:
            raise click.UsageError("Do not provide an agent name with --all.")
        _run_or_exit(lambda: _enqueue_all_agents(workspace_root, force))
        return
    if agent_name is None:
        raise click.UsageError("Provide an agent name or pass --all.")
    _run_or_exit(lambda: _enqueue_agent(agent_name, workspace_root, force))


_admin_agent_command_family = build_admin_agent_command_family(
    workspace_root_option,
    agent_names=TRIGGERABLE_BACKGROUND_TASK_NAMES,
    run_agent=_admin_agent_run_command_action,
)
admin_agent_group = _admin_agent_command_family.group
admin_run_agent_command = _admin_agent_command_family.run_command
admin_group.add_command(admin_agent_group)


def _admin_conversation_list_command_action(
    workspace_root: str | None,
    scope: str,
    workspace_id: str | None,
    task_name: str | None,
    status: str | None,
    limit: int,
    json_output: bool,
) -> None:
    def _run(service: ManagementService) -> None:
        effective_workspace_id = _resolve_operator_workspace_filter(service, scope=scope, workspace_id=workspace_id)
        payload = service.list_ai_conversations(
            workspace_id=effective_workspace_id,
            task_name=task_name,
            status=status,
            limit=limit,
        )
        if json_output:
            click.echo(json.dumps(payload.model_dump(), sort_keys=True))
            return
        _render_ai_conversation_table(payload)

    _with_management_service(workspace_root, _run, workspace_id=None)


def _admin_conversation_show_command_action(
    request_id: str,
    workspace_root: str | None,
    scope: str,
    workspace_id: str | None,
    json_output: bool,
) -> None:
    def _run(service: ManagementService) -> None:
        effective_workspace_id = _resolve_operator_workspace_filter(service, scope=scope, workspace_id=workspace_id)
        payload = service.list_ai_conversations(workspace_id=effective_workspace_id, request_id=request_id, limit=200)
        if json_output:
            click.echo(json.dumps(payload.model_dump(), sort_keys=True))
            return
        _render_ai_conversation_table(payload)
        for conversation in payload.conversations:
            console.print(f"\n[bold]Request {conversation.request_id} attempt {conversation.attempt}[/]")
            console.print(f"status={conversation.status} pid={conversation.subprocess_pid or '-'} task={conversation.task_name or '-'}")
            console.print("[bold]Prompt[/]")
            console.print(conversation.prompt_text or "-")
            console.print("[bold]Response[/]")
            console.print(conversation.response_text or "-")
            if conversation.error_text:
                console.print(f"[red]Error:[/] {conversation.error_text}")

    _with_management_service(workspace_root, _run, workspace_id=None)


_admin_conversation_command_family = build_admin_conversation_command_family(
    workspace_root_option,
    list_conversations=_admin_conversation_list_command_action,
    show_conversation=_admin_conversation_show_command_action,
)
admin_conversation_group = _admin_conversation_command_family.group
list_conversations_command = _admin_conversation_command_family.list_command
show_conversation_command = _admin_conversation_command_family.show_command
admin_group.add_command(admin_conversation_group)


def _admin_log_list_command_action(
    workspace_root: str | None,
    scope: str,
    workspace_id: str | None,
    limit: int,
    level: str | None,
    logger_name: str | None,
    source: str | None,
    query: str | None,
    after: float | None,
    before: float | None,
    json_output: bool,
) -> None:
    _show_logs(workspace_root, scope, workspace_id, limit, level, logger_name, source, query, after, before, json_output)


def _admin_log_summary_command_action(
    workspace_root: str | None,
    scope: str,
    workspace_id: str | None,
    level: str | None,
    logger_name: str | None,
    source: str | None,
    query: str | None,
    after: float | None,
    before: float | None,
    json_output: bool,
) -> None:
    _summarize_logs(workspace_root, scope, workspace_id, level, logger_name, source, query, after, before, json_output)


def _admin_log_prune_command_action(
    workspace_root: str | None,
    scope: str,
    workspace_id: str | None,
    max_runtime_logs: int | None,
    max_log_age_days: int | None,
    json_output: bool,
) -> None:
    _prune_logs(workspace_root, scope, workspace_id, max_runtime_logs, max_log_age_days, json_output)


_admin_log_command_family = build_admin_log_command_family(
    workspace_root_option,
    show_logs=_admin_log_list_command_action,
    summarize_logs=_admin_log_summary_command_action,
    prune_logs=_admin_log_prune_command_action,
)
admin_log_group = _admin_log_command_family.group
admin_logs = _admin_log_command_family.list_command
admin_log_summary = _admin_log_command_family.summary_command
admin_log_prune = _admin_log_command_family.prune_command
admin_group.add_command(admin_log_group)


def _admin_task_list_command_action(
    workspace_root: str | None,
    status: str | None,
    limit: int,
    json_output: bool,
) -> None:
    _list_tasks(workspace_root, status, limit, json_output)


def _admin_task_recent_runs_command_action(
    workspace_root: str | None,
    limit: int,
    json_output: bool,
) -> None:
    def _run(service: ManagementService) -> None:
        payload = service.list_recent_agent_runs(
            limit=limit,
            detail_level="full",
        )
        if json_output:
            click.echo(json.dumps(payload.model_dump(), sort_keys=True))
            return
        _render_recent_agent_runs_table(payload)

    _with_management_service(workspace_root, _run, workspace_id=None)


def _admin_task_sampling_summary_command_action(
    workspace_root: str | None,
    limit: int,
    json_output: bool,
) -> None:
    def _run(service: ManagementService) -> None:
        summary = service.get_task_sampling_summary(limit=limit)
        if json_output:
            click.echo(json.dumps(summary.model_dump(), sort_keys=True))
            return
        _render_sampling_summary(summary)

    _with_management_service(workspace_root, _run, workspace_id=None)


def _admin_task_cancel_command_action(
    task_id: str,
    workspace_root: str | None,
    reason: str,
    json_output: bool,
) -> None:
    _run_or_exit(lambda: _cancel_task(task_id, workspace_root, reason, json_output))


def _admin_task_show_command_action(task_id: str, workspace_root: str | None, json_output: bool) -> None:
    _run_or_exit(lambda: _show_task(task_id, workspace_root, json_output))


_admin_task_command_family = build_admin_task_command_family(
    workspace_root_option,
    list_tasks=_admin_task_list_command_action,
    list_recent_task_runs=_admin_task_recent_runs_command_action,
    show_task_sampling_summary=_admin_task_sampling_summary_command_action,
    cancel_task=_admin_task_cancel_command_action,
    show_task=_admin_task_show_command_action,
)
admin_task_group = _admin_task_command_family.group
admin_list_tasks_command = _admin_task_command_family.list_command
admin_recent_task_runs_command = _admin_task_command_family.recent_runs_command
admin_task_sampling_summary_command = _admin_task_command_family.sampling_summary_command
admin_cancel_task_command = _admin_task_command_family.cancel_command
admin_show_task_command = _admin_task_command_family.show_command
admin_group.add_command(admin_task_group)


def _admin_search_health_command_action(workspace_root: str | None, json_output: bool) -> None:
    _show_search_health(workspace_root, json_output)


def _admin_search_repair_command_action(workspace_root: str | None, json_output: bool) -> None:
    _run_or_exit(lambda: _repair_search_index(workspace_root, json_output))


def _admin_search_debug_command_action(
    workspace_root: str | None,
    query: str,
    limit: int,
    json_output: bool,
) -> None:
    _run_or_exit(lambda: _show_search_debug(workspace_root, query, limit, json_output))


_admin_search_command_family = build_admin_search_command_family(
    workspace_root_option,
    show_search_health=_admin_search_health_command_action,
    repair_search_index=_admin_search_repair_command_action,
    show_search_debug=_admin_search_debug_command_action,
)
admin_search_group = _admin_search_command_family.group
search_health_command = _admin_search_command_family.health_command
search_repair_command = _admin_search_command_family.repair_command
search_debug_command = _admin_search_command_family.debug_command
admin_group.add_command(admin_search_group)


def _admin_dashboard_open_command_action(workspace_root: str | None) -> None:
    _run_or_exit(lambda: _print_dashboard_url(workspace_root, open_browser=True))


def _admin_dashboard_build_command_action() -> None:
    _run_or_exit(_build_dashboard_frontend)


_admin_dashboard_command_family = build_admin_dashboard_command_family(
    workspace_root_option,
    open_dashboard=_admin_dashboard_open_command_action,
    build_dashboard_frontend=_admin_dashboard_build_command_action,
)
admin_dashboard_group = _admin_dashboard_command_family.group
admin_dashboard_open_command = _admin_dashboard_command_family.open_command
admin_dashboard_build_command = _admin_dashboard_command_family.build_command
admin_group.add_command(admin_dashboard_group)


def _admin_overview_command_action(
    workspace_root: str | None,
    watch: bool,
    interval: float,
    verbose: bool,
) -> None:
    _run_or_exit(lambda: _show_stats_command(workspace_root, watch, interval, verbose))


def _admin_health_command_action(workspace_root: str | None, json_output: bool) -> None:
    _run_or_exit(lambda: _show_operator_health_snapshot(workspace_root, json_output))


def _admin_monitor_command_action(workspace_root: str | None, interval: float) -> None:
    _run_or_exit(lambda: run_monitor_tui(workspace_root, interval))


def _show_quality_cleanup_candidates(
    workspace_root: str | None,
    window_hours: int,
    bucket_minutes: int,
    limit: int,
    json_output: bool,
) -> None:
    def _run(service: ManagementService) -> None:
        payload = service.list_quality_cleanup_candidates(
            window_hours=window_hours,
            bucket_minutes=bucket_minutes,
            limit=limit,
        )
        if json_output:
            click.echo(json.dumps(payload.model_dump(), sort_keys=True))
            return
        _render_quality_cleanup_candidates(payload)

    _with_management_service(workspace_root, _run, workspace_id=None)


def _admin_quality_cleanup_command_action(
    workspace_root: str | None,
    window_hours: int,
    bucket_minutes: int,
    limit: int,
    json_output: bool,
) -> None:
    _show_quality_cleanup_candidates(workspace_root, window_hours, bucket_minutes, limit, json_output)


_admin_operator_command_cluster = build_admin_operator_command_cluster(
    workspace_root_option,
    show_overview=_admin_overview_command_action,
    show_health=_admin_health_command_action,
    open_monitor=_admin_monitor_command_action,
    show_quality_cleanup=_admin_quality_cleanup_command_action,
)
admin_overview_command = _admin_operator_command_cluster.overview_command
admin_health_command = _admin_operator_command_cluster.health_command
admin_monitor_command = _admin_operator_command_cluster.monitor_command
admin_quality_cleanup_command = _admin_operator_command_cluster.quality_cleanup_command
admin_group.add_command(admin_overview_command)
admin_group.add_command(admin_health_command)
admin_group.add_command(admin_monitor_command)
admin_group.add_command(admin_quality_cleanup_command)


def _admin_install_command_action(
    tools: tuple[str, ...],
    components: tuple[str, ...],
    scope: str,
    workspace_root: str | None,
) -> None:
    try:
        result = install_integrations(
            tools=tools,
            components=components,
            scope=scope,
            workspace_root=workspace_root,
        )
    except ValueError as exc:
        console.print(f"[red]Error:[/] {exc}")
        sys.exit(1)

    console.print(f"[green]Install target:[/] {result.workspace_root}")
    for action in result.actions:
        console.print(
            f"- {action.tool}/{action.component}: {action.status} → {action.path}"
        )


def _hook_runner_command_action(workspace_root: str | None) -> None:
    try:
        payload = load_hook_payload(sys.stdin)
    except (json.JSONDecodeError, ValueError) as exc:
        click.echo(f"mcp-memory hook-runner: {exc}", err=True)
        click.echo("{}")
        return

    response, error_message = safe_forward_hook_event(payload, workspace_root=workspace_root)
    if error_message is not None:
        click.echo(f"mcp-memory hook-runner: {error_message}", err=True)
    click.echo(json.dumps(response, sort_keys=True))


hook_runner = build_hook_runner_command(hook_runner_command_action=_hook_runner_command_action)
main.add_command(hook_runner)


def _admin_prefetch_model_command_action(workspace_root: str | None) -> None:
    _run_or_exit(lambda: _prefetch_embedding_model(workspace_root))


def _admin_migrate_sqlite_to_postgres_command_action(
    sqlite_path: Path | None,
    postgres_dsn: str | None,
    dry_run: bool,
    allow_non_empty_target: bool,
    json_output: bool,
) -> None:
    _run_or_exit(
        lambda: _migrate_sqlite_to_postgres_command(
            sqlite_path,
            postgres_dsn,
            dry_run,
            allow_non_empty_target,
            json_output,
        )
    )


_admin_utility_command_cluster = build_admin_utility_command_cluster(
    workspace_root_option,
    install_command_action=_admin_install_command_action,
    prefetch_model=_admin_prefetch_model_command_action,
    migrate_sqlite_to_postgres=_admin_migrate_sqlite_to_postgres_command_action,
)
install = _admin_utility_command_cluster.install_command
prefetch_model = _admin_utility_command_cluster.prefetch_model_command
migrate_sqlite_to_postgres_cli = _admin_utility_command_cluster.migrate_sqlite_to_postgres_command
admin_group.add_command(install)
admin_group.add_command(prefetch_model)
admin_group.add_command(migrate_sqlite_to_postgres_cli)


def _build_management_service(runtime, workspace_id: str | None | object = ... ) -> ManagementService:
    ctx = runtime if workspace_id is ... else replace(runtime, workspace_id=workspace_id)
    return ManagementService(ctx, SimpleNamespace(has_runtime=True, client_count=1))


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
