from __future__ import annotations

import logging
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

from mcp_memory.mcp.runtime import resolve_runtime_spec
from mcp_memory.runtime_log_store import RuntimeLogRepository
from mcp_memory.utils.db import DatabaseManager


_LOG_CONSOLE = Console(stderr=True)
_BASE_LOG_RECORD_KEYS = frozenset(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}


class SQLiteStructuredLogHandler(logging.Handler):
    def __init__(
        self,
        *,
        db_manager: DatabaseManager | None = None,
        db_path: Path | None = None,
        workspace_id: str | None = None,
        source: str = "runtime",
    ):
        super().__init__()
        if db_manager is None and db_path is None:
            raise ValueError("db_manager_or_db_path_required")
        if db_manager is not None:
            self._db_manager = db_manager
        else:
            assert db_path is not None
            self._db_manager = DatabaseManager(db_path)
        self._owns_db_manager = db_manager is None
        self._source = source
        self._repository = RuntimeLogRepository(self._db_manager, workspace_id=workspace_id)

    def emit(self, record: logging.LogRecord):
        try:
            self._repository.write_log(
                source=self._source,
                logger_name=record.name,
                level=record.levelname,
                message=record.getMessage(),
                created_at=float(record.created),
                data=_build_log_data(record),
            )
        except Exception:
            self.handleError(record)

    def close(self):
        try:
            if self._owns_db_manager:
                self._db_manager.close()
        finally:
            super().close()


def configure_cli_logging(debug: bool):
    _configure_root_logging(
        debug,
        handlers=[RichHandler(rich_tracebacks=True, console=_LOG_CONSOLE)],
    )


def configure_workspace_logging(
    debug: bool,
    *,
    workspace_root_override: str | None,
    cwd: Path | None = None,
    console_output: bool,
    source: str,
):
    _configure_root_logging(debug, handlers=[logging.NullHandler()])
    spec = resolve_runtime_spec(workspace_root_override, cwd)
    handlers: list[logging.Handler] = []
    if console_output:
        handlers.append(RichHandler(rich_tracebacks=True, console=_LOG_CONSOLE))
    handlers.append(
        SQLiteStructuredLogHandler(
            db_path=spec.memory_path / "indices" / "memory.db",
            workspace_id=spec.workspace_id,
            source=source,
        )
    )
    _configure_root_logging(debug, handlers=handlers)


def _configure_root_logging(debug: bool, *, handlers: list[logging.Handler]):
    level = logging.DEBUG if debug else logging.INFO
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    root.setLevel(level)
    for handler in handlers or [logging.NullHandler()]:
        root.addHandler(handler)


def _build_log_data(record: logging.LogRecord):
    payload: dict[str, object] = {
        "pathname": record.pathname,
        "lineno": record.lineno,
        "module": record.module,
        "funcName": record.funcName,
        "process": record.process,
        "threadName": record.threadName,
    }
    if record.exc_info is not None:
        payload["exception"] = logging.Formatter().formatException(record.exc_info)

    extras = {
        key: _make_json_safe(value)
        for key, value in record.__dict__.items()
        if key not in _BASE_LOG_RECORD_KEYS
    }
    if extras:
        payload["extra"] = extras
    return payload


def _make_json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_make_json_safe(item) for item in value]
    return str(value)