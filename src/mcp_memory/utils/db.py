from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path

from mcp_memory.utils.db_schema import SCHEMA_VERSION, initialize_schema

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Thread-safe SQLite database manager with WAL mode."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._local = threading.local()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # Initialize schema via a temporary connection
        conn = self._open_connection()
        self._initialize_schema_on(conn)
        conn.close()

    def _open_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.row_factory = sqlite3.Row
        return conn

    def get_connection(self) -> sqlite3.Connection:
        """Return a per-thread SQLite connection."""
        conn = getattr(self._local, "connection", None)
        if conn is None:
            conn = self._open_connection()
            self._local.connection = conn
        return conn

    @property
    def db_path(self):
        return self._db_path

    def _initialize_schema_on(self, conn: sqlite3.Connection) -> None:
        initialize_schema(conn)

    def initialize_schema(self) -> None:
        self._initialize_schema_on(self.get_connection())

    def get_schema_version(self) -> int | None:
        row = self.get_connection().execute(
            "SELECT value FROM schema_metadata WHERE key = ?",
            ("schema_version",),
        ).fetchone()
        if row is None:
            return None
        return int(row[0])

    def close(self) -> None:
        conn = getattr(self._local, "connection", None)
        if conn is not None:
            conn.close()
            self._local.connection = None


__all__ = ["DatabaseManager", "SCHEMA_VERSION"]
