from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path

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

    def _initialize_schema_on(self, conn: sqlite3.Connection) -> None:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                file_path TEXT,
                content_hash TEXT,
                mtime REAL,
                status TEXT,
                indexed_at REAL
            );

            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT,
                content TEXT,
                metadata TEXT,
                vector BLOB,
                indexed_at REAL
            );

            CREATE TABLE IF NOT EXISTS kv_store (
                key TEXT PRIMARY KEY,
                value TEXT
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS search_index USING fts5(
                chunk_id UNINDEXED,
                doc_id UNINDEXED,
                content,
                title,
                headers,
                tags,
                source_file UNINDEXED
            );

            CREATE TABLE IF NOT EXISTS graph_nodes (
                node_id TEXT PRIMARY KEY,
                metadata TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS graph_edges (
                source TEXT NOT NULL,
                target TEXT NOT NULL,
                edge_type TEXT NOT NULL DEFAULT 'related_to',
                edge_context TEXT DEFAULT '',
                PRIMARY KEY (source, target, edge_type),
                FOREIGN KEY (source) REFERENCES graph_nodes(node_id) ON DELETE CASCADE,
                FOREIGN KEY (target) REFERENCES graph_nodes(node_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_graph_edges_source ON graph_edges(source);
            CREATE INDEX IF NOT EXISTS idx_graph_edges_target ON graph_edges(target);
            CREATE INDEX IF NOT EXISTS idx_graph_edges_type ON graph_edges(edge_type);

            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                task_name TEXT NOT NULL,
                data TEXT DEFAULT '{}',
                status TEXT NOT NULL DEFAULT 'pending',
                created_at REAL NOT NULL,
                updated_at REAL
            );

            CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);

            CREATE TABLE IF NOT EXISTS system1_journal (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                workspace_id TEXT,
                author TEXT,
                timestamp REAL NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending'
            );

            CREATE INDEX IF NOT EXISTS idx_system1_journal_status ON system1_journal(status);

            CREATE TABLE IF NOT EXISTS system_state (
                key TEXT PRIMARY KEY,
                value TEXT
            );

            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                summary TEXT,
                type TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                access_score REAL NOT NULL DEFAULT 0,
                last_accessed_at TEXT,
                last_surfaced_at TEXT,
                metadata TEXT NOT NULL DEFAULT '{}'
            );

            CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type);
            CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status);
            CREATE INDEX IF NOT EXISTS idx_memories_updated_at ON memories(updated_at);
            CREATE INDEX IF NOT EXISTS idx_memories_last_accessed_at ON memories(last_accessed_at);
            CREATE INDEX IF NOT EXISTS idx_memories_last_surfaced_at ON memories(last_surfaced_at);

            CREATE TABLE IF NOT EXISTS memory_workspaces (
                memory_id TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                PRIMARY KEY (memory_id, workspace_id),
                FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_memory_workspaces_workspace_id ON memory_workspaces(workspace_id);

            CREATE TABLE IF NOT EXISTS tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE
            );

            CREATE TABLE IF NOT EXISTS memory_tags (
                memory_id TEXT NOT NULL,
                tag_id INTEGER NOT NULL,
                PRIMARY KEY (memory_id, tag_id),
                FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE,
                FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_memory_tags_tag_id ON memory_tags(tag_id);

            CREATE TABLE IF NOT EXISTS links (
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                type TEXT NOT NULL,
                context TEXT DEFAULT '',
                PRIMARY KEY (source_id, target_id, type),
                FOREIGN KEY (source_id) REFERENCES memories(id) ON DELETE CASCADE,
                FOREIGN KEY (target_id) REFERENCES memories(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_links_source_id ON links(source_id);
            CREATE INDEX IF NOT EXISTS idx_links_target_id ON links(target_id);
            CREATE INDEX IF NOT EXISTS idx_links_type ON links(type);
        """)
        self._ensure_column(conn, "system1_journal", "workspace_id", "TEXT")
        self._ensure_column(conn, "system1_journal", "author", "TEXT")
        self._ensure_column(conn, "memories", "summary", "TEXT")
        self._ensure_column(conn, "memories", "status", "TEXT NOT NULL DEFAULT 'active'")
        self._ensure_column(conn, "memories", "created_at", "TEXT")
        self._ensure_column(conn, "memories", "updated_at", "TEXT")
        self._ensure_column(conn, "memories", "access_score", "REAL NOT NULL DEFAULT 0")
        self._ensure_column(conn, "memories", "last_accessed_at", "TEXT")
        self._ensure_column(conn, "memories", "last_surfaced_at", "TEXT")
        self._ensure_column(conn, "memories", "metadata", "TEXT NOT NULL DEFAULT '{}'")
        conn.commit()

    def _ensure_column(
        self,
        conn: sqlite3.Connection,
        table_name: str,
        column_name: str,
        column_definition: str,
    ):
        rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        existing_columns = {row[1] for row in rows}
        if column_name in existing_columns:
            return
        conn.execute(
            f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}"
        )

    def initialize_schema(self) -> None:
        self._initialize_schema_on(self.get_connection())

    def close(self) -> None:
        conn = getattr(self._local, "connection", None)
        if conn is not None:
            conn.close()
            self._local.connection = None
