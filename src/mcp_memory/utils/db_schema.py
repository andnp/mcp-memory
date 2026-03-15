from __future__ import annotations

import sqlite3


SCHEMA_VERSION = 3


def initialize_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS tasks (
            id TEXT PRIMARY KEY,
            task_name TEXT NOT NULL,
            workspace_id TEXT,
            data TEXT DEFAULT '{}',
            status TEXT NOT NULL DEFAULT 'pending',
            priority INTEGER NOT NULL DEFAULT 100,
            retries_count INTEGER NOT NULL DEFAULT 0,
            max_retries INTEGER NOT NULL DEFAULT 3,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            available_at REAL NOT NULL,
            claimed_at REAL,
            started_at REAL,
            completed_at REAL,
            last_error TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
        CREATE INDEX IF NOT EXISTS idx_tasks_ready ON tasks(status, available_at, priority, created_at);

        CREATE TABLE IF NOT EXISTS task_runs (
            id TEXT PRIMARY KEY,
            task_id TEXT NOT NULL,
            task_name TEXT NOT NULL,
            workspace_id TEXT,
            status TEXT NOT NULL,
            started_at REAL NOT NULL,
            completed_at REAL NOT NULL,
            duration_seconds REAL NOT NULL DEFAULT 0,
            result_json TEXT NOT NULL DEFAULT '{}',
            error_text TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_task_runs_task_name_completed_at
            ON task_runs(task_name, completed_at DESC);
        CREATE INDEX IF NOT EXISTS idx_task_runs_workspace_task_name_completed_at
            ON task_runs(workspace_id, task_name, completed_at DESC);

        CREATE TABLE IF NOT EXISTS system1_journal (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            workspace_id TEXT,
            author TEXT,
            timestamp REAL NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending'
        );

        CREATE INDEX IF NOT EXISTS idx_system1_journal_status ON system1_journal(status);

        CREATE TABLE IF NOT EXISTS schema_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
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
            FOREIGN KEY (source_id) REFERENCES memories(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_links_source_id ON links(source_id);
        CREATE INDEX IF NOT EXISTS idx_links_target_id ON links(target_id);
        CREATE INDEX IF NOT EXISTS idx_links_type ON links(type);
        """
    )
    ensure_column(conn, "system1_journal", "workspace_id", "TEXT")
    ensure_column(conn, "system1_journal", "author", "TEXT")
    ensure_column(conn, "tasks", "workspace_id", "TEXT")
    ensure_column(conn, "tasks", "priority", "INTEGER NOT NULL DEFAULT 100")
    ensure_column(conn, "tasks", "retries_count", "INTEGER NOT NULL DEFAULT 0")
    ensure_column(conn, "tasks", "max_retries", "INTEGER NOT NULL DEFAULT 3")
    ensure_column(conn, "tasks", "available_at", "REAL NOT NULL DEFAULT 0")
    ensure_column(conn, "tasks", "claimed_at", "REAL")
    ensure_column(conn, "tasks", "started_at", "REAL")
    ensure_column(conn, "tasks", "completed_at", "REAL")
    ensure_column(conn, "tasks", "last_error", "TEXT")
    conn.execute(
        "UPDATE tasks SET updated_at = COALESCE(updated_at, created_at, 0) WHERE updated_at IS NULL"
    )
    conn.execute(
        "UPDATE tasks SET available_at = COALESCE(available_at, created_at, 0) WHERE available_at IS NULL"
    )
    ensure_column(conn, "memories", "summary", "TEXT")
    ensure_column(conn, "memories", "status", "TEXT NOT NULL DEFAULT 'active'")
    ensure_column(conn, "memories", "created_at", "TEXT")
    ensure_column(conn, "memories", "updated_at", "TEXT")
    ensure_column(conn, "memories", "access_score", "REAL NOT NULL DEFAULT 0")
    ensure_column(conn, "memories", "last_accessed_at", "TEXT")
    ensure_column(conn, "memories", "last_surfaced_at", "TEXT")
    ensure_column(conn, "memories", "metadata", "TEXT NOT NULL DEFAULT '{}'"
    )
    conn.execute(
        "INSERT OR REPLACE INTO schema_metadata (key, value) VALUES (?, ?)",
        ("schema_version", str(SCHEMA_VERSION)),
    )
    conn.commit()


def ensure_column(
    conn: sqlite3.Connection,
    table_name: str,
    column_name: str,
    column_definition: str,
) -> None:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    existing_columns = {row[1] for row in rows}
    if column_name in existing_columns:
        return
    conn.execute(
        f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}"
    )