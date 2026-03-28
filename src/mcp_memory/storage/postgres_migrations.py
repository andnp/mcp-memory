from __future__ import annotations

from dataclasses import dataclass


POSTGRES_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class PostgresMigration:
    version: int
    name: str
    statements: tuple[str, ...]


POSTGRES_MIGRATIONS = (
    PostgresMigration(
        version=1,
        name="bootstrap_core_memory_schema",
        statements=(
            """
            CREATE TABLE IF NOT EXISTS schema_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                summary TEXT,
                type TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                read_count INTEGER NOT NULL DEFAULT 0,
                access_score DOUBLE PRECISION NOT NULL DEFAULT 0,
                last_accessed_at TEXT,
                last_surfaced_at TEXT,
                metadata JSONB NOT NULL DEFAULT '{}'::jsonb
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS memory_workspaces (
                memory_id TEXT NOT NULL,
                workspace_id TEXT NOT NULL,
                PRIMARY KEY (memory_id, workspace_id),
                FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS tags (
                id BIGSERIAL PRIMARY KEY,
                name TEXT NOT NULL UNIQUE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS memory_tags (
                memory_id TEXT NOT NULL,
                tag_id BIGINT NOT NULL,
                PRIMARY KEY (memory_id, tag_id),
                FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE,
                FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS links (
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                type TEXT NOT NULL,
                context TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (source_id, target_id, type),
                FOREIGN KEY (source_id) REFERENCES memories(id) ON DELETE CASCADE
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_memory_workspaces_workspace_id ON memory_workspaces(workspace_id)",
            "CREATE INDEX IF NOT EXISTS idx_memory_tags_tag_id ON memory_tags(tag_id)",
            "CREATE INDEX IF NOT EXISTS idx_links_target_id ON links(target_id)",
        ),
    ),
    PostgresMigration(
        version=2,
        name="add_runtime_logs_and_task_execution_attempts",
        statements=(
            """
            CREATE TABLE IF NOT EXISTS runtime_logs (
                id BIGSERIAL PRIMARY KEY,
                workspace_id TEXT,
                source TEXT NOT NULL,
                logger_name TEXT NOT NULL,
                level TEXT NOT NULL,
                message TEXT NOT NULL,
                created_at DOUBLE PRECISION NOT NULL,
                data_json JSONB NOT NULL DEFAULT '{}'::jsonb
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_runtime_logs_created_at ON runtime_logs(created_at DESC, id DESC)",
            "CREATE INDEX IF NOT EXISTS idx_runtime_logs_workspace_id ON runtime_logs(workspace_id)",
            "CREATE INDEX IF NOT EXISTS idx_runtime_logs_level ON runtime_logs(level)",
            """
            CREATE TABLE IF NOT EXISTS task_execution_attempts (
                id BIGSERIAL PRIMARY KEY,
                task_id TEXT NOT NULL,
                execution_epoch INTEGER NOT NULL,
                workspace_id TEXT,
                task_name TEXT,
                request_id TEXT,
                subprocess_pid INTEGER,
                provider_key TEXT,
                provider_name TEXT,
                model_name TEXT,
                status TEXT NOT NULL,
                started_at DOUBLE PRECISION NOT NULL,
                last_heartbeat_at DOUBLE PRECISION,
                completed_at DOUBLE PRECISION,
                error_text TEXT,
                termination_reason TEXT,
                UNIQUE (task_id, execution_epoch)
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_task_execution_attempts_workspace_started ON task_execution_attempts(workspace_id, started_at DESC, id DESC)",
            "CREATE INDEX IF NOT EXISTS idx_task_execution_attempts_status_started ON task_execution_attempts(status, started_at DESC, id DESC)",
        ),
    ),
)


def apply_postgres_migrations(cursor, *, current_version: int | None) -> int:
    effective_version = 0 if current_version is None else current_version
    for migration in POSTGRES_MIGRATIONS:
        if migration.version <= effective_version:
            continue
        for statement in migration.statements:
            cursor.execute(statement)
        cursor.execute(
            """
            INSERT INTO schema_metadata(key, value)
            VALUES (%s, %s)
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
            """,
            ("schema_version", str(migration.version)),
        )
        effective_version = migration.version
    return effective_version
