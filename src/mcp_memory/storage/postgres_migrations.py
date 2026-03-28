from __future__ import annotations

from dataclasses import dataclass


POSTGRES_SCHEMA_VERSION = 4


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
    PostgresMigration(
        version=3,
        name="add_provider_usage_and_ai_conversations",
        statements=(
            """
            CREATE TABLE IF NOT EXISTS provider_usage (
                id BIGSERIAL PRIMARY KEY,
                workspace_id TEXT,
                task_name TEXT,
                task_id TEXT,
                request_id TEXT,
                subprocess_pid INTEGER,
                provider_key TEXT NOT NULL,
                provider_name TEXT NOT NULL,
                model_name TEXT NOT NULL,
                status TEXT NOT NULL,
                duration_seconds DOUBLE PRECISION NOT NULL,
                created_at DOUBLE PRECISION NOT NULL,
                error_text TEXT,
                reason_category TEXT,
                reason_code TEXT,
                retry_delay_seconds DOUBLE PRECISION
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_provider_usage_workspace_created ON provider_usage(workspace_id, created_at DESC, id DESC)",
            "CREATE INDEX IF NOT EXISTS idx_provider_usage_provider_created ON provider_usage(provider_key, created_at DESC, id DESC)",
            "CREATE INDEX IF NOT EXISTS idx_provider_usage_model_created ON provider_usage(model_name, created_at DESC, id DESC)",
            """
            CREATE TABLE IF NOT EXISTS ai_conversations (
                id BIGSERIAL PRIMARY KEY,
                request_id TEXT NOT NULL,
                attempt INTEGER NOT NULL,
                workspace_id TEXT,
                task_name TEXT,
                task_id TEXT,
                provider_key TEXT NOT NULL,
                provider_name TEXT NOT NULL,
                model_name TEXT NOT NULL,
                subprocess_pid INTEGER,
                prompt_text TEXT NOT NULL,
                response_text TEXT NOT NULL,
                parsed_json JSONB,
                status TEXT NOT NULL,
                error_text TEXT,
                reason_category TEXT,
                reason_code TEXT,
                retry_delay_seconds DOUBLE PRECISION,
                started_at DOUBLE PRECISION NOT NULL,
                completed_at DOUBLE PRECISION NOT NULL,
                duration_seconds DOUBLE PRECISION NOT NULL,
                UNIQUE (request_id, attempt)
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_ai_conversations_workspace_completed ON ai_conversations(workspace_id, completed_at DESC, id DESC)",
            "CREATE INDEX IF NOT EXISTS idx_ai_conversations_task_completed ON ai_conversations(task_name, completed_at DESC, id DESC)",
            "CREATE INDEX IF NOT EXISTS idx_ai_conversations_status_completed ON ai_conversations(status, completed_at DESC, id DESC)",
            """
            CREATE TABLE IF NOT EXISTS provider_admission_state (
                provider_key TEXT NOT NULL,
                model_name TEXT NOT NULL,
                reason_category TEXT NOT NULL,
                reason_code TEXT NOT NULL,
                error_text TEXT,
                retry_delay_seconds DOUBLE PRECISION,
                active_until DOUBLE PRECISION NOT NULL,
                updated_at DOUBLE PRECISION NOT NULL,
                PRIMARY KEY (provider_key, model_name)
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_provider_admission_state_active_until ON provider_admission_state(active_until DESC)",
        ),
    ),
    PostgresMigration(
        version=4,
        name="add_provider_policy_events",
        statements=(
            """
            CREATE TABLE IF NOT EXISTS provider_policy_events (
                id BIGSERIAL PRIMARY KEY,
                workspace_id TEXT,
                task_name TEXT NOT NULL,
                task_id TEXT,
                event_kind TEXT NOT NULL,
                warning_kind TEXT,
                provider_key TEXT,
                provider_name TEXT,
                model_name TEXT,
                route_key TEXT,
                candidate_routes_json JSONB NOT NULL,
                reason_category TEXT,
                reason_code TEXT,
                retry_delay_seconds DOUBLE PRECISION,
                warning_suppressed BOOLEAN NOT NULL DEFAULT FALSE,
                created_at DOUBLE PRECISION NOT NULL
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_provider_policy_events_task_created ON provider_policy_events(task_name, created_at DESC, id DESC)",
            "CREATE INDEX IF NOT EXISTS idx_provider_policy_events_task_id_created ON provider_policy_events(task_id, created_at DESC, id DESC)",
            "CREATE INDEX IF NOT EXISTS idx_provider_policy_events_workspace_created ON provider_policy_events(workspace_id, created_at DESC, id DESC)",
            "CREATE INDEX IF NOT EXISTS idx_provider_policy_events_event_kind_created ON provider_policy_events(event_kind, created_at DESC, id DESC)",
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
