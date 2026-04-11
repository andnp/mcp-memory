from __future__ import annotations

import sqlite3


SCHEMA_VERSION = 21


def initialize_schema(conn: sqlite3.Connection) -> None:
    create_current_schema(conn)
    apply_legacy_additive_migrations(conn)
    finalize_schema_setup(conn)


def create_current_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS tasks (
            id TEXT PRIMARY KEY,
            task_name TEXT NOT NULL,
            workspace_id TEXT,
            data TEXT DEFAULT '{}',
            status TEXT NOT NULL DEFAULT 'pending',
            execution_epoch INTEGER NOT NULL DEFAULT 0,
            priority INTEGER NOT NULL DEFAULT 100,
            retries_count INTEGER NOT NULL DEFAULT 0,
            max_retries INTEGER NOT NULL DEFAULT 3,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            available_at REAL NOT NULL,
            claimed_at REAL,
            started_at REAL,
            completed_at REAL,
            last_error TEXT,
            subprocess_pid INTEGER,
            active_request_id TEXT,
            cancellation_requested_at REAL,
            cancelled_at REAL,
            cancellation_reason TEXT,
            cancelled_by TEXT
        );

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

        CREATE TABLE IF NOT EXISTS system1_journal (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            workspace_id TEXT,
            author TEXT,
            timestamp REAL NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            claim_task_id TEXT,
            claimed_at REAL,
            recoverable_until REAL
        );

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
            read_count INTEGER NOT NULL DEFAULT 0,
            access_score REAL NOT NULL DEFAULT 0,
            last_accessed_at TEXT,
            last_surfaced_at TEXT,
            metadata TEXT NOT NULL DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS memory_workspaces (
            memory_id TEXT NOT NULL,
            workspace_id TEXT NOT NULL,
            PRIMARY KEY (memory_id, workspace_id),
            FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
        );

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

        CREATE TABLE IF NOT EXISTS links (
            source_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            type TEXT NOT NULL,
            context TEXT DEFAULT '',
            PRIMARY KEY (source_id, target_id, type),
            FOREIGN KEY (source_id) REFERENCES memories(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS embeddings (
            source_kind TEXT NOT NULL,
            source_id TEXT NOT NULL,
            workspace_id TEXT,
            model_name TEXT NOT NULL,
            embedding_json TEXT NOT NULL,
            updated_at REAL NOT NULL,
            PRIMARY KEY (source_kind, source_id, model_name)
        );

        CREATE TABLE IF NOT EXISTS hook_conversations (
            conversation_id TEXT PRIMARY KEY,
            workspace_id TEXT,
            started_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            last_ping_at REAL,
            last_reminder_at REAL,
            last_tool_name TEXT,
            last_payload_json TEXT NOT NULL DEFAULT '{}',
            ended_at REAL
        );

        CREATE TABLE IF NOT EXISTS runtime_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            workspace_id TEXT,
            source TEXT NOT NULL,
            logger_name TEXT NOT NULL,
            level TEXT NOT NULL,
            message TEXT NOT NULL,
            created_at REAL NOT NULL,
            data_json TEXT NOT NULL DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS provider_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            workspace_id TEXT,
            task_name TEXT,
            task_id TEXT,
            request_id TEXT,
            subprocess_pid INTEGER,
            provider_key TEXT NOT NULL,
            provider_name TEXT NOT NULL,
            model_name TEXT NOT NULL,
            status TEXT NOT NULL,
            duration_seconds REAL NOT NULL DEFAULT 0,
            created_at REAL NOT NULL,
            error_text TEXT,
            reason_category TEXT,
            reason_code TEXT,
            retry_delay_seconds REAL
        );

        CREATE TABLE IF NOT EXISTS task_execution_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
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
            started_at REAL NOT NULL,
            last_heartbeat_at REAL,
            completed_at REAL,
            error_text TEXT,
            termination_reason TEXT,
            UNIQUE(task_id, execution_epoch)
        );

        CREATE TABLE IF NOT EXISTS ai_conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id TEXT NOT NULL,
            attempt INTEGER NOT NULL DEFAULT 1,
            workspace_id TEXT,
            task_name TEXT,
            task_id TEXT,
            provider_key TEXT NOT NULL,
            provider_name TEXT NOT NULL,
            model_name TEXT NOT NULL,
            subprocess_pid INTEGER,
            prompt_text TEXT NOT NULL,
            response_text TEXT NOT NULL DEFAULT '',
            parsed_json TEXT,
            status TEXT NOT NULL,
            error_text TEXT,
            reason_category TEXT,
            reason_code TEXT,
            retry_delay_seconds REAL,
            started_at REAL NOT NULL,
            completed_at REAL NOT NULL,
            duration_seconds REAL NOT NULL DEFAULT 0,
            UNIQUE(request_id, attempt)
        );

        CREATE TABLE IF NOT EXISTS provider_admission_state (
            provider_key TEXT NOT NULL,
            model_name TEXT NOT NULL,
            reason_category TEXT NOT NULL,
            reason_code TEXT NOT NULL,
            error_text TEXT,
            retry_delay_seconds REAL,
            active_until REAL NOT NULL,
            updated_at REAL NOT NULL,
            PRIMARY KEY (provider_key, model_name)
        );

        CREATE TABLE IF NOT EXISTS memory_tool_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            invocation_id TEXT NOT NULL,
            workspace_id TEXT,
            caller_kind TEXT NOT NULL,
            event_kind TEXT NOT NULL,
            memory_id TEXT,
            query_text TEXT,
            result_rank INTEGER,
            result_count INTEGER,
            duration_ms REAL,
            created_at REAL NOT NULL,
            FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE SET NULL
        );

        CREATE TABLE IF NOT EXISTS provider_policy_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            workspace_id TEXT,
            task_name TEXT NOT NULL,
            task_id TEXT,
            event_kind TEXT NOT NULL,
            warning_kind TEXT,
            provider_key TEXT,
            provider_name TEXT,
            model_name TEXT,
            route_key TEXT,
            candidate_routes_json TEXT NOT NULL DEFAULT '[]',
            reason_category TEXT,
            reason_code TEXT,
            retry_delay_seconds REAL,
            warning_suppressed INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS work_items (
            id TEXT PRIMARY KEY,
            family_key TEXT NOT NULL,
            execution_lane TEXT NOT NULL,
            workspace_id TEXT,
            payload_json TEXT NOT NULL DEFAULT '{}',
            status TEXT NOT NULL DEFAULT 'pending',
            priority INTEGER NOT NULL DEFAULT 100,
            attempt_count INTEGER NOT NULL DEFAULT 0,
            available_at REAL NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            claimed_at REAL,
            completed_at REAL,
            lease_owner TEXT,
            lease_expires_at REAL,
            idempotency_key TEXT UNIQUE,
            last_error TEXT
        );

        CREATE TABLE IF NOT EXISTS embedding_repair_queue (
            id TEXT PRIMARY KEY,
            memory_id TEXT NOT NULL,
            workspace_id TEXT,
            model_name TEXT NOT NULL,
            memory_updated_at TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            attempt_count INTEGER NOT NULL DEFAULT 0,
            available_at REAL NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            claimed_at REAL,
            completed_at REAL,
            lease_owner TEXT,
            lease_expires_at REAL,
            last_error TEXT,
            UNIQUE(memory_id, model_name, memory_updated_at)
        );

        CREATE TABLE IF NOT EXISTS embedding_integrity_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            workspace_id TEXT,
            event_kind TEXT NOT NULL,
            model_name TEXT,
            source_kind TEXT,
            source_id TEXT,
            scanned_row_count INTEGER,
            invalid_row_count INTEGER,
            mixed_dimension_group_count INTEGER,
            details_json TEXT NOT NULL DEFAULT '{}',
            created_at REAL NOT NULL
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
            memory_id UNINDEXED,
            title,
            summary,
            content,
            tags,
            tokenize='unicode61'
        );
        """
    )


def apply_legacy_additive_migrations(conn: sqlite3.Connection) -> None:
    ensure_column(conn, "system1_journal", "workspace_id", "TEXT")
    ensure_column(conn, "system1_journal", "author", "TEXT")
    ensure_column(conn, "system1_journal", "claim_task_id", "TEXT")
    ensure_column(conn, "system1_journal", "claimed_at", "REAL")
    ensure_column(conn, "system1_journal", "recoverable_until", "REAL")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_system1_journal_claim_task_id ON system1_journal(claim_task_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_system1_journal_recoverable_until ON system1_journal(status, recoverable_until)"
    )
    ensure_column(conn, "tasks", "workspace_id", "TEXT")
    ensure_column(conn, "tasks", "execution_epoch", "INTEGER NOT NULL DEFAULT 0")
    ensure_column(conn, "tasks", "priority", "INTEGER NOT NULL DEFAULT 100")
    ensure_column(conn, "tasks", "retries_count", "INTEGER NOT NULL DEFAULT 0")
    ensure_column(conn, "tasks", "max_retries", "INTEGER NOT NULL DEFAULT 3")
    ensure_column(conn, "tasks", "available_at", "REAL NOT NULL DEFAULT 0")
    ensure_column(conn, "tasks", "claimed_at", "REAL")
    ensure_column(conn, "tasks", "started_at", "REAL")
    ensure_column(conn, "tasks", "completed_at", "REAL")
    ensure_column(conn, "tasks", "last_error", "TEXT")
    ensure_column(conn, "tasks", "subprocess_pid", "INTEGER")
    ensure_column(conn, "tasks", "active_request_id", "TEXT")
    ensure_column(conn, "tasks", "cancellation_requested_at", "REAL")
    ensure_column(conn, "tasks", "cancelled_at", "REAL")
    ensure_column(conn, "tasks", "cancellation_reason", "TEXT")
    ensure_column(conn, "tasks", "cancelled_by", "TEXT")
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
    ensure_column(conn, "memories", "read_count", "INTEGER NOT NULL DEFAULT 0")
    ensure_column(conn, "memories", "access_score", "REAL NOT NULL DEFAULT 0")
    ensure_column(conn, "memories", "last_accessed_at", "TEXT")
    ensure_column(conn, "memories", "last_surfaced_at", "TEXT")
    ensure_column(conn, "memories", "metadata", "TEXT NOT NULL DEFAULT '{}'"
    )
    ensure_column(conn, "hook_conversations", "workspace_id", "TEXT")
    ensure_column(conn, "hook_conversations", "last_tool_name", "TEXT")
    ensure_column(conn, "hook_conversations", "last_payload_json", "TEXT NOT NULL DEFAULT '{}'"
    )
    ensure_column(conn, "hook_conversations", "ended_at", "REAL")
    ensure_column(conn, "runtime_logs", "workspace_id", "TEXT")
    ensure_column(conn, "runtime_logs", "source", "TEXT NOT NULL DEFAULT 'runtime'")
    ensure_column(conn, "runtime_logs", "logger_name", "TEXT NOT NULL DEFAULT ''")
    ensure_column(conn, "runtime_logs", "level", "TEXT NOT NULL DEFAULT 'INFO'")
    ensure_column(conn, "runtime_logs", "message", "TEXT NOT NULL DEFAULT ''")
    ensure_column(conn, "runtime_logs", "created_at", "REAL NOT NULL DEFAULT 0")
    ensure_column(conn, "runtime_logs", "data_json", "TEXT NOT NULL DEFAULT '{}'"
    )
    ensure_column(conn, "provider_usage", "workspace_id", "TEXT")
    ensure_column(conn, "provider_usage", "task_name", "TEXT")
    ensure_column(conn, "provider_usage", "task_id", "TEXT")
    ensure_column(conn, "provider_usage", "request_id", "TEXT")
    ensure_column(conn, "provider_usage", "subprocess_pid", "INTEGER")
    ensure_column(conn, "provider_usage", "provider_key", "TEXT NOT NULL DEFAULT ''")
    ensure_column(conn, "provider_usage", "provider_name", "TEXT NOT NULL DEFAULT ''")
    ensure_column(conn, "provider_usage", "model_name", "TEXT NOT NULL DEFAULT ''")
    ensure_column(conn, "provider_usage", "status", "TEXT NOT NULL DEFAULT 'success'")
    ensure_column(conn, "provider_usage", "duration_seconds", "REAL NOT NULL DEFAULT 0")
    ensure_column(conn, "provider_usage", "created_at", "REAL NOT NULL DEFAULT 0")
    ensure_column(conn, "provider_usage", "error_text", "TEXT")
    ensure_column(conn, "provider_usage", "reason_category", "TEXT")
    ensure_column(conn, "provider_usage", "reason_code", "TEXT")
    ensure_column(conn, "provider_usage", "retry_delay_seconds", "REAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS task_execution_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
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
            started_at REAL NOT NULL,
            last_heartbeat_at REAL,
            completed_at REAL,
            error_text TEXT,
            termination_reason TEXT,
            UNIQUE(task_id, execution_epoch)
        )
        """
    )
    ensure_column(conn, "ai_conversations", "reason_category", "TEXT")
    ensure_column(conn, "ai_conversations", "reason_code", "TEXT")
    ensure_column(conn, "ai_conversations", "retry_delay_seconds", "REAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS provider_admission_state (
            provider_key TEXT NOT NULL,
            model_name TEXT NOT NULL,
            reason_category TEXT NOT NULL,
            reason_code TEXT NOT NULL,
            error_text TEXT,
            retry_delay_seconds REAL,
            active_until REAL NOT NULL,
            updated_at REAL NOT NULL,
            PRIMARY KEY (provider_key, model_name)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_tool_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            invocation_id TEXT NOT NULL,
            workspace_id TEXT,
            caller_kind TEXT NOT NULL,
            event_kind TEXT NOT NULL,
            memory_id TEXT,
            query_text TEXT,
            result_rank INTEGER,
            result_count INTEGER,
            duration_ms REAL,
            created_at REAL NOT NULL,
            FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE SET NULL
        )
        """
    )
    ensure_column(conn, "memory_tool_events", "duration_ms", "REAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS provider_policy_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            workspace_id TEXT,
            task_name TEXT NOT NULL,
            task_id TEXT,
            event_kind TEXT NOT NULL,
            warning_kind TEXT,
            provider_key TEXT,
            provider_name TEXT,
            model_name TEXT,
            route_key TEXT,
            candidate_routes_json TEXT NOT NULL DEFAULT '[]',
            reason_category TEXT,
            reason_code TEXT,
            retry_delay_seconds REAL,
            warning_suppressed INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS work_items (
            id TEXT PRIMARY KEY,
            family_key TEXT NOT NULL,
            execution_lane TEXT NOT NULL,
            workspace_id TEXT,
            payload_json TEXT NOT NULL DEFAULT '{}',
            status TEXT NOT NULL DEFAULT 'pending',
            priority INTEGER NOT NULL DEFAULT 100,
            attempt_count INTEGER NOT NULL DEFAULT 0,
            available_at REAL NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            claimed_at REAL,
            completed_at REAL,
            lease_owner TEXT,
            lease_expires_at REAL,
            idempotency_key TEXT UNIQUE,
            last_error TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS embedding_repair_queue (
            id TEXT PRIMARY KEY,
            memory_id TEXT NOT NULL,
            workspace_id TEXT,
            model_name TEXT NOT NULL,
            memory_updated_at TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            attempt_count INTEGER NOT NULL DEFAULT 0,
            available_at REAL NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            claimed_at REAL,
            completed_at REAL,
            lease_owner TEXT,
            lease_expires_at REAL,
            last_error TEXT,
            UNIQUE(memory_id, model_name, memory_updated_at)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS embedding_integrity_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            workspace_id TEXT,
            event_kind TEXT NOT NULL,
            model_name TEXT,
            source_kind TEXT,
            source_id TEXT,
            scanned_row_count INTEGER,
            invalid_row_count INTEGER,
            mixed_dimension_group_count INTEGER,
            details_json TEXT NOT NULL DEFAULT '{}',
            created_at REAL NOT NULL
        )
        """
    )


def finalize_schema_setup(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
        CREATE INDEX IF NOT EXISTS idx_tasks_ready ON tasks(status, available_at, priority, created_at);

        CREATE INDEX IF NOT EXISTS idx_task_runs_task_name_completed_at
            ON task_runs(task_name, completed_at DESC);
        CREATE INDEX IF NOT EXISTS idx_task_runs_workspace_task_name_completed_at
            ON task_runs(workspace_id, task_name, completed_at DESC);

        CREATE INDEX IF NOT EXISTS idx_system1_journal_status ON system1_journal(status);
        CREATE INDEX IF NOT EXISTS idx_system1_journal_claim_task_id ON system1_journal(claim_task_id);
        CREATE INDEX IF NOT EXISTS idx_system1_journal_recoverable_until ON system1_journal(status, recoverable_until);

        CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type);
        CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status);
        CREATE INDEX IF NOT EXISTS idx_memories_updated_at ON memories(updated_at);
        CREATE INDEX IF NOT EXISTS idx_memories_last_accessed_at ON memories(last_accessed_at);
        CREATE INDEX IF NOT EXISTS idx_memories_last_surfaced_at ON memories(last_surfaced_at);

        CREATE INDEX IF NOT EXISTS idx_memory_workspaces_workspace_id ON memory_workspaces(workspace_id);
        CREATE INDEX IF NOT EXISTS idx_memory_tags_tag_id ON memory_tags(tag_id);

        CREATE INDEX IF NOT EXISTS idx_links_source_id ON links(source_id);
        CREATE INDEX IF NOT EXISTS idx_links_target_id ON links(target_id);
        CREATE INDEX IF NOT EXISTS idx_links_type ON links(type);

        CREATE INDEX IF NOT EXISTS idx_embeddings_source_kind_id
            ON embeddings(source_kind, source_id);
        CREATE INDEX IF NOT EXISTS idx_embeddings_workspace_kind
            ON embeddings(workspace_id, source_kind);

        CREATE INDEX IF NOT EXISTS idx_hook_conversations_workspace_id
            ON hook_conversations(workspace_id);

        CREATE INDEX IF NOT EXISTS idx_runtime_logs_workspace_created_at
            ON runtime_logs(workspace_id, created_at DESC, id DESC);
        CREATE INDEX IF NOT EXISTS idx_runtime_logs_level_created_at
            ON runtime_logs(level, created_at DESC, id DESC);

        CREATE INDEX IF NOT EXISTS idx_provider_usage_workspace_created_at
            ON provider_usage(workspace_id, created_at DESC, id DESC);
        CREATE INDEX IF NOT EXISTS idx_provider_usage_provider_created_at
            ON provider_usage(provider_key, created_at DESC, id DESC);
        CREATE INDEX IF NOT EXISTS idx_provider_usage_reason_created_at
            ON provider_usage(reason_code, created_at DESC, id DESC);
        CREATE INDEX IF NOT EXISTS idx_task_execution_attempts_workspace_started_at
            ON task_execution_attempts(workspace_id, started_at DESC, id DESC);
        CREATE INDEX IF NOT EXISTS idx_task_execution_attempts_task_started_at
            ON task_execution_attempts(task_id, started_at DESC, id DESC);
        CREATE INDEX IF NOT EXISTS idx_task_execution_attempts_status_started_at
            ON task_execution_attempts(status, started_at DESC, id DESC);
        CREATE INDEX IF NOT EXISTS idx_task_execution_attempts_request_id
            ON task_execution_attempts(request_id);

        CREATE INDEX IF NOT EXISTS idx_ai_conversations_request_id
            ON ai_conversations(request_id);
        CREATE INDEX IF NOT EXISTS idx_ai_conversations_workspace_created_at
            ON ai_conversations(workspace_id, completed_at DESC, id DESC);
        CREATE INDEX IF NOT EXISTS idx_ai_conversations_task_created_at
            ON ai_conversations(task_name, completed_at DESC, id DESC);

        CREATE INDEX IF NOT EXISTS idx_provider_admission_state_active_until
            ON provider_admission_state(active_until DESC, updated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_memory_tool_events_workspace_kind_created_at
            ON memory_tool_events(workspace_id, event_kind, created_at DESC, id DESC);
        CREATE INDEX IF NOT EXISTS idx_memory_tool_events_memory_kind_created_at
            ON memory_tool_events(memory_id, event_kind, created_at DESC, id DESC);
        CREATE INDEX IF NOT EXISTS idx_memory_tool_events_invocation_kind
            ON memory_tool_events(invocation_id, event_kind);
        CREATE INDEX IF NOT EXISTS idx_provider_policy_events_workspace_created_at
            ON provider_policy_events(workspace_id, created_at DESC, id DESC);
        CREATE INDEX IF NOT EXISTS idx_provider_policy_events_task_created_at
            ON provider_policy_events(task_name, created_at DESC, id DESC);
        CREATE INDEX IF NOT EXISTS idx_provider_policy_events_kind_created_at
            ON provider_policy_events(event_kind, created_at DESC, id DESC);
        CREATE INDEX IF NOT EXISTS idx_provider_policy_events_provider_created_at
            ON provider_policy_events(provider_key, created_at DESC, id DESC);
        CREATE INDEX IF NOT EXISTS idx_work_items_ready
            ON work_items(status, family_key, execution_lane, available_at, priority, created_at);
        CREATE INDEX IF NOT EXISTS idx_work_items_workspace_ready
            ON work_items(workspace_id, status, family_key, execution_lane, available_at, priority, created_at);
        CREATE INDEX IF NOT EXISTS idx_work_items_lease_owner
            ON work_items(lease_owner, status, lease_expires_at);
        CREATE INDEX IF NOT EXISTS idx_embedding_repair_queue_ready
            ON embedding_repair_queue(status, available_at, created_at);
        CREATE INDEX IF NOT EXISTS idx_embedding_repair_queue_workspace_ready
            ON embedding_repair_queue(workspace_id, status, available_at, created_at);
        CREATE INDEX IF NOT EXISTS idx_embedding_repair_queue_lease_owner
            ON embedding_repair_queue(lease_owner, status, lease_expires_at);
        CREATE INDEX IF NOT EXISTS idx_embedding_integrity_events_workspace_created_at
            ON embedding_integrity_events(workspace_id, created_at DESC, id DESC);
        CREATE INDEX IF NOT EXISTS idx_embedding_integrity_events_kind_created_at
            ON embedding_integrity_events(event_kind, created_at DESC, id DESC);
        CREATE INDEX IF NOT EXISTS idx_embedding_integrity_events_model_created_at
            ON embedding_integrity_events(model_name, created_at DESC, id DESC);
        """
    )
    conn.execute(
        "INSERT OR REPLACE INTO schema_metadata (key, value) VALUES (?, ?)",
        ("schema_version", str(SCHEMA_VERSION)),
    )
    rebuild_memories_fts(conn)
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


def rebuild_memories_fts(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM memories_fts")
    conn.execute(
        """
        INSERT INTO memories_fts(memory_id, title, summary, content, tags)
        SELECT
            memories.id,
            memories.title,
            COALESCE(memories.summary, ''),
            memories.content,
            COALESCE(GROUP_CONCAT(DISTINCT tags.name), '')
        FROM memories
        LEFT JOIN memory_tags ON memory_tags.memory_id = memories.id
        LEFT JOIN tags ON tags.id = memory_tags.tag_id
        GROUP BY memories.id, memories.title, memories.summary, memories.content
        """
    )
