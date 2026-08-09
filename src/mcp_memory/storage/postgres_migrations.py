from __future__ import annotations

from dataclasses import dataclass


POSTGRES_SCHEMA_VERSION = 34


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
                archived_at TEXT,
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
    PostgresMigration(
        version=5,
        name="add_operational_runtime_stores",
        statements=(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                task_name TEXT NOT NULL,
                workspace_id TEXT,
                data JSONB NOT NULL DEFAULT '{}'::jsonb,
                status TEXT NOT NULL DEFAULT 'pending',
                execution_epoch INTEGER NOT NULL DEFAULT 0,
                priority INTEGER NOT NULL DEFAULT 100,
                retries_count INTEGER NOT NULL DEFAULT 0,
                max_retries INTEGER NOT NULL DEFAULT 3,
                created_at DOUBLE PRECISION NOT NULL,
                updated_at DOUBLE PRECISION NOT NULL,
                available_at DOUBLE PRECISION NOT NULL,
                claimed_at DOUBLE PRECISION,
                started_at DOUBLE PRECISION,
                completed_at DOUBLE PRECISION,
                last_error TEXT,
                subprocess_pid INTEGER,
                active_request_id TEXT,
                cancellation_requested_at DOUBLE PRECISION,
                cancelled_at DOUBLE PRECISION,
                cancellation_reason TEXT,
                cancelled_by TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS task_runs (
                id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                task_name TEXT NOT NULL,
                workspace_id TEXT,
                status TEXT NOT NULL,
                started_at DOUBLE PRECISION NOT NULL,
                completed_at DOUBLE PRECISION NOT NULL,
                duration_seconds DOUBLE PRECISION NOT NULL DEFAULT 0,
                result_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                error_text TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS system1_journal (
                id BIGSERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                workspace_id TEXT,
                author TEXT,
                timestamp DOUBLE PRECISION NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                claim_task_id TEXT,
                claimed_at DOUBLE PRECISION,
                recoverable_until DOUBLE PRECISION
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                source_kind TEXT NOT NULL,
                source_id TEXT NOT NULL,
                workspace_id TEXT,
                model_name TEXT NOT NULL,
                embedding_json JSONB NOT NULL,
                memory_updated_at TEXT,
                updated_at DOUBLE PRECISION NOT NULL,
                PRIMARY KEY (source_kind, source_id, model_name)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS work_items (
                id TEXT PRIMARY KEY,
                family_key TEXT NOT NULL,
                execution_lane TEXT NOT NULL,
                workspace_id TEXT,
                payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                status TEXT NOT NULL DEFAULT 'pending',
                priority INTEGER NOT NULL DEFAULT 100,
                attempt_count INTEGER NOT NULL DEFAULT 0,
                available_at DOUBLE PRECISION NOT NULL,
                created_at DOUBLE PRECISION NOT NULL,
                updated_at DOUBLE PRECISION NOT NULL,
                claimed_at DOUBLE PRECISION,
                completed_at DOUBLE PRECISION,
                lease_owner TEXT,
                lease_expires_at DOUBLE PRECISION,
                idempotency_key TEXT UNIQUE,
                last_error TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS embedding_repair_queue (
                id TEXT PRIMARY KEY,
                memory_id TEXT NOT NULL,
                workspace_id TEXT,
                model_name TEXT NOT NULL,
                memory_updated_at TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                attempt_count INTEGER NOT NULL DEFAULT 0,
                available_at DOUBLE PRECISION NOT NULL,
                created_at DOUBLE PRECISION NOT NULL,
                updated_at DOUBLE PRECISION NOT NULL,
                claimed_at DOUBLE PRECISION,
                completed_at DOUBLE PRECISION,
                lease_owner TEXT,
                lease_expires_at DOUBLE PRECISION,
                last_error TEXT,
                UNIQUE (memory_id, model_name, memory_updated_at)
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)",
            "CREATE INDEX IF NOT EXISTS idx_tasks_ready ON tasks(status, available_at, priority, created_at)",
            "CREATE INDEX IF NOT EXISTS idx_task_runs_task_name_completed_at ON task_runs(task_name, completed_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_task_runs_workspace_task_name_completed_at ON task_runs(workspace_id, task_name, completed_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_system1_journal_status ON system1_journal(status)",
            "CREATE INDEX IF NOT EXISTS idx_system1_journal_claim_task_id ON system1_journal(claim_task_id)",
            "CREATE INDEX IF NOT EXISTS idx_system1_journal_recoverable_until ON system1_journal(status, recoverable_until)",
            "CREATE INDEX IF NOT EXISTS idx_embeddings_source_kind_id ON embeddings(source_kind, source_id)",
            "CREATE INDEX IF NOT EXISTS idx_embeddings_workspace_kind ON embeddings(workspace_id, source_kind)",
            "CREATE INDEX IF NOT EXISTS idx_work_items_ready ON work_items(status, family_key, execution_lane, available_at, priority, created_at)",
            "CREATE INDEX IF NOT EXISTS idx_work_items_workspace_ready ON work_items(workspace_id, status, family_key, execution_lane, available_at, priority, created_at)",
            "CREATE INDEX IF NOT EXISTS idx_work_items_lease_owner ON work_items(lease_owner, status, lease_expires_at)",
            "CREATE INDEX IF NOT EXISTS idx_embedding_repair_queue_ready ON embedding_repair_queue(status, available_at, created_at)",
            "CREATE INDEX IF NOT EXISTS idx_embedding_repair_queue_workspace_ready ON embedding_repair_queue(workspace_id, status, available_at, created_at)",
            "CREATE INDEX IF NOT EXISTS idx_embedding_repair_queue_lease_owner ON embedding_repair_queue(lease_owner, status, lease_expires_at)",
        ),
    ),
    PostgresMigration(
        version=6,
        name="add_hook_conversations_and_memory_tool_events",
        statements=(
            """
            CREATE TABLE IF NOT EXISTS hook_conversations (
                conversation_id TEXT PRIMARY KEY,
                workspace_id TEXT,
                started_at DOUBLE PRECISION NOT NULL,
                updated_at DOUBLE PRECISION NOT NULL,
                last_ping_at DOUBLE PRECISION,
                last_reminder_at DOUBLE PRECISION,
                last_tool_name TEXT,
                last_payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                ended_at DOUBLE PRECISION
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_hook_conversations_workspace_id ON hook_conversations(workspace_id)",
            """
            CREATE TABLE IF NOT EXISTS memory_tool_events (
                id BIGSERIAL PRIMARY KEY,
                invocation_id TEXT NOT NULL,
                workspace_id TEXT,
                caller_kind TEXT NOT NULL,
                event_kind TEXT NOT NULL,
                memory_id TEXT,
                query_text TEXT,
                result_rank INTEGER,
                result_count INTEGER,
                duration_ms DOUBLE PRECISION,
                created_at DOUBLE PRECISION NOT NULL,
                FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE SET NULL
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_memory_tool_events_workspace_kind_created ON memory_tool_events(workspace_id, event_kind, created_at DESC, id DESC)",
            "CREATE INDEX IF NOT EXISTS idx_memory_tool_events_memory_kind_created ON memory_tool_events(memory_id, event_kind, created_at DESC, id DESC)",
            "CREATE INDEX IF NOT EXISTS idx_memory_tool_events_invocation_kind ON memory_tool_events(invocation_id, event_kind)",
        ),
    ),
    PostgresMigration(
        version=7,
        name="add_memory_tool_event_latency",
        statements=(
            "ALTER TABLE memory_tool_events ADD COLUMN IF NOT EXISTS duration_ms DOUBLE PRECISION",
        ),
    ),
    PostgresMigration(
        version=8,
        name="add_memory_search_projection",
        statements=(
            """
            CREATE TABLE IF NOT EXISTS memory_search_documents (
                memory_id TEXT PRIMARY KEY,
                title TEXT NOT NULL DEFAULT '',
                summary TEXT NOT NULL DEFAULT '',
                content TEXT NOT NULL DEFAULT '',
                tags_text TEXT NOT NULL DEFAULT '',
                search_document TSVECTOR NOT NULL,
                FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_memory_search_documents_search_document ON memory_search_documents USING GIN(search_document)",
            """
            INSERT INTO memory_search_documents (memory_id, title, summary, content, tags_text, search_document)
            SELECT
                memories.id,
                COALESCE(memories.title, ''),
                COALESCE(memories.summary, ''),
                COALESCE(memories.content, ''),
                COALESCE(tag_agg.tags_text, ''),
                (
                    setweight(to_tsvector('simple', COALESCE(memories.title, '')), 'A')
                    || setweight(to_tsvector('simple', COALESCE(memories.summary, '')), 'A')
                    || setweight(to_tsvector('simple', COALESCE(tag_agg.tags_text, '')), 'B')
                    || setweight(to_tsvector('simple', COALESCE(memories.content, '')), 'C')
                )
            FROM memories
            LEFT JOIN (
                SELECT memory_tags.memory_id, STRING_AGG(tags.name, ' ' ORDER BY tags.name) AS tags_text
                FROM memory_tags
                JOIN tags ON tags.id = memory_tags.tag_id
                GROUP BY memory_tags.memory_id
            ) AS tag_agg ON tag_agg.memory_id = memories.id
            ON CONFLICT (memory_id)
            DO UPDATE SET
                title = EXCLUDED.title,
                summary = EXCLUDED.summary,
                content = EXCLUDED.content,
                tags_text = EXCLUDED.tags_text,
                search_document = EXCLUDED.search_document
            """,
        ),
    ),
    PostgresMigration(
        version=9,
        name="add_optional_pgvector_embedding_column",
        statements=(
            """
            DO $$
            BEGIN
                IF EXISTS (
                    SELECT 1
                    FROM pg_extension
                    WHERE extname = 'vector'
                ) THEN
                    EXECUTE 'ALTER TABLE embeddings ADD COLUMN IF NOT EXISTS embedding_vector vector';
                END IF;
            END
            $$
            """,
        ),
    ),
    PostgresMigration(
        version=10,
        name="add_embedding_integrity_events",
        statements=(
            """
            CREATE TABLE IF NOT EXISTS embedding_integrity_events (
                id BIGSERIAL PRIMARY KEY,
                workspace_id TEXT,
                event_kind TEXT NOT NULL,
                model_name TEXT,
                source_kind TEXT,
                source_id TEXT,
                scanned_row_count INTEGER,
                invalid_row_count INTEGER,
                mixed_dimension_group_count INTEGER,
                details_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at DOUBLE PRECISION NOT NULL
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_embedding_integrity_events_workspace_created ON embedding_integrity_events(workspace_id, created_at DESC, id DESC)",
            "CREATE INDEX IF NOT EXISTS idx_embedding_integrity_events_kind_created ON embedding_integrity_events(event_kind, created_at DESC, id DESC)",
            "CREATE INDEX IF NOT EXISTS idx_embedding_integrity_events_model_created ON embedding_integrity_events(model_name, created_at DESC, id DESC)",
        ),
    ),
    PostgresMigration(
        version=11,
        name="add_mutation_history_and_protections",
        statements=(
            """
            CREATE TABLE IF NOT EXISTS memory_mutation_events (
                id TEXT PRIMARY KEY,
                operation TEXT NOT NULL,
                actor_kind TEXT NOT NULL,
                actor_id TEXT,
                family TEXT,
                task_id TEXT,
                curation_run_id TEXT,
                plan_id TEXT,
                action_id TEXT,
                provider_id TEXT,
                reason_code TEXT,
                rationale TEXT,
                policy_version TEXT,
                schema_version INTEGER NOT NULL DEFAULT 1,
                status TEXT NOT NULL,
                restores_event_id TEXT REFERENCES memory_mutation_events(id),
                idempotency_key TEXT,
                created_at TEXT NOT NULL,
                terminalized_at TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS memory_record_revisions (
                id BIGSERIAL PRIMARY KEY,
                event_id TEXT NOT NULL REFERENCES memory_mutation_events(id) ON DELETE CASCADE,
                memory_id TEXT NOT NULL,
                role TEXT NOT NULL,
                before_exists BOOLEAN NOT NULL,
                before_snapshot JSONB,
                after_exists BOOLEAN NOT NULL,
                after_snapshot JSONB,
                before_token TEXT,
                after_token TEXT,
                UNIQUE (event_id, memory_id, role)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS memory_link_revisions (
                id BIGSERIAL PRIMARY KEY,
                event_id TEXT NOT NULL REFERENCES memory_mutation_events(id) ON DELETE CASCADE,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                link_type TEXT NOT NULL,
                context TEXT,
                before_exists BOOLEAN NOT NULL,
                after_exists BOOLEAN NOT NULL,
                UNIQUE (event_id, source_id, target_id, link_type)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS memory_protections (
                memory_id TEXT NOT NULL,
                mode TEXT NOT NULL,
                reason TEXT NOT NULL,
                actor_id TEXT,
                created_at TEXT NOT NULL,
                expires_at TEXT,
                PRIMARY KEY (memory_id, mode)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS memory_restore_requests (
                id TEXT PRIMARY KEY,
                target_event_id TEXT NOT NULL REFERENCES memory_mutation_events(id),
                scope TEXT NOT NULL,
                expected_record_tokens JSONB NOT NULL DEFAULT '{}'::jsonb,
                expected_link_tokens JSONB NOT NULL DEFAULT '{}'::jsonb,
                actor_id TEXT,
                reason TEXT NOT NULL,
                idempotency_key TEXT NOT NULL UNIQUE,
                confirmation BOOLEAN NOT NULL DEFAULT FALSE,
                status TEXT NOT NULL,
                event_id TEXT REFERENCES memory_mutation_events(id),
                conflict_reason TEXT,
                conflict_details JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TEXT NOT NULL,
                terminalized_at TEXT
            )
            """,
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_memory_mutation_events_action ON memory_mutation_events(curation_run_id, action_id) WHERE curation_run_id IS NOT NULL AND action_id IS NOT NULL",
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_memory_mutation_events_idempotency ON memory_mutation_events(idempotency_key) WHERE idempotency_key IS NOT NULL",
            "CREATE INDEX IF NOT EXISTS idx_memory_mutation_events_created_at ON memory_mutation_events(created_at DESC, id DESC)",
            "CREATE INDEX IF NOT EXISTS idx_memory_mutation_events_curation_run ON memory_mutation_events(curation_run_id, action_id)",
            "CREATE INDEX IF NOT EXISTS idx_memory_mutation_events_restores_event ON memory_mutation_events(restores_event_id)",
            "CREATE INDEX IF NOT EXISTS idx_memory_record_revisions_event ON memory_record_revisions(event_id, id)",
            "CREATE INDEX IF NOT EXISTS idx_memory_record_revisions_memory ON memory_record_revisions(memory_id, id DESC)",
            "CREATE INDEX IF NOT EXISTS idx_memory_link_revisions_event ON memory_link_revisions(event_id, id)",
            "CREATE INDEX IF NOT EXISTS idx_memory_link_revisions_endpoint ON memory_link_revisions(source_id, target_id, id DESC)",
            "CREATE INDEX IF NOT EXISTS idx_memory_protections_memory ON memory_protections(memory_id, mode)",
            "CREATE INDEX IF NOT EXISTS idx_memory_restore_requests_target ON memory_restore_requests(target_event_id, created_at DESC)",
        ),
    ),
    PostgresMigration(
        version=12,
        name="add_curation_ledger",
        statements=(
            """
            CREATE TABLE IF NOT EXISTS curation_runs (
                run_id TEXT PRIMARY KEY,
                task_id TEXT,
                execution_epoch INTEGER,
                work_item_id TEXT,
                frontier_key TEXT NOT NULL,
                selector_strategy TEXT,
                context_fingerprint TEXT NOT NULL,
                planner_id TEXT,
                provider_id TEXT,
                model_id TEXT,
                policy_version TEXT NOT NULL DEFAULT '1',
                schema_version INTEGER NOT NULL DEFAULT 1,
                state TEXT NOT NULL,
                outcome TEXT,
                plan_id TEXT,
                rejection_codes_json JSONB NOT NULL DEFAULT '[]'::jsonb,
                retry_reason TEXT,
                budget_usage_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                disclosure_audit_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TEXT NOT NULL,
                terminalized_at TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS curation_action_receipts (
                run_id TEXT NOT NULL,
                action_id TEXT NOT NULL,
                operation TEXT NOT NULL,
                affected_ids_json JSONB NOT NULL DEFAULT '[]'::jsonb,
                status TEXT NOT NULL,
                before_token TEXT,
                after_token TEXT,
                mutation_event_id TEXT,
                intent_hash TEXT NOT NULL,
                verification_descriptor_json JSONB,
                error_code TEXT,
                applied_at TEXT,
                verified_at TEXT,
                PRIMARY KEY (run_id, action_id),
                FOREIGN KEY (run_id) REFERENCES curation_runs(run_id) ON DELETE CASCADE,
                FOREIGN KEY (mutation_event_id) REFERENCES memory_mutation_events(id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS curation_candidate_state (
                memory_id TEXT PRIMARY KEY,
                last_observed_revision_token TEXT,
                disposition TEXT NOT NULL DEFAULT 'pending',
                consecutive_no_op_count INTEGER NOT NULL DEFAULT 0,
                cooldown_until TEXT,
                last_disposition_reason TEXT,
                last_frontier_key TEXT,
                last_run_id TEXT,
                escalation_count INTEGER NOT NULL DEFAULT 0,
                last_escalated_strategy TEXT
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_curation_runs_created_at ON curation_runs(created_at DESC, run_id DESC)",
            "CREATE INDEX IF NOT EXISTS idx_curation_action_receipts_run ON curation_action_receipts(run_id, action_id)",
            "CREATE INDEX IF NOT EXISTS idx_curation_candidate_state_cooldown ON curation_candidate_state(cooldown_until, memory_id)",
        ),
    ),
    PostgresMigration(
        version=13,
        name="add_embedding_memory_versions",
        statements=(
            "ALTER TABLE embeddings ADD COLUMN IF NOT EXISTS memory_updated_at TEXT",
        ),
    ),
    PostgresMigration(
        version=14,
        name="add_curation_disclosure_audit",
        statements=(
            "ALTER TABLE curation_runs ADD COLUMN IF NOT EXISTS disclosure_audit_json JSONB NOT NULL DEFAULT '{}'::jsonb",
        ),
    ),
    PostgresMigration(
        version=15,
        name="add_curation_action_intent_hash",
        statements=(
            "ALTER TABLE curation_action_receipts ADD COLUMN IF NOT EXISTS intent_hash TEXT",
        ),
    ),
    PostgresMigration(
        version=16,
        name="add_memories_updated_at_index",
        statements=(
            "CREATE INDEX IF NOT EXISTS idx_memories_updated_at ON memories(updated_at)",
        ),
    ),
    PostgresMigration(
        version=17,
        name="add_curation_verification_descriptor",
        statements=(
            "ALTER TABLE curation_action_receipts ADD COLUMN IF NOT EXISTS verification_descriptor_json JSONB",
        ),
    ),
    PostgresMigration(
        version=18,
        name="add_memory_references",
        statements=(
            "ALTER TABLE memories ADD COLUMN IF NOT EXISTS memory_ref BIGINT",
            "CREATE SEQUENCE IF NOT EXISTS memory_ref_seq",
            (
                "SELECT setval("
                "'memory_ref_seq', "
                "GREATEST(COALESCE(MAX(memory_ref), 1), 1), "
                "COUNT(memory_ref) > 0"
                ") FROM memories"
            ),
            "UPDATE memories SET memory_ref = nextval('memory_ref_seq') WHERE memory_ref IS NULL",
            (
                "SELECT setval("
                "'memory_ref_seq', "
                "GREATEST(COALESCE(MAX(memory_ref), 1), 1), "
                "COUNT(memory_ref) > 0"
                ") FROM memories"
            ),
            "ALTER TABLE memories ALTER COLUMN memory_ref SET DEFAULT nextval('memory_ref_seq')",
            "ALTER TABLE memories ALTER COLUMN memory_ref SET NOT NULL",
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_memories_memory_ref ON memories(memory_ref)",
        ),
    ),
    PostgresMigration(
        version=19,
        name="add_search_mutation_epochs",
        statements=(
            "INSERT INTO schema_metadata(key, value) VALUES ('search_epoch_keyword', '0') ON CONFLICT (key) DO NOTHING",
            "INSERT INTO schema_metadata(key, value) VALUES ('search_epoch_vector', '0') ON CONFLICT (key) DO NOTHING",
            "INSERT INTO schema_metadata(key, value) VALUES ('search_epoch_graph', '0') ON CONFLICT (key) DO NOTHING",
            """
            CREATE OR REPLACE FUNCTION mcp_memory_bump_search_epoch()
            RETURNS trigger
            LANGUAGE plpgsql
            AS $function$
            BEGIN
                UPDATE schema_metadata
                SET value = (value::bigint + 1)::text
                WHERE key = TG_ARGV[0];
                IF TG_OP = 'DELETE' THEN
                    RETURN OLD;
                END IF;
                RETURN NEW;
            END;
            $function$
            """,
            "DROP TRIGGER IF EXISTS memories_search_epoch ON memories",
            """
            CREATE TRIGGER memories_search_epoch
            AFTER INSERT OR UPDATE OR DELETE ON memories
            FOR EACH ROW EXECUTE FUNCTION mcp_memory_bump_search_epoch('search_epoch_keyword')
            """,
            "DROP TRIGGER IF EXISTS memory_workspaces_search_epoch ON memory_workspaces",
            """
            CREATE TRIGGER memory_workspaces_search_epoch
            AFTER INSERT OR UPDATE OR DELETE ON memory_workspaces
            FOR EACH ROW EXECUTE FUNCTION mcp_memory_bump_search_epoch('search_epoch_keyword')
            """,
            "DROP TRIGGER IF EXISTS links_search_epoch ON links",
            """
            CREATE TRIGGER links_search_epoch
            AFTER INSERT OR UPDATE OR DELETE ON links
            FOR EACH ROW EXECUTE FUNCTION mcp_memory_bump_search_epoch('search_epoch_graph')
            """,
            "DROP TRIGGER IF EXISTS embeddings_search_epoch ON embeddings",
            """
            CREATE TRIGGER embeddings_search_epoch
            AFTER INSERT OR UPDATE OR DELETE ON embeddings
            FOR EACH ROW EXECUTE FUNCTION mcp_memory_bump_search_epoch('search_epoch_vector')
            """,
        ),
    ),
    PostgresMigration(
        version=20,
        name="add_curation_quality_evidence",
        statements=(
            """
            CREATE TABLE IF NOT EXISTS curation_quality_evidence (
                run_id TEXT NOT NULL,
                action_id TEXT NOT NULL,
                operation TEXT NOT NULL,
                affected_memory_ids_json JSONB NOT NULL DEFAULT '[]'::jsonb,
                policy_version TEXT NOT NULL,
                query_id TEXT,
                status TEXT NOT NULL,
                before_ranked_ids_json JSONB NOT NULL DEFAULT '[]'::jsonb,
                after_ranked_ids_json JSONB NOT NULL DEFAULT '[]'::jsonb,
                retrieval_regression_count INTEGER,
                zero_result_change INTEGER,
                payload_size_change INTEGER,
                useful_work INTEGER,
                created_at TEXT NOT NULL,
                retrieval_utility_delta DOUBLE PRECISION,
                acceptance_met INTEGER,
                neutral_reason TEXT,
                content_quality_score_before DOUBLE PRECISION,
                content_quality_score_after DOUBLE PRECISION,
                content_quality_delta DOUBLE PRECISION,
                content_quality_improved INTEGER,
                PRIMARY KEY (run_id, action_id),
                FOREIGN KEY (run_id) REFERENCES curation_runs(run_id) ON DELETE CASCADE
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_curation_quality_evidence_created_at ON curation_quality_evidence(created_at DESC, run_id, action_id)",
        ),
    ),
    PostgresMigration(
        version=21,
        name="add_curation_candidate_coverage",
        statements=(
            "ALTER TABLE curation_candidate_state ADD COLUMN IF NOT EXISTS last_considered_at TEXT",
            "ALTER TABLE curation_candidate_state ADD COLUMN IF NOT EXISTS last_considered_strategy TEXT",
            "ALTER TABLE curation_candidate_state ADD COLUMN IF NOT EXISTS last_mutation_family TEXT",
            "ALTER TABLE curation_candidate_state ADD COLUMN IF NOT EXISTS last_mutated_at TEXT",
            "ALTER TABLE curation_candidate_state ADD COLUMN IF NOT EXISTS coverage_evidence_json JSONB NOT NULL DEFAULT '{}'::jsonb",
        ),
    ),
    PostgresMigration(
        version=22,
        name="add_curation_quality_acceptance_evidence",
        statements=(
            "ALTER TABLE curation_quality_evidence ADD COLUMN IF NOT EXISTS retrieval_utility_delta DOUBLE PRECISION",
            "ALTER TABLE curation_quality_evidence ADD COLUMN IF NOT EXISTS acceptance_met INTEGER",
            "ALTER TABLE curation_quality_evidence ADD COLUMN IF NOT EXISTS neutral_reason TEXT",
        ),
    ),
    PostgresMigration(
        version=23,
        name="add_curation_content_quality_evidence",
        statements=(
            "ALTER TABLE curation_quality_evidence ADD COLUMN IF NOT EXISTS content_quality_score_before DOUBLE PRECISION",
            "ALTER TABLE curation_quality_evidence ADD COLUMN IF NOT EXISTS content_quality_score_after DOUBLE PRECISION",
            "ALTER TABLE curation_quality_evidence ADD COLUMN IF NOT EXISTS content_quality_delta DOUBLE PRECISION",
            "ALTER TABLE curation_quality_evidence ADD COLUMN IF NOT EXISTS content_quality_improved INTEGER",
        ),
    ),
    PostgresMigration(
        version=24,
        name="add_memory_tool_graph_provenance",
        statements=(
            "ALTER TABLE memory_tool_events ADD COLUMN IF NOT EXISTS graph_provenance_json JSONB",
        ),
    ),
    PostgresMigration(
        version=25,
        name="add_curation_engagement_evidence",
        statements=(
            "ALTER TABLE curation_quality_evidence ADD COLUMN IF NOT EXISTS engagement_utility_delta DOUBLE PRECISION",
            "ALTER TABLE curation_quality_evidence ADD COLUMN IF NOT EXISTS engagement_evidence_json JSONB NOT NULL DEFAULT '{}'::jsonb",
        ),
    ),
    PostgresMigration(
        version=26,
        name="add_provider_token_usage",
        statements=(
            "ALTER TABLE provider_usage ADD COLUMN IF NOT EXISTS input_tokens BIGINT",
            "ALTER TABLE provider_usage ADD COLUMN IF NOT EXISTS output_tokens BIGINT",
            "ALTER TABLE provider_usage ADD COLUMN IF NOT EXISTS cached_input_tokens BIGINT",
            "ALTER TABLE provider_usage ADD COLUMN IF NOT EXISTS cache_write_tokens BIGINT",
            "ALTER TABLE provider_usage ADD COLUMN IF NOT EXISTS reasoning_tokens BIGINT",
            "ALTER TABLE provider_usage ADD COLUMN IF NOT EXISTS total_tokens BIGINT",
            "ALTER TABLE provider_usage ADD COLUMN IF NOT EXISTS token_usage_source TEXT",
            "ALTER TABLE ai_conversations ADD COLUMN IF NOT EXISTS input_tokens BIGINT",
            "ALTER TABLE ai_conversations ADD COLUMN IF NOT EXISTS output_tokens BIGINT",
            "ALTER TABLE ai_conversations ADD COLUMN IF NOT EXISTS cached_input_tokens BIGINT",
            "ALTER TABLE ai_conversations ADD COLUMN IF NOT EXISTS cache_write_tokens BIGINT",
            "ALTER TABLE ai_conversations ADD COLUMN IF NOT EXISTS reasoning_tokens BIGINT",
            "ALTER TABLE ai_conversations ADD COLUMN IF NOT EXISTS total_tokens BIGINT",
            "ALTER TABLE ai_conversations ADD COLUMN IF NOT EXISTS token_usage_source TEXT",
        ),
    ),
    PostgresMigration(
        version=27,
        name="add_memory_archived_at",
        statements=(
            "ALTER TABLE memories ADD COLUMN IF NOT EXISTS archived_at TEXT",
            "UPDATE memories SET archived_at = updated_at WHERE status = 'archived' AND archived_at IS NULL",
        ),
    ),
    PostgresMigration(
        version=28,
        name="reconcile_legacy_memory_links",
        statements=(
            "DELETE FROM links WHERE NOT EXISTS (SELECT 1 FROM memories WHERE memories.id = links.source_id) OR (links.target_id NOT LIKE 'ext:%' AND NOT EXISTS (SELECT 1 FROM memories WHERE memories.id = links.target_id))",
        ),
    ),
    PostgresMigration(
        version=29,
        name="add_direct_mutation_evidence",
        statements=(
            """
            CREATE TABLE IF NOT EXISTS direct_mutation_evidence (
                evidence_id TEXT PRIMARY KEY,
                task_id TEXT,
                execution_epoch INTEGER,
                session_id TEXT,
                call_id TEXT,
                sequence INTEGER,
                tool_name TEXT NOT NULL,
                operation TEXT NOT NULL,
                idempotency_key TEXT NOT NULL UNIQUE,
                payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                ledger_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                outcome TEXT,
                error_code TEXT,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                finalized_at TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS direct_mutation_entity_deltas (
                evidence_id TEXT NOT NULL REFERENCES direct_mutation_evidence(evidence_id) ON DELETE CASCADE,
                ordinal INTEGER NOT NULL,
                entity_kind TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                before_revision TEXT,
                after_revision TEXT,
                before_exists INTEGER NOT NULL,
                after_exists INTEGER NOT NULL,
                transition TEXT NOT NULL,
                snapshot_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                PRIMARY KEY (evidence_id, ordinal)
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_direct_mutation_evidence_execution ON direct_mutation_evidence(task_id, execution_epoch, sequence)",
        ),
    ),
    PostgresMigration(
        version=30,
        name="add_provider_attempt_identity",
        statements=(
            "ALTER TABLE provider_usage ADD COLUMN IF NOT EXISTS execution_epoch INTEGER",
            "ALTER TABLE provider_usage ADD COLUMN IF NOT EXISTS attempt INTEGER",
            "ALTER TABLE provider_usage ADD COLUMN IF NOT EXISTS attempt_identity TEXT",
            "CREATE INDEX IF NOT EXISTS idx_provider_usage_attempt_identity ON provider_usage(attempt_identity)",
        ),
    ),
    PostgresMigration(
        version=31,
        name="add_curation_quality_evidence_identity",
        statements=(
            "ALTER TABLE curation_quality_evidence ADD COLUMN IF NOT EXISTS evidence_id TEXT",
            "CREATE INDEX IF NOT EXISTS idx_curation_quality_evidence_identity ON curation_quality_evidence(evidence_id)",
        ),
    ),
    PostgresMigration(
        version=32,
        name="add_curation_run_execution_identity",
        statements=(
            "ALTER TABLE curation_runs ADD COLUMN IF NOT EXISTS execution_epoch INTEGER",
            "CREATE INDEX IF NOT EXISTS idx_curation_runs_execution_identity ON curation_runs(task_id, execution_epoch, created_at DESC, run_id DESC)",
        ),
    ),
    PostgresMigration(
        version=33,
        name="add_ingress_evidence",
        statements=(
            """
            CREATE TABLE IF NOT EXISTS ingress_batch_evidence (
                batch_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                execution_epoch INTEGER NOT NULL,
                batch_sequence INTEGER NOT NULL,
                claimed_entry_ids_json JSONB NOT NULL DEFAULT '[]'::jsonb,
                source_fingerprint TEXT NOT NULL,
                source_entries_json JSONB NOT NULL DEFAULT '[]'::jsonb,
                provider_route TEXT NOT NULL,
                execution_mode TEXT NOT NULL,
                policy_version TEXT NOT NULL,
                schema_version TEXT NOT NULL,
                claimed_at TEXT NOT NULL,
                finalized_at TEXT,
                grouping_strategy TEXT,
                grouping_fallback_reason TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS ingress_action_receipts (
                action_id TEXT PRIMARY KEY,
                batch_id TEXT NOT NULL REFERENCES ingress_batch_evidence(batch_id) ON DELETE CASCADE,
                operation TEXT NOT NULL,
                entry_ids_json JSONB NOT NULL DEFAULT '[]'::jsonb,
                target_ids_json JSONB NOT NULL DEFAULT '[]'::jsonb,
                canonical_payload_digest TEXT NOT NULL,
                status TEXT NOT NULL,
                mutation_evidence_id TEXT,
                created_at TEXT NOT NULL,
                terminalized_at TEXT,
                error_code TEXT,
                before_revision_tokens_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                after_revision_tokens_json JSONB NOT NULL DEFAULT '{}'::jsonb
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS ingress_source_coverage (
                entry_id TEXT PRIMARY KEY,
                action_id TEXT REFERENCES ingress_action_receipts(action_id) ON DELETE SET NULL,
                outcome TEXT NOT NULL,
                reason TEXT
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_ingress_batch_evidence_execution ON ingress_batch_evidence(task_id, execution_epoch, batch_sequence)",
            "CREATE INDEX IF NOT EXISTS idx_ingress_action_receipts_batch ON ingress_action_receipts(batch_id, action_id)",
        ),
    ),
    PostgresMigration(
        version=34,
        name="add_ingress_quality_evidence",
        statements=(
            """
            CREATE TABLE IF NOT EXISTS ingress_quality_evidence (
                action_id TEXT PRIMARY KEY,
                disposition TEXT NOT NULL,
                reason TEXT,
                evaluated_at TEXT NOT NULL,
                evaluator TEXT,
                query_provenance_json JSONB NOT NULL DEFAULT '{}'::jsonb
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_ingress_quality_evidence_disposition ON ingress_quality_evidence(disposition, evaluated_at DESC, action_id)",
            "CREATE INDEX IF NOT EXISTS idx_ingress_quality_evidence_evaluated_at ON ingress_quality_evidence(evaluated_at DESC, action_id)",
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
