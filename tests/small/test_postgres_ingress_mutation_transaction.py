"""Fake-connection coverage for the Postgres ingress transaction primitive.

The fake translates the exercised placeholders and JSONB casts to SQLite. It
proves transaction ownership, SQL ordering, replay, and rollback behavior;
driver-level Postgres locking and JSONB adaptation still require an integration
test with a live Postgres server.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from typing import Any, cast
from uuid import uuid4

import pytest

from mcp_memory.core.ports.ingress import IngressActionReceiptIdentityConflictError
from mcp_memory.storage.postgres_ingress_mutation_transaction import (
    IngressMutationInjectedFailure,
    IngressMutationResult,
    PostgresIngressMutationStore,
)
from mcp_memory.storage.session import DbConnectionLike, SessionManager


pytestmark = pytest.mark.small


class _Cursor:
    def __init__(self, connection: _Connection) -> None:
        self._connection = connection
        self._cursor = connection._database.cursor()

    def __enter__(self) -> _Cursor:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        self._cursor.close()
        return False

    @property
    def rowcount(self) -> int:
        return self._cursor.rowcount

    def execute(self, query: str, params: tuple[object, ...] | None = None) -> None:
        self._connection.statements.append(query)
        values = tuple(() if params is None else params)
        if "INSERT INTO memory_search_documents" in query:
            self._cursor.execute(
                """
                INSERT INTO memory_search_documents (
                    memory_id, title, summary, content, tags_text, search_document
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (memory_id) DO UPDATE SET
                    title = excluded.title, summary = excluded.summary,
                    content = excluded.content, tags_text = excluded.tags_text,
                    search_document = excluded.search_document
                """,
                (*values[:5], " ".join(str(value or "") for value in values[5:])),
            )
            return
        self._cursor.execute(_sqlite_query(query), values)

    def fetchone(self) -> tuple[object, ...] | None:
        return self._cursor.fetchone()

    def fetchall(self) -> list[tuple[object, ...]]:
        return self._cursor.fetchall()


class _Connection:
    def __init__(self) -> None:
        self._database = sqlite3.connect(":memory:")
        self._database.execute("PRAGMA foreign_keys = ON")
        self.autocommit = False
        self.commits = 0
        self.rollbacks = 0
        self.statements: list[str] = []
        self._database.executescript(
            """
            CREATE TABLE ingress_batch_evidence (batch_id TEXT PRIMARY KEY);
            CREATE TABLE memories (
                id TEXT PRIMARY KEY, title TEXT NOT NULL, content TEXT NOT NULL,
                summary TEXT, type TEXT NOT NULL, status TEXT NOT NULL,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                read_count INTEGER NOT NULL DEFAULT 0,
                access_score REAL NOT NULL DEFAULT 0,
                last_accessed_at TEXT, last_surfaced_at TEXT,
                metadata TEXT NOT NULL DEFAULT '{}', memory_ref INTEGER, archived_at TEXT
            );
            CREATE TABLE memory_workspaces (
                memory_id TEXT NOT NULL, workspace_id TEXT NOT NULL,
                PRIMARY KEY (memory_id, workspace_id),
                FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
            );
            CREATE TABLE tags (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL UNIQUE);
            CREATE TABLE memory_tags (
                memory_id TEXT NOT NULL, tag_id INTEGER NOT NULL,
                PRIMARY KEY (memory_id, tag_id),
                FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE,
                FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE
            );
            CREATE TABLE memory_search_documents (
                memory_id TEXT PRIMARY KEY, title TEXT NOT NULL, summary TEXT NOT NULL,
                content TEXT NOT NULL, tags_text TEXT NOT NULL, search_document TEXT NOT NULL,
                FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
            );
            CREATE TABLE ingress_action_receipts (
                action_id TEXT PRIMARY KEY, batch_id TEXT NOT NULL,
                operation TEXT NOT NULL, entry_ids_json TEXT NOT NULL,
                target_ids_json TEXT NOT NULL, canonical_payload_digest TEXT NOT NULL,
                status TEXT NOT NULL, mutation_evidence_id TEXT, created_at TEXT NOT NULL,
                terminalized_at TEXT, error_code TEXT,
                before_revision_tokens_json TEXT NOT NULL,
                after_revision_tokens_json TEXT NOT NULL,
                FOREIGN KEY (batch_id) REFERENCES ingress_batch_evidence(batch_id)
            );
            CREATE TABLE ingress_source_coverage (
                entry_id TEXT PRIMARY KEY, action_id TEXT, outcome TEXT NOT NULL, reason TEXT,
                FOREIGN KEY (action_id) REFERENCES ingress_action_receipts(action_id)
            );
            CREATE TABLE embedding_repair_queue (
                id TEXT PRIMARY KEY, memory_id TEXT NOT NULL, workspace_id TEXT,
                model_name TEXT NOT NULL, memory_updated_at TEXT NOT NULL,
                status TEXT NOT NULL, attempt_count INTEGER NOT NULL,
                available_at REAL NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL,
                claimed_at REAL, completed_at REAL, lease_owner TEXT,
                lease_expires_at REAL, last_error TEXT,
                UNIQUE (memory_id, model_name, memory_updated_at)
            );
            CREATE TABLE memory_mutation_events (
                id TEXT PRIMARY KEY, operation TEXT NOT NULL, actor_kind TEXT NOT NULL,
                family TEXT, action_id TEXT, schema_version INTEGER NOT NULL,
                status TEXT NOT NULL, created_at TEXT NOT NULL
            );
            CREATE TABLE memory_record_revisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT NOT NULL,
                memory_id TEXT NOT NULL, role TEXT NOT NULL, before_exists INTEGER NOT NULL,
                before_snapshot TEXT, after_exists INTEGER NOT NULL, after_snapshot TEXT,
                before_token TEXT, after_token TEXT,
                FOREIGN KEY (event_id) REFERENCES memory_mutation_events(id)
            );
            """
        )

    def cursor(self) -> _Cursor:
        return _Cursor(self)

    def commit(self) -> None:
        self.commits += 1
        self._database.commit()

    def rollback(self) -> None:
        self.rollbacks += 1
        self._database.rollback()


class _Lease:
    def __init__(self, connection: _Connection) -> None:
        self.connection = connection

    def __enter__(self) -> _Connection:
        return self.connection

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return False


class _Sessions:
    def __init__(self) -> None:
        self.connection = _Connection()
        self.leases = 0

    def open_connection(self) -> _Lease:
        self.leases += 1
        return _Lease(self.connection)


def _sqlite_query(query: str) -> str:
    return query.replace(" FOR UPDATE", "").replace("%s::jsonb", "?").replace("%s", "?")


def _store(
    sessions: _Sessions,
    *,
    embedding_model: str = "default",
    fault_stage: str | None = None,
    fault_injector: Callable[[str], None] | None = None,
) -> PostgresIngressMutationStore:
    return PostgresIngressMutationStore(
        cast(SessionManager[DbConnectionLike], sessions),
        embedding_model=embedding_model,
        fault_stage=fault_stage,
        fault_injector=fault_injector,
    )


def _seed_batch(sessions: _Sessions) -> None:
    sessions.connection._database.execute("INSERT INTO ingress_batch_evidence (batch_id) VALUES (?)", ("batch-1",))
    sessions.connection.commit()


def _count(sessions: _Sessions, table: str) -> int:
    row = sessions.connection._database.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    assert row is not None
    return int(row[0])


def _seed_memory(sessions: _Sessions, memory_id: str) -> None:
    sessions.connection._database.execute(
        """
        INSERT INTO memories (
            id, title, content, summary, type, status, created_at, updated_at,
            metadata
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (memory_id, "Existing", "Before", "Existing", "observation", "active", "created", "updated", "{}"),
    )
    sessions.connection._database.execute(
        "INSERT INTO memory_workspaces (memory_id, workspace_id) VALUES (?, ?)", (memory_id, "workspace")
    )
    sessions.connection.commit()


def test_create_commits_domain_evidence_and_repair_on_one_postgres_lease() -> None:
    """A create commits the memory and all ingress artifacts together."""
    sessions = _Sessions()
    _seed_batch(sessions)
    store = _store(sessions, embedding_model="test-model")
    memory_id = str(uuid4())

    receipt = store.execute(
        batch_id="batch-1",
        operation="create",
        entry_ids=["entry-2", "entry-1"],
        target_ids=[],
        payload={"content": "Important context"},
        apply=lambda transaction: IngressMutationResult(
            "create",
            [transaction.create_memory(
                memory_id=memory_id,
                title="A durable note",
                content="Important context",
                workspace_ids=["workspace"],
            ).id],
        ),
    )

    assert receipt.status.value == "applied_unverified"
    assert receipt.mutation_evidence_id is not None
    assert _count(sessions, "memories") == 1
    assert _count(sessions, "ingress_action_receipts") == 1
    assert _count(sessions, "ingress_source_coverage") == 2
    assert _count(sessions, "memory_mutation_events") == 1
    assert _count(sessions, "memory_record_revisions") == 1
    assert _count(sessions, "embedding_repair_queue") == 1
    assert sessions.leases == 1
    assert sessions.connection.autocommit is False
    assert any("%s::jsonb" in statement for statement in sessions.connection.statements)


def test_exact_replay_returns_receipt_without_running_callback() -> None:
    """A replay returns the original receipt and skips domain work."""
    sessions = _Sessions()
    _seed_batch(sessions)
    store = _store(sessions)
    calls = 0

    def apply(transaction: Any) -> IngressMutationResult:
        nonlocal calls
        calls += 1
        record = transaction.create_memory(
            memory_id=str(uuid4()), title="Replayable", content="Only once", workspace_ids=["workspace"]
        )
        return IngressMutationResult("create", [record.id])

    arguments = {
        "batch_id": "batch-1",
        "operation": "create",
        "entry_ids": ["entry-1"],
        "target_ids": [],
        "payload": {"content": "Only once"},
        "apply": apply,
    }
    first = store.execute(**arguments)
    replay = store.execute(**arguments)

    assert replay == first
    assert calls == 1
    assert _count(sessions, "memories") == 1
    assert _count(sessions, "ingress_action_receipts") == 1
    assert sessions.leases == 2


def test_payload_collision_is_rejected_before_replay_callback() -> None:
    """A changed payload for one action identity fails before callback entry."""
    sessions = _Sessions()
    _seed_batch(sessions)
    store = _store(sessions)
    applied = store.execute(
        batch_id="batch-1",
        operation="create",
        entry_ids=["entry-1"],
        target_ids=[],
        payload={"content": "original"},
        apply=lambda transaction: IngressMutationResult(
            "create",
            [transaction.create_memory(title="Original", content="original", workspace_ids=["workspace"]).id],
        ),
    )
    called = False

    def unexpected_apply(transaction: Any) -> IngressMutationResult:
        nonlocal called
        called = True
        raise AssertionError("collision callback must not run")

    with pytest.raises(IngressActionReceiptIdentityConflictError):
        store.execute(
            batch_id="batch-1",
            operation="create",
            entry_ids=["entry-1"],
            target_ids=[],
            payload={"content": "changed"},
            apply=unexpected_apply,
        )

    assert not called
    assert store.execute(
        batch_id="batch-1",
        operation="create",
        entry_ids=["entry-1"],
        target_ids=[],
        payload={"content": "original"},
        apply=unexpected_apply,
    ) == applied


@pytest.mark.parametrize("fault_stage", ["after_domain_mutation", "before_receipt_coverage_commit"])
def test_fault_stages_roll_back_domain_and_ingress_artifacts(fault_stage: str) -> None:
    """Failures after mutation leave no committed domain or evidence rows."""
    sessions = _Sessions()
    _seed_batch(sessions)
    memory_id = str(uuid4())
    store = _store(sessions, fault_stage=fault_stage)

    with pytest.raises(IngressMutationInjectedFailure):
        store.execute(
            batch_id="batch-1",
            operation="create",
            entry_ids=["entry-1"],
            target_ids=[],
            payload={"content": "rolled back"},
            apply=lambda transaction: IngressMutationResult(
                "create",
                [transaction.create_memory(
                    memory_id=memory_id,
                    title="Rolled back",
                    content="rolled back",
                    workspace_ids=["workspace"],
                ).id],
            ),
        )

    assert _count(sessions, "memories") == 0
    for table in (
        "ingress_action_receipts",
        "ingress_source_coverage",
        "memory_mutation_events",
        "memory_record_revisions",
        "embedding_repair_queue",
    ):
        assert _count(sessions, table) == 0
    assert sessions.connection.rollbacks == 1


def test_append_updates_memory_and_assigns_appended_coverage_atomically() -> None:
    """An append updates one target and writes its terminal coverage."""
    sessions = _Sessions()
    _seed_batch(sessions)
    memory_id = str(uuid4())
    _seed_memory(sessions, memory_id)
    store = _store(sessions)

    receipt = store.execute(
        batch_id="batch-1",
        operation="append",
        entry_ids=["entry-1"],
        target_ids=[memory_id],
        payload={"content": "After"},
        apply=lambda transaction: IngressMutationResult(
            "append", [transaction.append_memory(memory_id, "After").id]
        ),
    )

    row = sessions.connection._database.execute("SELECT content FROM memories WHERE id = ?", (memory_id,)).fetchone()
    assert row is not None
    assert row[0] == "Before\n\nAfter"
    coverage = sessions.connection._database.execute(
        "SELECT outcome, action_id FROM ingress_source_coverage WHERE entry_id = ?", ("entry-1",)
    ).fetchone()
    assert coverage is not None
    assert tuple(coverage) == ("appended", receipt.action_id)
    assert _count(sessions, "memory_record_revisions") == 1
