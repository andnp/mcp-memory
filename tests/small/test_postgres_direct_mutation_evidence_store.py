from __future__ import annotations

from typing import Any, cast

from mcp_memory.core.direct_mutation_evidence import DirectMutationEntityDelta, DirectMutationEvidence
from mcp_memory.storage.postgres_direct_mutation_evidence_store import PostgresDirectMutationEvidenceStore
from mcp_memory.storage.session import DbConnectionLike, SessionManager


class _Cursor:
    def __init__(self, connection: _Connection) -> None:
        self.connection = connection

    def __enter__(self) -> _Cursor:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return False

    def execute(self, query: str, params: tuple[object, ...] | None = None) -> None:
        self.connection.calls.append((query, params))

    def fetchone(self) -> tuple[object, ...] | None:
        return None

    def fetchall(self) -> list[tuple[object, ...]]:
        return []


class _Connection:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...] | None]] = []
        self.commits = 0

    def cursor(self) -> _Cursor:
        return _Cursor(self)

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        pass


class _Lease:
    def __init__(self, connection: _Connection) -> None:
        self.connection = connection

    def __enter__(self) -> _Connection:
        return self.connection

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return False

    def close(self) -> None:
        pass


class _Sessions:
    def __init__(self) -> None:
        self.connection = _Connection()

    def open_connection(self) -> _Lease:
        return _Lease(self.connection)

    def __enter__(self) -> _Sessions:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return False

    def close(self) -> None:
        pass


def test_postgres_evidence_store_normalizes_integer_existence_flags() -> None:
    """PostgreSQL integer flags receive integer values from boolean deltas."""
    evidence = DirectMutationEvidence.start(
        task_id="task-1", execution_epoch=1, session_id="session-1", call_id="call-1",
        sequence=1, tool_name="internal_update_memory_record", arguments={"memory_id": "m-1"},
    ).finish(
        payload={"status": "ok"}, ledger_entry={},
        deltas=(DirectMutationEntityDelta("record", "m-1", before_exists=False, after_exists=True),),
    )
    sessions = _Sessions()

    PostgresDirectMutationEvidenceStore(cast(SessionManager[DbConnectionLike], sessions)).save(evidence)

    delta_params = sessions.connection.calls[2][1]
    assert delta_params is not None
    assert delta_params[6:8] == (0, 1)
    assert sessions.connection.commits == 1
