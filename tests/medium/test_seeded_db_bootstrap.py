import sqlite3

import pytest


pytestmark = pytest.mark.medium


def test_seeded_db_populates_documents_and_chunks(seeded_db) -> None:
    conn = seeded_db.get_connection()

    document_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    assert document_count == 3
    assert chunk_count == 3


def test_seeded_db_populates_journal_entries(seeded_db) -> None:
    conn = seeded_db.get_connection()

    pending_count = conn.execute(
        "SELECT COUNT(*) FROM system1_journal WHERE status = 'pending'"
    ).fetchone()[0]
    processed_count = conn.execute(
        "SELECT COUNT(*) FROM system1_journal WHERE status = 'processed'"
    ).fetchone()[0]

    assert pending_count == 1
    assert processed_count == 1


def test_seeded_db_populates_system_state(seeded_db) -> None:
    conn = seeded_db.get_connection()

    state = conn.execute(
        "SELECT value FROM system_state WHERE key = 'last_boot_state'"
    ).fetchone()[0]

    assert state == "fresh"