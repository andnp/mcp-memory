import pytest


pytestmark = pytest.mark.medium


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