from __future__ import annotations

import pytest
from mcp_memory.utils.migration_runner import Migration, apply_migrations

pytestmark = pytest.mark.small


class FakeCursor:
    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple[object, ...] | None]] = []

    def execute(self, query: str, params: tuple[object, ...] | None = None) -> None:
        self.executed.append((query, params))


RECORD_PROGRESS_STATEMENT = "UPSERT schema_metadata"


def test_applies_all_migrations_from_none_version() -> None:
    cursor = FakeCursor()
    migrations = (
        Migration(version=1, name="first", statements=("CREATE TABLE a", "CREATE INDEX ia")),
        Migration(version=2, name="second", statements=("CREATE TABLE b",)),
    )

    result = apply_migrations(
        cursor,
        migrations,
        current_version=None,
        record_progress_statement=RECORD_PROGRESS_STATEMENT,
        record_progress_params=("schema_version",),
    )

    assert result == 2
    assert cursor.executed == [
        ("CREATE TABLE a", None),
        ("CREATE INDEX ia", None),
        (RECORD_PROGRESS_STATEMENT, ("schema_version", "1")),
        ("CREATE TABLE b", None),
        (RECORD_PROGRESS_STATEMENT, ("schema_version", "2")),
    ]


def test_skips_migrations_at_or_below_current_version() -> None:
    cursor = FakeCursor()
    migrations = (
        Migration(version=1, name="first", statements=("CREATE TABLE a",)),
        Migration(version=2, name="second", statements=("CREATE TABLE b",)),
        Migration(version=3, name="third", statements=("CREATE TABLE c",)),
    )

    result = apply_migrations(
        cursor,
        migrations,
        current_version=2,
        record_progress_statement=RECORD_PROGRESS_STATEMENT,
        record_progress_params=("schema_version",),
    )

    assert result == 3
    assert cursor.executed == [
        ("CREATE TABLE c", None),
        (RECORD_PROGRESS_STATEMENT, ("schema_version", "3")),
    ]


def test_no_migrations_pending_leaves_cursor_untouched() -> None:
    cursor = FakeCursor()
    migrations = (Migration(version=1, name="first", statements=("CREATE TABLE a",)),)

    result = apply_migrations(
        cursor,
        migrations,
        current_version=1,
        record_progress_statement=RECORD_PROGRESS_STATEMENT,
    )

    assert result == 1
    assert cursor.executed == []


def test_record_progress_params_default_to_empty() -> None:
    cursor = FakeCursor()
    migrations = (Migration(version=1, name="first", statements=()),)

    apply_migrations(
        cursor,
        migrations,
        current_version=None,
        record_progress_statement=RECORD_PROGRESS_STATEMENT,
    )

    assert cursor.executed == [(RECORD_PROGRESS_STATEMENT, ("1",))]
