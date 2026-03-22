from pathlib import Path
import sqlite3

import pytest

from mcp_memory.sqlite_backup import create_and_prune_sqlite_backup, detect_shared_storage_risks, prune_old_backups


pytestmark = pytest.mark.small


def test_create_and_prune_sqlite_backup_creates_snapshot(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.db"
    backup_dir = tmp_path / "backups"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("CREATE TABLE memories (id INTEGER PRIMARY KEY, title TEXT)")
        conn.execute("INSERT INTO memories (title) VALUES ('hello')")
        conn.commit()
    finally:
        conn.close()

    result = create_and_prune_sqlite_backup(db_path, backup_dir, max_snapshots=2, now=1_700_000_000)

    assert result.backup_path.exists()
    backup_conn = sqlite3.connect(result.backup_path)
    try:
        title = backup_conn.execute("SELECT title FROM memories").fetchone()[0]
    finally:
        backup_conn.close()
    assert title == "hello"
    assert result.pruned_paths == ()


def test_prune_old_backups_keeps_latest_files(tmp_path: Path) -> None:
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()
    first = backup_dir / "memory-20260320T000000Z.sqlite3"
    second = backup_dir / "memory-20260320T010000Z.sqlite3"
    third = backup_dir / "memory-20260320T020000Z.sqlite3"
    for index, path in enumerate((first, second, third), start=1):
        path.write_text(str(index), encoding="utf-8")

    pruned = prune_old_backups(backup_dir, max_snapshots=2)

    assert pruned == (first,)
    assert not first.exists()
    assert second.exists()
    assert third.exists()


def test_detect_shared_storage_risks_flags_syncthing_markers(tmp_path: Path) -> None:
    app_data_dir = tmp_path / "mcp-memory"
    indices_dir = app_data_dir / "memories" / "indices"
    indices_dir.mkdir(parents=True)
    (app_data_dir / ".stfolder").write_text("marker", encoding="utf-8")
    (indices_dir / "memory.sync-conflict-20260320.db").write_text("conflict", encoding="utf-8")

    warnings = detect_shared_storage_risks(app_data_dir)

    assert warnings == ("syncthing_marker_detected", "sqlite_sync_conflicts_detected:1")