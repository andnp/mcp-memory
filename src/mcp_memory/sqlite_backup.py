from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)
_SHARED_STORAGE_WARNING_MIN_INTERVAL_SECONDS = 6 * 3600.0


@dataclass(frozen=True)
class SQLiteBackupResult:
    backup_path: Path
    pruned_paths: tuple[Path, ...] = ()


def create_sqlite_backup(db_path: Path, backup_dir: Path, *, now: float | None = None) -> Path:
    if not db_path.exists():
        raise FileNotFoundError(db_path)

    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.fromtimestamp(time.time() if now is None else now, tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    destination = backup_dir / f"memory-{timestamp}.sqlite3"
    temp_destination = destination.with_suffix(".sqlite3.tmp")
    if temp_destination.exists():
        temp_destination.unlink()

    source = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    target = sqlite3.connect(temp_destination)
    try:
        source.backup(target)
        target.commit()
    finally:
        target.close()
        source.close()

    temp_destination.replace(destination)
    return destination


def prune_old_backups(backup_dir: Path, *, max_snapshots: int) -> tuple[Path, ...]:
    if max_snapshots < 1:
        raise ValueError("max_snapshots must be >= 1")
    if not backup_dir.exists():
        return ()

    snapshots = sorted(
        (path for path in backup_dir.iterdir() if path.is_file() and path.name.startswith("memory-") and path.suffix == ".sqlite3"),
        key=lambda path: (path.stat().st_mtime, path.name),
        reverse=True,
    )
    pruned_paths: list[Path] = []
    for path in snapshots[max_snapshots:]:
        path.unlink(missing_ok=True)
        pruned_paths.append(path)
    return tuple(pruned_paths)


def create_and_prune_sqlite_backup(
    db_path: Path,
    backup_dir: Path,
    *,
    max_snapshots: int,
    now: float | None = None,
) -> SQLiteBackupResult:
    backup_path = create_sqlite_backup(db_path, backup_dir, now=now)
    pruned_paths = prune_old_backups(backup_dir, max_snapshots=max_snapshots)
    return SQLiteBackupResult(backup_path=backup_path, pruned_paths=pruned_paths)


def detect_shared_storage_risks(app_data_dir: Path) -> tuple[str, ...]:
    warnings: list[str] = []
    if (app_data_dir / ".stfolder").exists():
        warnings.append("syncthing_marker_detected")

    indices_dir = app_data_dir / "memories" / "indices"
    if indices_dir.exists():
        conflict_files = sorted(path.name for path in indices_dir.iterdir() if ".sync-conflict-" in path.name)
        if conflict_files:
            warnings.append(f"sqlite_sync_conflicts_detected:{len(conflict_files)}")

    return tuple(warnings)


def log_shared_storage_risks(app_data_dir: Path) -> tuple[str, ...]:
    warnings = detect_shared_storage_risks(app_data_dir)
    if not warnings:
        return warnings
    now = time.time()
    state_path = _shared_storage_warning_state_path()
    fingerprint = "|".join(warnings)
    previous = _read_shared_storage_warning_state(state_path)
    last_logged_at = previous.get("last_logged_at")
    last_fingerprint = previous.get("fingerprint")
    should_log = (
        not isinstance(last_logged_at, int | float)
        or last_fingerprint != fingerprint
        or now - float(last_logged_at) >= _SHARED_STORAGE_WARNING_MIN_INTERVAL_SECONDS
    )
    if should_log:
        logger.warning(
            "Shared-storage SQLite safety warning(s) at %s: %s. Prefer one writer, local DB ownership, and periodic logical backups instead of syncing a live SQLite store.",
            app_data_dir,
            ", ".join(warnings),
        )
        _write_shared_storage_warning_state(
            state_path,
            {
                "fingerprint": fingerprint,
                "last_logged_at": now,
            },
        )
    return warnings


def _shared_storage_warning_state_path() -> Path:
    from mcp_memory.config import resolve_state_dir

    state_dir = resolve_state_dir()
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir / "shared-storage-warnings.json"


def _read_shared_storage_warning_state(state_path: Path) -> dict[str, object]:
    if not state_path.exists():
        return {}
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_shared_storage_warning_state(state_path: Path, payload: dict[str, object]) -> None:
    try:
        state_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    except OSError:
        logger.debug("Failed to persist shared-storage warning state", exc_info=True)