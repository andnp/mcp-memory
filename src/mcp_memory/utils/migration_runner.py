from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


class MigrationCursorLike(Protocol):
    def execute(self, query: str, params: tuple[object, ...] | None = None) -> object: ...


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    statements: tuple[str, ...]


def apply_migrations(
    cursor: MigrationCursorLike,
    migrations: Sequence[Migration],
    *,
    current_version: int | None,
    record_progress_statement: str,
    record_progress_params: tuple[object, ...] = (),
) -> int:
    """Apply migrations sequentially past `current_version`, recording progress after each.

    Generic and dependency-free: works with any cursor exposing `execute(sql, params)`,
    regardless of backend or placeholder style. `record_progress_statement` is executed
    after each migration with `(*record_progress_params, str(migration.version))`.
    """
    effective_version = 0 if current_version is None else current_version
    for migration in migrations:
        if migration.version <= effective_version:
            continue
        for statement in migration.statements:
            cursor.execute(statement)
        cursor.execute(record_progress_statement, (*record_progress_params, str(migration.version)))
        effective_version = migration.version
    return effective_version


__all__ = ["Migration", "MigrationCursorLike", "apply_migrations"]
