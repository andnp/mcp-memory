from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast


def _row_to_mapping(row: object, columns: list[str] | None = None) -> dict[str, object]:
    if isinstance(row, dict):
        return row
    if isinstance(row, Mapping):
        return {str(key): value for key, value in row.items()}
    row_keys = getattr(row, "keys", None)
    if callable(row_keys):
        row_like = cast(Any, row)
        return {str(key): row_like[key] for key in cast(Any, row_keys)()}
    if isinstance(row, tuple) and columns is not None:
        row_values = cast(tuple[object, ...], row)
        return dict(zip(columns, row_values, strict=False))
    raise TypeError(f"Unsupported row type: {type(row)!r}")


class ManagementQueryRunner:
    def __init__(self, db_manager) -> None:
        self._db_manager = db_manager

    @property
    def available(self) -> bool:
        return self._db_manager is not None

    def uses_sqlite_connection_api(self) -> bool:
        return self.available and hasattr(self._db_manager, "get_connection")

    def adapt_query(self, query: str) -> str:
        if self.uses_sqlite_connection_api():
            return query
        return (
            query.replace("CHAR(10)", "CHR(10)")
            .replace("GROUP_CONCAT(DISTINCT workspace_id)", "STRING_AGG(DISTINCT workspace_id, ',')")
            .replace("GROUP_CONCAT(DISTINCT tags.name)", "STRING_AGG(DISTINCT tags.name, ',')")
            .replace("?", "%s")
        )

    def fetchall(self, query: str, params: Sequence[object] | None = None) -> list[dict[str, object]]:
        if not self.available:
            return []
        effective_params = list(params or [])
        adapted_query = self.adapt_query(query)
        if self.uses_sqlite_connection_api():
            conn = self._db_manager.get_connection()
            rows = conn.execute(adapted_query, effective_params).fetchall()
            return [_row_to_mapping(row) for row in rows]
        with self._db_manager.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(adapted_query, tuple(effective_params))
                rows = cursor.fetchall()
                description = getattr(cursor, "description", None)
                columns = [column.name for column in description] if description is not None else []
                return [_row_to_mapping(row, columns) for row in rows]

    def fetchone(self, query: str, params: Sequence[object] | None = None) -> dict[str, object] | None:
        rows = self.fetchall(query, params)
        return rows[0] if rows else None