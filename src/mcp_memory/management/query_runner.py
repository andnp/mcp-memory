from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, cast


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


class ManagementQueryAdapter(Protocol):
    def adapt_query(self, query: str) -> str: ...

    def fetchall(
        self,
        db_manager: Any,
        query: str,
        params: Sequence[object] | None = None,
    ) -> list[dict[str, object]]: ...

    def uses_sqlite_connection_api(self) -> bool: ...


class SQLiteManagementQueryAdapter:
    def adapt_query(self, query: str) -> str:
        return query

    def uses_sqlite_connection_api(self) -> bool:
        return True

    def fetchall(
        self,
        db_manager: Any,
        query: str,
        params: Sequence[object] | None = None,
    ) -> list[dict[str, object]]:
        conn = db_manager.get_connection()
        rows = conn.execute(self.adapt_query(query), list(params or [])).fetchall()
        return [_row_to_mapping(row) for row in rows]


class PostgresManagementQueryAdapter:
    def adapt_query(self, query: str) -> str:
        return (
            query.replace("CHAR(10)", "CHR(10)")
            .replace("GROUP_CONCAT(DISTINCT workspace_id)", "STRING_AGG(DISTINCT workspace_id, ',')")
            .replace("GROUP_CONCAT(DISTINCT tags.name)", "STRING_AGG(DISTINCT tags.name, ',')")
            .replace("?", "%s")
        )

    def uses_sqlite_connection_api(self) -> bool:
        return False

    def fetchall(
        self,
        db_manager: Any,
        query: str,
        params: Sequence[object] | None = None,
    ) -> list[dict[str, object]]:
        with db_manager.open_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(self.adapt_query(query), tuple(params or []))
                rows = cursor.fetchall()
                description = getattr(cursor, "description", None)
                columns = [column.name for column in description] if description is not None else []
                return [_row_to_mapping(row, columns) for row in rows]


class ManagementQueryRunner:
    def __init__(
        self,
        db_manager,
        adapter: ManagementQueryAdapter | None = None,
    ) -> None:
        self._db_manager = db_manager
        self._adapter = adapter if adapter is not None else self._select_adapter(db_manager)

    @staticmethod
    def _select_adapter(db_manager) -> ManagementQueryAdapter:
        if db_manager is not None and hasattr(db_manager, "get_connection"):
            return SQLiteManagementQueryAdapter()
        return PostgresManagementQueryAdapter()

    @property
    def available(self) -> bool:
        return self._db_manager is not None

    def uses_sqlite_connection_api(self) -> bool:
        return self.available and self._adapter.uses_sqlite_connection_api()

    def adapt_query(self, query: str) -> str:
        return self._adapter.adapt_query(query)

    def fetchall(self, query: str, params: Sequence[object] | None = None) -> list[dict[str, object]]:
        if not self.available:
            return []
        return self._adapter.fetchall(self._db_manager, query, params)

    def fetchone(self, query: str, params: Sequence[object] | None = None) -> dict[str, object] | None:
        rows = self.fetchall(query, params)
        return rows[0] if rows else None
