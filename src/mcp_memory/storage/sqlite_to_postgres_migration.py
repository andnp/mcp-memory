from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sqlite3

from mcp_memory.config import PostgresStorageConfig
from mcp_memory.storage.postgres import ensure_postgres_schema
from mcp_memory.storage.postgres_connection import PostgresConnectionManager


@dataclass(frozen=True)
class SQLiteSourceMemory:
    id: str
    title: str
    content: str
    summary: str | None
    memory_type: str
    status: str
    created_at: str
    updated_at: str
    read_count: int
    access_score: float
    last_accessed_at: str | None
    last_surfaced_at: str | None
    metadata_json: str
    workspace_ids: tuple[str, ...]
    tags: tuple[str, ...]


@dataclass(frozen=True)
class SQLiteSourceLink:
    source_id: str
    target_id: str
    link_type: str
    context: str


@dataclass(frozen=True)
class SQLiteToPostgresMigrationSummary:
    sqlite_path: str
    postgres_dsn: str
    dry_run: bool
    allow_non_empty_target: bool
    target_existing_memories: int
    memories: int
    workspace_mappings: int
    tag_names: int
    tag_mappings: int
    links: int
    skipped_links: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def migrate_sqlite_to_postgres(
    sqlite_path: Path,
    postgres_config: PostgresStorageConfig,
    *,
    dry_run: bool = False,
    allow_non_empty_target: bool = False,
) -> SQLiteToPostgresMigrationSummary:
    source_export = _read_sqlite_export(sqlite_path)
    ensure_postgres_schema(postgres_config)
    summary: SQLiteToPostgresMigrationSummary | None = None

    with PostgresConnectionManager(postgres_config) as manager:
        with manager.open_connection() as connection:
            with connection.cursor() as cursor:
                target_existing_memories = _count_target_memories(cursor)
                if target_existing_memories > 0 and not allow_non_empty_target:
                    connection.rollback()
                    raise ValueError(
                        "Refusing to import into a non-empty Postgres target. "
                        "Pass allow_non_empty_target=True only for controlled reruns."
                    )

                summary = SQLiteToPostgresMigrationSummary(
                    sqlite_path=str(sqlite_path),
                    postgres_dsn=postgres_config.dsn,
                    dry_run=dry_run,
                    allow_non_empty_target=allow_non_empty_target,
                    target_existing_memories=target_existing_memories,
                    memories=len(source_export.memories),
                    workspace_mappings=sum(len(memory.workspace_ids) for memory in source_export.memories),
                    tag_names=len(source_export.tag_names),
                    tag_mappings=sum(len(memory.tags) for memory in source_export.memories),
                    links=len(source_export.links),
                    skipped_links=source_export.skipped_links,
                )
                if dry_run:
                    connection.rollback()
                    return summary

                _import_memories(cursor, source_export.memories)
                _import_links(cursor, source_export.links)
            connection.commit()
    assert summary is not None
    return summary


@dataclass(frozen=True)
class _SQLiteExport:
    memories: tuple[SQLiteSourceMemory, ...]
    links: tuple[SQLiteSourceLink, ...]
    tag_names: tuple[str, ...]
    skipped_links: int


def _read_sqlite_export(sqlite_path: Path) -> _SQLiteExport:
    resolved_path = sqlite_path.expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"SQLite source database was not found: {resolved_path}")

    connection = sqlite3.connect(f"file:{resolved_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        workspace_rows = connection.execute(
            "SELECT memory_id, workspace_id FROM memory_workspaces ORDER BY memory_id ASC, workspace_id ASC"
        ).fetchall()
        workspaces_by_memory: dict[str, list[str]] = defaultdict(list)
        for row in workspace_rows:
            workspaces_by_memory[str(row["memory_id"])].append(str(row["workspace_id"]))

        tag_rows = connection.execute("SELECT id, name FROM tags ORDER BY id ASC").fetchall()
        tag_name_by_id = {int(row["id"]): str(row["name"]) for row in tag_rows}
        memory_tag_rows = connection.execute(
            "SELECT memory_id, tag_id FROM memory_tags ORDER BY memory_id ASC, tag_id ASC"
        ).fetchall()
        tags_by_memory: dict[str, list[str]] = defaultdict(list)
        for row in memory_tag_rows:
            tag_name = tag_name_by_id.get(int(row["tag_id"]))
            if tag_name is None:
                continue
            tags_by_memory[str(row["memory_id"])].append(tag_name)

        memory_rows = connection.execute(
            """
            SELECT
                id,
                title,
                content,
                summary,
                type,
                status,
                created_at,
                updated_at,
                read_count,
                access_score,
                last_accessed_at,
                last_surfaced_at,
                metadata
            FROM memories
            ORDER BY created_at ASC, id ASC
            """
        ).fetchall()

        memories: list[SQLiteSourceMemory] = []
        for row in memory_rows:
            memory_id = str(row["id"])
            memories.append(
                SQLiteSourceMemory(
                    id=memory_id,
                    title=str(row["title"]),
                    content=str(row["content"]),
                    summary=None if row["summary"] is None else str(row["summary"]),
                    memory_type=str(row["type"]),
                    status=str(row["status"]),
                    created_at=str(row["created_at"]),
                    updated_at=str(row["updated_at"]),
                    read_count=int(row["read_count"] or 0),
                    access_score=float(row["access_score"] or 0.0),
                    last_accessed_at=None if row["last_accessed_at"] is None else str(row["last_accessed_at"]),
                    last_surfaced_at=None if row["last_surfaced_at"] is None else str(row["last_surfaced_at"]),
                    metadata_json=_normalize_metadata_json(row["metadata"]),
                    workspace_ids=tuple(sorted(dict.fromkeys(workspaces_by_memory.get(memory_id, [])))),
                    tags=tuple(sorted(dict.fromkeys(tags_by_memory.get(memory_id, [])))),
                )
            )

        valid_memory_ids = {memory.id for memory in memories}
        link_rows = connection.execute(
            "SELECT source_id, target_id, type, context FROM links ORDER BY source_id ASC, target_id ASC, type ASC"
        ).fetchall()
        skipped_links = 0
        valid_links: list[SQLiteSourceLink] = []
        for row in link_rows:
            source_id = str(row["source_id"])
            target_id = str(row["target_id"])
            if source_id not in valid_memory_ids or target_id not in valid_memory_ids:
                skipped_links += 1
                continue
            valid_links.append(
                SQLiteSourceLink(
                    source_id=source_id,
                    target_id=target_id,
                    link_type=str(row["type"]),
                    context=str(row["context"] or ""),
                )
            )
        links = tuple(valid_links)

        return _SQLiteExport(
            memories=tuple(memories),
            links=links,
            tag_names=tuple(sorted({tag for memory in memories for tag in memory.tags})),
            skipped_links=skipped_links,
        )
    finally:
        connection.close()


def _normalize_metadata_json(raw_metadata: object) -> str:
    if raw_metadata is None:
        return "{}"
    if isinstance(raw_metadata, str):
        decoded = json.loads(raw_metadata or "{}")
    else:
        decoded = raw_metadata
    if not isinstance(decoded, dict):
        raise ValueError("SQLite memory metadata must be a JSON object")
    return json.dumps(decoded, sort_keys=True)


def _count_target_memories(cursor) -> int:
    cursor.execute("SELECT COUNT(*) FROM memories")
    row = cursor.fetchone()
    return 0 if row is None or row[0] is None else int(row[0])


def _import_memories(cursor, memories: tuple[SQLiteSourceMemory, ...]) -> None:
    for memory in memories:
        cursor.execute(
            """
            INSERT INTO memories (
                id,
                title,
                content,
                summary,
                type,
                status,
                created_at,
                updated_at,
                read_count,
                access_score,
                last_accessed_at,
                last_surfaced_at,
                metadata
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (id)
            DO UPDATE SET
                title = EXCLUDED.title,
                content = EXCLUDED.content,
                summary = EXCLUDED.summary,
                type = EXCLUDED.type,
                status = EXCLUDED.status,
                created_at = EXCLUDED.created_at,
                updated_at = EXCLUDED.updated_at,
                read_count = EXCLUDED.read_count,
                access_score = EXCLUDED.access_score,
                last_accessed_at = EXCLUDED.last_accessed_at,
                last_surfaced_at = EXCLUDED.last_surfaced_at,
                metadata = EXCLUDED.metadata
            """,
            (
                memory.id,
                memory.title,
                memory.content,
                memory.summary,
                memory.memory_type,
                memory.status,
                memory.created_at,
                memory.updated_at,
                memory.read_count,
                memory.access_score,
                memory.last_accessed_at,
                memory.last_surfaced_at,
                memory.metadata_json,
            ),
        )
        cursor.execute("DELETE FROM memory_workspaces WHERE memory_id = %s", (memory.id,))
        for workspace_id in memory.workspace_ids:
            cursor.execute(
                "INSERT INTO memory_workspaces (memory_id, workspace_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                (memory.id, workspace_id),
            )

        cursor.execute("DELETE FROM memory_tags WHERE memory_id = %s", (memory.id,))
        for tag_name in memory.tags:
            tag_id = _ensure_tag(cursor, tag_name)
            cursor.execute(
                "INSERT INTO memory_tags (memory_id, tag_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                (memory.id, tag_id),
            )


def _ensure_tag(cursor, tag_name: str) -> int:
    cursor.execute(
        "INSERT INTO tags (name) VALUES (%s) ON CONFLICT (name) DO NOTHING",
        (tag_name,),
    )
    cursor.execute("SELECT id FROM tags WHERE name = %s", (tag_name,))
    row = cursor.fetchone()
    if row is None or row[0] is None:
        raise ValueError(f"Failed to resolve imported tag: {tag_name}")
    return int(row[0])


def _import_links(cursor, links: tuple[SQLiteSourceLink, ...]) -> None:
    for link in links:
        cursor.execute(
            """
            INSERT INTO links (source_id, target_id, type, context)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (source_id, target_id, type)
            DO UPDATE SET context = EXCLUDED.context
            """,
            (link.source_id, link.target_id, link.link_type, link.context),
        )
