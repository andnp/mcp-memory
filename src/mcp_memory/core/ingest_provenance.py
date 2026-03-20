from __future__ import annotations

from typing import Any, Mapping


INGEST_CREATED_VIA_METADATA_KEY = "created_via_ingest"
INGEST_APPENDED_VIA_METADATA_KEY = "appended_via_ingest"
INGEST_TASK_ID_METADATA_KEY = "ingest_task_id"
INGEST_SOURCE_ENTRY_IDS_METADATA_KEY = "source_entry_ids"
INGEST_APPENDED_ENTRY_IDS_METADATA_KEY = "appended_entry_ids"


def build_ingest_created_metadata(
    *,
    task_id: str,
    entry_ids: list[int],
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    merged = dict(metadata or {})
    merged[INGEST_CREATED_VIA_METADATA_KEY] = True
    merged[INGEST_TASK_ID_METADATA_KEY] = task_id
    merged[INGEST_SOURCE_ENTRY_IDS_METADATA_KEY] = _merge_int_list(
        merged.get(INGEST_SOURCE_ENTRY_IDS_METADATA_KEY),
        entry_ids,
    )
    return merged


def build_ingest_appended_metadata(
    *,
    task_id: str,
    entry_ids: list[int],
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    merged = dict(metadata or {})
    merged[INGEST_APPENDED_VIA_METADATA_KEY] = True
    merged[INGEST_TASK_ID_METADATA_KEY] = task_id
    merged[INGEST_APPENDED_ENTRY_IDS_METADATA_KEY] = _merge_int_list(
        merged.get(INGEST_APPENDED_ENTRY_IDS_METADATA_KEY),
        entry_ids,
    )
    return merged


def _merge_int_list(existing: object, incoming: list[int]) -> list[int]:
    values = {
        int(item)
        for item in incoming
        if isinstance(item, int) and not isinstance(item, bool)
    }
    if isinstance(existing, list):
        values.update(
            int(item)
            for item in existing
            if isinstance(item, int) and not isinstance(item, bool)
        )
    return sorted(values)
