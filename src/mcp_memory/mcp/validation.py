from __future__ import annotations


def validate_ingest_mutation_payload(
    *,
    entry_ids: list[int],
    content: str,
    title: str | None = None,
) -> None:
    if not entry_ids:
        raise ValueError("ingest mutations require at least one claimed entry id")
    if not content.strip():
        raise ValueError("ingest mutations require non-empty content")
    if title is not None and not title.strip():
        raise ValueError("ingest mutations require non-empty title")
