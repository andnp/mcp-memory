from datetime import datetime, timezone
from pathlib import Path

import pytest

from mcp_memory.indices.keyword import KeywordIndex
from mcp_memory.models import Chunk


pytestmark = pytest.mark.medium


def _chunk_from_seed(chunk_data: dict) -> Chunk:
    metadata = chunk_data["metadata"]
    file_path = f"/workspace/{chunk_data['doc_id'].replace(':', '_')}.md"
    return Chunk(
        chunk_id=chunk_data["chunk_id"],
        doc_id=chunk_data["doc_id"],
        content=chunk_data["content"],
        metadata={
            "title": chunk_data["doc_id"],
            "tags": metadata.get("memory_tags", []),
            "summary": chunk_data["content"],
            "source_file": file_path,
        },
        chunk_index=0,
        header_path="",
        start_pos=0,
        end_pos=len(chunk_data["content"]),
        file_path=file_path,
        modified_time=datetime.now(timezone.utc),
    )


def test_keyword_index_search_returns_seeded_result(
    seeded_db,
    seed_data: dict,
) -> None:
    index = KeywordIndex(seeded_db)
    index.add_chunks([_chunk_from_seed(chunk) for chunk in seed_data["chunks"]])

    results = index.search("hybrid retrieval", top_k=3)

    assert results
    assert results[0]["doc_id"] == "memory:search-fact"
    assert results[0]["chunk_id"] == "memory:search-fact_chunk_0"


def test_keyword_index_honors_excluded_files(
    seeded_db,
    seed_data: dict,
) -> None:
    index = KeywordIndex(seeded_db)
    index.add_chunks([_chunk_from_seed(chunk) for chunk in seed_data["chunks"]])

    results = index.search(
        "bootstrap tests",
        top_k=3,
        excluded_files={"memory_cleanup-observation.md"},
        docs_root=Path("/workspace"),
    )

    assert all(result["doc_id"] != "memory:cleanup-observation" for result in results)