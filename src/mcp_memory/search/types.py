from typing import TypedDict


class SearchResultDict(TypedDict, total=False):
    chunk_id: str
    doc_id: str
    score: float
    header_path: str
    file_path: str
    content: str
    metadata: dict