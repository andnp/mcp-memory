from dataclasses import dataclass, field
from typing import Any


@dataclass
class SearchResult:
    doc_id: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)