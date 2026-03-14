from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from mcp_memory.config import Config
from mcp_memory.core.repository import MemoryLink, RelationalMemoryRecord, RelationalMemoryRepository


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_:-]+")
ACCESS_HALF_LIFE_DAYS = 7
DEGRADATION_PENALTY = 0.3
WORKSPACE_BOOST = 1.2


@dataclass(slots=True)
class RelationalSearchResult:
    memory_id: str
    title: str
    summary: str
    memory_type: str
    status: str
    tags: list[str] = field(default_factory=list)
    workspace_ids: list[str] = field(default_factory=list)
    score: float = 0.0


@dataclass(slots=True)
class RelationalReadResult:
    record: RelationalMemoryRecord
    relationships: dict[str, list[MemoryLink]]
    superseded: list[RelationalMemoryRecord]


class RelationalMemorySearchService:
    def __init__(
        self,
        repository: RelationalMemoryRepository,
        config: Config,
    ) -> None:
        self._repository = repository
        self._config = config

    def search_memories(
        self,
        query: str,
        workspace_id: str | None = None,
        limit: int = 5,
        memory_type: str | None = None,
        status: str | None = None,
        include_superseded: bool = False,
    ):
        tokens = _tokenize(query)
        if not tokens:
            return []

        candidates = self._repository.list_memories(memory_type=memory_type, status=status, limit=500)
        ranked: list[RelationalSearchResult] = []
        surfaced_ids: list[str] = []

        for candidate in candidates:
            if not include_superseded and self._repository.has_incoming_link(candidate.id, "SUPERSEDES"):
                continue

            base_score = _base_match_score(candidate, tokens)
            if base_score <= 0:
                continue

            score = _normalize_base_score(base_score)
            score = _apply_recency_boost(score, candidate, self._config)
            score = _apply_access_boost(score, candidate)

            if workspace_id and workspace_id in candidate.workspace_ids:
                score *= WORKSPACE_BOOST

            if candidate.status in {"stale", "degraded"}:
                score *= DEGRADATION_PENALTY

            ranked.append(
                RelationalSearchResult(
                    memory_id=candidate.id,
                    title=candidate.title,
                    summary=candidate.summary or "",
                    memory_type=candidate.type,
                    status=candidate.status,
                    tags=list(candidate.tags),
                    workspace_ids=list(candidate.workspace_ids),
                    score=round(score, 6),
                )
            )

        ranked.sort(key=lambda item: item.score, reverse=True)
        final_results = ranked[:limit]
        surfaced_ids.extend(result.memory_id for result in final_results)
        if surfaced_ids:
            self._repository.touch_last_surfaced(surfaced_ids, _utc_now())
        return final_results

    def read_memory(self, memory_id: str):
        record = self._repository.get_memory(memory_id)
        if record is None:
            return None

        decayed_score = _decayed_access_score(record.access_score, record.last_accessed_at)
        updated_record = self._repository.record_access(memory_id, decayed_score + 1.0, _utc_now())
        current = updated_record or record

        outgoing = self._repository.get_links(memory_id, direction="outgoing")
        incoming = self._repository.get_links(memory_id, direction="incoming")
        superseded: list[RelationalMemoryRecord] = []
        for link in outgoing:
            if link.link_type != "SUPERSEDES":
                continue
            target = self._repository.get_memory(link.target_id)
            if target is not None:
                superseded.append(target)

        return RelationalReadResult(
            record=current,
            relationships={
                "outgoing": outgoing,
                "incoming": incoming,
            },
            superseded=superseded,
        )


def _tokenize(query: str):
    return [token.lower() for token in TOKEN_PATTERN.findall(query)]


def _base_match_score(record: RelationalMemoryRecord, tokens: list[str]):
    title = record.title.lower()
    summary = (record.summary or "").lower()
    content = record.content.lower()
    tags = {tag.lower() for tag in record.tags}

    score = 0.0
    for token in tokens:
        if token in title:
            score += 3.0
        if token in summary:
            score += 2.0
        if token in content:
            score += 1.0
        if token in tags:
            score += 1.5
    return score


def _normalize_base_score(score: float):
    return score / (score + 3.0)


def _apply_recency_boost(score: float, record: RelationalMemoryRecord, config: Config):
    try:
        created_at = datetime.fromisoformat(record.created_at)
    except ValueError:
        return score

    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)

    age_days = (datetime.now(timezone.utc) - created_at).days
    recency = config.memory.get_recency_config(record.type)
    if age_days > recency.boost_window_days:
        return score

    bonus = (recency.boost_decay_rate ** age_days) * recency.max_boost_amount
    return min(1.0, score + bonus)


def _apply_access_boost(score: float, record: RelationalMemoryRecord):
    decayed = _decayed_access_score(record.access_score, record.last_accessed_at)
    if decayed <= 0:
        return score
    return min(1.0, score + min(0.2, math.log1p(decayed) / 10.0))


def _decayed_access_score(access_score: float, last_accessed_at: str | None):
    if access_score <= 0 or not last_accessed_at:
        return max(access_score, 0.0)

    try:
        accessed_at = datetime.fromisoformat(last_accessed_at)
    except ValueError:
        return max(access_score, 0.0)

    if accessed_at.tzinfo is None:
        accessed_at = accessed_at.replace(tzinfo=timezone.utc)

    elapsed = datetime.now(timezone.utc) - accessed_at
    elapsed_days = max(elapsed / timedelta(days=1), 0.0)
    return access_score * (0.5 ** (elapsed_days / ACCESS_HALF_LIFE_DAYS))


def _utc_now():
    return datetime.now(timezone.utc).isoformat()
