"""mcp-memory's source-owned adapter for the searchkernel federation contract."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from typing import Any, cast

from searchkernel.domain import RecordStatus
from searchkernel.ports.federation import (
    FEDERATION_CONTRACT_VERSION,
    MAX_RERANK_TEXT_LENGTH,
    MAX_SNIPPET_LENGTH,
    MAX_TOP_K,
    SearchHit,
    SearchHitProvenance,
    SearchRequest,
    SearchResponse,
    SearchSource,
    SourceCapabilities,
    SourceIdentity,
)

from mcp_memory.core.ports.memory import MemoryReadPort
from mcp_memory.integrations.memory_retrieval import MemoryRetrievalPort

_SOURCE_KIND = "memory"
_SOURCE_ID = "mcp-memory"
_SAFE_FILTERS = frozenset({"workspace_id", "memory_type", "status", "include_superseded"})


class MemoryFederationSource(SearchSource):
    """Expose canonical memory retrieval without duplicating search policy."""

    def __init__(
        self,
        retrieval: MemoryRetrievalPort,
        repository: MemoryReadPort | None = None,
        *,
        source_identity: SourceIdentity | None = None,
        authorize: Callable[[SearchRequest, Any], bool] | None = None,
        health_provider: Any | None = None,
        side_effect_free: bool = False,
    ) -> None:
        self._retrieval = retrieval
        self._repository = repository
        self._identity = source_identity or SourceIdentity(_SOURCE_KIND, _SOURCE_ID)
        self._authorize = authorize
        self._health_provider = health_provider
        self._side_effect_free = side_effect_free
        self._capabilities = SourceCapabilities(
            supports_filters=True,
            supports_source_selection=True,
            supports_rerank_text=False,
            supports_partial_results=True,
            supports_cancellation=True,
            max_top_k=MAX_TOP_K,
            max_rerank_text_length=MAX_RERANK_TEXT_LENGTH,
        )

    @property
    def source_identity(self) -> SourceIdentity:
        return self._identity

    @property
    def source_kind(self) -> str:
        return self._identity.source_kind

    @property
    def source_id(self) -> str:
        return self._identity.source_id

    def capabilities(self) -> SourceCapabilities:
        return self._capabilities

    async def search(self, request: SearchRequest) -> SearchResponse:
        started = time.perf_counter()
        if request.source_selection and (
            self._identity.source_kind not in request.source_selection
            and self._identity.source_id not in request.source_selection
        ):
            return SearchResponse(
                source=self._identity,
                contract_version=FEDERATION_CONTRACT_VERSION,
                index_epoch=self.index_epoch(),
                capabilities=self._capabilities,
            )
        filters = {
            key: value
            for key, value in request.filters.items()
            if key in _SAFE_FILTERS
        }
        workspace_id = _string_filter(filters, "workspace_id")
        ranking_workspace_id = None
        if workspace_id is None and request.caller is not None:
            claim_workspace = request.caller.claims.get("workspace_id")
            if isinstance(claim_workspace, str) and claim_workspace:
                ranking_workspace_id = claim_workspace
        retrieval_filters: dict[str, object] = {}
        if ranking_workspace_id is not None:
            retrieval_filters["_ranking_workspace_id"] = ranking_workspace_id
        retrieval_kwargs: dict[str, object] = {
            "limit": request.top_k,
            "workspace_id": workspace_id,
            "memory_type": _string_filter(filters, "memory_type"),
            "status": _string_filter(filters, "status"),
            "include_superseded": bool(filters.get("include_superseded", False)),
        }
        if retrieval_filters:
            retrieval_kwargs["filters"] = retrieval_filters
        if self._side_effect_free:
            retrieval_kwargs["side_effect_free"] = True
        outcome = await cast(Any, self._retrieval).search(
            request.query,
            **retrieval_kwargs,
        )

        hits: list[SearchHit] = []
        for result in outcome.results:
            if self._authorize is not None and not self._authorize(request, result):
                continue
            hits.append(self._to_hit(request, result, len(hits) + 1))

        warnings = tuple(
            _bounded_warning(reason)
            for reason in (*outcome.diagnostics, *outcome.cache_diagnostics)
        )
        warnings += tuple(
            _bounded_warning(f"search failure: {type(failure).__name__}")
            for failure in outcome.failures
        )
        return SearchResponse(
            source=self._identity,
            contract_version=FEDERATION_CONTRACT_VERSION,
            hits=tuple(hits),
            index_epoch=self.index_epoch(),
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            partial=outcome.degraded,
            warnings=warnings,
            capabilities=self._capabilities,
        )

    def index_epoch(self) -> str | None:
        if self._repository is None:
            return None
        get_epochs = getattr(self._repository, "get_search_epochs", None)
        if not callable(get_epochs):
            return None
        epochs = get_epochs()
        if not isinstance(epochs, Mapping):
            return None
        return json.dumps(dict(sorted(epochs.items())), separators=(",", ":"), sort_keys=True)

    def health(self) -> dict[str, object]:
        degraded = False
        search_health = None
        get_health = getattr(self._health_provider, "get_health", None)
        if callable(get_health):
            health = get_health()
            degraded = bool(getattr(health, "degraded", False))
            search_health = {
                "available": bool(getattr(health, "available", False)),
                "semantic_enabled": bool(getattr(health, "semantic_enabled", False)),
                "degraded": degraded,
            }
        return {
            "contract_version": FEDERATION_CONTRACT_VERSION,
            "source": self._identity.to_dict(),
            "status": "degraded" if degraded else "ok",
            "index_epoch": self.index_epoch(),
            "capabilities": self._capabilities.to_dict(),
            "search": search_health,
        }

    def _to_hit(self, request: SearchRequest, result: Any, rank: int) -> SearchHit:
        record = result.record
        metadata = record.metadata
        memory_status = str(metadata.get("memory_status", record.status.value))
        lifecycle = (
            RecordStatus.ARCHIVED
            if memory_status == "archived"
            else RecordStatus.STALE
            if memory_status in {"stale", "degraded"}
            else RecordStatus.ACTIVE
        )
        memory_ref = metadata.get("memory_ref")
        source_id = str(record.source_id)
        uri = f"memory://{memory_ref}" if memory_ref is not None else f"memory://{source_id}"
        summary = metadata.get("summary")
        snippet = _bounded_text(summary if isinstance(summary, str) else record.title, MAX_SNIPPET_LENGTH)
        safe_metadata = {
            "memory_ref": memory_ref,
            "memory_type": metadata.get("memory_type"),
            "status": memory_status,
            "tags": metadata.get("tags", []),
            "workspace_ids": metadata.get("workspace_ids", []),
            "citation": uri,
        }
        return SearchHit(
            source_kind=_SOURCE_KIND,
            source_id=source_id,
            workspace_id=record.workspace_id,
            title=record.title,
            snippet=snippet,
            rerank_text=None,
            uri=uri,
            source_rank=rank,
            native_score=float(result.score),
            created_at=record.created_at,
            updated_at=record.updated_at,
            lifecycle=lifecycle,
            metadata=safe_metadata,
            provenance=SearchHitProvenance(
                source=self._identity,
                request_id=request.request_id or None,
                retrieval_method="mcp-memory.searchkernel",
                details={"policy": "mcp-memory-native"},
            ),
        )


def _string_filter(filters: Mapping[str, object], key: str) -> str | None:
    value = filters.get(key)
    return value if isinstance(value, str) and value else None


def _bounded_text(value: str, maximum: int) -> str:
    return value.strip()[:maximum]


def _bounded_warning(value: str) -> str:
    return value[:512]


__all__ = ["MemoryFederationSource"]
