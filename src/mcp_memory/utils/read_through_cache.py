"""Generic building blocks for validated read-through caching.

These pieces are domain-agnostic: they know nothing about memories, search
results, or any other mcp-memory concept. They only capture the recurring
shape of a validated read-through cache:

- ``ValidationTokenResolver`` resolves authoritative freshness tokens for a
  batch of keys through a duck-typed callable (looked up with ``getattr`` on
  some port), tolerating an absent resolver.
- ``InFlightCoalescer`` gates and delegates single-flight request coalescing
  to a cache object that exposes ``begin_inflight_search``/
  ``finish_inflight_search``.
- ``StaleFallbackLoader`` captures the "load a cached payload, log that it is
  being served stale after an authoritative failure, annotate it" pattern
  used for degraded-mode fallbacks.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from typing import Generic, Protocol, TypeVar

TKey = TypeVar("TKey")
TToken = TypeVar("TToken")
TRequest = TypeVar("TRequest")
TEntry = TypeVar("TEntry")
TPayload = TypeVar("TPayload")


def _dedupe(keys: Iterable[TKey], *, predicate: Callable[[TKey], bool] | None) -> list[TKey]:
    seen: set[TKey] = set()
    normalized: list[TKey] = []
    for key in keys:
        if predicate is not None and not predicate(key):
            continue
        if key in seen:
            continue
        seen.add(key)
        normalized.append(key)
    return normalized


class ValidationTokenResolver(Generic[TKey, TToken]):
    """Resolves authoritative validation tokens for a batch of keys.

    ``resolver`` is typically a method looked up with ``getattr`` on a port
    that may not implement it; passing ``None`` makes every lookup a no-op.
    Callers that need failure isolation should wrap ``resolve`` themselves —
    this class does not swallow exceptions raised by the resolver, so a
    caller's own try/except and logging around a validation step keeps
    working unchanged.
    """

    def __init__(self, resolver: Callable[[list[TKey]], dict[TKey, TToken]] | None) -> None:
        self._resolver = resolver

    @property
    def available(self) -> bool:
        return callable(self._resolver)

    def resolve(
        self,
        keys: Iterable[TKey],
        *,
        key_filter: Callable[[TKey], bool] | None = None,
        result_filter: Callable[[TKey, TToken], bool] | None = None,
    ) -> dict[TKey, TToken]:
        if not callable(self._resolver):
            return {}
        normalized_keys = _dedupe(keys, predicate=key_filter)
        if not normalized_keys:
            return {}
        tokens = self._resolver(normalized_keys)
        if not isinstance(tokens, dict):
            return {}
        if result_filter is None:
            return tokens
        return {key: value for key, value in tokens.items() if result_filter(key, value)}

    def resolve_one(self, key: TKey) -> TToken | None:
        return self.resolve([key]).get(key)


TRequest_contra = TypeVar("TRequest_contra", contravariant=True)
TPayload_contra = TypeVar("TPayload_contra", contravariant=True)


class InFlightSearchCache(Protocol[TRequest_contra, TEntry, TPayload_contra]):
    def begin_inflight_search(self, request: TRequest_contra) -> TEntry: ...

    def finish_inflight_search(
        self,
        entry: TEntry,
        *,
        payload: TPayload_contra | None = None,
        error: Exception | None = None,
    ) -> None: ...


class InFlightCoalescer(Generic[TRequest, TEntry, TPayload]):
    """Gates and delegates single-flight request coalescing to a cache.

    ``cache_provider`` is called on every operation rather than captured once,
    since the underlying cache (e.g. a shared read cache on a request-scoped
    port) may not exist yet, or may change between calls. ``enabled`` gates
    whether coalescing should even be attempted for this request.
    """

    def __init__(
        self,
        cache_provider: Callable[[], InFlightSearchCache[TRequest, TEntry, TPayload] | None],
        *,
        enabled: Callable[[], bool],
    ) -> None:
        self._cache_provider = cache_provider
        self._enabled = enabled

    def begin(self, request: TRequest) -> TEntry | None:
        if not self._enabled():
            return None
        cache = self._cache_provider()
        if cache is None:
            return None
        return cache.begin_inflight_search(request)

    def finish(
        self,
        entry: TEntry | None,
        *,
        payload: TPayload | None = None,
        error: Exception | None = None,
    ) -> None:
        if entry is None or not getattr(entry, "is_leader", False):
            return
        cache = self._cache_provider()
        if cache is None:
            return
        cache.finish_inflight_search(entry, payload=payload, error=error)


class StaleFallbackLoader(Generic[TPayload]):
    """Loads a cached payload as a degraded fallback after an authoritative failure.

    Returns ``None`` when nothing is cached; otherwise logs that a stale
    response is being served (with the triggering error attached) and applies
    an optional ``annotate`` transform (e.g. marking the payload degraded)
    before returning it.
    """

    def __init__(self, *, logger: logging.Logger, message: str) -> None:
        self._logger = logger
        self._message = message

    def load(
        self,
        loader: Callable[[], TPayload | None],
        *,
        error: Exception,
        annotate: Callable[[TPayload], TPayload] | None = None,
    ) -> TPayload | None:
        payload = loader()
        if payload is None:
            return None
        self._logger.warning(self._message, exc_info=error)
        return annotate(payload) if annotate is not None else payload
