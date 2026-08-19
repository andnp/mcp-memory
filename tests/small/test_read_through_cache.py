from __future__ import annotations

import logging

import pytest
from mcp_memory.utils.read_through_cache import (
    InFlightCoalescer,
    StaleFallbackLoader,
    ValidationTokenResolver,
)

pytestmark = pytest.mark.small


class FakeInFlightEntry:
    def __init__(self, *, is_leader: bool) -> None:
        self.is_leader = is_leader


class FakeInFlightCache:
    def __init__(self) -> None:
        self.begin_calls: list[str] = []
        self.finish_calls: list[tuple[object, object, object]] = []
        self._next_is_leader = True

    def begin_inflight_search(self, request: str) -> FakeInFlightEntry:
        self.begin_calls.append(request)
        entry = FakeInFlightEntry(is_leader=self._next_is_leader)
        self._next_is_leader = False
        return entry

    def finish_inflight_search(self, entry, *, payload=None, error=None) -> None:
        self.finish_calls.append((entry, payload, error))


class TestValidationTokenResolver:
    def test_returns_empty_when_resolver_is_none(self) -> None:
        resolver: ValidationTokenResolver[str, str] = ValidationTokenResolver(None)

        assert resolver.available is False
        assert resolver.resolve(["a", "b"]) == {}

    def test_dedupes_keys_before_calling_resolver(self) -> None:
        seen_batches: list[list[str]] = []

        def fake_resolver(keys: list[str]) -> dict[str, str]:
            seen_batches.append(keys)
            return {key: f"token-{key}" for key in keys}

        resolver: ValidationTokenResolver[str, str] = ValidationTokenResolver(fake_resolver)

        result = resolver.resolve(["a", "b", "a"])

        assert seen_batches == [["a", "b"]]
        assert result == {"a": "token-a", "b": "token-b"}

    def test_applies_key_filter_before_dedup(self) -> None:
        def fake_resolver(keys: list[str]) -> dict[str, str]:
            return dict.fromkeys(keys, "token")

        resolver: ValidationTokenResolver[str, str] = ValidationTokenResolver(fake_resolver)

        keys: list[str] = ["", "a", None]  # type: ignore[list-item]
        result = resolver.resolve(keys, key_filter=lambda key: bool(key))

        assert result == {"a": "token"}

    def test_applies_result_filter_to_returned_pairs(self) -> None:
        def fake_resolver(keys: list[str]) -> dict[str, str]:
            return {"a": "keep", "b": ""}

        resolver: ValidationTokenResolver[str, str] = ValidationTokenResolver(fake_resolver)

        result = resolver.resolve(
            ["a", "b"],
            result_filter=lambda key, token: bool(token),
        )

        assert result == {"a": "keep"}

    def test_returns_empty_when_resolver_result_is_not_a_dict(self) -> None:
        resolver: ValidationTokenResolver[str, str] = ValidationTokenResolver(
            lambda keys: "not-a-dict"  # type: ignore[return-value]
        )

        assert resolver.resolve(["a"]) == {}

    def test_returns_empty_when_no_keys_survive_dedup(self) -> None:
        calls: list[list[str]] = []

        def fake_resolver(keys: list[str]) -> dict[str, str]:
            calls.append(keys)
            return {}

        resolver: ValidationTokenResolver[str, str] = ValidationTokenResolver(fake_resolver)

        assert resolver.resolve([], key_filter=lambda key: bool(key)) == {}
        assert calls == []

    def test_resolve_one_returns_single_token(self) -> None:
        resolver: ValidationTokenResolver[str, str] = ValidationTokenResolver(
            lambda keys: dict.fromkeys(keys, "token")
        )

        assert resolver.resolve_one("a") == "token"
        assert resolver.resolve_one("missing-from-response") == "token"

    def test_propagates_resolver_exceptions(self) -> None:
        def failing_resolver(keys: list[str]) -> dict[str, str]:
            raise RuntimeError("boom")

        resolver: ValidationTokenResolver[str, str] = ValidationTokenResolver(failing_resolver)

        with pytest.raises(RuntimeError, match="boom"):
            resolver.resolve(["a"])


class TestInFlightCoalescer:
    def test_begin_returns_none_when_disabled(self) -> None:
        cache = FakeInFlightCache()
        coalescer: InFlightCoalescer[str, FakeInFlightEntry, dict] = InFlightCoalescer(
            cache_provider=lambda: cache, enabled=lambda: False
        )

        assert coalescer.begin("query") is None
        assert cache.begin_calls == []

    def test_begin_returns_none_when_cache_provider_returns_none(self) -> None:
        coalescer: InFlightCoalescer[str, FakeInFlightEntry, dict] = InFlightCoalescer(
            cache_provider=lambda: None, enabled=lambda: True
        )

        assert coalescer.begin("query") is None

    def test_begin_delegates_to_cache_when_enabled(self) -> None:
        cache = FakeInFlightCache()
        coalescer: InFlightCoalescer[str, FakeInFlightEntry, dict] = InFlightCoalescer(
            cache_provider=lambda: cache, enabled=lambda: True
        )

        entry = coalescer.begin("query")

        assert cache.begin_calls == ["query"]
        assert entry is not None and entry.is_leader is True

    def test_finish_is_noop_for_non_leader_entry(self) -> None:
        cache = FakeInFlightCache()
        coalescer: InFlightCoalescer[str, FakeInFlightEntry, dict] = InFlightCoalescer(
            cache_provider=lambda: cache, enabled=lambda: True
        )

        coalescer.finish(FakeInFlightEntry(is_leader=False), payload={"ok": True})

        assert cache.finish_calls == []

    def test_finish_is_noop_for_none_entry(self) -> None:
        cache = FakeInFlightCache()
        coalescer: InFlightCoalescer[str, FakeInFlightEntry, dict] = InFlightCoalescer(
            cache_provider=lambda: cache, enabled=lambda: True
        )

        coalescer.finish(None, payload={"ok": True})

        assert cache.finish_calls == []

    def test_finish_delegates_to_cache_for_leader_entry(self) -> None:
        cache = FakeInFlightCache()
        coalescer: InFlightCoalescer[str, FakeInFlightEntry, dict] = InFlightCoalescer(
            cache_provider=lambda: cache, enabled=lambda: True
        )
        entry = FakeInFlightEntry(is_leader=True)
        error = RuntimeError("failure")

        coalescer.finish(entry, payload=None, error=error)

        assert cache.finish_calls == [(entry, None, error)]


class TestStaleFallbackLoader:
    def test_returns_none_when_loader_returns_none(self) -> None:
        loader: StaleFallbackLoader[dict] = StaleFallbackLoader(
            logger=logging.getLogger("test"), message="serving stale"
        )

        result = loader.load(lambda: None, error=RuntimeError("boom"))

        assert result is None

    def test_returns_payload_unannotated_without_annotate(self) -> None:
        loader: StaleFallbackLoader[dict] = StaleFallbackLoader(
            logger=logging.getLogger("test"), message="serving stale"
        )

        result = loader.load(lambda: {"status": "ok"}, error=RuntimeError("boom"))

        assert result == {"status": "ok"}

    def test_applies_annotate_transform(self) -> None:
        loader: StaleFallbackLoader[dict] = StaleFallbackLoader(
            logger=logging.getLogger("test"), message="serving stale"
        )

        result = loader.load(
            lambda: {"status": "ok"},
            error=RuntimeError("boom"),
            annotate=lambda payload: payload | {"degraded": True},
        )

        assert result == {"status": "ok", "degraded": True}

    def test_logs_warning_with_triggering_error(self, caplog) -> None:
        loader: StaleFallbackLoader[dict] = StaleFallbackLoader(
            logger=logging.getLogger("test-stale-fallback"), message="serving stale"
        )
        error = RuntimeError("authoritative failure")

        with caplog.at_level(logging.WARNING, logger="test-stale-fallback"):
            loader.load(lambda: {"status": "ok"}, error=error)

        assert any("serving stale" in record.message for record in caplog.records)
        assert any(record.exc_info is not None for record in caplog.records)
