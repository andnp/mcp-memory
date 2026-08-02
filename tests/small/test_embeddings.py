from concurrent.futures import ThreadPoolExecutor
import builtins
import os
import sys
import threading
import time

from unittest import mock

from mcp_memory.config import EmbeddingsConfig
from mcp_memory.embeddings import (
    OllamaEmbedder,
    SentenceTransformerEmbedder,
    SQLiteVectorStore,
    _cap_torch_threads_before_sentence_transformer_load,
    build_embedder,
)
from searchkernel.ports import CandidateFilterSupport
from searchkernel.utils.similarity import cosine_similarity_lists


class FakeTorch:
    def __init__(
        self,
        *,
        num_threads: int,
        interop_threads: int,
        interop_error: bool = False,
    ) -> None:
        self.num_threads = num_threads
        self.interop_threads = interop_threads
        self.interop_error = interop_error
        self.set_num_threads_calls: list[int] = []
        self.set_num_interop_threads_calls: list[int] = []

    def get_num_threads(self) -> int:
        return self.num_threads

    def set_num_threads(self, value: int) -> None:
        self.set_num_threads_calls.append(value)
        self.num_threads = value

    def get_num_interop_threads(self) -> int:
        return self.interop_threads

    def set_num_interop_threads(self, value: int) -> None:
        self.set_num_interop_threads_calls.append(value)
        if self.interop_error:
            raise RuntimeError("interop threads already configured")
        self.interop_threads = value


def _reset_torch_thread_cap_state(monkeypatch) -> None:
    monkeypatch.setattr("mcp_memory.embeddings._TORCH_THREAD_CAP_INITIALIZED", False)
    monkeypatch.delenv("MCP_MEMORY_TORCH_NUM_THREADS", raising=False)
    monkeypatch.delenv("MCP_MEMORY_TORCH_INTEROP_THREADS", raising=False)
    monkeypatch.delenv("TOKENIZERS_PARALLELISM", raising=False)


def test_cosine_similarity_returns_expected_scores() -> None:
    assert cosine_similarity_lists([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert round(cosine_similarity_lists([1.0, 0.0], [0.0, 1.0]), 6) == 0.0


def test_sqlite_vector_store_round_trips_embeddings(db_manager) -> None:
    store = SQLiteVectorStore(db_manager)
    store.upsert(
        source_kind="memory",
        source_id="memory-1",
        workspace_id="workspace-a",
        model_name="test-model",
        embedding=[1.0, 0.0],
    )
    store.upsert(
        source_kind="memory",
        source_id="memory-2",
        workspace_id="workspace-a",
        model_name="test-model",
        embedding=[0.8, 0.2],
    )

    record = store.get(source_kind="memory", source_id="memory-1", model_name="test-model")
    results = store.search(
        source_kind="memory",
        model_name="test-model",
        workspace_id="workspace-a",
        query_embedding=[1.0, 0.0],
        limit=2,
    )

    assert record is not None
    assert record.embedding == [1.0, 0.0]
    assert [result[0] for result in results] == ["memory-1", "memory-2"]
    assert isinstance(store, CandidateFilterSupport)


def test_sqlite_vector_store_candidate_filtering_limits_rows_and_results(db_manager) -> None:
    store = SQLiteVectorStore(db_manager)
    for source_id, embedding in (
        ("memory-1", [1.0, 0.0]),
        ("memory-2", [0.8, 0.2]),
        ("memory-3", [0.0, 1.0]),
        ("memory-other-workspace", [1.0, 0.0]),
        ("memory-other-model", [1.0, 0.0]),
    ):
        store.upsert(
            source_kind="memory",
            source_id=source_id,
            workspace_id="workspace-b" if source_id == "memory-other-workspace" else "workspace-a",
            model_name="other-model" if source_id == "memory-other-model" else "test-model",
            embedding=embedding,
        )

    diagnostics: dict[str, object] = {}
    results = store.search(
        source_kind="memory",
        model_name="test-model",
        workspace_id="workspace-a",
        query_embedding=[1.0, 0.0],
        candidate_ids=["memory-3", "memory-other-workspace", "memory-other-model"],
        diagnostics=diagnostics,
        limit=20,
    )

    assert results == [("memory-3", 0.0)]
    assert diagnostics["row_count"] == 1
    assert diagnostics["candidate_filter_count"] == 3


def test_sqlite_vector_store_empty_candidate_filter_returns_no_results(db_manager) -> None:
    store = SQLiteVectorStore(db_manager)
    store.upsert(
        source_kind="memory",
        source_id="memory-1",
        workspace_id="workspace-a",
        model_name="test-model",
        embedding=[1.0, 0.0],
    )

    assert store.search(
        source_kind="memory",
        model_name="test-model",
        query_embedding=[1.0, 0.0],
        candidate_ids=[],
    ) == []


def test_build_embedder_defaults_to_ollama() -> None:
    with mock.patch("searchkernel.adapters.embedding.OllamaEmbeddingProvider"):
        embedder = build_embedder(EmbeddingsConfig())
    assert isinstance(embedder, OllamaEmbedder)


def test_build_embedder_uses_sentence_transformer_for_explicit_provider() -> None:
    embedder = build_embedder(
        EmbeddingsConfig(
            provider="sentence-transformers",
            model="sentence-transformers/all-MiniLM-L6-v2",
        )
    )
    assert isinstance(embedder, SentenceTransformerEmbedder)


def test_build_embedder_returns_ollama_embedder_for_ollama_provider() -> None:
    config = EmbeddingsConfig(provider="ollama", model="qwen3-embedding:0.6b")

    with mock.patch("searchkernel.adapters.embedding.OllamaEmbeddingProvider"):
        embedder = build_embedder(config)

    assert isinstance(embedder, OllamaEmbedder)


def test_embeddings_config_rejects_unknown_provider() -> None:
    try:
        EmbeddingsConfig(provider="unknown")
    except ValueError as exc:
        assert "provider" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown provider")


def test_ollama_embedder_model_name_and_status() -> None:
    config = EmbeddingsConfig(provider="ollama", model="qwen3-embedding:0.6b")

    with mock.patch("searchkernel.adapters.embedding.OllamaEmbeddingProvider"):
        embedder = OllamaEmbedder(config)

    assert embedder.model_name == "qwen3-embedding:0.6b"
    assert embedder.configured_model_name == "qwen3-embedding:0.6b"
    status = embedder.status()
    assert status.backend == "ollama"
    assert status.model_name == "qwen3-embedding:0.6b"
    assert status.model_cached is True


def test_ollama_embedder_embed_delegates_to_provider() -> None:
    config = EmbeddingsConfig(provider="ollama", model="qwen3-embedding:0.6b")

    with mock.patch("searchkernel.adapters.embedding.OllamaEmbeddingProvider") as mock_provider_cls:
        mock_provider = mock_provider_cls.return_value
        mock_provider.embed.return_value = [[0.1, 0.2]]
        embedder = OllamaEmbedder(config)

        result = embedder.embed(["hello"])

    assert result == [[0.1, 0.2]]
    mock_provider.embed.assert_called_once_with(["hello"])


def test_ollama_embedder_embed_returns_empty_list_for_no_texts() -> None:
    config = EmbeddingsConfig(provider="ollama", model="qwen3-embedding:0.6b")

    with mock.patch("searchkernel.adapters.embedding.OllamaEmbeddingProvider"):
        embedder = OllamaEmbedder(config)

    assert embedder.embed([]) == []


def test_sentence_transformer_cache_model_uses_local_cache_before_network(monkeypatch) -> None:
    calls: list[bool] = []

    class DummyModel:
        def encode(self, texts, **_kwargs):
            return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr("mcp_memory.embeddings._is_model_cached_locally", lambda _model_name: True)

    def fake_load(model_name: str, *, local_files_only: bool):
        calls.append(local_files_only)
        assert model_name == "sentence-transformers/all-MiniLM-L6-v2"
        return DummyModel()

    monkeypatch.setattr("mcp_memory.embeddings._load_sentence_transformer", fake_load)

    embedder = SentenceTransformerEmbedder(
        EmbeddingsConfig(
            provider="sentence-transformers",
            model="sentence-transformers/all-MiniLM-L6-v2",
        )
    )

    assert embedder.cache_model() is True
    assert calls == [True]


def test_sentence_transformer_cache_model_skips_uncached_download_in_test_mode(monkeypatch) -> None:
    monkeypatch.setenv("MCP_MEMORY_TEST_MODE", "1")
    monkeypatch.setattr("mcp_memory.embeddings._is_model_cached_locally", lambda _model_name: False)

    def fail_load(*_args, **_kwargs):
        raise AssertionError("test mode must not attempt an uncached model load")

    monkeypatch.setattr("mcp_memory.embeddings._load_sentence_transformer", fail_load)

    assert SentenceTransformerEmbedder(EmbeddingsConfig()).cache_model() is False


def test_sentence_transformer_embedder_switches_to_effective_fallback_model_name(monkeypatch) -> None:
    config = EmbeddingsConfig(model="sentence-transformers/all-MiniLM-L6-v2")
    embedder = SentenceTransformerEmbedder(config)

    monkeypatch.setattr("mcp_memory.embeddings._is_model_cached_locally", lambda _model_name: False)

    assert embedder.model_name == config.model

    vectors = embedder.embed(["hello world"])
    status = embedder.status()

    assert len(vectors) == 1
    assert len(vectors[0]) == 64
    assert embedder.model_name == f"hash:{config.model}"
    assert status.model_name == f"hash:{config.model}"
    assert status.configured_model_name == config.model
    assert status.backend == "fallback"
    assert status.model_cached is False


def test_sentence_transformer_embedder_restores_configured_model_name_after_successful_cache(monkeypatch) -> None:
    config = EmbeddingsConfig(model="sentence-transformers/all-MiniLM-L6-v2")
    embedder = SentenceTransformerEmbedder(config)

    monkeypatch.delenv("MCP_MEMORY_TEST_MODE", raising=False)
    monkeypatch.setattr("mcp_memory.embeddings._is_model_cached_locally", lambda _model_name: False)

    embedder.embed(["hello world"])

    class DummyModel:
        def encode(self, texts, **_kwargs):
            return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(
        "mcp_memory.embeddings._load_sentence_transformer",
        lambda model_name, *, local_files_only: DummyModel(),
    )

    assert embedder.cache_model() is True
    assert embedder.model_name == config.model
    assert embedder.status().configured_model_name == config.model
    assert embedder.embed(["hello again"]) == [[1.0, 0.0]]


def test_cap_torch_threads_before_sentence_transformer_load_uses_default_caps(monkeypatch) -> None:
    _reset_torch_thread_cap_state(monkeypatch)
    fake_torch = FakeTorch(num_threads=12, interop_threads=8)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    _cap_torch_threads_before_sentence_transformer_load()

    assert fake_torch.set_num_threads_calls == [4]
    assert fake_torch.set_num_interop_threads_calls == [1]
    assert fake_torch.num_threads == 4
    assert fake_torch.interop_threads == 1
    assert os.environ["TOKENIZERS_PARALLELISM"] == "false"


def test_cap_torch_threads_before_sentence_transformer_load_honors_env_overrides(monkeypatch) -> None:
    _reset_torch_thread_cap_state(monkeypatch)
    fake_torch = FakeTorch(num_threads=9, interop_threads=5)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setenv("MCP_MEMORY_TORCH_NUM_THREADS", "3")
    monkeypatch.setenv("MCP_MEMORY_TORCH_INTEROP_THREADS", "2")

    _cap_torch_threads_before_sentence_transformer_load()

    assert fake_torch.set_num_threads_calls == [3]
    assert fake_torch.set_num_interop_threads_calls == [2]
    assert fake_torch.num_threads == 3
    assert fake_torch.interop_threads == 2


def test_cap_torch_threads_before_sentence_transformer_load_only_caps_downward_and_preserves_tokenizers_env(
    monkeypatch,
) -> None:
    _reset_torch_thread_cap_state(monkeypatch)
    fake_torch = FakeTorch(num_threads=2, interop_threads=1)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setenv("MCP_MEMORY_TORCH_NUM_THREADS", "8")
    monkeypatch.setenv("MCP_MEMORY_TORCH_INTEROP_THREADS", "4")
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "true")

    _cap_torch_threads_before_sentence_transformer_load()

    assert fake_torch.set_num_threads_calls == []
    assert fake_torch.set_num_interop_threads_calls == []
    assert fake_torch.num_threads == 2
    assert fake_torch.interop_threads == 1
    assert os.environ["TOKENIZERS_PARALLELISM"] == "true"


def test_cap_torch_threads_before_sentence_transformer_load_is_idempotent_and_tolerates_late_interop_errors(
    monkeypatch,
) -> None:
    _reset_torch_thread_cap_state(monkeypatch)
    fake_torch = FakeTorch(num_threads=10, interop_threads=6, interop_error=True)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    _cap_torch_threads_before_sentence_transformer_load()
    _cap_torch_threads_before_sentence_transformer_load()

    assert fake_torch.set_num_threads_calls == [4]
    assert fake_torch.set_num_interop_threads_calls == [1]
    assert fake_torch.num_threads == 4
    assert os.environ["TOKENIZERS_PARALLELISM"] == "false"


def test_cap_torch_threads_before_sentence_transformer_load_tolerates_missing_torch(monkeypatch) -> None:
    _reset_torch_thread_cap_state(monkeypatch)
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch":
            raise ImportError("torch unavailable")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, "torch", raising=False)

    _cap_torch_threads_before_sentence_transformer_load()

    assert os.environ["TOKENIZERS_PARALLELISM"] == "false"


def test_sentence_transformer_concurrent_load_only_initializes_once(monkeypatch) -> None:
    entered_load = threading.Event()
    release_load = threading.Event()
    call_count = 0

    class DummyModel:
        def encode(self, texts, **_kwargs):
            return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr("mcp_memory.embeddings._is_model_cached_locally", lambda _model_name: True)

    def fake_load(_model_name: str, *, local_files_only: bool):
        nonlocal call_count
        assert local_files_only is True
        call_count += 1
        entered_load.set()
        release_load.wait(timeout=1.0)
        return DummyModel()

    monkeypatch.setattr("mcp_memory.embeddings._load_sentence_transformer", fake_load)

    embedder = SentenceTransformerEmbedder(EmbeddingsConfig())

    with ThreadPoolExecutor(max_workers=2) as executor:
        warmup_future = executor.submit(embedder.cache_model)
        assert entered_load.wait(timeout=1.0)
        embed_future = executor.submit(embedder.embed, ["hello"])
        time.sleep(0.05)
        release_load.set()

        assert warmup_future.result(timeout=1.0) is True
        assert embed_future.result(timeout=1.0) == [[1.0, 0.0]]

    assert call_count == 1