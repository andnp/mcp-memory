from concurrent.futures import ThreadPoolExecutor
import builtins
import os
import sys
import threading
import time

from mcp_memory.config import EmbeddingsConfig
from mcp_memory.embeddings import (
    SentenceTransformerEmbedder,
    SQLiteVectorStore,
    _cap_torch_threads_before_sentence_transformer_load,
    cosine_similarity,
)


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
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert round(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 6) == 0.0


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

    embedder = SentenceTransformerEmbedder(EmbeddingsConfig())

    assert embedder.cache_model() is True
    assert calls == [True]


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