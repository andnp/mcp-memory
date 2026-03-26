from concurrent.futures import ThreadPoolExecutor
import threading
import time

from mcp_memory.config import EmbeddingsConfig
from mcp_memory.embeddings import SentenceTransformerEmbedder, SQLiteVectorStore, cosine_similarity


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