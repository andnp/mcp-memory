from mcp_memory.embeddings import SQLiteVectorStore, cosine_similarity


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