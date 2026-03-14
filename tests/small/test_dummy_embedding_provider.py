import pytest

from tests.fixtures import SAMPLE_EMBEDDING_TEXTS


pytestmark = pytest.mark.small


def test_dummy_embedding_provider_returns_fixed_dimension(dummy_embedding_provider) -> None:
    vector = dummy_embedding_provider.get_text_embedding(SAMPLE_EMBEDDING_TEXTS[0])

    assert len(vector) == dummy_embedding_provider.dimension
    assert all(0.0 <= value <= 1.0 for value in vector)


def test_dummy_embedding_provider_is_deterministic(dummy_embedding_provider) -> None:
    text = SAMPLE_EMBEDDING_TEXTS[1]

    first = dummy_embedding_provider.get_text_embedding(text)
    second = dummy_embedding_provider.get_text_embedding(text)

    assert first == second


def test_dummy_embedding_provider_supports_overrides() -> None:
    from tests.sdk.providers import DummyEmbeddingProvider

    provider = DummyEmbeddingProvider(dimension=4, overrides={"exact": [0.1, 0.2, 0.3, 0.4]})

    assert provider.get_text_embedding("exact") == [0.1, 0.2, 0.3, 0.4]