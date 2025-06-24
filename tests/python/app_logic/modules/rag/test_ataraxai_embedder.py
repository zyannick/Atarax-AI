import pytest
from ataraxai.app_logic.modules.rag.ataraxai_embedder import AtaraxAIEmbedder

@pytest.fixture
def embedder():
    return AtaraxAIEmbedder()

def test_embed_returns_correct_shape(embedder):
    texts = ["Hello world!", "Test sentence."]
    embeddings = embedder.embed(texts)
    assert isinstance(embeddings, list)
    assert len(embeddings) == len(texts)
    assert all(isinstance(vec, list) for vec in embeddings)
    assert all(isinstance(val, float) for vec in embeddings for val in vec)
    assert all(len(vec) == 384 for vec in embeddings)

def test_call_method_equivalent_to_embed(embedder):
    texts = ["foo", "bar"]
    assert embedder(texts) == embedder.embed(texts)

def test_embed_empty_list(embedder):
    embeddings = embedder.embed([])
    assert embeddings == []

def test_embed_single_text(embedder):
    text = ["A local privacy assistant."]
    embeddings = embedder.embed(text)
    assert isinstance(embeddings, list)
    assert len(embeddings) == 1
    assert isinstance(embeddings[0], list)
    assert len(embeddings[0]) == 384