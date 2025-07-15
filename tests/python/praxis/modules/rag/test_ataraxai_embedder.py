import pytest
from ataraxai.praxis.modules.rag.ataraxai_embedder import AtaraxAIEmbedder
import numpy as np
from typing_extensions import Union

@pytest.fixture
def embedder():
    return AtaraxAIEmbedder()

def test_embed_returns_correct_shape(embedder):
    texts = ["Hello world!", "Test sentence."]
    embeddings = embedder(texts)
    assert isinstance(embeddings, list)
    assert len(embeddings) == len(texts)
    assert all(isinstance(vec, np.ndarray) for vec in embeddings)
    assert all(isinstance(val, Union[float, np.float32]) for vec in embeddings for val in vec)


def test_embed_empty_list(embedder):
    with pytest.raises(ValueError, match="Expected Embeddings to be non-empty"):
        embedder([])

def test_embed_single_text(embedder):
    text = ["A local privacy assistant."]
    embeddings = embedder(text)
    assert isinstance(embeddings, list)
    assert len(embeddings) == 1
    assert isinstance(embeddings[0], np.ndarray)
