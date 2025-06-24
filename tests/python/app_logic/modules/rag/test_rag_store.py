import pytest
from unittest.mock import MagicMock, patch
from ataraxai.app_logic.modules.rag.rag_store import RAGStore

class DummyEmbedder:
    def embed(self, texts):
        return [[float(i)] * 3 for i in range(len(texts))]

@pytest.fixture
def dummy_collection():
    collection = MagicMock()
    collection.name = "test_collection"
    collection.count.return_value = 0
    return collection

@pytest.fixture
def rag_store(tmp_path, dummy_collection):
    embedder = DummyEmbedder()
    with patch("chromadb.PersistentClient") as mock_client:
        mock_client.return_value.get_or_create_collection.return_value = dummy_collection
        store = RAGStore(str(tmp_path), "test_collection", embedder)
        store.collection = dummy_collection  
        return store

def test_add_chunks_with_embeddings(rag_store):
    ids = ["id1", "id2"]
    texts = ["text one", "text two"]
    metadatas = [{"meta": 1}, {"meta": 2}]
    embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    rag_store.add_chunks(ids, texts, metadatas, embeddings_list=embeddings)

    rag_store.collection.add.assert_called_once_with(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

def test_add_chunks_without_embeddings_uses_embedder(rag_store):
    ids = ["id1"]
    texts = ["text one"]
    metadatas = [{"meta": 1}]

    rag_store.add_chunks(ids, texts, metadatas)

    rag_store.collection.add.assert_called_once_with(
        ids=ids,
        embeddings=[[0.0, 0.0, 0.0]],
        documents=texts,
        metadatas=metadatas,
    )

def test_add_chunks_raises_if_no_embedder_and_no_embeddings(tmp_path, dummy_collection):
    with patch("chromadb.PersistentClient") as mock_client:
        mock_client.return_value.get_or_create_collection.return_value = dummy_collection
        store = RAGStore(str(tmp_path), "test_collection", embedder=None)
        store.collection = dummy_collection
        ids = ["id1"]
        texts = ["text one"]
        metadatas = [{"meta": 1}]
        with pytest.raises(ValueError, match="Embedder not available in RAGStore"):
            store.add_chunks(ids, texts, metadatas)

def test_add_chunks_raises_on_embedding_generation_failure(rag_store):
    rag_store.embedder.embed = MagicMock(return_value=None)
    ids = ["id1"]
    texts = ["text one"]
    metadatas = [{"meta": 1}]
    with pytest.raises(ValueError, match="Embedding generation failed"):
        rag_store.add_chunks(ids, texts, metadatas)

def test_add_chunks_raises_on_embedding_count_mismatch(rag_store):
    rag_store.embedder.embed = MagicMock(return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    ids = ["id1"]
    texts = ["text one"]
    metadatas = [{"meta": 1}]
    with pytest.raises(ValueError, match="Embedding generation failed"):
        rag_store.add_chunks(ids, texts, metadatas)