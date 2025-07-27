import pytest
from pathlib import Path

from ataraxai.praxis.modules.rag.ataraxai_embedder import AtaraxAIEmbedder
from ataraxai.praxis.modules.rag.rag_store import RAGStore

@pytest.mark.integration
def test_rag_embedding_and_retrieval(tmp_path: Path):

    correct_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    db_path = tmp_path / "chroma_test_db"


    embedder = AtaraxAIEmbedder(model_name=correct_model_name)
    rag_store = RAGStore(
        db_path_str=str(db_path),
        collection_name="integration_test",
        embedder=embedder
    )

    documents = ["The sky is blue.", "The grass is green."]
    ids = ["doc1", "doc2"]
    rag_store.add_chunks(ids=ids, texts=documents, metadatas=[{'color': 'blue'}, {'color': 'green'}])

    query_text = "What color is the sky?"
    results = rag_store.query(query_text=query_text, n_results=1)

    assert results is not None
    assert "documents" in results and len(results["documents"][0]) == 1 # type: ignore
    assert results["documents"][0][0] == "The sky is blue." # type: ignore