from sentence_transformers import (
    SentenceTransformer,
)


from chromadb.api.types import EmbeddingFunction, Documents, Embeddings


class AtaraxAIEmbedder(EmbeddingFunction):  # type: ignore
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        print(f"AtaraxAIEmbedder: Initialized with model '{model_name}'.")

    def __call__(self, input: Documents) -> Embeddings:
        return self.model.encode(input).tolist()  # type: ignore
