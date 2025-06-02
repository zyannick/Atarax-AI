from sentence_transformers import (
    SentenceTransformer,
)


class AtaraxAIEmbedder:

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, convert_to_tensor=True).tolist()

    def __call__(self, texts: list[str]) -> list[list[float]]:
        return self.embed(texts)
