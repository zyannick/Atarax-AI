from sentence_transformers import (
    SentenceTransformer,
)
from typing import Dict, Any
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings


class AtaraxAIEmbedder(EmbeddingFunction):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initializes the AtaraxAIEmbedder with a specified SentenceTransformer model.

        Args:
            model_name (str, optional): The name or path of the pre-trained SentenceTransformer model to use.
                Defaults to "sentence-transformers/all-MiniLM-L6-v2".

        Attributes:
            model (SentenceTransformer): The loaded SentenceTransformer model.
            model_name (str): The name or path of the model used.

        Prints:
            Confirmation message indicating which model has been initialized.
        """
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        print(f"AtaraxAIEmbedder: Initialized with model '{self.model_name}'.")

    def __call__(self, input: Documents) -> Embeddings:
        return self.model.encode(list(input)).tolist() # type: ignore

    def name(self) -> str: # type: ignore[override]
        return self.model_name

    def get_config(self) -> Dict[str, Any]: # type: ignore[override]
        return {
            "model_name": self.model_name,
            "model_type": "sentence-transformers",
        }
