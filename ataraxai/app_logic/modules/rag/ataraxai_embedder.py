from sentence_transformers import (
    SentenceTransformer,
)


class AtaraxAIEmbedder:

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initializes the embedder with a specified SentenceTransformer model.

        Args:
            model_name (str, optional): The name or path of the pre-trained SentenceTransformer model to use.
                Defaults to "sentence-transformers/all-MiniLM-L6-v2".

        Attributes:
            model (SentenceTransformer): The loaded SentenceTransformer model instance.
        """
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embeds a list of input texts into vector representations.

        Args:
            texts (list[str]): A list of strings to be embedded.

        Returns:
            list[list[float]]: A list of embedding vectors, where each vector corresponds to an input text.
        """
        return self.model.encode(texts, convert_to_tensor=True).tolist()

    def __call__(self, texts: list[str]) -> list[list[float]]:
        """
        Invokes the embedder on a list of input texts and returns their corresponding embeddings.

        Args:
            texts (list[str]): A list of input strings to be embedded.

        Returns:
            list[list[float]]: A list of embeddings, where each embedding is a list of floats corresponding to an input text.
        """
        return self.embed(texts)
