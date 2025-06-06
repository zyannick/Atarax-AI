import chromadb

from pathlib import Path
from platformdirs import user_data_dir
from ataraxai.app_logic.modules.rag.ataraxai_embedder import AtaraxAIEmbedder

APP_NAME = "AtaraxAI"
APP_AUTHOR = "AtaraxAI"

CHROMA_DATA_PATH = (
    Path(user_data_dir(appname=APP_NAME, appauthor=APP_AUTHOR)) / "chroma_db"
)
CHROMA_DATA_PATH.mkdir(parents=True, exist_ok=True)

COLLECTION_NAME = "ataraxai_knowledge"


import chromadb
from pathlib import Path


class RAGStore:
    def __init__(
        self, db_path_str: str, collection_name: str, embedder: AtaraxAIEmbedder
    ):
        """
        Initializes a RAGStore instance with a persistent ChromaDB collection.

        Args:
            db_path_str (str): The file system path where the ChromaDB database will be stored.
            collection_name (str): The name of the collection to load or create within the database.
            embedder (AtaraxAIEmbedder): An embedding function or model used for vectorizing data.

        Side Effects:
            - Creates the database directory if it does not exist.
            - Loads or creates a ChromaDB collection with the specified embedding function and cosine similarity.
            - Prints the collection name and item count upon initialization.
        """
        self.db_path = Path(db_path_str)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.embedder = embedder
        self.client = chromadb.PersistentClient(path=str(self.db_path))

        chroma_ef = self.embedder

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=chroma_ef,
            metadata={"hnsw:space": "cosine"},
        )
        print(
            f"RAGStore: Collection '{self.collection.name}' loaded/created with {self.collection.count()} items."
        )

    def add_chunks(
        self,
        ids: list[str],
        texts: list[str],
        metadatas: list[dict],
        embeddings_list: list[list[float]] = None,
    ):
        """
        Adds text chunks along with their metadata and embeddings to the collection.

        Args:
            ids (list[str]): List of unique identifiers for each text chunk.
            texts (list[str]): List of text chunks to be added.
            metadatas (list[dict]): List of metadata dictionaries corresponding to each text chunk.
            embeddings_list (list[list[float]], optional): Precomputed embeddings for the text chunks. 
                If not provided, embeddings will be generated using the embedder.

        Raises:
            ValueError: If embeddings_list is not provided and no embedder is available.
            ValueError: If embedding generation fails or returns an incorrect number of embeddings.

        Side Effects:
            Adds or updates items in the collection and prints status messages.
        """
        if not embeddings_list:
            if not self.embedder:
                raise ValueError(
                    "Embedder not available in RAGStore to generate embeddings."
                )
            print(f"RAGStore: Generating embeddings for {len(texts)} texts...")
            embeddings_list = self.embedder.embed(texts)
            if embeddings_list is None or len(embeddings_list) != len(texts):
                raise ValueError(
                    "Embedding generation failed or returned an incorrect number of embeddings."
                )

        self.collection.add(
            ids=ids,
            embeddings=embeddings_list,
            documents=texts,
            metadatas=metadatas,
        )
        print(
            f"Added/updated {len(ids)} items to collection '{self.collection.name}'. New count: {self.collection.count()}"
        )

    def query(
        self,
        query_text: str = None,
        query_embedding: list[float] = None,
        n_results: int = 5,
        filter_metadata: dict = None,
    ):
        """
        Queries the RAGStore for relevant documents based on a query text or embedding.

        Args:
            query_text (str, optional): The text query to search for. If provided and
                `query_embedding` is not given, an embedding will be generated using the
                store's embedder.
            query_embedding (list[float], optional): Precomputed embedding vector for the query.
                If provided, this will be used directly for the search.
            n_results (int, optional): The maximum number of results to return. Defaults to 5.
            filter_metadata (dict, optional): Metadata filters to apply to the search.

        Returns:
            dict: The query results from the collection, including metadatas, documents,
                and distances.

        Raises:
            ValueError: If neither `query_text` nor `query_embedding` is provided, or if
                embedding generation fails, or if the embedder is not available.
        """
        if not query_embedding and query_text:
            if not self.embedder:
                raise ValueError(
                    "Embedder not available in RAGStore to generate query embedding."
                )
            print(f"RAGStore: Generating embedding for query: '{query_text[:50]}...'")
            embedding_list = self.embedder.embed([query_text])
            if not embedding_list:
                raise ValueError("Query embedding generation failed.")
            query_embedding = embedding_list[0]

        if query_embedding:
            return self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=filter_metadata,
                include=["metadatas", "documents", "distances"],  # Request desired info
            )
        else:
            raise ValueError("Either query_text or query_embedding must be provided.")

    def delete_by_metadata(self, metadata_filter: dict):
        """
        Deletes entries from the collection that match the specified metadata filter.

        Args:
            metadata_filter (dict): A dictionary specifying the metadata criteria for deletion.
                If the filter is empty, no action is taken.

        Returns:
            None

        Side Effects:
            - Deletes matching entries from the collection.
            - Prints a message indicating the result of the deletion attempt.
        """
        if not metadata_filter:
            print(
                "RAGStore: Delete_by_metadata called with empty filter. No action taken."
            )
            return
        self.collection.delete(where=metadata_filter)
        print(
            f"RAGStore: Attempted deletion with filter {metadata_filter}. New count: {self.collection.count()}"
        )

    def delete_by_ids(self, ids: list[str]):
        """
        Deletes items from the collection by their IDs.

        Args:
            ids (list[str]): A list of string IDs corresponding to the items to be deleted.

        Returns:
            None

        Notes:
            - If the provided list of IDs is empty, no action is taken and a message is printed.
            - After deletion, prints the number of items deleted and the new count of items in the collection.
        """
        if not ids:
            print("RAGStore: Delete_by_ids called with empty ID list. No action taken.")
            return
        self.collection.delete(ids=ids)
        print(
            f"RAGStore: Deleted {len(ids)} items by ID. New count: {self.collection.count()}"
        )
