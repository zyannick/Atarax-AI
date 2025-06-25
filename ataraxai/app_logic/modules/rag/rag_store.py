import chromadb

from pathlib import Path
from typing import Optional
from platformdirs import user_data_dir
from ataraxai.app_logic.modules.rag.ataraxai_embedder import AtaraxAIEmbedder
from chromadb.api.types import Embeddings, Metadata, Document, Documents
from typing_extensions import List

APP_NAME = "AtaraxAI"
APP_AUTHOR = "AtaraxAI"

CHROMA_DATA_PATH = (
    Path(user_data_dir(appname=APP_NAME, appauthor=APP_AUTHOR)) / "chroma_db"
)
CHROMA_DATA_PATH.mkdir(parents=True, exist_ok=True)

COLLECTION_NAME = "ataraxai_knowledge"


class RAGStore:
    def __init__(
        self, db_path_str: str, collection_name: str, embedder: AtaraxAIEmbedder
    ):
        self.db_path = Path(db_path_str)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.embedder = embedder if embedder else AtaraxAIEmbedder()
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
        metadatas: List[Metadata],
        embeddings_list: Optional[Embeddings] = None,
    ):
        
        if len(metadatas) != len(texts):
            raise ValueError(
                "Length of metadatas must match length of texts. "
                f"Got {len(metadatas)} metadata entries for {len(texts)} texts."
            )
        
        if not embeddings_list:
            if not self.embedder:
                raise ValueError(
                    "Embedder not available in RAGStore to generate embeddings."
                )
            print(f"RAGStore: Generating embeddings for {len(texts)} texts...")
            embeddings_list = self.embedder(texts)
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
        query_text: Optional[str] = None,
        query_embedding: Optional[Embeddings] = None,
            n_results: int = 5,
            filter_metadata: Optional[dict] = None,
    ):
        if not query_embedding and query_text:
            if not self.embedder:
                raise ValueError(
                    "Embedder not available in RAGStore to generate query embedding."
                )
            print(f"RAGStore: Generating embedding for query: '{query_text[:50]}...'")
            document_query = Document(query_text)
            documents_query = Documents([document_query])
            query_embedding = self.embedder(documents_query)

        if query_embedding:
            return self.collection.query(
                query_embeddings=query_embedding,
                n_results=n_results,
                where=filter_metadata,
                include=["metadatas", "documents", "distances"],  # Request desired info
            )
        else:
            raise ValueError("Either query_text or query_embedding must be provided.")

    def delete_by_metadata(self, metadata_filter: dict):
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
        if not ids:
            print("RAGStore: Delete_by_ids called with empty ID list. No action taken.")
            return
        self.collection.delete(ids=ids)
        print(
            f"RAGStore: Deleted {len(ids)} items by ID. New count: {self.collection.count()}"
        )
