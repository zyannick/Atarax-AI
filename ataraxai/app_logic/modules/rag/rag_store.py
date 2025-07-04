import chromadb
from pathlib import Path
from typing import Optional, Any

from ataraxai.app_logic.modules.rag.ataraxai_embedder import AtaraxAIEmbedder
from typing import Dict, List, Union, Mapping

MetadataDict = Mapping[str, Union[str, int, float, bool, None]]


class RAGStore:
    def __init__(self, db_path_str: str, collection_name: str, embedder: AtaraxAIEmbedder):
        self.db_path = Path(db_path_str)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name


        if not embedder:
            raise ValueError("An embedder instance must be provided to RAGStore.")
        self.embedder = embedder

        self.client = chromadb.PersistentClient(path=str(self.db_path))

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedder,
            metadata={"hnsw:space": "cosine"},
        )
        print(
            f"RAGStore: Collection '{self.collection.name}' loaded/created with {self.collection.count()} items."
        )

    def add_chunks(
        self,
        ids: List[str],
        texts: List[str],
        metadatas: List[MetadataDict],
    ):

        if len(metadatas) != len(texts):
            raise ValueError("Length of metadatas must match length of texts.")


        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
        )
        print(
            f"Added/updated {len(ids)} items to collection '{self.collection.name}'. New count: {self.collection.count()}"
        )

    def query(
        self,
        query_text: str, 
        n_results: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ):
        if not query_text:
            raise ValueError("query_text must be provided.")
            
        print(f"RAGStore: Querying with text: '{query_text[:50]}...'")


        return self.collection.query(
            query_texts=[query_text], 
            n_results=n_results,
            where=filter_metadata,
            include=["metadatas", "documents", "distances"],
        )

    def delete_by_metadata(self, metadata_filter: Dict[str, Any]):
        if not metadata_filter:
            print("RAGStore: Delete_by_metadata called with empty filter. No action taken.")
            return
        self.collection.delete(where=metadata_filter)
        print(
            f"RAGStore: Attempted deletion with filter {metadata_filter}. New count: {self.collection.count()}"
        )

    def delete_by_ids(self, ids: List[str]):
        if not ids:
            print("RAGStore: Delete_by_ids called with empty ID list. No action taken.")
            return
        self.collection.delete(ids=ids)
        print(
            f"RAGStore: Deleted {len(ids)} items by ID. New count: {self.collection.count()}"
        )