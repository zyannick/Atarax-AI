from pathlib import Path
from typing import Tuple
from ataraxai.app_logic.modules.rag.rag_store import AtaraxAIEmbedder
from ataraxai.app_logic.modules.rag.resilient_indexer import start_rag_file_monitoring
from ataraxai.app_logic.modules.rag.rag_store import RAGStore
from ataraxai.app_logic.modules.rag.rag_manifest import RAGManifest
from ataraxai.app_logic.preferences_manager import PreferencesManager
from typing_extensions import Optional, List, Dict, Any
from sentence_transformers import CrossEncoder
from chromadb import QueryResult
import numpy as np
from functools import lru_cache


class AtaraxAIRAGManager:
    def __init__(
        self,
        preferences_manager_instance: PreferencesManager,
        app_data_root_path: Path,
        core_ai_service: Any,
    ):
        self.app_data_root_path = app_data_root_path
        self.preferences_manager_instance = preferences_manager_instance
        self.llm_engine = core_ai_service

        rag_store_db_path = self.app_data_root_path / "rag_chroma_store"
        rag_store_db_path.mkdir(parents=True, exist_ok=True)

        self.manifest_file_path = self.app_data_root_path / "rag_manifest.json"

        self.embedder = AtaraxAIEmbedder(
            model_name=self.preferences_manager_instance.get(  # type: ignore
                "rag_embedder_model", "sentence-transformers/all-MiniLM-L6-v2"
            )
        )
        self.rag_store = RAGStore(
            db_path_str=str(rag_store_db_path),
            collection_name="ataraxai_knowledge",
            embedder=self.embedder,  # type: ignore
        )
        self.manifest = RAGManifest(self.manifest_file_path)

        self.rag_use_reranking: bool = bool(self.preferences_manager_instance.get("rag_use_reranking", False))  # type: ignore

        self.n_result: int = int(self.preferences_manager_instance.get("n_result", 5))  # type: ignore
        self.n_result_final: int = int(self.preferences_manager_instance.get("n_result_final", 3))  # type: ignore
        self.use_hyde: bool = bool(self.preferences_manager_instance.get("use_hyde", True))  # type: ignore

        self.file_observer = None

        print("AtaraxAIRAGManager initialized.")

    def start_file_monitoring(self, watched_directories: List[str]):
        if self.file_observer and self.file_observer.is_alive():
            self.file_observer.stop()
            self.file_observer.join()

        if watched_directories:
            self.file_observer = start_rag_file_monitoring(
                paths_to_watch=watched_directories,
                manifest=self.manifest,
                chroma_collection=self.rag_store.collection,
            )
            print("File monitoring started via AtaraxAIRAGManager.")
        else:
            print("No directories specified to watch for RAG updates.")

    def stop_file_monitoring(self):
        if self.file_observer and self.file_observer.is_alive():
            self.file_observer.stop()
            self.file_observer.join()
            print("File monitoring stopped via AtaraxAIRAGManager.")
        else:
            print("No active file monitoring to stop.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # type: ignore
        self.cleanup()

    def cleanup(self):
        self.stop_file_monitoring()
        if self._cross_encoder:
            del self._cross_encoder

    @property
    def cross_encoder(self):
        if not hasattr(self, "_cross_encoder"):
            if self.rag_use_reranking and self.llm_engine:
                cross_encoder_model: str = self.preferences_manager_instance.get(  # type: ignore
                    "rag_cross_encoder_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"
                )
                self._cross_encoder = CrossEncoder(cross_encoder_model)
            else:
                self._cross_encoder = None
        return self._cross_encoder

    @lru_cache(maxsize=128)
    def _generate_hypothetical_document(self, query_text: str) -> str:
        print("Generating hypothetical document for HyDE...")
        prompt = f"Please write a short paragraph that provides a clear and direct answer to the following question:\n\nQuestion: {query_text}\n\nAnswer:"
        try:
            hypothetical_doc = self.llm_engine.generate_completion(prompt)
            return hypothetical_doc
        except Exception as e:
            print(f"Error generating hypothetical document: {e}")
            return query_text

    def _rerank_documents(self, query_text: str, documents: List[str]) -> List[str]:
        if not self.cross_encoder:
            print(
                "Warning: Re-ranking was requested, but the cross-encoder model is not loaded. Returning original order."
            )
            return documents

        query_doc_pairs: List[tuple[str, str]] = [
            (query_text, doc) for doc in documents
        ]

        scores: np.ndarray = self.cross_encoder.predict(query_doc_pairs, convert_to_numpy=True)  # type: ignore

        scored_docs: List[Tuple[Any, str]] = sorted(
            zip(scores, documents), key=lambda x: x[0], reverse=True
        )

        return [doc for score, doc in scored_docs]

    def query_knowledge(
        self, query_text: str, filter_metadata: Optional[Dict[Any, Any]] = None
    ) -> List[str]:
        if not query_text or not query_text.strip():
            raise ValueError("Query text cannot be empty")

        if not self._should_use_advanced_retrieval():
            return self._simple_query(query_text, filter_metadata)

        return self._advanced_query(query_text, filter_metadata)

    def _should_use_advanced_retrieval(self) -> bool:
        return self.use_hyde or self.rag_use_reranking

    def _simple_query(
        self, query_text: str, filter_metadata: Optional[Dict[Any, Any]]
    ) -> List[str]:
        results = self.rag_store.query(
            query_text=query_text,
            n_results=self.n_result,
            filter_metadata=filter_metadata,
        )
        documents = results.get("documents") if results else None
        if documents and len(documents) > 0:
            return documents[0]
        return []

    def _advanced_query(
        self, query_text: str, filter_metadata: Optional[Dict[Any, Any]]
    ) -> List[str]:
        search_query = query_text
        if self.use_hyde:
            search_query = self._generate_hypothetical_document(query_text)

        n_initial_retrieval = 20 if self.rag_use_reranking else self.n_result_final
        results: QueryResult = self.rag_store.query(
            query_text=search_query, n_results=n_initial_retrieval
        )

        documents = results.get("documents") if results else None
        initial_docs: List[str] = documents[0] if documents else []

        if not initial_docs:
            return []

        final_docs: List[str] = initial_docs
        if self.rag_use_reranking:
            final_docs = self._rerank_documents(query_text, initial_docs)

        return final_docs[: self.n_result_final]
