from pathlib import Path
import queue
from types import TracebackType
from typing import Tuple, Type, Optional, List, Dict, Any
from ataraxai.app_logic.modules.rag.ataraxai_embedder import AtaraxAIEmbedder
from ataraxai.app_logic.modules.rag.resilient_indexer import start_rag_file_monitoring
from ataraxai.app_logic.modules.rag.rag_store import RAGStore
from ataraxai.app_logic.modules.rag.rag_manifest import RAGManifest
# from ataraxai.app_logic.modules.rag.rag_updater import process_new_file
# from ataraxai.app_logic.preferences_manager import PreferencesManager
from sentence_transformers import CrossEncoder
from chromadb import QueryResult
import numpy as np
from functools import lru_cache
import os
import threading
import logging
from ataraxai.app_logic.modules.rag.rag_updater import rag_update_worker
from ataraxai.app_logic.utils.rag_config_manager import RAGConfigManager


class AtaraxAIRAGManager:
    """
    Manages RAG (Retrieval-Augmented Generation) operations including document indexing,
    file monitoring, and advanced retrieval with HyDE and re-ranking capabilities.
    """

    def __init__(
        self,
        rag_config_manager: RAGConfigManager,
        app_data_root_path: Path,
        core_ai_service: Any,
    ):
        self.app_data_root_path = app_data_root_path
        self.rag_config_manager = rag_config_manager
        self.core_ai_service = core_ai_service

        self.logger = logging.getLogger(__name__)

        rag_store_db_path = self.app_data_root_path / "rag_chroma_store"
        rag_store_db_path.mkdir(parents=True, exist_ok=True)
        self.manifest_file_path = self.app_data_root_path / "rag_manifest.json"

        self.embedder = AtaraxAIEmbedder(
            model_name=self.rag_config_manager.get(
                "rag_embedder_model", "sentence-transformers/all-MiniLM-L6-v2"
            )  # type: ignore
        )

        self.rag_store = RAGStore(
            db_path_str=str(rag_store_db_path),
            collection_name="ataraxai_knowledge",
            embedder=self.embedder,
        )

        self.manifest = RAGManifest(self.manifest_file_path)

        self.rag_use_reranking: bool = bool(
            self.rag_config_manager.get("rag_use_reranking", False)
        )
        self.n_result: int = self.rag_config_manager.get("n_result", 5)  # type: ignore
        self.n_result_final: int = self.rag_config_manager.get("n_result_final", 3)  # type: ignore
        self.use_hyde: bool = self.rag_config_manager.get("use_hyde", True)  # type: ignore

        self.processing_queue: queue.Queue[Dict[str, Any]] = queue.Queue()
        self.file_observer = None
        # self.worker_thread : threading.Thread = None

        self._cross_encoder = None
        self._cross_encoder_initialized = False

        self.logger.info("AtaraxAIRAGManager initialized successfully.")

    def check_manifest_validity(self) -> bool:
        try:
            is_valid = self.manifest.is_valid(self.rag_store)
            self.logger.info(
                f"Manifest validity check: {'Valid' if is_valid else 'Invalid'}"
            )
            return is_valid
        except Exception as e:
            self.logger.error(f"Error checking manifest validity: {e}")
            return False

    def rebuild_index(self, directories_to_scan: Optional[List[str]]) -> bool:
        self.logger.info("Rebuilding RAG index from scratch...")
        if not directories_to_scan:
            self.logger.warning(
                "Unable to rebuild RAG index: No directories specified."
            )
            return False
        try:
            self.rag_store.client.delete_collection(name=self.rag_store.collection_name)
            self.rag_store.collection = self.rag_store.client.get_or_create_collection(
                name=self.rag_store.collection_name,
                embedding_function=self.embedder,
                metadata={"hnsw:space": "cosine"},
            )
            self.manifest.clear()
            self.perform_initial_scan(directories_to_scan)
            self.logger.info("RAG index rebuild completed successfully.")
            return True

        except Exception as e:
            self.logger.error(f"Error during index rebuild: {e}")
            return False

    def perform_initial_scan(self, directories_to_scan: Optional[List[str]]) -> int:
        if not directories_to_scan:
            self.logger.info("No directories to scan.")
            return 0

        files_found = 0
        self.logger.info(
            f"Performing initial scan of directories: {directories_to_scan}"
        )

        for directory in directories_to_scan:
            if not os.path.exists(directory):
                self.logger.warning(f"Directory does not exist: {directory}")
                continue

            try:
                for root, _, files in os.walk(directory):
                    for filename in files:
                        file_path = os.path.join(root, filename)
                        if not self.manifest.is_file_in_manifest(file_path):
                            task = {"event_type": "created", "path": file_path}
                            self.processing_queue.put(task)
                            files_found += 1
                            self.logger.debug(f"Queued file for processing: {filename}")

            except Exception as e:
                self.logger.error(f"Error scanning directory {directory}: {e}")

        self.logger.info(
            f"Initial scan completed. Found {files_found} files to process."
        )
        return files_found

    def start_file_monitoring(self, watched_directories: Optional[List[str]]) -> bool:
        if not watched_directories:
            self.logger.warning("No directories specified for monitoring.")
            return False
        try:
            if self.file_observer and self.file_observer.is_alive():
                self.file_observer.stop()
                self.file_observer.join()
            if (
                not hasattr(self, "worker_thread")
                or not self.worker_thread
                or not self.worker_thread.is_alive()
            ):
                self.worker_thread : threading.Thread = threading.Thread(
                    target=rag_update_worker,
                    args=(
                        self.processing_queue,
                        self.manifest,
                        self.rag_store,
                        self.rag_config_manager.get("rag_chunk_config", {}),
                    ),
                    daemon=True,
                )
                self.worker_thread.start()
                self.logger.info("RAG update worker thread started.")

            # Start file monitoring
            self.file_observer = start_rag_file_monitoring(
                paths_to_watch=watched_directories,
                manifest=self.manifest,
                rag_store=self.rag_store,
                chunk_config=self.rag_config_manager.get("rag_chunk_config", {}),  # type: ignore
            )

            self.logger.info("File monitoring started successfully.")
            return True

        except Exception as e:
            self.logger.error(f"Error starting file monitoring: {e}")
            return False

    def stop_file_monitoring(self) -> None:
        try:
            if self.file_observer and self.file_observer.is_alive():
                self.file_observer.stop()
                self.file_observer.join()
                self.logger.info("File monitoring stopped successfully.")
            else:
                self.logger.info("No active file monitoring to stop.")
        except Exception as e:
            self.logger.error(f"Error stopping file monitoring: {e}")

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ):
        self.cleanup()

    def cleanup(self) -> None:
        self.logger.info("Cleaning up RAG Manager resources...")
        self.stop_file_monitoring()

        if hasattr(self, "_cross_encoder") and self._cross_encoder:
            del self._cross_encoder
            self._cross_encoder = None
            self._cross_encoder_initialized = False

    @property
    def cross_encoder(self) -> Optional[CrossEncoder]:
        """Lazy initialization of cross-encoder for re-ranking."""
        if not self._cross_encoder_initialized:
            self._cross_encoder_initialized = True
            if self.rag_use_reranking and self.core_ai_service:
                try:
                    cross_encoder_model: str = self.rag_config_manager.get(
                        "rag_cross_encoder_model",
                        "cross-encoder/ms-marco-MiniLM-L-6-v2",
                    )  # type: ignore
                    self._cross_encoder = CrossEncoder(cross_encoder_model)
                    self.logger.info(
                        f"Cross-encoder model loaded: {cross_encoder_model}"
                    )
                except Exception as e:
                    self.logger.error(f"Error loading cross-encoder model: {e}")
                    self._cross_encoder = None
            else:
                self._cross_encoder = None
        return self._cross_encoder

    @lru_cache(maxsize=128)
    def _generate_hypothetical_document(self, query_text: str) -> str:
        self.logger.debug("Generating hypothetical document for HyDE...")
        prompt = f"""Please write a short paragraph that provides a clear and direct answer to the following question:
                    Question: {query_text}
                    Answer:"""

        try:
            hypothetical_doc = self.core_ai_service.generate_completion(prompt)
            self.logger.debug("Successfully generated hypothetical document.")
            return hypothetical_doc
        except Exception as e:
            self.logger.error(f"Error generating hypothetical document: {e}")
            return query_text

    def _rerank_documents(self, query_text: str, documents: List[str]) -> List[str]:
        if not self.cross_encoder:
            self.logger.warning(
                "Re-ranking was requested, but the cross-encoder model is not loaded. "
                "Returning original order."
            )
            return documents

        if not documents:
            return documents

        try:
            query_doc_pairs: List[Tuple[str, str]] = [
                (query_text, doc) for doc in documents
            ]

            scores: np.ndarray = self.cross_encoder.predict(  # type: ignore
                query_doc_pairs, convert_to_numpy=True
            )

            scored_docs: List[Tuple[float, str]] = sorted(
                zip(scores, documents), key=lambda x: x[0], reverse=True
            )

            reranked_docs = [doc for score, doc in scored_docs]
            self.logger.debug(f"Re-ranked {len(documents)} documents.")
            return reranked_docs

        except Exception as e:
            self.logger.error(f"Error during document re-ranking: {e}")
            return documents

    def query_knowledge(
        self, query_text: str, filter_metadata: Optional[Dict[Any, Any]] = None
    ) -> List[str]:
        if not query_text or not query_text.strip():
            raise ValueError("Query text cannot be empty")

        self.logger.debug(f"Querying knowledge base with query: '{query_text[:50]}...'")

        try:
            if not self._should_use_advanced_retrieval():
                return self._simple_query(query_text, filter_metadata)
            else:
                return self._advanced_query(query_text, filter_metadata)
        except Exception as e:
            self.logger.error(f"Error during knowledge query: {e}")
            return []

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
            self.logger.debug(f"Simple query returned {len(documents[0])} documents.")
            return documents[0]

        self.logger.debug("Simple query returned no documents.")
        return []

    def _advanced_query(
        self, query_text: str, filter_metadata: Optional[Dict[Any, Any]]
    ) -> List[str]:
        search_query = query_text

        if self.use_hyde:
            search_query = self._generate_hypothetical_document(query_text)

        n_initial_retrieval = 20 if self.rag_use_reranking else self.n_result_final

        results: QueryResult = self.rag_store.query(
            query_text=search_query,
            n_results=n_initial_retrieval,
            filter_metadata=filter_metadata,
        )

        documents = results.get("documents") if results else None
        initial_docs: List[str] = documents[0] if documents else []

        if not initial_docs:
            self.logger.debug("Advanced query returned no documents.")
            return []

        final_docs: List[str] = initial_docs
        if self.rag_use_reranking:
            final_docs = self._rerank_documents(query_text, initial_docs)

        result = final_docs[: self.n_result_final]
        self.logger.debug(
            f"Advanced query returned {len(result)} documents after processing."
        )
        return result

    def get_stats(self) -> Dict[str, Any]:
        try:
            collection_count = self.rag_store.collection.count()
            manifest_files = (
                len(self.manifest.get_all_files())
                if hasattr(self.manifest, "get_all_files")
                else 0
            )

            return {
                "total_documents": collection_count,
                "manifest_files": manifest_files,
                "use_hyde": self.use_hyde,
                "use_reranking": self.rag_use_reranking,
                "n_result": self.n_result,
                "n_result_final": self.n_result_final,
                "monitoring_active": self.file_observer is not None
                and self.file_observer.is_alive(),
                "worker_thread_active": self.worker_thread is not None
                and self.worker_thread.is_alive(),
            }
        except Exception as e:
            self.logger.error(f"Error getting RAG stats: {e}")
            return {}
