from pathlib import Path
import queue
from types import TracebackType
from typing import Tuple, Type, Optional, List, Dict, Any
from ataraxai.praxis.modules.rag.ataraxai_embedder import AtaraxAIEmbedder
from ataraxai.praxis.modules.rag.resilient_indexer import start_rag_file_monitoring
from ataraxai.praxis.modules.rag.rag_store import RAGStore
from ataraxai.praxis.modules.rag.rag_manifest import RAGManifest
# from ataraxai.praxis.modules.rag.rag_updater import process_new_file
# from ataraxai.praxis.preferences_manager import PreferencesManager
from sentence_transformers import CrossEncoder
from chromadb import QueryResult
import numpy as np
from functools import lru_cache
import os
import threading
import logging
from ataraxai.praxis.modules.rag.rag_updater import rag_update_worker
from ataraxai.praxis.utils.rag_config_manager import RAGConfigManager


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
        """
        Initializes the AtaraxAIRAGManager instance.

        Args:
            rag_config_manager (RAGConfigManager): Manager for RAG configuration settings.
            app_data_root_path (Path): Root path for application data storage.
            core_ai_service (Any): Core AI service instance used by the manager.

        Attributes:
            app_data_root_path (Path): Root directory for storing RAG-related data.
            rag_config_manager (RAGConfigManager): Configuration manager for RAG settings.
            core_ai_service (Any): Reference to the core AI service.
            logger (logging.Logger): Logger for the manager.
            manifest_file_path (Path): Path to the RAG manifest JSON file.
            embedder (AtaraxAIEmbedder): Embedder instance for text embeddings.
            rag_store (RAGStore): RAGStore instance for managing knowledge storage.
            manifest (RAGManifest): Manifest manager for RAG documents.
            rag_use_reranking (bool): Whether to use reranking in retrieval.
            n_result (int): Number of results to retrieve initially.
            n_result_final (int): Number of final results after reranking/filtering.
            use_hyde (bool): Whether to use the HyDE technique.
            processing_queue (queue.Queue): Queue for processing tasks.
            file_observer: Observer for file changes (currently None).
            _cross_encoder: Cross-encoder model instance (lazy initialization).
            _cross_encoder_initialized (bool): Flag indicating if cross-encoder is initialized.

        Logs:
            Logs successful initialization of the AtaraxAIRAGManager.
        """
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
        """
        Checks the validity of the manifest against the RAG store.

        Returns:
            bool: True if the manifest is valid, False otherwise.

        Logs:
            - Info message indicating whether the manifest is valid or invalid.
            - Error message if an exception occurs during the validity check.
        """
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
        """
        Rebuilds the RAG (Retrieval-Augmented Generation) index from scratch using the specified directories.

        This method deletes the existing RAG index collection, creates a new one with the configured embedding function,
        clears the manifest, and performs an initial scan of the provided directories to repopulate the index.

        Args:
            directories_to_scan (Optional[List[str]]): List of directory paths to scan and index. If None or empty, the rebuild is aborted.

        Returns:
            bool: True if the index was rebuilt successfully, False otherwise.
        """
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
        """
        Scans the specified directories for files not present in the manifest and queues them for processing.

        Args:
            directories_to_scan (Optional[List[str]]): A list of directory paths to scan. If None or empty, no scanning is performed.

        Returns:
            int: The number of new files found and queued for processing.

        Logs:
            - Info when starting and completing the scan.
            - Warning if a directory does not exist.
            - Debug for each file queued for processing.
            - Error if an exception occurs during scanning.
        """
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
        """
        Starts monitoring the specified directories for file changes and initializes the RAG update worker thread.

        Args:
            watched_directories (Optional[List[str]]): List of directory paths to monitor for file changes.

        Returns:
            bool: True if file monitoring started successfully, False otherwise.

        Side Effects:
            - Starts or restarts a background thread for processing RAG updates.
            - Initializes or restarts a file observer to monitor the specified directories.
            - Logs informational and error messages.

        Notes:
            - If no directories are specified, logs a warning and returns False.
            - If a file observer or worker thread is already running, they are stopped and restarted as needed.
            - Any exceptions during initialization are logged and result in a return value of False.
        """
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
        """
        Stops the file monitoring process if it is currently active.

        This method checks if the file observer is running. If so, it stops and joins the observer thread,
        logging the outcome. If no file monitoring is active, it logs that information. Any exceptions
        encountered during the process are caught and logged as errors.
        """
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
        """
        Lazily initializes and returns a CrossEncoder instance for re-ranking, if enabled.

        Returns:
            Optional[CrossEncoder]: The initialized CrossEncoder instance if re-ranking is enabled and initialization succeeds; otherwise, None.

        Raises:
            Logs any exceptions encountered during CrossEncoder initialization.
        """
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
        """
        Generates a hypothetical document that provides a clear and direct answer to the given query text.

        This method constructs a prompt using the provided query and requests a completion from the core AI service.
        If the generation is successful, the hypothetical document is returned. If an error occurs during generation,
        the original query text is returned as a fallback.

        Args:
            query_text (str): The question or query for which a hypothetical answer document should be generated.

        Returns:
            str: The generated hypothetical document or the original query text if an error occurs.
        """
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
        """
        Re-rank a list of documents based on their relevance to a given query using a cross-encoder model.

        Args:
            query_text (str): The query string to compare against the documents.
            documents (List[str]): A list of document strings to be re-ranked.

        Returns:
            List[str]: The list of documents re-ordered by relevance to the query. If the cross-encoder
                model is not loaded or an error occurs, returns the documents in their original order.

        Logs:
            - Warning if the cross-encoder model is not loaded.
            - Error if an exception occurs during re-ranking.
            - Debug message indicating the number of documents re-ranked.
        """
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
        """
        Queries the knowledge base using the provided query text and optional metadata filters.

        Depending on the retrieval strategy, either a simple or advanced query method is used.
        Logs the query and handles exceptions gracefully.

        Args:
            query_text (str): The text query to search in the knowledge base.
            filter_metadata (Optional[Dict[Any, Any]]): Optional dictionary of metadata to filter the search.

        Returns:
            List[str]: A list of strings representing the retrieved knowledge entries. Returns an empty list if an error occurs.

        Raises:
            ValueError: If the query_text is empty or only whitespace.
        """
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
        """
        Determines whether advanced retrieval techniques should be used.

        Returns:
            bool: True if either the HyDE method or reranking is enabled; False otherwise.
        """
        return self.use_hyde or self.rag_use_reranking

    def _simple_query(
        self, query_text: str, filter_metadata: Optional[Dict[Any, Any]]
    ) -> List[str]:
        """
        Executes a simple query against the RAG store and returns a list of document strings.

        Args:
            query_text (str): The text query to search for relevant documents.
            filter_metadata (Optional[Dict[Any, Any]]): Optional metadata filters to apply to the query.

        Returns:
            List[str]: A list of document strings matching the query. Returns an empty list if no documents are found.
        """
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
        """
        Executes an advanced query against the RAG (Retrieval-Augmented Generation) store, optionally using hypothetical document generation (HyDE) and reranking.

        Args:
            query_text (str): The input query string to search for relevant documents.
            filter_metadata (Optional[Dict[Any, Any]]): Optional metadata filters to apply to the search.

        Returns:
            List[str]: A list of relevant document strings retrieved and optionally reranked based on the query.

        Notes:
            - If HyDE is enabled (`self.use_hyde`), a hypothetical document is generated from the query to improve retrieval.
            - If reranking is enabled (`self.rag_use_reranking`), the initially retrieved documents are reranked for relevance.
            - The number of final results is limited by `self.n_result_final`.
            - Logs debug information about the retrieval process.
        """
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
        """
        Retrieve various statistics and status information about the RAG manager.

        Returns:
            Dict[str, Any]: A dictionary containing the following keys:
                - "total_documents" (int): The total number of documents in the RAG store collection.
                - "manifest_files" (int): The number of files in the manifest, or 0 if unavailable.
                - "use_hyde" (bool): Whether the HYDE feature is enabled.
                - "use_reranking" (bool): Whether reranking is enabled.
                - "n_result" (int): The number of results to return.
                - "n_result_final" (int): The final number of results after processing.
                - "monitoring_active" (bool): Whether the file observer monitoring is active.
                - "worker_thread_active" (bool): Whether the worker thread is currently active.

        If an error occurs, logs the error and returns an empty dictionary.
        """
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
