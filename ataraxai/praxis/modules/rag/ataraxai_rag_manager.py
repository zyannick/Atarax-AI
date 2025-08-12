import asyncio
from pathlib import Path
from async_lru import alru_cache
from typing import Optional, List, Dict, Any
from ataraxai.praxis.modules.rag.ataraxai_embedder import AtaraxAIEmbedder
from ataraxai.praxis.modules.rag.rag_store import RAGStore
from ataraxai.praxis.modules.rag.rag_manifest import RAGManifest

from sentence_transformers import CrossEncoder
from chromadb import QueryResult
import numpy as np
from functools import lru_cache
from asyncio import Queue
from typing import Optional

import logging
from ataraxai.praxis.modules.rag.rag_updater import rag_update_worker_async
from ataraxai.praxis.utils.configs.rag_config_manager import RAGConfigManager
from ataraxai.praxis.modules.rag.resilient_indexer import ResilientIndexer
from ataraxai.praxis.modules.rag.watch_directories_manager import (
    WatchedDirectoriesManager,
)


class AtaraxAIRAGManager:

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
            model_name=self.rag_config_manager.config.rag_embedder_model
        )
        self.rag_store = RAGStore(
            db_path_str=str(rag_store_db_path),
            collection_name="ataraxai_knowledge",
            embedder=self.embedder,
        )
        self.manifest = RAGManifest(self.manifest_file_path)
        self.processing_queue: Queue[Dict[str, Any]] = Queue()

        self.rag_use_reranking = self.rag_config_manager.config.rag_use_reranking
        self.n_result: int = self.rag_config_manager.config.rag_n_result
        self.n_result_final: int = self.rag_config_manager.config.rag_n_result_final
        self.use_hyde: bool = self.rag_config_manager.config.rag_use_hyde

        self.directory_manager = WatchedDirectoriesManager(
            rag_config_manager=self.rag_config_manager,
            manifest=self.manifest,
            processing_queue=self.processing_queue,
            logger=self.logger,
        )
        self.file_watcher_manager = ResilientIndexer(
            rag_config_manager=self.rag_config_manager,
            processing_queue=self.processing_queue,
            logger=self.logger,
        )

        self._rag_worker_task: Optional[asyncio.Task[Any]] = None
        self._cross_encoder: Optional[CrossEncoder] = None
        self._cross_encoder_lock = asyncio.Lock()
        self.logger.info("Async AtaraxAIRAGManager initialized successfully.")

    async def check_manifest_validity(self) -> bool:
        try:
            is_valid = await asyncio.to_thread(self.manifest.is_valid, self.rag_store)
            self.logger.info(f"Manifest validity check: {'Valid' if is_valid else 'Invalid'}")
            return is_valid
        except Exception as e:
            self.logger.error(f"Error checking manifest validity: {e}")
            return False

    async def health_check(self) -> bool:
        try:
            await asyncio.to_thread(self.rag_store.collection.count)
            return True
        except Exception:
            return False
        

    async def start(self):
        self.logger.info("Starting RAG manager services...")
        self._rag_worker_task = asyncio.create_task(
            rag_update_worker_async(
                processing_queue=self.processing_queue,
                manifest=self.manifest,
                rag_store=self.rag_store,
                chunk_config=self.rag_config_manager.config.model_dump(),
            )
        )
        await self.file_watcher_manager.start()

        watch_dirs = self.rag_config_manager.config.rag_watched_directories
        if watch_dirs:
            await self.directory_manager.add_directories(watch_dirs)

        self.logger.info("RAG manager services started successfully.")

    async def rebuild_index(self):
        watch_dirs = self.rag_config_manager.config.rag_watched_directories
        if not watch_dirs:
            self.logger.warning(
                "No watched directories configured for RAG index rebuild."
            )
            return False
        try:
            await asyncio.to_thread(
                self.rag_store.client.delete_collection,
                name=self.rag_store.collection_name,
            )
            self.rag_store.collection = await asyncio.to_thread(
                self.rag_store.client.get_or_create_collection,
                name=self.rag_store.collection_name,
                embedding_function=self.embedder,
                metadata={"hnsw:space": "cosine"},
            )
            self.manifest.clear()
            await self.directory_manager.add_directories(watch_dirs)
            self.logger.info("RAG index rebuild completed successfully.")
            return True

        except Exception as e:
            self.logger.error(f"Error during index rebuild: {e}")
            return False

    async def stop(self):
        self.logger.info("Stopping RAG manager services...")
        await self.file_watcher_manager.stop()
        if self._rag_worker_task and not self._rag_worker_task.done():
            await self.processing_queue.put({"event_type": "stop"})
            await self._rag_worker_task
        self.logger.info("RAG manager services stopped.")

    async def add_watch_directories(self, directories: List[str]) -> bool:
        return await self.directory_manager.add_directories(set(directories))

    async def remove_watch_directories(self, directories: List[str]) -> bool:
        return await self.directory_manager.remove_directories(set(directories))

    async def query_knowledge(
        self, query_text: str, filter_metadata: Optional[Dict] = None
    ) -> List[str]:
        if not query_text or not query_text.strip():
            raise ValueError("Query text cannot be empty")

        use_advanced = (
            self.rag_config_manager.config.rag_use_reranking
            or self.rag_config_manager.config.rag_use_hyde
        )
        try:
            if use_advanced:
                return await self._advanced_query(query_text, filter_metadata)
            else:
                return await self._simple_query(query_text, filter_metadata)
        except Exception as e:
            self.logger.error(f"Error during knowledge query: {e}", exc_info=True)
            return []

    async def _simple_query(
        self, query_text: str, filter_metadata: Optional[Dict[str, Any]]
    ) -> List[str]:
        results: QueryResult = await asyncio.to_thread(
            self.rag_store.query,
            query_text=query_text,
            n_results=self.rag_config_manager.config.rag_n_result,
            filter_metadata=filter_metadata,
        )
        documents = results.get("documents", [[]])
        return documents[0] if documents else []

    async def _advanced_query(
        self, query_text: str, filter_metadata: Optional[Dict[str, Any]]
    ) -> List[str]:
        search_query = query_text
        if self.rag_config_manager.config.rag_use_hyde:
            search_query = await self._generate_hypothetical_document(query_text)

        n_initial = (
            20
            if self.rag_config_manager.config.rag_use_reranking
            else self.rag_config_manager.config.rag_n_result_final
        )

        results: QueryResult = await asyncio.to_thread(
            self.rag_store.query,
            query_text=search_query,
            n_results=n_initial,
            filter_metadata=filter_metadata,
        )

        documents = results.get("documents", [[]])
        initial_docs = documents[0] if documents else []

        if not initial_docs:
            return []

        final_docs = initial_docs
        if self.rag_config_manager.config.rag_use_reranking:
            final_docs = await self._rerank_documents(query_text, initial_docs)

        return final_docs[: self.rag_config_manager.config.rag_n_result_final]

    @alru_cache(maxsize=128)
    async def _generate_hypothetical_document(self, query_text: str) -> str:
        prompt = f"Please write a short paragraph that provides a clear and direct answer to the following question:\nQuestion: {query_text}\nAnswer:"
        try:
            hypothetical_doc = await asyncio.to_thread(
                self.core_ai_service.generate_completion, prompt
            )
            return hypothetical_doc or query_text
        except Exception as e:
            self.logger.error(f"Error generating hypothetical document: {e}")
            return query_text

    async def get_cross_encoder(self) -> Optional[CrossEncoder]:
        if (
            self._cross_encoder is None
            and self.rag_config_manager.config.rag_use_reranking
        ):
            async with self._cross_encoder_lock:
                if self._cross_encoder is None:
                    self.logger.info("Initializing cross-encoder model...")
                    try:
                        model_name = (
                            self.rag_config_manager.config.rag_cross_encoder_model
                        )
                        self._cross_encoder = await asyncio.to_thread(
                            CrossEncoder, model_name
                        )
                        self.logger.info(f"Cross-encoder '{model_name}' loaded.")
                    except Exception as e:
                        self.logger.error(f"Failed to load cross-encoder model: {e}")
        return self._cross_encoder

    async def _rerank_documents(
        self, query_text: str, documents: List[str]
    ) -> List[str]:
        cross_encoder = await self.get_cross_encoder()
        if not cross_encoder or not documents:
            return documents

        query_doc_pairs = [(query_text, doc) for doc in documents]

        scores: List[float] = await asyncio.to_thread(  # type: ignore
            cross_encoder.predict, query_doc_pairs, convert_to_numpy=True  # type: ignore
        )

        scored_docs = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)  # type: ignore
        return [doc for _, doc in scored_docs]  # type: ignore
