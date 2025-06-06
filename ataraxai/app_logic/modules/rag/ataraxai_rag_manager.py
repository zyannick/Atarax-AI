from pathlib import Path
import chromadb
from .rag_store import AtaraxAIEmbedder
from platformdirs import user_data_dir, user_config_dir
import json

from .resilient_indexer import start_rag_file_monitoring
from .rag_store import RAGStore
from .rag_manifest import RAGManifest
from ataraxai import __version__

APP_NAME = "AtaraxAI"
APP_AUTHOR = "AtaraxAI"


class AtaraxAIRAGManager:
    def __init__(self, core_ai_service_instance=None):
        self.app_config_dir = Path(
            user_config_dir(appname=APP_NAME, appauthor=APP_AUTHOR)
        )
        self.app_data_dir = Path(user_data_dir(appname=APP_NAME, appauthor=APP_AUTHOR))

        self.app_config_dir.mkdir(parents=True, exist_ok=True)
        self.app_data_dir.mkdir(parents=True, exist_ok=True)

        self.first_launch_marker_file = (
            self.app_config_dir / ".ataraxai_setup_" + __version__ + "_complete"
        )

        rag_store_db_path = self.app_data_dir / "rag_chroma_store"
        rag_store_db_path.mkdir(parents=True, exist_ok=True)

        self.manifest_file_path = self.app_data_dir / "rag_manifest.json"

        self.embedder = AtaraxAIEmbedder(core_ai_service=core_ai_service_instance)
        self.rag_store = RAGStore(
            db_path_str=str(rag_store_db_path),
            collection_name="ataraxai_knowledge",
            embedder_instance=self.embedder,
        )
        self.manifest = RAGManifest(self.manifest_file_path)

        self.file_observer = None
        print("AtaraxAIRAGManager initialized.")

    def start_file_monitoring(self, watched_directories: list[str]):
        if self.file_observer and self.file_observer.is_alive():
            self.file_observer.stop()
            self.file_observer.join()

        if watched_directories:
            self.file_observer = start_rag_file_monitoring(
                watched_directories=watched_directories,
                manifest=self.manifest,
                rag_store=self.rag_store,
                embedder=self.embedder,
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

    def query_knowledge(
        self, query_text: str, n_results: int = 3, filter_metadata: dict = None
    ):
        """High-level method to query the RAG store."""
        return self.rag_store.query(
            query_text=query_text, n_results=n_results, filter_metadata=filter_metadata
        )
