from watchdog.observers import Observer
from watchdog.events import (
    FileSystemEventHandler,
    DirCreatedEvent,
    FileCreatedEvent,
    DirModifiedEvent,
    FileModifiedEvent,
    DirDeletedEvent,
    FileDeletedEvent,
    DirMovedEvent,
    FileMovedEvent,
)
from pathlib import Path
import queue
from ataraxai.app_logic.modules.rag.rag_manifest import RAGManifest
from typing_extensions import Union
from typing_extensions import Dict, Any
import threading

from ataraxai.app_logic.modules.rag.rag_store import RAGStore
from ataraxai.app_logic.modules.rag.rag_updater import rag_update_worker


class ResilientFileIndexer(FileSystemEventHandler):
    def __init__(self,  processing_queue: queue.Queue):
        self.processing_queue: queue.Queue[Dict[str, Any]] = processing_queue

    def on_created(self, event: Union[DirCreatedEvent, FileCreatedEvent]):
        if not event.is_directory:
            task: Dict[str, Any] = {"event_type": "created", "file_path": event.src_path}
            self.processing_queue.put(task)


    def on_modified(self, event: Union[DirModifiedEvent, FileModifiedEvent]):
        if not event.is_directory:
            task : Dict[str, Any] = {"event_type": "modified", "file_path": event.src_path}
            self.processing_queue.put(task)

    def on_deleted(self, event: Union[DirDeletedEvent, FileDeletedEvent]):
        if not event.is_directory:
            task : Dict[str, Any] = {"event_type": "deleted", "file_path": event.src_path}
            self.processing_queue.put(task)

    def on_moved(self, event: Union[DirMovedEvent, FileMovedEvent]):
        if not event.is_directory:
            task : Dict[str, Any] = {
                "event_type": "moved",
                "src_path": event.src_path,
                "dest_path": event.dest_path,
            }
            self.processing_queue.put(task)




def start_rag_file_monitoring(
    paths_to_watch: list[str],
    manifest: RAGManifest,
    rag_store: RAGStore,
    chunk_config: Dict[str, Any] 
) :
    """
    Starts monitoring the specified file system paths for changes to support RAG (Retrieval-Augmented Generation) updates.

    Args:
        paths_to_watch (list[str]): A list of directory paths to monitor for file changes.
        manifest (RAGManifest): The manifest object containing RAG configuration and state.
        chroma_collection (chromadb.Collection): The ChromaDB collection used for indexing and retrieval.

    Returns:
        Observer: An instance of the Observer that is actively monitoring the specified paths.

    Notes:
        - Only existing paths will be monitored; non-existent paths will trigger a warning.
        - Monitoring is recursive for each directory in `paths_to_watch`.
        - The observer must be stopped manually when monitoring is no longer needed.
    """
    processing_queue: queue.Queue[Dict[str, Any]] = queue.Queue()

    worker_thread = threading.Thread(
        target=rag_update_worker,
        args=(processing_queue, manifest, rag_store, chunk_config),
        daemon=True  
    )
    worker_thread.start() 
 
    event_handler = ResilientFileIndexer(processing_queue)
    observer = Observer(timeout=30)
    for path_str in paths_to_watch:
        path = Path(path_str)
        if path.exists():
            observer.schedule(event_handler, str(path), recursive=True)
            print(f"Watching directory: {path}")
        else:
            print(f"Warning: Path not found, cannot watch: {path}")

    observer.start()
    print("File system monitoring started for RAG updates...")
    return observer


