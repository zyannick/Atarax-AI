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
from ataraxai.app_logic.modules.rag.rag_store import RAGStore
from ataraxai.app_logic.modules.rag.rag_manifest import RAGManifest
import chromadb
from typing_extensions import Union
from typing_extensions import Optional, List, Dict, Any


class ResilientFileIndexer(FileSystemEventHandler):
    def __init__(self, manifest: RAGManifest, chroma_collection: chromadb.Collection):
        self.manifest = manifest
        self.chroma_collection = chroma_collection
        self.processing_queue: queue.Queue[Dict[str, Any]] = queue.Queue()

    def on_created(self, event: Union[DirCreatedEvent, FileCreatedEvent]):
        if not event.is_directory:
            print(f"New file created: {event.src_path}")
            task: Dict[str, Any] = {"event_type": "created", "file_path": event.src_path}
            self.processing_queue.put(task)


    def on_modified(self, event: Union[DirModifiedEvent, FileModifiedEvent]):
        if not event.is_directory:
            print(f"File modified: {event.src_path}")
            task : Dict[str, Any] = {"event_type": "modified", "file_path": event.src_path}
            self.processing_queue.put(task)

    def on_deleted(self, event: Union[DirDeletedEvent, FileDeletedEvent]):
        if not event.is_directory:
            print(f"File deleted: {event.src_path}")
            task : Dict[str, Any] = {"event_type": "deleted", "file_path": event.src_path}
            self.processing_queue.put(task)

    def on_moved(self, event: Union[DirMovedEvent, FileMovedEvent]):
        if not event.is_directory:
            print(f"File moved: {event.src_path} to {event.dest_path}")
            task : Dict[str, Any] = {
                "event_type": "moved",
                "src_path": event.src_path,
                "dest_path": event.dest_path,
            }
            self.processing_queue.put(task)


# class ResilientMailIndexer:
#     def __init__(self, manifest: RAGManifest, rag_store: RAGStore):
#         self.manifest = manifest
#         self.rag_store = rag_store

#     def process_email(self, file_path: str):
#         print(f"Processing email for indexing: {file_path}")


# class ResilientCalendarIndexer:
#     def __init__(self, manifest: RAGManifest, rag_store: RAGStore):
#         self.manifest = manifest
#         self.rag_store = rag_store

#     def process_calendar_event(self, file_path: str):
#         print(f"Processing calendar event for indexing: {file_path}")


def start_rag_file_monitoring(
    paths_to_watch: list[str],
    manifest: RAGManifest,
    chroma_collection: chromadb.Collection,
):
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
    event_handler = ResilientFileIndexer(manifest, chroma_collection)
    observer = Observer(timeout=5)
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


