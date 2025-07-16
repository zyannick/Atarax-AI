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
from ataraxai.praxis.modules.rag.rag_manifest import RAGManifest
from typing_extensions import Union
from typing_extensions import Dict, Any
import threading

from ataraxai.praxis.modules.rag.rag_store import RAGStore
from ataraxai.praxis.modules.rag.rag_updater import rag_update_worker


class ResilientFileIndexer(FileSystemEventHandler):
    def __init__(self,  processing_queue: queue.Queue):
        """
        Initializes the class with a processing queue.

        Args:
            processing_queue (queue.Queue): A queue containing items to be processed, where each item is a dictionary with string keys and values of any type.
        """
        self.processing_queue: queue.Queue[Dict[str, Any]] = processing_queue

    def on_created(self, event: Union[DirCreatedEvent, FileCreatedEvent]):
        """
        Handles file or directory creation events.

        If the created event is for a file (not a directory), constructs a task dictionary
        containing the event type and file path, and puts it into the processing queue.

        Args:
            event (Union[DirCreatedEvent, FileCreatedEvent]): The event object representing
                the creation of a file or directory.

        Returns:
            None
        """
        if not event.is_directory:
            task: Dict[str, Any] = {"event_type": "created", "file_path": event.src_path}
            self.processing_queue.put(task)


    def on_modified(self, event: Union[DirModifiedEvent, FileModifiedEvent]):
        """
        Handles file or directory modification events.

        When a file (not a directory) is modified, this method creates a task dictionary
        containing the event type and the file path, and puts it into the processing queue
        for further handling.

        Args:
            event (Union[DirModifiedEvent, FileModifiedEvent]): The event object representing
                the modification, which includes information about whether the event pertains
                to a directory or a file, and the source path of the modified item.
        """
        if not event.is_directory:
            task : Dict[str, Any] = {"event_type": "modified", "file_path": event.src_path}
            self.processing_queue.put(task)

    def on_deleted(self, event: Union[DirDeletedEvent, FileDeletedEvent]):
        """
        Handles file or directory deletion events.

        If the deleted event corresponds to a file (not a directory), creates a task describing the deletion
        and adds it to the processing queue for further handling.

        Args:
            event (Union[DirDeletedEvent, FileDeletedEvent]): The event object representing the deletion,
                containing information about the deleted file or directory.
        """
        if not event.is_directory:
            task : Dict[str, Any] = {"event_type": "deleted", "file_path": event.src_path}
            self.processing_queue.put(task)

    def on_moved(self, event: Union[DirMovedEvent, FileMovedEvent]):
        """
        Handles file or directory move events.

        If the event corresponds to a file (not a directory), creates a task dictionary
        containing the event type ('moved'), the source path, and the destination path.
        The task is then added to the processing queue for further handling.

        Args:
            event (Union[DirMovedEvent, FileMovedEvent]): The move event object containing
                information about the source and destination paths, and whether the event
                pertains to a directory.
        """
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


