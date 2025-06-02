import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from pathlib import Path
import queue

class ResilientIndexer(FileSystemEventHandler):
    def __init__(self, manifest, chroma_collection):
        self.manifest = manifest
        self.chroma_collection = chroma_collection
        self.processing_queue = queue.Queue()

    def on_created(self, event):
        if not event.is_directory:
            print(f"New file created: {event.src_path}")
            task = {"event_type": "created", "file_path": event.src_path}
            self.processing_queue.put(task)

    def on_modified(self, event):
        if not event.is_directory:
            print(f"File modified: {event.src_path}")
            task = {"event_type": "modified", "file_path": event.src_path}
            self.processing_queue.put(task)

    def on_deleted(self, event):
        if not event.is_directory:
            print(f"File deleted: {event.src_path}")
            task = {"event_type": "deleted", "file_path": event.src_path}
            self.processing_queue.put(task)

    def on_moved(self, event):  
        if not event.is_directory:
            print(f"File moved: {event.src_path} to {event.dest_path}")
            task = {"event_type": "moved", "src_path": event.src_path, "dest_path": event.dest_path}
            self.processing_queue.put(task)

     


def start_rag_file_monitoring(paths_to_watch, manifest, chroma_collection):
    event_handler = ResilientIndexer(manifest, chroma_collection)
    observer = Observer()
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