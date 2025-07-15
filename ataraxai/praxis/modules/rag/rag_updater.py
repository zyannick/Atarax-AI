from pathlib import Path
from ataraxai.praxis.modules.rag.rag_store import RAGStore
from ataraxai.praxis.modules.rag.rag_manifest import (
    RAGManifest,
)
from ataraxai.praxis.modules.rag.smart_chunker import SmartChunker
import threading
import os
import queue
from ataraxai.praxis.modules.rag.parser.base_meta_data import (
    set_base_metadata,
    get_file_hash,
)
from typing import Dict, List, Union, Mapping, Any


MetadataDict = Mapping[str, Union[str, int, float, bool, None]]


def process_new_file(
    file_path_str: str,
    manifest: RAGManifest,
    rag_store: RAGStore,
    chunker: SmartChunker,
):
    """
    Processes a new file for ingestion into the RAG (Retrieval-Augmented Generation) system.

    This function performs the following steps:
    1. Converts the file path string to a Path object and sets base metadata.
    2. Uses the provided chunker to split the file into document chunks.
    3. If chunks are generated, extracts their content and metadata.
    4. Generates unique chunk IDs and adds the chunks to the RAG store.
    5. Updates the manifest with file metadata, chunk IDs, and status.
    6. Handles errors by updating the manifest status accordingly.

    Args:
        file_path_str (str): The path to the file to be processed.
        manifest (RAGManifest): The manifest object for tracking file and chunk metadata.
        rag_store (RAGStore): The storage backend for adding document chunks.
        chunker (SmartChunker): The chunker used to split the file into manageable pieces.

    Returns:
        None
    """
    file_path = Path(file_path_str)
    print(f"WORKER: Processing NEW file: {file_path}")

    base_metadata: Dict[str, Any] = set_base_metadata(file_path)
    chunked_document_objects = chunker.ingest_file(
        file_path=file_path,
    )

    if not chunked_document_objects:
        print(f"WORKER: No chunks generated for {file_path}. Skipping.")
        return

    try:
        final_texts = [cd.content for cd in chunked_document_objects]

        final_metadatas: List[MetadataDict] = []
        for cd in chunked_document_objects:
            dict_value: Dict[str, Any] = {}
            for k, v in cd.metadata.items():
                dict_value[k] = v
            final_metadatas.append(dict_value)

        chunk_ids = [
            f"{str(file_path)}_{base_metadata['file_hash'][:8]}_chunk_{i}"
            for i in range(len(final_texts))
        ]
        print(f"WORKER: Adding {len(final_texts)} chunks for {file_path} to RAG store.")
        rag_store.add_chunks(
            ids=chunk_ids,
            texts=final_texts,
            metadatas=final_metadatas,
        )

        print(base_metadata)

        manifest.add_file(
            str(file_path),
            metadata={
                "timestamp": base_metadata["file_timestamp"],
                "hash": base_metadata["file_hash"],
                "chunk_ids": chunk_ids,
                "status": "indexed",
            },
        )
        manifest.save()
        print(f"WORKER: Successfully processed and indexed {file_path}")

    except Exception as e:
        # print(f"WORKER: Error processing new file {file_path}: {e}")
        if manifest.data.get(str(file_path)):
            manifest.data[str(file_path)]["status"] = f"error: {e}"
            manifest.save()


def process_modified_file(
    file_path_str: str,
    manifest: RAGManifest,
    rag_store: RAGStore,
    chunker: SmartChunker,
):
    """
    Processes a modified file by updating its entry in the RAG manifest and RAG store.

    This function checks if the specified file exists and is not a directory. If the file is missing,
    it treats the event as a deletion and processes it accordingly. If the file exists, it computes
    the file's hash and compares it with the hash stored in the manifest. If the content is unchanged,
    it updates the timestamp if necessary. If the content has changed or the file needs re-indexing,
    it deletes old chunks from the RAG store and processes the file as a new file.

    Args:
        file_path_str (str): The path to the file to process.
        manifest (RAGManifest): The manifest object tracking file metadata and chunk IDs.
        rag_store (RAGStore): The RAG store object responsible for storing and deleting chunks.
        chunker (SmartChunker): The chunker used to process and split the file content.

    Raises:
        Exception: Any exception encountered during processing is caught, logged, and the manifest is updated with the error status.
    """
    file_path = Path(file_path_str)
    print(f"WORKER: Processing MODIFIED file: {file_path}")

    if not file_path.exists() or file_path.is_dir():
        print(
            f"WORKER: Modified file {file_path} not found (maybe deleted quickly). Treating as delete."
        )
        process_deleted_file(file_path_str, manifest, rag_store)
        return

    try:
        current_timestamp = file_path.stat().st_mtime
        current_hash = get_file_hash(file_path)
        if not current_hash:
            return

        manifest_entry = manifest.data.get(str(file_path))

        if manifest_entry and manifest_entry.get("hash") == current_hash:
            print(
                f"WORKER: File {file_path} content unchanged (hash match). Updating timestamp if newer."
            )
            if current_timestamp > manifest_entry.get("timestamp", 0):
                manifest_entry["timestamp"] = current_timestamp
                manifest.save()
            return

        print(
            f"WORKER: File {file_path} has changed or needs re-indexing. Re-processing..."
        )

        if manifest_entry and manifest_entry.get("chunk_ids"):
            print(f"WORKER: Deleting old chunks for {file_path} from RAG store.")
            rag_store.delete_by_ids(ids=manifest_entry["chunk_ids"])

        process_new_file(file_path_str, manifest, rag_store, chunker)

    except Exception as e:
        print(f"WORKER: Error processing modified file {file_path}: {e}")
        if manifest.data.get(str(file_path)):
            manifest.data[str(file_path)]["status"] = f"error: {e}"
            manifest.save()


def process_deleted_file(
    file_path_str: str, manifest: "RAGManifest", rag_store: "RAGStore"
):
    """
    Processes the deletion of a file from the RAG system by removing its entry from the manifest
    and deleting associated chunks from the RAG store.

    Args:
        file_path_str (str): The file path of the deleted file as a string.
        manifest (RAGManifest): The manifest object containing metadata about files and their chunks.
        rag_store (RAGStore): The RAG store object responsible for storing and deleting chunks.

    Behavior:
        - Removes the file entry from the manifest.
        - Deletes associated chunk IDs from the RAG store if present.
        - Saves the updated manifest.
        - Logs the process and handles exceptions gracefully.
    """
    print(f"WORKER: Processing DELETED file: {file_path_str}")

    try:
        manifest_entry = manifest.data.pop(str(file_path_str), None)

        if manifest_entry and manifest_entry.get("chunk_ids"):
            print(f"WORKER: Deleting chunks for {file_path_str} from RAG store.")
            rag_store.delete_by_ids(ids=manifest_entry["chunk_ids"])
            manifest.save()
            print(f"WORKER: Successfully processed deletion of {file_path_str}")
        else:
            print(
                f"WORKER: File {file_path_str} not found in manifest or no chunk IDs to delete."
            )
    except Exception as e:
        print(f"WORKER: Error processing deleted file {file_path_str}: {e}")


def rag_update_worker(
    processing_queue: queue.Queue[Dict[str, Any]],
    manifest: RAGManifest,
    rag_store: RAGStore,
    chunk_config: Dict[str, Any],
):
    """
    Worker function that processes file system events from a queue and updates the RAG (Retrieval-Augmented Generation) store accordingly.

    This function continuously listens for tasks from the provided processing queue, handling file creation, modification, deletion, and movement events. For each event, it updates the RAG manifest and store using a configurable chunking strategy.

    Args:
        processing_queue (queue.Queue[Dict[str, Any]]): Queue containing file event tasks to process.
        manifest (RAGManifest): The manifest object representing the current state of indexed files.
        rag_store (RAGStore): The storage backend for RAG chunks and metadata.
        chunk_config (Dict[str, Any]): Configuration for chunking, including size, overlap, model name, separators, etc.

    Task Dict Format:
        {
            "event_type": str,   # One of "created", "modified", "deleted", "moved"
            "path": str,         # Path to the affected file
            "dest_path": str,    # (Optional) Destination path for "moved" events
            ...
        }

    Behavior:
        - Waits for tasks from the queue.
        - Processes each task according to its event type.
        - Uses SmartChunker to split file contents as needed.
        - Calls appropriate processing functions for each event.
        - Exits cleanly when a sentinel (None) is received.
        - Handles errors gracefully and ensures queue task completion.
    """
    print(
        f"RAG Update Worker started. PID: {os.getpid()}, Thread: {threading.get_ident()}"
    )
    chunk_size = chunk_config.get("size", 400)
    chunk_overlap = chunk_config.get("overlap", 50)

    chunker = SmartChunker(
        model_name_for_tiktoken=chunk_config.get("model_name_for_tiktoken", "gpt-4"),
        chunk_size_tokens=chunk_size,
        chunk_overlap_tokens=chunk_overlap,
        separators=chunk_config.get("separators", None),
        keep_separator=chunk_config.get("keep_separator", True),
    )

    while True:
        # print("RAG Update Worker: Waiting for tasks...")
        try:
            task: Dict[str, Any] = processing_queue.get(timeout=1)
            if task is None:
                print("RAG Update Worker: Received sentinel. Exiting.")
                processing_queue.task_done()
                break

            print(f"RAG Update Worker: Got task {task}")
            event_type = task.get("event_type")
            file_path = task.get("path")
            dest_path = task.get("dest_path")

            if not file_path:
                print(f"RAG Update Worker: Invalid task, missing path: {task}")
                processing_queue.task_done()
                continue

            if event_type == "created":
                process_new_file(file_path, manifest, rag_store, chunker)
            elif event_type == "modified":
                process_modified_file(file_path, manifest, rag_store, chunker)
            elif event_type == "deleted":
                process_deleted_file(file_path, manifest, rag_store)
            elif event_type == "moved":
                if dest_path:
                    process_deleted_file(file_path, manifest, rag_store)
                    process_new_file(
                        dest_path,
                        manifest,
                        rag_store,
                        chunker,
                    )
                else:
                    print(
                        f"RAG Update Worker: Invalid 'moved' task, missing dest_path: {task}"
                    )
            else:
                print(
                    f"RAG Update Worker: Unknown event type '{event_type}' for task: {task}"
                )
            processing_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"RAG Update Worker: Error processing task: {e}")
            # It's good practice to still call task_done() even on an error
            # to prevent the queue from getting stuck.
            processing_queue.task_done()
        # finally:
        #     # if "task" in locals() and task is not None:
        #     #     processing_queue.task_done()
        #     print(f"RAG Update Worker: Error processing task: {e}")
        #     processing_queue.task_done()
