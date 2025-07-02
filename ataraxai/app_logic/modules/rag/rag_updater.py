from pathlib import Path
from .rag_store import RAGStore
from .ataraxai_rag_manager import (
    RAGManifest,
)
from ataraxai.app_logic.modules.rag.ataraxai_embedder import AtaraxAIEmbedder
from ataraxai.app_logic.modules.rag.smart_chunker import SmartChunker
import threading
import os
import queue
from ataraxai.app_logic.modules.rag.parser.base_meta_data import (
    set_base_metadata,
    get_file_hash,
)
from typing import Optional, Dict, Any
from typing_extensions import Union



def process_new_file(
    file_path_str: str,
    manifest: RAGManifest,
    rag_store: RAGStore,
    chunker: SmartChunker,
):
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
        final_metadatas = [v for cd in chunked_document_objects for k, v in cd.metadata.items()]
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
        print(f"WORKER: Error processing new file {file_path}: {e}")
        if manifest.data.get(str(file_path)):
            manifest.data[str(file_path)]["status"] = f"error: {e}"
            manifest.save()


def process_modified_file(
    file_path_str: str,
    manifest: RAGManifest,
    rag_store: RAGStore,
    chunker: SmartChunker,
):
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

        except queue.Empty:
            continue
        except Exception as e:
            print(
                f"RAG Update Worker: Unhandled error processing task {task if 'task' in locals() else 'UNKNOWN_TASK'}: {e}"
            )
        finally:
            if "task" in locals() and task is not None:
                processing_queue.task_done()
