import asyncio
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


async def process_new_file(
    file_path_str: str,
    manifest: RAGManifest,
    rag_store: RAGStore,
    chunker: SmartChunker,
):
    file_path = Path(file_path_str)

    base_metadata: Dict[str, Any] = set_base_metadata(file_path)
    chunked_document_objects = await asyncio.to_thread(
        chunker.ingest_file, file_path=file_path
    )

    if not chunked_document_objects:
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
        await asyncio.to_thread(
            rag_store.add_chunks,
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


    except Exception as e:
        if manifest.data.get(str(file_path)):
            manifest.data[str(file_path)]["status"] = f"error: {e}"
            manifest.save()


async def process_modified_file(
    file_path_str: str,
    manifest: RAGManifest,
    rag_store: RAGStore,
    chunker: SmartChunker,
):
    file_path = Path(file_path_str)

    if not file_path.exists() or file_path.is_dir():
        await process_deleted_file(file_path_str, manifest, rag_store)
        return

    try:
        current_timestamp = file_path.stat().st_mtime
        current_hash = get_file_hash(file_path)
        if not current_hash:
            return

        manifest_entry = manifest.data.get(str(file_path))

        if manifest_entry and manifest_entry.get("hash") == current_hash:
            if current_timestamp > manifest_entry.get("timestamp", 0):
                manifest_entry["timestamp"] = current_timestamp
                manifest.save()
            return


        if manifest_entry and manifest_entry.get("chunk_ids"):
            print(f"WORKER: Deleting old chunks for {file_path} from RAG store.")
            rag_store.delete_by_ids(ids=manifest_entry["chunk_ids"])

        await process_new_file(file_path_str, manifest, rag_store, chunker)

    except Exception as e:
        if manifest.data.get(str(file_path)):
            manifest.data[str(file_path)]["status"] = f"error: {e}"
            manifest.save()


async def process_deleted_file(
    file_path_str: str, manifest: "RAGManifest", rag_store: "RAGStore"
):

    try:
        manifest_entry = manifest.data.pop(str(file_path_str), None)

        if manifest_entry and manifest_entry.get("chunk_ids"):
            rag_store.delete_by_ids(ids=manifest_entry["chunk_ids"])
            manifest.save()
        else:
            pass
    except Exception as e:
        print(f"WORKER: Error processing deleted file {file_path_str}: {e}")


async def rag_update_worker(
    processing_queue: asyncio.Queue[Dict[str, Any]],
    manifest: RAGManifest,
    rag_store: RAGStore,
    chunk_config: Dict[str, Any],
):
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
            task: Dict[str, Any] = await processing_queue.get()
            if task is None:
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
                await process_new_file(file_path, manifest, rag_store, chunker)
            elif event_type == "modified":
                await process_modified_file(file_path, manifest, rag_store, chunker)
            elif event_type == "deleted":
                await process_deleted_file(file_path, manifest, rag_store)
            elif event_type == "moved":
                if dest_path:
                    await process_deleted_file(file_path, manifest, rag_store)
                    await process_new_file(
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
