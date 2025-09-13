import asyncio
import time
from logging import Logger
from pathlib import Path
from typing import Any, Dict, List, Mapping, Union

# import threading
# import os
# import queue
from ataraxai.praxis.modules.rag.parser.base_meta_data import (
    get_file_hash,
    set_base_metadata,
)
from ataraxai.praxis.modules.rag.rag_manifest import (
    RAGManifest,
)

# from ataraxai.praxis.modules.rag.parser.document_base_parser import DocumentChunk
from ataraxai.praxis.modules.rag.rag_store import RAGStore
from ataraxai.praxis.modules.rag.smart_chunker import SmartChunker

MetadataDict = Mapping[str, Union[str, int, float, bool, None]]


async def process_new_file(
    file_path_str: str,
    manifest: RAGManifest,
    rag_store: RAGStore,
    chunker: SmartChunker,
    logger: Logger = Logger(__name__),
):
    file_path = Path(file_path_str)

    base_metadata: Dict[str, Any] = set_base_metadata(file_path)
    chunked_document_objects = await asyncio.to_thread(
        chunker.ingest_file, file_path=file_path
    )

    if not chunked_document_objects:
        return

    try:
        final_texts = [cd.content for cd in chunked_document_objects]  # type: ignore

        final_metadatas: List[MetadataDict] = []
        for cd in chunked_document_objects:  # type: ignore
            dict_value: Dict[str, Any] = {}
            for k, v in cd.metadata.items():  # type: ignore
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
        await asyncio.to_thread(manifest.save)

    except Exception as e:
        logger.error(f"Failed to process new file {file_path}: {e}", exc_info=True)

        manifest.add_file(
            str(file_path),
            metadata={
                "timestamp": time.time(),
                "hash": "unknown_due_to_error",
                "chunk_ids": [],
                "status": f"error: {e}",
            },
        )
        await asyncio.to_thread(manifest.save)


async def process_modified_file(
    file_path_str: str,
    manifest: RAGManifest,
    rag_store: RAGStore,
    chunker: SmartChunker,
    logger: Logger = Logger(__name__),
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
            logger.info(f"WORKER: Deleting old chunks for {file_path} from RAG store.")
            rag_store.delete_by_ids(ids=manifest_entry["chunk_ids"])

        await process_new_file(file_path_str, manifest, rag_store, chunker)

    except Exception as e:
        if manifest.data.get(str(file_path)):
            manifest.data[str(file_path)]["status"] = f"error: {e}"
            manifest.save()


async def process_deleted_file(
    file_path_str: str,
    manifest: "RAGManifest",
    rag_store: "RAGStore",
    logger: Logger = Logger(__name__),
):

    try:
        manifest_entry = manifest.data.pop(str(file_path_str), None)

        if manifest_entry and manifest_entry.get("chunk_ids"):
            rag_store.delete_by_ids(ids=manifest_entry["chunk_ids"])
            manifest.save()
        else:
            pass
    except Exception as e:
        logger.error(f"WORKER: Error processing deleted file {file_path_str}: {e}")


async def rag_update_worker_async(
    processing_queue: asyncio.Queue[Dict[str, Any]],
    manifest: RAGManifest,
    rag_store: RAGStore,
    chunk_config: Dict[str, Any],
    logger: Logger = Logger(__name__),
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

            logger.info(f"RAG Update Worker: Got task {task}")
            event_type = task.get("event_type")
            file_path = task.get("path")
            dest_path = task.get("dest_path")

            if event_type == "stop":
                logger.info("RAG Update Worker: Stopping as per 'stop' task.")
                processing_queue.task_done()
                break

            if not file_path:
                logger.warning(f"RAG Update Worker: Invalid task, missing path: {task}")
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
                    logger.warning(
                        f"RAG Update Worker: Invalid 'moved' task, missing dest_path: {task}"
                    )
            else:
                logger.warning(
                    f"RAG Update Worker: Unknown event type '{event_type}' for task: {task}"
                )
            processing_queue.task_done()
        except Exception as e:
            logger.error(f"RAG Update Worker: Error processing task: {e}")
            processing_queue.task_done()
