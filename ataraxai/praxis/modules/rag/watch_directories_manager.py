import asyncio
from typing import Set, Dict, Any
from ataraxai.praxis.modules.rag.rag_manifest import RAGManifest

import os
from asyncio import Queue

import logging
from ataraxai.praxis.utils.configs.rag_config_manager import RAGConfigManager


class WatchedDirectoriesManager:
    def __init__(
        self,
        rag_config_manager: RAGConfigManager,
        manifest: RAGManifest,
        processing_queue: Queue[Dict[str, Any]],
        logger: logging.Logger,
    ):
        self.rag_config_manager = rag_config_manager
        self.manifest = manifest
        self.processing_queue = processing_queue
        self.logger = logger

    async def _update_config(self, new_dirs: Set[str]):
        await asyncio.to_thread(
            self.rag_config_manager.set, "rag_watched_directories", list(new_dirs)
        )

    async def _scan_and_queue_files(self, directory: str, action: str):
        loop = asyncio.get_running_loop()
        for root, _, files in os.walk(directory):
            for filename in files:
                file_path = os.path.join(root, filename)
                if action == "add" and not self.manifest.is_file_in_manifest(file_path):
                    self.manifest.add_file(file_path)
                    task = {"event_type": "created", "path": file_path}
                    asyncio.run_coroutine_threadsafe(
                        self.processing_queue.put(task), loop
                    )
                elif action == "remove" and self.manifest.is_file_in_manifest(
                    file_path
                ):
                    self.manifest.remove_file(file_path)
                    task = {"event_type": "deleted", "path": file_path}
                    asyncio.run_coroutine_threadsafe(
                        self.processing_queue.put(task), loop
                    )

    async def add_directories(self, directories_to_add: Set[str]) -> bool:
        current_dirs = set(self.rag_config_manager.config.rag_watched_directories or [])
        new_dirs = current_dirs.union(directories_to_add)
        if new_dirs != current_dirs:
            await self._update_config(new_dirs)
        else:
            self.logger.info("No new directories to add, skipping update.")
            return False

        for directory in directories_to_add:
            await asyncio.to_thread(self._scan_and_queue_files, directory, "add")

        self.logger.info(f"Added directories to watch: {directories_to_add}")
        return True

    async def remove_directories(self, directories_to_remove: Set[str]) -> bool:
        current_dirs = set(self.rag_config_manager.config.rag_watched_directories or [])
        new_dirs = current_dirs.difference(directories_to_remove)
        await self._update_config(new_dirs)

        for directory in directories_to_remove:
            await asyncio.to_thread(self._scan_and_queue_files, directory, "remove")

        self.logger.info(f"Removed directories from watch: {directories_to_remove}")
        return True
