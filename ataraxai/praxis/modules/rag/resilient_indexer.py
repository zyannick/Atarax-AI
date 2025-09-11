import asyncio
from pathlib import Path
from logging import Logger
from typing import Dict, Any, Set
from watchfiles import awatch, Change # type: ignore

from asyncio import Queue

from ataraxai.praxis.utils.configs.config_schemas.rag_config_schema import RAGConfig
from ataraxai.praxis.utils.configs.rag_config_manager import RAGConfigManager



class ResilientIndexer:
    def __init__(
        self,
        rag_config_manager: RAGConfigManager,
        processing_queue: Queue,
        logger: Logger,
    ):
        self._config_manager = rag_config_manager
        self._processing_queue: Queue[Dict[str, Any]] = processing_queue
        self._logger = logger
        
        self._watched_paths: Set[str] = set()
        self._content_watcher_task: asyncio.Task[None] | None = None
        self._config_watcher_task: asyncio.Task[None] | None = None
        self._is_reloading = asyncio.Lock()

    async def _run_content_watcher(self):
        if not self._watched_paths:
            self._logger.info("Content watcher: No paths to watch. Task will sleep.")
            return

        self._logger.info(f"Content watcher: Starting for paths: {list(self._watched_paths)}")
        try:
            async for changes in awatch(*self._watched_paths, stop_event=self._shutdown_event):
                for change_type, path_str in changes:
                    path = Path(path_str)
                    if path.is_dir():
                        continue

                    task = None
                    if change_type == Change.added:
                        task = {"event_type": "created", "path": path_str}
                    elif change_type == Change.modified:
                        task = {"event_type": "modified", "path": path_str}
                    elif change_type == Change.deleted:
                        task = {"event_type": "deleted", "path": path_str}

                    if task:
                        self._logger.info(f"Content watcher detected event: {task}")
                        await self._processing_queue.put(task)
        except asyncio.CancelledError:
            self._logger.info("Content watcher task was cancelled.")
        except Exception as e:
            self._logger.error(f"Content watcher error: {e}", exc_info=True)

    async def _run_config_watcher(self):
        config_path = self._config_manager.config_path
        self._logger.info(f"Config watcher: Starting for file: {config_path}")
        try:
            async for changes in awatch(config_path.parent, stop_event=self._shutdown_event):
                for _, path_str in changes:
                    if Path(path_str) == config_path:
                        self._logger.info("Config watcher: Config file change detected. Triggering reload.")
                        asyncio.create_task(self.reload_and_restart())
                        break
        except asyncio.CancelledError:
            self._logger.info("Config watcher task was cancelled.")
        except Exception as e:
            self._logger.error(f"Config watcher error: {e}", exc_info=True)
            
    async def reload_and_restart(self):
        async with self._is_reloading:
            self._logger.info("Reloading configuration...")
            config : RAGConfig = await asyncio.to_thread(self._config_manager.reload)
            new_paths = set(config.rag_watched_directories or [])

            if new_paths == self._watched_paths:
                self._logger.info("No change in watched paths. Watcher not restarted.")
                return

            self._logger.info(f"Watched paths changed. Old: {self._watched_paths}, New: {new_paths}")
            self._watched_paths = new_paths

            if self._content_watcher_task and not self._content_watcher_task.done():
                self._content_watcher_task.cancel()
                await asyncio.sleep(0.1)

            self._content_watcher_task = asyncio.create_task(self._run_content_watcher())
            self._logger.info("Content watcher has been restarted with new paths.")

    async def start(self):
        self._logger.info("Starting DirectoryWatcherManager...")
        self._shutdown_event = asyncio.Event()
        
        await self.reload_and_restart()
        
        self._config_watcher_task = asyncio.create_task(self._run_config_watcher())

    async def stop(self):
        self._logger.info("Stopping DirectoryWatcherManager...")
        self._shutdown_event.set() 
        
        tasks_to_cancel = [self._content_watcher_task, self._config_watcher_task]
        for task in tasks_to_cancel:
            if task and not task.done():
                task.cancel()
        
        await asyncio.gather(*[t for t in tasks_to_cancel if t], return_exceptions=True)
        self._logger.info("DirectoryWatcherManager stopped.")

