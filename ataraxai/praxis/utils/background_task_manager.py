import logging
import threading
import time
from typing import Set

logger = logging.getLogger(__name__)


class BackgroundTaskManager:

    def __init__(self):
        self._lock = threading.Lock()
        self._active_tasks: Set[threading.Thread] = set()

    def register_task(self, thread: threading.Thread):
        with self._lock:
            self._active_tasks.add(thread)
        logger.info(
            f"Registered background task: {thread.name} (Total: {len(self._active_tasks)})"
        )

    def unregister_task(self, thread: threading.Thread):
        with self._lock:
            self._active_tasks.discard(thread)
        logger.info(
            f"Unregistered background task: {thread.name} (Remaining: {len(self._active_tasks)})"
        )

    def wait_for_all_tasks(self, timeout: int = 180):
        logger.info(
            f"Waiting for {len(self._active_tasks)} background task(s) to complete..."
        )
        start_time = time.time()

        with self._lock:
            tasks_to_join = list(self._active_tasks)

        for thread in tasks_to_join:
            remaining_time = timeout - (time.time() - start_time)
            if remaining_time <= 0:
                logger.warning("Timeout reached while waiting for background tasks.")
                break
            thread.join(timeout=remaining_time)

        with self._lock:
            remaining_tasks = [t for t in self._active_tasks if t.is_alive()]

        if not remaining_tasks:
            logger.info("All background tasks completed successfully.")
        else:
            logger.warning(
                f"{len(remaining_tasks)} background task(s) did not complete within the timeout."
            )
