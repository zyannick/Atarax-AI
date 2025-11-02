import asyncio
import uuid
from enum import Enum
from typing import Any, Dict, Optional


class TaskStatus(Enum):
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    ERROR = "error"
    DELETED = "deleted"


class Task:
    def __init__(self, task_id: str, future: asyncio.Future):
        self.id = task_id
        self.future = future
        self.status = TaskStatus.PENDING
        self.result: Optional[Any] = None
        self.error: Optional[str] = None


class GatewayTaskManager:
    def __init__(self):
        self._tasks: Dict[str, Task] = {}

    def _on_task_done(self, task_id: str, future: asyncio.Future):
        task = self._tasks.get(task_id)
        if not task:
            return
        try:
            task.result = future.result()
            task.status = TaskStatus.SUCCESS
        except Exception as e:
            task.error = str(e)
            task.status = TaskStatus.FAILED

    def create_task(self, future: asyncio.Future) -> str:
        task_id = str(uuid.uuid4())
        task = Task(task_id, future)
        self._tasks[task_id] = task

        future.add_done_callback(lambda f: self._on_task_done(task_id, f))

        return task_id

    def cancel_task(self, task_id: str) -> bool:
        task = self._tasks.get(task_id)
        if not task or task.status != TaskStatus.PENDING:
            return False

        task.future.cancel()
        return True

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        task = self._tasks.get(task_id)
        if not task:
            return None

        return {
            "task_id": task.id,
            "status": task.status,
            "result": task.result,
            "error": task.error,
        }
