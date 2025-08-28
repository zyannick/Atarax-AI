import uuid
import asyncio
from typing import Dict, Any, Optional
from enum import Enum, auto

class ProgressTracker:
    def __init__(self):
        self._progress: Dict[str, Dict[str, Any]] = {}

    def get_progress(self, task_id: str) -> Optional[Dict[str, Any]]:
        return self._progress.get(task_id)

    def set_progress(self, task_id: str, status_data: Dict[str, Any]):
        self._progress[task_id] = status_data

    def clear_progress(self, task_id: str):
        if task_id in self._progress:
            del self._progress[task_id]


class TaskStatus(Enum):
    PENDING = auto()
    SUCCESS = auto()
    FAILED = auto()
    ERROR = auto()

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
            "error": task.error
        }