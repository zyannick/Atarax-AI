import os
import importlib
import inspect
from pathlib import Path
from typing import Dict, List, Type
from ataraxai.app_logic.modules.prompt_engine.specific_tasks.base_task import BaseTask


class TaskManager:
    def __init__(
        self,
        tasks_directory: str = "ataraxai/app_logic/modules/prompt_engine/specific_tasks",
    ):
        self.tasks: Dict[str, BaseTask] = {}
        self._discover_and_load_tasks(Path(tasks_directory))

    def _discover_and_load_tasks(self, tasks_path: Path):
        print(f"TaskManager: Discovering tasks in '{tasks_path}'...")
        for file_path in tasks_path.glob("*.py"):
            if file_path.name.startswith("_") or file_path.name == "base_task.py":
                continue

            module_path = ".".join(file_path.with_suffix("").parts)

            try:
                module = importlib.import_module(module_path)
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, BaseTask) and obj is not BaseTask:
                        task_instance = obj()
                        self.register_task(task_instance)
            except Exception as e:
                print(f"Warning: Could not load tasks from {file_path}: {e}")

    def register_task(self, task: BaseTask):
        if task.id in self.tasks:
            print(
                f"Warning: Task with id '{task.id}' is already registered. Overwriting."
            )
        self.tasks[task.id] = task
        print(f"  -> Registered task: '{task.id}'")

    def get_task(self, task_id: str) -> BaseTask:
        task = self.tasks.get(task_id)
        if not task:
            raise ValueError(f"Task with id '{task_id}' not found.")
        return task

    def list_tasks(self) -> List[Dict[str, str]]:
        return [task.metadata for task in self.tasks.values()]
