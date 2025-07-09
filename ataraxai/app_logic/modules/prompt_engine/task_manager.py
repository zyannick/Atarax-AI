import importlib
import inspect
from pathlib import Path
from typing import Dict, List
from ataraxai.app_logic.modules.prompt_engine.specific_tasks.base_task import BaseTask


class TaskManager:
    def __init__(
        self,
        tasks_directory: str = "ataraxai/app_logic/modules/prompt_engine/specific_tasks",
    ):
        """
        Initializes the TaskManager by setting up the tasks dictionary and discovering available tasks.

        Args:
            tasks_directory (str, optional): Path to the directory containing specific task modules.
                Defaults to "ataraxai/app_logic/modules/prompt_engine/specific_tasks".

        Attributes:
            tasks (Dict[str, BaseTask]): A dictionary mapping task names to their corresponding BaseTask instances.

        Raises:
            Any exceptions raised by _discover_and_load_tasks if the tasks directory is invalid or loading fails.
        """
        self.tasks: Dict[str, BaseTask] = {}
        self._discover_and_load_tasks(Path(tasks_directory))

    def _discover_and_load_tasks(self, tasks_path: Path):
        """
        Discovers and loads task classes from Python files in the specified directory.

        This method scans the given `tasks_path` for all Python files (excluding those starting with an underscore or named 'base_task.py'),
        dynamically imports each module, and registers any class that is a subclass of `BaseTask` (excluding `BaseTask` itself).

        Args:
            tasks_path (Path): The directory path to search for task modules.

        Side Effects:
            - Dynamically imports modules found in the directory.
            - Instantiates and registers discovered task classes.
            - Prints status and warning messages to the console if tasks cannot be loaded.
        """
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
        """
        Registers a new task in the task manager.

        If a task with the same ID already exists, it will be overwritten and a warning will be printed.

        Args:
            task (BaseTask): The task instance to register.

        Side Effects:
            Prints a warning if the task ID is already registered.
            Prints a confirmation message upon registration.
        """
        if task.id in self.tasks:
            print(
                f"Warning: Task with id '{task.id}' is already registered. Overwriting."
            )
        self.tasks[task.id] = task
        print(f"  -> Registered task: '{task.id}'")

    def get_task(self, task_id: str) -> BaseTask:
        """
        Retrieve a task by its unique identifier.

        Args:
            task_id (str): The unique identifier of the task to retrieve.

        Returns:
            BaseTask: The task object associated with the given task_id.

        Raises:
            ValueError: If no task with the specified task_id is found.
        """
        task = self.tasks.get(task_id)
        if not task:
            raise ValueError(f"Task with id '{task_id}' not found.")
        return task

    def list_tasks(self) -> List[Dict[str, str]]:
        """
        Returns a list of metadata dictionaries for all tasks.

        Each dictionary in the returned list contains string key-value pairs representing the metadata of a task.

        Returns:
            List[Dict[str, str]]: A list of metadata dictionaries for each task.
        """
        return [task.metadata for task in self.tasks.values()]
