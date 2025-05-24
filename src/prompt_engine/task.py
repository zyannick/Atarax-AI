


class BaseTask:
    def __init__(self, task_name: str):
        self.task_name = task_name

    def __str__(self):
        return f"Task: {self.task_name}"

    def __repr__(self):
        return f"BaseTask({self.task_name})"
        self._registry[task_class.__name__] = task_class


class Task:
    def __init__(self, task_name: str):
        self.task_name = task_name

    def __str__(self):
        return f"Task: {self.task_name}"

    def __repr__(self):
        return f"Task({self.task_name})"