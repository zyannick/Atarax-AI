import pytest
from unittest import mock
from ataraxai.app_logic.modules.prompt_engine.task_manager import TaskManager
from ataraxai.app_logic.modules.prompt_engine.specific_tasks.base_task import BaseTask


class DummyTask(BaseTask):
    id = "dummy"
    description = "Dummy Task"
    metadata = {"id": "dummy", "description": "Dummy Task"}

    def _load_resources(self):
        pass

    def execute(self, processed_input, context, prompt_loader):
        pass

    def run(self, *args, **kwargs):
        return "ran"


def test_register_and_get_task(monkeypatch):
    tm = TaskManager(
        tasks_directory="ataraxai/app_logic/modules/prompt_engine/specific_tasks"
    )
    tm.tasks = {}

    dummy = DummyTask()
    tm.register_task(dummy)
    assert tm.get_task("dummy") is dummy


def test_register_task_overwrites(monkeypatch, capsys):
    tm = TaskManager(
        tasks_directory="ataraxai/app_logic/modules/prompt_engine/specific_tasks"
    )
    tm.tasks = {}

    dummy1 = DummyTask()
    dummy2 = DummyTask()
    tm.register_task(dummy1)
    tm.register_task(dummy2)
    captured = capsys.readouterr()
    assert "already registered" in captured.out


def test_get_task_not_found():
    tm = TaskManager(
        tasks_directory="ataraxai/app_logic/modules/prompt_engine/specific_tasks"
    )
    tm.tasks = {}
    with pytest.raises(ValueError):
        tm.get_task("not_exist")


def test_list_tasks():
    tm = TaskManager(
        tasks_directory="ataraxai/app_logic/modules/prompt_engine/specific_tasks"
    )
    tm.tasks = {}
    dummy = DummyTask()
    tm.register_task(dummy)
    tasks = tm.list_tasks()
    assert isinstance(tasks, list)
    assert tasks[0]["id"] == "dummy"


def test_discover_and_load_tasks(monkeypatch):
    # Patch Path.glob to simulate .py files
    fake_path = mock.Mock()
    fake_file = mock.Mock()
    fake_file.name = "dummy_task.py"
    fake_file.with_suffix.return_value.parts = [
        "ataraxai",
        "app_logic",
        "modules",
        "prompt_engine",
        "specific_tasks",
        "dummy_task",
    ]
    fake_path.glob.return_value = [fake_file]

    # Patch importlib.import_module to return a module with DummyTask
    dummy_module = mock.Mock()
    dummy_module.DummyTask = DummyTask

    with mock.patch(
        "ataraxai.app_logic.modules.prompt_engine.task_manager.Path",
        return_value=fake_path,
    ), mock.patch(
        "ataraxai.app_logic.modules.prompt_engine.task_manager.importlib.import_module",
        return_value=dummy_module,
    ), mock.patch(
        "ataraxai.app_logic.modules.prompt_engine.task_manager.inspect.getmembers",
        return_value=[("DummyTask", DummyTask)],
    ):
        tm = TaskManager(
            tasks_directory="ataraxai/app_logic/modules/prompt_engine/specific_tasks"
        )
        assert "dummy" in tm.tasks
        assert isinstance(tm.get_task("dummy"), DummyTask)
