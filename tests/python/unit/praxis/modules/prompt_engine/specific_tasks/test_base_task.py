import pytest
from unittest import mock
from ataraxai.praxis.modules.prompt_engine.specific_tasks.base_task import BaseTask
from ataraxai.praxis.modules.prompt_engine.specific_tasks.task_dependencies import TaskDependencies



class DummyTaskDependencies(TaskDependencies):
    pass

class DummyTask(BaseTask):
    id = "dummy"
    description = "A dummy task for testing"
    required_inputs = ["foo", "bar"]
    prompt_template_name = "dummy_template"

    def _load_resources(self):
        self.resources_loaded = True

    async def execute(self, processed_input, dependencies):
        return {"result": processed_input["foo"] + processed_input["bar"]}

def test_init_missing_id():
    class NoIdTask(BaseTask):
        description = "desc"
        def _load_resources(self): pass
        async def execute(self, processed_input, dependencies): pass
    with pytest.raises(NotImplementedError):
        NoIdTask()

def test_init_missing_description():
    class NoDescTask(BaseTask):
        id = "id"
        def _load_resources(self): pass
        async def execute(self, processed_input, dependencies): pass
    with pytest.raises(NotImplementedError):
        NoDescTask()

def test_load_if_needed(monkeypatch):
    task = DummyTask()
    task._initialized = False
    called = {}
    def fake_load_resources():
        called["loaded"] = True
    task._load_resources = fake_load_resources
    task.load_if_needed()
    assert called["loaded"]
    assert task._initialized

def test_load_if_needed_already_initialized():
    task = DummyTask()
    task._initialized = True
    task._load_resources = mock.Mock()
    task.load_if_needed()
    task._load_resources.assert_not_called()

def test_validate_inputs_success():
    task = DummyTask()
    input_data = {"foo": 1, "bar": 2}
    task.validate_inputs(input_data)  

def test_validate_inputs_missing():
    task = DummyTask()
    input_data = {"foo": 1}
    with pytest.raises(ValueError) as exc:
        task.validate_inputs(input_data)
    assert "missing required inputs" in str(exc.value)

def test_preprocess_returns_input():
    task = DummyTask()
    data = {"foo": 1, "bar": 2}
    assert task.preprocess(data) == data

def test_postprocess_returns_output():
    task = DummyTask()
    output = {"result": 3}
    assert task.postprocess(output) == output

def test_metadata_property():
    task = DummyTask()
    meta = task.metadata
    assert meta["id"] == "dummy"
    assert meta["description"] == "A dummy task for testing"
    assert meta["required_inputs"] == ["foo", "bar"]
    assert meta["prompt_template"] == "dummy_template"

@pytest.mark.asyncio
async def test_run_success(monkeypatch):
    task = DummyTask()
    task._initialized = False
    input_data = {"foo": 2, "bar": 3}
    dependencies = DummyTaskDependencies()
    result = await task.run(input_data, dependencies)
    assert result == {"result": 5}

@pytest.mark.asyncio
async def test_run_missing_input(monkeypatch):
    task = DummyTask()
    input_data = {"foo": 2}
    dependencies = DummyTaskDependencies()
    with pytest.raises(ValueError):
        await task.run(input_data, dependencies)

@pytest.mark.asyncio
async def test_run_execute_raises(monkeypatch):
    class ErrorTask(DummyTask):
        async def execute(self, processed_input, dependencies):
            raise RuntimeError("fail!")
    task = ErrorTask()
    input_data = {"foo": 1, "bar": 2}
    dependencies = DummyTaskDependencies()
    with pytest.raises(RuntimeError):
        await task.run(input_data, dependencies)

def test_handle_error_raises():
    task = DummyTask()
    with pytest.raises(ValueError):
        task.handle_error(ValueError("fail"))
