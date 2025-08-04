import pytest
from unittest import mock
from ataraxai.praxis.modules.prompt_engine.specific_tasks.base_task import BaseTask

class DummyTaskDependencies:
    pass

class DummyTask(BaseTask):
    id = "dummy"
    description = "A dummy task for testing."
    required_inputs = ["foo", "bar"]
    prompt_template_name = "dummy_template"

    def _load_resources(self):
        self.resources_loaded = True

    def execute(self, processed_input, dependencies):
        return {"result": processed_input["foo"] + processed_input["bar"]}

def test_base_task_init_requires_id_and_description():
    class NoIdTask(BaseTask):
        description = "desc"
        def _load_resources(self): pass
        def execute(self, processed_input, dependencies): pass
    with pytest.raises(NotImplementedError):
        NoIdTask()
    class NoDescTask(BaseTask):
        id = "id"
        def _load_resources(self): pass
        def execute(self, processed_input, dependencies): pass
    with pytest.raises(NotImplementedError):
        NoDescTask()

def test_load_if_needed_loads_once(monkeypatch):
    task = DummyTask()
    task._initialized = False
    called = []
    def fake_load_resources():
        called.append(True)
    monkeypatch.setattr(task, "_load_resources", fake_load_resources)
    task.load_if_needed()
    assert called == [True]
    called.clear()
    task.load_if_needed()
    assert called == []

def test_validate_inputs_raises_on_missing():
    task = DummyTask()
    with pytest.raises(ValueError) as exc:
        task.validate_inputs({"foo": 1})
    assert "missing required inputs" in str(exc.value)

def test_validate_inputs_passes_on_all_present():
    task = DummyTask()
    task.validate_inputs({"foo": 1, "bar": 2})

def test_preprocess_returns_input():
    task = DummyTask()
    data = {"foo": 1, "bar": 2}
    assert task.preprocess(data) == data

def test_postprocess_returns_raw_output():
    task = DummyTask()
    raw = {"a": 1}
    assert task.postprocess(raw) == raw

def test_handle_error_prints_and_raises():
    task = DummyTask()
    err = RuntimeError("fail")
    with pytest.raises(RuntimeError):
        task.handle_error(err)

def test_run_success(monkeypatch):
    task = DummyTask()
    monkeypatch.setattr(task, "_load_resources", lambda: None)
    input_data = {"foo": 2, "bar": 3}
    deps = DummyTaskDependencies()
    result = task.run(input_data, deps)
    assert result == {"result": 5}

def test_run_handles_error(monkeypatch):
    class ErrorTask(DummyTask):
        def execute(self, processed_input, dependencies):
            raise ValueError("fail")
        def handle_error(self, error):
            return "handled"
    task = ErrorTask()
    monkeypatch.setattr(task, "_load_resources", lambda: None)
    input_data = {"foo": 1, "bar": 2}
    deps = DummyTaskDependencies()
    assert task.run(input_data, deps) == "handled"

def test_metadata_property():
    task = DummyTask()
    meta = task.metadata
    assert meta["id"] == "dummy"
    assert meta["description"] == "A dummy task for testing."
    assert meta["required_inputs"] == ["foo", "bar"]
    assert meta["prompt_template"] == "dummy_template"