import pytest
from ataraxai.app_logic.modules.base_task import BaseTask

class DummyPromptLoader:
    pass

class DummyTaskContext:
    pass

class ConcreteTask(BaseTask):
    id = "test_task"
    description = "A test task"
    required_inputs = ["foo"]
    prompt_template_name = "test_template"

    def execute(self, processed_input, context, prompt_loader):
        return {"executed": True, "input": processed_input}

def test_base_task_requires_id_and_description():
    class NoIdTask(BaseTask):
        description = "desc"
        def execute(self, processed_input, context, prompt_loader):
            pass
    with pytest.raises(NotImplementedError):
        NoIdTask()

    class NoDescriptionTask(BaseTask):
        id = "id"
        def execute(self, processed_input, context, prompt_loader):
            pass
    with pytest.raises(NotImplementedError):
        NoDescriptionTask()

def test_preprocess_returns_input_unchanged():
    task = ConcreteTask()
    input_data = {"foo": "bar"}
    context = DummyTaskContext()
    assert task.preprocess(input_data, context) == input_data

def test_postprocess_returns_output_unchanged():
    task = ConcreteTask()
    raw_output = {"result": 42}
    context = DummyTaskContext()
    assert task.postprocess(raw_output, context) == raw_output

def test_execute_called_and_returns_expected():
    task = ConcreteTask()
    processed_input = {"foo": "bar"}
    context = DummyTaskContext()
    prompt_loader = DummyPromptLoader()
    result = task.execute(processed_input, context, prompt_loader)
    assert result["executed"] is True
    assert result["input"] == processed_input

def test_base_task_is_abstract():
    with pytest.raises(TypeError):
        BaseTask()