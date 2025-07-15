import pytest
from ataraxai.praxis.modules.prompt_engine.specific_tasks.base_task import BaseTask
from unittest import mock


class DummyPromptLoader:
    pass


class DummyTaskContext:
    pass


class ConcreteTask(BaseTask):
    id = "test_task"
    description = "A test task"
    required_inputs = ["foo"]
    prompt_template_name = "test_template"

    def _load_resources(self):
        return None

    def execute(self, processed_input, context, prompt_loader):
        return {"executed": True, "input": processed_input}


def test_base_task_requires_id_and_description():
    class NoIdTask(BaseTask):
        description = "desc"

        def _load_resources(self):
            pass

        def execute(self, processed_input, context, prompt_loader):
            pass

    with pytest.raises(NotImplementedError):
        NoIdTask()

    class NoDescriptionTask(BaseTask):
        id = "id"

        def _load_resources(self):
            pass

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

        def test_run_success(monkeypatch):
            task = ConcreteTask()
            context = DummyTaskContext()
            dependencies = {"dep": "value"}
            input_data = {"foo": "bar"}

            monkeypatch.setattr(task, "load_if_needed", mock.Mock())
            monkeypatch.setattr(task, "validate_inputs", mock.Mock())
            monkeypatch.setattr(
                task, "preprocess", mock.Mock(return_value={"foo": "baz"})
            )
            monkeypatch.setattr(
                task, "execute", mock.Mock(return_value={"executed": True})
            )
            monkeypatch.setattr(
                task, "postprocess", mock.Mock(return_value={"final": 123})
            )

            result = task.run(input_data, context, dependencies)

            task.load_if_needed.assert_called_once()
            task.validate_inputs.assert_called_once_with(input_data)
            task.preprocess.assert_called_once_with(input_data, context)
            task.execute.assert_called_once_with({"foo": "baz"}, context, dependencies)
            task.postprocess.assert_called_once_with({"executed": True}, context)
            assert result == {"final": 123}

        def test_run_handles_exception_and_calls_handle_error(monkeypatch):
            task = ConcreteTask()
            context = DummyTaskContext()
            dependencies = {}
            input_data = {"foo": "bar"}

            exc = ValueError("fail!")
            monkeypatch.setattr(task, "load_if_needed", mock.Mock())
            monkeypatch.setattr(task, "validate_inputs", mock.Mock(side_effect=exc))
            monkeypatch.setattr(
                task, "handle_error", mock.Mock(return_value="error_handled")
            )

            result = task.run(input_data, context, dependencies)

            task.handle_error.assert_called_once_with(exc, context)
            assert result == "error_handled"

        def test_run_raises_if_handle_error_raises(monkeypatch):
            task = ConcreteTask()
            context = DummyTaskContext()
            dependencies = {}
            input_data = {"foo": "bar"}

            exc = RuntimeError("fail!")
            monkeypatch.setattr(task, "load_if_needed", mock.Mock())
            monkeypatch.setattr(task, "validate_inputs", mock.Mock(side_effect=exc))
            monkeypatch.setattr(task, "handle_error", mock.Mock(side_effect=exc))

            with pytest.raises(RuntimeError):
                task.run(input_data, context, dependencies)
