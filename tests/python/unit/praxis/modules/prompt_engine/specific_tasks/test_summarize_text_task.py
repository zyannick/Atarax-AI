import pytest
from unittest import mock
from ataraxai.praxis.modules.prompt_engine.specific_tasks.summarize_text_task import SummarizeTextTask

@pytest.fixture
def mock_dependencies():
    prompt_manager = mock.Mock()
    prompt_manager.load_template.return_value = "Summarize: This is a test text."
    core_ai_service = mock.Mock()
    core_ai_service.process_prompt.return_value = "A summary."
    return {
        "prompt_manager": prompt_manager,
        "core_ai_service": core_ai_service,
        "generation_params": {"temperature": 0.5}
    }

@pytest.fixture
def mock_context():
    return mock.Mock()

def test_init_sets_attributes():
    task = SummarizeTextTask()
    assert task.id == "summarize_text"
    assert task.description == "Summarizes a given block of text."
    assert task.required_inputs == ["text"]
    assert task.prompt_template_name == "summarize_text"

def test_load_resources_prints_message(capsys):
    task = SummarizeTextTask()
    task._load_resources()
    captured = capsys.readouterr()
    assert f"Task '{task.id}' requires no special resources to load." in captured.out

def test_execute_success(mock_dependencies, mock_context):
    task = SummarizeTextTask()
    processed_input = {"text": "This is a test text."}
    result = task.execute(processed_input, mock_context, mock_dependencies)
    assert result == "A summary."
    mock_dependencies["prompt_manager"].load_template.assert_called_once_with(
        "summarize_text", text_to_summarize="This is a test text."
    )
    mock_dependencies["core_ai_service"].process_prompt.assert_called_once()

def test_execute_missing_text_raises():
    task = SummarizeTextTask()
    with pytest.raises(ValueError, match="Input dictionary must contain 'text'."):
        task.execute({}, mock.Mock(), {"prompt_manager": mock.Mock(), "core_ai_service": mock.Mock()})

def test_execute_strips_summary(mock_dependencies, mock_context):
    task = SummarizeTextTask()
    mock_dependencies["core_ai_service"].process_prompt.return_value = "  summary with spaces  "
    processed_input = {"text": "Some text"}
    result = task.execute(processed_input, mock_context, mock_dependencies)
    assert result == "summary with spaces"