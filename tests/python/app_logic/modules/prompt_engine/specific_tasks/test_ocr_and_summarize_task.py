import pytest
from unittest import mock
from ataraxai.app_logic.modules.prompt_engine.specific_tasks.ocr_and_summarize_task import (
    OCRandSummarizeTask,
)


@pytest.fixture
def ocr_task():
    return OCRandSummarizeTask()


def test_init_sets_attributes(ocr_task):
    assert ocr_task.id == "ocr_and_summarize"
    assert "Extracts text" in ocr_task.description
    assert ocr_task.required_inputs == ["image_path"]
    assert ocr_task.prompt_template_name == "ocr_summarize"


def test_load_resources_success(monkeypatch, ocr_task):
    monkeypatch.setattr("pytesseract.get_tesseract_version", lambda: "5.0.0")
    ocr_task._load_resources()


def test_load_resources_tesseract_not_found(monkeypatch, ocr_task):
    class DummyError(Exception):
        pass

    monkeypatch.setattr(
        "pytesseract.get_tesseract_version",
        lambda: (_ for _ in ()).throw(
            getattr(__import__("pytesseract"), "TesseractNotFoundError", Exception)(
                "not found"
            )
        ),
    )
    with pytest.raises(Exception):
        ocr_task._load_resources()


def test_execute_raises_on_missing_image_path(ocr_task):
    with pytest.raises(ValueError):
        dependencies = {"prompt_manager": mock.Mock(), "core_ai_service": mock.Mock()}
        ocr_task.execute({}, mock.Mock(), dependencies)


@mock.patch(
    "ataraxai.app_logic.modules.prompt_engine.specific_tasks.ocr_and_summarize_task.Image"
)
@mock.patch(
    "ataraxai.app_logic.modules.prompt_engine.specific_tasks.ocr_and_summarize_task.pytesseract"
)
def test_execute_returns_no_text_message(mock_pytesseract, mock_image, ocr_task):
    mock_image.open.return_value = "img"
    mock_pytesseract.image_to_string.return_value = "   "
    dependencies = {"prompt_manager": mock.Mock(), "core_ai_service": mock.Mock()}
    result = ocr_task.execute({"image_path": "fake_path"}, mock.Mock(), dependencies)
    assert result == "No text could be extracted from the image."


@mock.patch(
    "ataraxai.app_logic.modules.prompt_engine.specific_tasks.ocr_and_summarize_task.Image"
)
@mock.patch(
    "ataraxai.app_logic.modules.prompt_engine.specific_tasks.ocr_and_summarize_task.pytesseract"
)
def test_execute_successful_flow(mock_pytesseract, mock_image, ocr_task):
    mock_image.open.return_value = "img"
    mock_pytesseract.image_to_string.return_value = "Some extracted text."
    
    mock_prompt_manager = mock.Mock()
    mock_prompt_manager.load_template.return_value = "Formatted Prompt: Some extracted text."
    
    mock_core_ai_service = mock.Mock()
    mock_core_ai_service.process_prompt.return_value = "Summary."

    mock_gen_params = mock.Mock() 
    dependencies = {
        "prompt_manager": mock_prompt_manager,
        "core_ai_service": mock_core_ai_service,
        "generation_params": mock_gen_params, 
    }
    
    result = ocr_task.execute(
        processed_input={"image_path": "fake_path"}, 
        context=mock.Mock(), 
        dependencies=dependencies
    )
    
    assert result == "Summary."
    mock_prompt_manager.load_template.assert_called_once_with(
        "ocr_summarize", 
        ocr_text="Some extracted text."
    )
    mock_core_ai_service.process_prompt.assert_called_once_with(
        "Formatted Prompt: Some extracted text.", 
        mock_gen_params
    )


@mock.patch(
    "ataraxai.app_logic.modules.prompt_engine.specific_tasks.ocr_and_summarize_task.Image"
)
@mock.patch(
    "ataraxai.app_logic.modules.prompt_engine.specific_tasks.ocr_and_summarize_task.pytesseract"
)
def test_execute_ocr_failure(mock_pytesseract, mock_image, ocr_task):
    mock_image.open.side_effect = Exception("bad image")
    with pytest.raises(IOError):
        dependencies = {"prompt_manager": mock.Mock(), "core_ai_service": mock.Mock()}
        ocr_task.execute(
            {"image_path": "fake_path"}, mock.Mock(), dependencies
        )


@mock.patch(
    "ataraxai.app_logic.modules.prompt_engine.specific_tasks.ocr_and_summarize_task.Image"
)
@mock.patch(
    "ataraxai.app_logic.modules.prompt_engine.specific_tasks.ocr_and_summarize_task.pytesseract"
)
def test_execute_llm_failure(mock_pytesseract, mock_image, ocr_task):
    mock_image.open.return_value = "img"
    mock_pytesseract.image_to_string.return_value = "Some extracted text."
    mock_prompt_manager = mock.Mock()
    mock_prompt_manager.load_template.return_value = "Summarize: {ocr_text}"
    mock_core_ai_service = mock.Mock()
    mock_core_ai_service.process_prompt.side_effect = Exception("LLM error")
    dependencies = {
        "prompt_manager": mock_prompt_manager,
        "core_ai_service": mock_core_ai_service,
    }
    with pytest.raises(RuntimeError):
        ocr_task.execute({"image_path": "fake_path"}, mock.Mock(), dependencies)
