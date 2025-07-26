import pytest
from unittest import mock
from pathlib import Path
from ataraxai.praxis.utils.exceptions import ServiceInitializationError, ValidationError
from ataraxai.praxis.utils.service_status import ServiceStatus
from ataraxai.praxis.utils.core_ai_service_manager import CoreAIServiceManager




@pytest.fixture
def mock_config_manager():
    mock_llama_params = mock.Mock()
    mock_llama_params.model_path.return_value = "/fake/llama/model.bin"
    mock_llama_params.model_dump.return_value = {"model_path": "/fake/llama/model.bin"}
    mock_generation_params = mock.Mock()
    mock_generation_params.model_dump.return_value = {}
    mock_llama_config_manager = mock.Mock()
    mock_llama_config_manager.get_llama_cpp_params.return_value = mock_llama_params
    mock_llama_config_manager.get_generation_params.return_value = mock_generation_params

    mock_whisper_params = mock.Mock()
    mock_whisper_params.model = "/fake/whisper/model.bin"
    mock_whisper_params.model_dump.return_value = {"model": "/fake/whisper/model.bin"}
    mock_transcription_params = mock.Mock()
    mock_transcription_params.model_dump.return_value = {}
    mock_whisper_config_manager = mock.Mock()
    mock_whisper_config_manager.get_whisper_params.return_value = mock_whisper_params
    mock_whisper_config_manager.get_transcription_params.return_value = mock_transcription_params

    config_manager = mock.Mock()
    config_manager.llama_config_manager = mock_llama_config_manager
    config_manager.whisper_config_manager = mock_whisper_config_manager
    return config_manager

@pytest.fixture
def mock_logger():
    return mock.Mock()

@pytest.fixture
def mock_hegemonikon(monkeypatch):
    class DummyLlamaModelParams:
        @staticmethod
        def from_dict(d): return d
    class DummyGenerationParams:
        @staticmethod
        def from_dict(d): return d
    class DummyWhisperModelParams:
        @staticmethod
        def from_dict(d): return d
    class DummyWhisperGenerationParams:
        @staticmethod
        def from_dict(d): return d
    class DummyCoreAIService:
        def initialize_llama_model(self, params): pass
        def initialize_whisper_model(self, params): pass
    dummy_module = mock.Mock()
    dummy_module.LlamaModelParams = DummyLlamaModelParams
    dummy_module.GenerationParams = DummyGenerationParams
    dummy_module.WhisperModelParams = DummyWhisperModelParams
    dummy_module.WhisperGenerationParams = DummyWhisperGenerationParams
    dummy_module.CoreAIService = DummyCoreAIService
    monkeypatch.setattr("ataraxai.hegemonikon_py", dummy_module)

def test_initialization_success(monkeypatch, mock_config_manager, mock_logger, mock_hegemonikon):
    monkeypatch.setattr(Path, "exists", lambda self: True)
    manager = CoreAIServiceManager(mock_config_manager, mock_logger)
    manager.initialize()
    assert manager.status == ServiceStatus.INITIALIZED
    assert manager.is_initialized
    assert manager.service is not None

def test_initialization_failure_llama_path(monkeypatch, mock_config_manager, mock_logger):
    mock_config_manager.llama_config_manager.get_llama_cpp_params.return_value.model_path.return_value = ""
    manager = CoreAIServiceManager(mock_config_manager, mock_logger)
    with pytest.raises(ServiceInitializationError):
        manager.initialize()
    assert manager.status == ServiceStatus.FAILED

def test_initialization_failure_whisper_path(monkeypatch, mock_config_manager, mock_logger):
    monkeypatch.setattr(Path, "exists", lambda self: True)
    mock_config_manager.whisper_config_manager.get_whisper_params.return_value.model = ""
    manager = CoreAIServiceManager(mock_config_manager, mock_logger)
    with pytest.raises(ServiceInitializationError):
        manager.initialize()
    assert manager.status == ServiceStatus.FAILED

def test_is_configured_true(monkeypatch, mock_config_manager, mock_logger):
    monkeypatch.setattr(Path, "exists", lambda self: True)
    manager = CoreAIServiceManager(mock_config_manager, mock_logger)
    assert manager.is_configured() is True

def test_is_configured_false(monkeypatch, mock_config_manager, mock_logger):
    mock_config_manager.llama_config_manager.get_llama_cpp_params.return_value.model_path.return_value = ""
    manager = CoreAIServiceManager(mock_config_manager, mock_logger)
    assert manager.is_configured() is False

def test_get_configuration_status(monkeypatch, mock_config_manager, mock_logger):
    monkeypatch.setattr(Path, "exists", lambda self: True)
    manager = CoreAIServiceManager(mock_config_manager, mock_logger)
    status = manager.get_configuration_status()
    assert status["llama_configured"] is True
    assert status["whisper_configured"] is True
    assert status["llama_path_exists"] is True
    assert status["whisper_path_exists"] is True
    assert status["initialization_status"] == ServiceStatus.NOT_INITIALIZED.value

def test_shutdown_resets_service_and_status(mock_config_manager, mock_logger):
    manager = CoreAIServiceManager(mock_config_manager, mock_logger)
    manager.service = mock.Mock()
    manager.status = ServiceStatus.INITIALIZED
    manager.shutdown()
    assert manager.service is None
    assert manager.status == ServiceStatus.NOT_INITIALIZED
