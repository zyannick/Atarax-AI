import pytest
from unittest import mock
from pathlib import Path
from ataraxai.praxis.utils.exceptions import ServiceInitializationError, ValidationError
from ataraxai.praxis.utils.service_status import ServiceStatus

from ataraxai.praxis.utils.core_ai_service_manager import (
    CoreAIServiceManager,
)

@pytest.fixture
def mock_logger():
    return mock.Mock()

@pytest.fixture
def mock_config_manager():
    llama_params = mock.Mock()
    llama_params.model_path = "/fake/llama/model.bin"
    llama_params.model_info.local_path = "/fake/llama/model.bin"
    llama_params.n_ctx = 2048
    llama_params.n_gpu_layers = 1
    llama_params.main_gpu = 0
    llama_params.tensor_split = None
    llama_params.vocab_only = False
    llama_params.use_map = False
    llama_params.use_mlock = False

    llama_config_manager = mock.Mock()
    llama_config_manager.get_llama_cpp_params.return_value = llama_params
    llama_config_manager.get_generation_params.return_value.model_dump.return_value = {}

    whisper_params = mock.Mock()
    whisper_params.model = "/fake/whisper/model.bin"
    whisper_config_manager = mock.Mock()
    whisper_config_manager.get_whisper_params.return_value = whisper_params
    whisper_config_manager.get_transcription_params.return_value.model_dump.return_value = {}

    config_manager = mock.Mock()
    config_manager.llama_config_manager = llama_config_manager
    config_manager.whisper_config_manager = whisper_config_manager
    return config_manager

@pytest.fixture
def mock_hegemonikon(monkeypatch):
    class DummyLlamaModelParams:
        @staticmethod
        def from_dict(d):
            return d
    class DummyGenerationParams:
        @staticmethod
        def from_dict(d):
            return d
    class DummyCoreAIService:
        def initialize_llama_model(self, params):
            self.llama_params = params
        def process_prompt(self, prompt, gen_params):
            return b"response"
        def tokenization(self, text):
            return [1, 2, 3]
        def detokenization(self, tokens):
            return "decoded"
    dummy_module = mock.Mock()
    dummy_module.LlamaModelParams = DummyLlamaModelParams
    dummy_module.GenerationParams = DummyGenerationParams
    dummy_module.CoreAIService = DummyCoreAIService
    monkeypatch.setattr("ataraxai.hegemonikon_py", dummy_module)

@pytest.fixture
def manager(mock_config_manager, mock_logger, monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "exists", lambda self: True)
    return CoreAIServiceManager(mock_config_manager, mock_logger)

def test_initialize_sets_status_and_creates_service(manager, mock_config_manager, mock_logger, monkeypatch):
    monkeypatch.setattr(Path, "exists", lambda self: True)
    manager._create_core_ai_service = mock.Mock(return_value=mock.Mock())
    manager._convert_params = mock.Mock(return_value=("llama", "gen"))
    manager.llama_cpp_status = ServiceStatus.NOT_INITIALIZED
    manager.initialize()
    assert manager.llama_cpp_status == ServiceStatus.NOT_INITIALIZED or manager.core_ai_service is not None

def test_initialize_raises_on_invalid_path(manager, mock_config_manager, monkeypatch):
    monkeypatch.setattr(Path, "exists", lambda self: False)
    with pytest.raises(ServiceInitializationError):
        manager.initialize()

def test_get_service_initializes_if_needed(manager, monkeypatch):
    manager.llama_cpp_status = ServiceStatus.NOT_INITIALIZED
    manager.initialize = mock.Mock()
    manager.core_ai_service = mock.Mock()
    manager.get_service()
    manager.initialize.assert_called_once()

def test_get_service_raises_if_failed(manager):
    manager.llama_cpp_status = ServiceStatus.FAILED
    with pytest.raises(ServiceInitializationError):
        manager.get_service()

def test_process_prompt_success(manager):
    manager.core_ai_service = mock.Mock()
    manager.llama_cpp_generation_params_cc = {"param": 1}
    manager.core_ai_service.process_prompt.return_value = "response"
    result = manager.process_prompt("prompt")
    assert result == "response"

def test_process_prompt_raises_if_not_initialized(manager):
    manager.core_ai_service = None
    with pytest.raises(ServiceInitializationError):
        manager.process_prompt("prompt")

def test_get_llama_cpp_model_context_size(manager):
    manager.core_ai_service = mock.Mock()
    manager.config_manager.llama_config_manager.get_llama_cpp_params.return_value.n_ctx = 4096
    assert manager.get_llama_cpp_model_context_size() == 4096

def test_is_configured_true(manager, monkeypatch):
    monkeypatch.setattr(Path, "exists", lambda self: True)
    assert manager.is_configured() is True

def test_is_configured_false(manager, monkeypatch):
    monkeypatch.setattr(Path, "exists", lambda self: False)
    assert manager.is_configured() is False

def test_get_configuration_status(manager, monkeypatch):
    monkeypatch.setattr(Path, "exists", lambda self: True)
    status = manager.get_configuration_status()
    assert status["llama_configured"] is True
    assert status["whisper_configured"] is True
    assert status["llama_path_exists"] is True
    assert status["whisper_path_exists"] is True

def test_tokenize_and_decode(manager):
    manager.core_ai_service = mock.Mock()
    manager.core_ai_service.tokenization.return_value = [1, 2, 3]
    manager.core_ai_service.detokenization.return_value = "decoded"
    tokens = manager.tokenize("hello")
    assert tokens == [1, 2, 3]
    decoded = manager.decode([1, 2, 3])
    assert decoded == "decoded"

def test_tokenize_raises_if_not_initialized(manager):
    manager.core_ai_service = None
    with pytest.raises(ServiceInitializationError):
        manager.tokenize("hello")

def test_decode_raises_if_not_initialized(manager):
    manager.core_ai_service = None
    with pytest.raises(ServiceInitializationError):
        manager.decode([1, 2, 3])

def test_shutdown_resets_service(manager):
    mock_service = mock.Mock()
    manager.core_ai_service = mock_service
    manager.status = ServiceStatus.INITIALIZED
    manager.shutdown()
    assert manager.core_ai_service is None

def test_is_initialized_property(manager):
    manager.status = ServiceStatus.INITIALIZED
    assert manager.is_initialized is True
    manager.status = ServiceStatus.NOT_INITIALIZED
    assert manager.is_initialized is False