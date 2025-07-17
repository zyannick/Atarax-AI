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
    cm = mock.Mock()
    cm.llama_config.get_llama_cpp_params.return_value = mock.Mock(
        model_path="/fake/llama/model.bin",
        model_dump=lambda: {"model_path": "/fake/llama/model.bin"}
    )
    cm.llama_config.get_generation_params.return_value = mock.Mock(
        model_dump=lambda: {}
    )
    cm.whisper_config.get_whisper_params.return_value = mock.Mock(
        model="/fake/whisper/model.bin",
        model_dump=lambda: {"model": "/fake/whisper/model.bin"}
    )
    cm.whisper_config.get_transcription_params.return_value = mock.Mock(
        model_dump=lambda: {}
    )
    return cm

@pytest.fixture
def manager(mock_config_manager, mock_logger):
    with mock.patch("ataraxai.praxis.utils.core_ai_service_manager.hegemonikon_py"):
        return CoreAIServiceManager(mock_config_manager, mock_logger)

def test_is_configured_true(manager):
    with mock.patch.object(Path, "exists", return_value=True):
        assert manager.is_configured() is True

def test_is_configured_false_llama_missing(manager):
    manager.config_manager.llama_config.get_llama_cpp_params.return_value.model_path = ""
    with mock.patch.object(Path, "exists", return_value=True):
        assert manager.is_configured() is False

def test_is_configured_false_whisper_missing(manager):
    manager.config_manager.whisper_config.get_whisper_params.return_value.model = ""
    with mock.patch.object(Path, "exists", return_value=True):
        assert manager.is_configured() is False

def test_is_configured_false_path_missing(manager):
    with mock.patch.object(Path, "exists", return_value=False):
        assert manager.is_configured() is False

def test_get_configuration_status_success(manager):
    with mock.patch.object(Path, "exists", return_value=True):
        status = manager.get_configuration_status()
        assert status["llama_configured"] is True
        assert status["whisper_configured"] is True
        assert status["llama_path_exists"] is True
        assert status["whisper_path_exists"] is True
        assert status["initialization_status"] == ServiceStatus.NOT_INITIALIZED.value

def test_get_configuration_status_llama_exception(manager, mock_logger):
    manager.config_manager.llama_config.get_llama_cpp_params.side_effect = Exception("fail")
    with mock.patch.object(Path, "exists", return_value=True):
        status = manager.get_configuration_status()
        assert status["llama_configured"] is False
        assert status["llama_model_path"] is None
        mock_logger.warning.assert_any_call(mock.ANY)

def test_get_configuration_status_whisper_exception(manager, mock_logger):
    manager.config_manager.whisper_config.get_whisper_params.side_effect = Exception("fail")
    with mock.patch.object(Path, "exists", return_value=True):
        status = manager.get_configuration_status()
        assert status["whisper_configured"] is False
        assert status["whisper_model_path"] is None
        mock_logger.warning.assert_any_call(mock.ANY)

def test_initialize_success(manager):
    with mock.patch.object(Path, "exists", return_value=True), \
         mock.patch("ataraxai.praxis.utils.core_ai_service_manager.hegemonikon_py") as mock_heg:
        mock_service = mock.Mock()
        mock_heg.CoreAIService.return_value = mock_service
        manager.initialize()
        assert manager.status == ServiceStatus.INITIALIZED
        assert manager.service is not None

def test_initialize_already_initialized(manager):
    manager.status = ServiceStatus.INITIALIZED
    manager.initialize()
    manager.logger.info.assert_any_call("Core AI services already initialized")

def test_initialize_failure(manager):
    with mock.patch.object(manager, "_validate_model_paths", side_effect=Exception("fail")):
        with pytest.raises(ServiceInitializationError):
            manager.initialize()
        assert manager.status == ServiceStatus.FAILED

def test_get_service_initializes(manager):
    with mock.patch.object(manager, "initialize") as mock_init:
        manager.status = ServiceStatus.NOT_INITIALIZED
        manager.get_service()
        mock_init.assert_called_once()

def test_get_service_failed(manager):
    manager.status = ServiceStatus.FAILED
    with pytest.raises(ServiceInitializationError):
        manager.get_service()

def test_shutdown_resets_service(manager):
    manager.service = mock.Mock()
    manager.status = ServiceStatus.INITIALIZED
    manager.shutdown()
    assert manager.service is None
    assert manager.status == ServiceStatus.NOT_INITIALIZED

def test_is_initialized_property(manager):
    manager.status = ServiceStatus.INITIALIZED
    assert manager.is_initialized is True
    manager.status = ServiceStatus.NOT_INITIALIZED
    assert manager.is_initialized is False