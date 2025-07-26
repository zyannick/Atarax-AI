import pytest
from unittest import mock
from pathlib import Path
from ataraxai.praxis.utils.configuration_manager import ConfigurationManager
from ataraxai.praxis.utils.exceptions import ServiceInitializationError

@pytest.fixture
def mock_logger():
    return mock.Mock()

@pytest.fixture
def mock_config_dir(tmp_path):
    return tmp_path

@pytest.fixture
def patch_config_managers(monkeypatch):
    monkeypatch.setattr(
        "ataraxai.praxis.utils.configuration_manager.UserPreferencesManager",
        mock.Mock(return_value=mock.Mock())
    )
    monkeypatch.setattr(
        "ataraxai.praxis.utils.configuration_manager.LlamaConfigManager",
        mock.Mock(return_value=mock.Mock())
    )
    monkeypatch.setattr(
        "ataraxai.praxis.utils.configuration_manager.WhisperConfigManager",
        mock.Mock(return_value=mock.Mock())
    )
    monkeypatch.setattr(
        "ataraxai.praxis.utils.configuration_manager.RAGConfigManager",
        mock.Mock(return_value=mock.Mock())
    )

def test_initialization_success(mock_config_dir, mock_logger, patch_config_managers):
    cm = ConfigurationManager(config_dir=mock_config_dir, logger=mock_logger)
    assert hasattr(cm, "preferences_manager")
    assert hasattr(cm, "llama_config_manager")
    assert hasattr(cm, "whisper_config_manager")
    assert hasattr(cm, "rag_config_manager")
    mock_logger.info.assert_called_with("Configuration managers initialized successfully")

def test_initialization_failure(mock_config_dir, mock_logger, monkeypatch):
    def raise_exc(*args, **kwargs):
        raise Exception("fail")
    monkeypatch.setattr(
        "ataraxai.praxis.utils.configuration_manager.UserPreferencesManager",
        raise_exc
    )
    with pytest.raises(ServiceInitializationError) as excinfo:
        ConfigurationManager(config_dir=mock_config_dir, logger=mock_logger)
    assert "Configuration initialization failed" in str(excinfo.value)
    mock_logger.error.assert_called()

def test_get_watched_directories_returns_value(mock_config_dir, mock_logger, patch_config_managers):
    cm = ConfigurationManager(config_dir=mock_config_dir, logger=mock_logger)
    mock_config = mock.Mock()
    mock_config.rag_watched_directories = ["/dir1", "/dir2"]
    cm.rag_config_manager.get_config.return_value = mock_config
    result = cm.get_watched_directories()
    assert result == ["/dir1", "/dir2"]

def test_get_watched_directories_returns_none(mock_config_dir, mock_logger, patch_config_managers):
    cm = ConfigurationManager(config_dir=mock_config_dir, logger=mock_logger)
    mock_config = mock.Mock()
    delattr(mock_config, "rag_watched_directories")
    cm.rag_config_manager.get_config.return_value = mock_config
    result = cm.get_watched_directories()
    assert result is None

def test_add_watched_directory_adds_new(mock_config_dir, mock_logger, patch_config_managers):
    cm = ConfigurationManager(config_dir=mock_config_dir, logger=mock_logger)
    mock_config = mock.Mock()
    mock_config.rag_watched_directories = ["/dir1"]
    cm.rag_config_manager.get_config.return_value = mock_config
    cm.rag_config_manager.set = mock.Mock()
    cm.add_watched_directory("/dir2")
    cm.rag_config_manager.set.assert_called_with("rag_watched_directories", ["/dir1", "/dir2"])

def test_add_watched_directory_does_not_add_existing(mock_config_dir, mock_logger, patch_config_managers):
    cm = ConfigurationManager(config_dir=mock_config_dir, logger=mock_logger)
    mock_config = mock.Mock()
    mock_config.rag_watched_directories = ["/dir1"]
    cm.rag_config_manager.get_config.return_value = mock_config
    cm.rag_config_manager.set = mock.Mock()
    cm.add_watched_directory("/dir1")
    cm.rag_config_manager.set.assert_not_called()

def test_add_watched_directory_initializes_list(mock_config_dir, mock_logger, patch_config_managers):
    cm = ConfigurationManager(config_dir=mock_config_dir, logger=mock_logger)
    mock_config = mock.Mock()
    delattr(mock_config, "rag_watched_directories")
    cm.rag_config_manager.get_config.return_value = mock_config
    cm.rag_config_manager.set = mock.Mock()
    cm.add_watched_directory("/dir3")
    cm.rag_config_manager.set.assert_called_with("rag_watched_directories", ["/dir3"])