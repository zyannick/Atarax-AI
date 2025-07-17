import pytest
from unittest import mock
from pathlib import Path
from ataraxai.praxis.utils.configuration_manager import ConfigurationManager
from ataraxai.praxis.utils.exceptions import ServiceInitializationError

@pytest.fixture
def mock_logger():
    return mock.Mock()

@pytest.fixture
def mock_preferences_manager():
    with mock.patch("ataraxai.praxis.utils.configuration_manager.PreferencesManager") as m:
        yield m

@pytest.fixture
def mock_llama_config_manager():
    with mock.patch("ataraxai.praxis.utils.configuration_manager.LlamaConfigManager") as m:
        yield m

@pytest.fixture
def mock_whisper_config_manager():
    with mock.patch("ataraxai.praxis.utils.configuration_manager.WhisperConfigManager") as m:
        yield m

@pytest.fixture
def mock_rag_config_manager():
    with mock.patch("ataraxai.praxis.utils.configuration_manager.RAGConfigManager") as m:
        yield m

@pytest.fixture
def config_manager(
    mock_logger,
    mock_preferences_manager,
    mock_llama_config_manager,
    mock_whisper_config_manager,
    mock_rag_config_manager,
):
    return ConfigurationManager(config_dir=Path("/fake/config"), logger=mock_logger)

def test_init_successful(
    mock_logger,
    mock_preferences_manager,
    mock_llama_config_manager,
    mock_whisper_config_manager,
    mock_rag_config_manager,
):
    ConfigurationManager(config_dir=Path("/fake/config"), logger=mock_logger)
    mock_logger.info.assert_called_with("Configuration managers initialized successfully")

def test_init_failure(mock_logger):
    with mock.patch(
        "ataraxai.praxis.utils.configuration_manager.PreferencesManager",
        side_effect=Exception("fail"),
    ):
        with pytest.raises(ServiceInitializationError) as excinfo:
            ConfigurationManager(config_dir=Path("/fake/config"), logger=mock_logger)
        assert "Configuration initialization failed" in str(excinfo.value)
        mock_logger.error.assert_called()

def test_get_watched_directories_returns_value(config_manager):
    fake_config = mock.Mock()
    fake_config.rag_watched_directories = ["/dir1", "/dir2"]
    config_manager.rag_config.get_config.return_value = fake_config
    result = config_manager.get_watched_directories()
    assert result == ["/dir1", "/dir2"]

def test_get_watched_directories_returns_none(config_manager):
    fake_config = mock.Mock()
    delattr(fake_config, "rag_watched_directories")
    config_manager.rag_config.get_config.return_value = fake_config
    result = config_manager.get_watched_directories()
    assert result is None

def test_add_watched_directory_adds_new(config_manager):
    config_manager.get_watched_directories = mock.Mock(return_value=["/dir1"])
    config_manager.rag_config.set = mock.Mock()
    config_manager.add_watched_directory("/dir2")
    config_manager.rag_config.set.assert_called_with("rag_watched_directories", ["/dir1", "/dir2"])

def test_add_watched_directory_no_duplicates(config_manager):
    config_manager.get_watched_directories = mock.Mock(return_value=["/dir1"])
    config_manager.rag_config.set = mock.Mock()
    config_manager.add_watched_directory("/dir1")
    config_manager.rag_config.set.assert_not_called()

def test_add_watched_directory_when_none(config_manager):
    config_manager.get_watched_directories = mock.Mock(return_value=None)
    config_manager.rag_config.set = mock.Mock()
    config_manager.add_watched_directory("/dir3")
    config_manager.rag_config.set.assert_called_with("rag_watched_directories", ["/dir3"])