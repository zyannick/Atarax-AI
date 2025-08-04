import pytest
from unittest import mock
from pathlib import Path
from ataraxai.praxis.utils.setup_manager import SetupManager

@pytest.fixture
def mock_directories():
    mock_dirs = mock.Mock()
    mock_dirs.config = Path("/tmp/config")
    return mock_dirs

@pytest.fixture
def mock_config():
    mock_cfg = mock.Mock()
    mock_cfg.get_setup_marker_filename.return_value = "markerfile"
    return mock_cfg

@pytest.fixture
def mock_logger():
    return mock.Mock()

@pytest.fixture
def setup_manager(mock_directories, mock_config, mock_logger):
    with mock.patch("ataraxai.praxis.utils.setup_manager.__version__", "1.0.0"):
        return SetupManager(mock_directories, mock_config, mock_logger)

def test_is_first_launch_true(setup_manager):
    with mock.patch.object(Path, "exists", return_value=False):
        assert setup_manager.is_first_launch() is True

def test_is_first_launch_false(setup_manager):
    with mock.patch.object(Path, "exists", return_value=True):
        assert setup_manager.is_first_launch() is False

def test_perform_first_launch_setup_creates_marker(setup_manager):
    with mock.patch.object(setup_manager, "is_first_launch", return_value=True), \
         mock.patch.object(setup_manager, "_create_setup_marker") as mock_create_marker:
        setup_manager.perform_first_launch_setup()
        mock_create_marker.assert_called_once()
        setup_manager.logger.info.assert_any_call("Performing first launch setup...")
        setup_manager.logger.info.assert_any_call("First launch setup completed successfully")

def test_perform_first_launch_setup_skips_if_not_first(setup_manager):
    with mock.patch.object(setup_manager, "is_first_launch", return_value=False), \
         mock.patch.object(setup_manager, "_create_setup_marker") as mock_create_marker:
        setup_manager.perform_first_launch_setup()
        mock_create_marker.assert_not_called()
        setup_manager.logger.info.assert_called_with("Skipping first launch setup - already completed")

def test_perform_first_launch_setup_logs_and_raises_on_error(setup_manager):
    with mock.patch.object(setup_manager, "is_first_launch", return_value=True), \
         mock.patch.object(setup_manager, "_create_setup_marker", side_effect=Exception("fail")):
        with pytest.raises(Exception, match="fail"):
            setup_manager.perform_first_launch_setup()
        setup_manager.logger.error.assert_called()
        assert "First launch setup failed" in setup_manager.logger.error.call_args[0][0]

def test_create_setup_marker_calls_touch(setup_manager):
    with mock.patch("pathlib.Path.touch") as mock_touch:
        setup_manager._create_setup_marker()
        mock_touch.assert_called_once_with(exist_ok=False)

def test_marker_file_path_uses_config_and_version(mock_directories, mock_config, mock_logger):
    mock_directories.config = Path("/tmp/config")
    mock_config.get_setup_marker_filename.return_value = "markerfile-1.2.3"
    with mock.patch("ataraxai.praxis.utils.setup_manager.__version__", "1.2.3"):
        manager = SetupManager(mock_directories, mock_config, mock_logger)
        expected_path = Path("/tmp/config") / "markerfile-1.2.3"
        assert manager._marker_file == expected_path

def test_create_setup_marker_raises_if_file_exists(setup_manager):
    with mock.patch("pathlib.Path.touch", side_effect=FileExistsError):
        with pytest.raises(FileExistsError):
            setup_manager._create_setup_marker()
