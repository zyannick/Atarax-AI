import pytest
from unittest import mock
from pathlib import Path
from ataraxai.praxis.utils.app_directories import AppDirectories

@pytest.fixture
def fake_settings():
    class FakeSettings:
        app_name = "AtaraxAI"
        app_author = "AtaraxAI"
    return FakeSettings()

@mock.patch("ataraxai.praxis.utils.app_directories.user_config_dir")
@mock.patch("ataraxai.praxis.utils.app_directories.user_data_dir")
@mock.patch("ataraxai.praxis.utils.app_directories.user_cache_dir")
@mock.patch("ataraxai.praxis.utils.app_directories.user_log_dir")
@mock.patch.object(AppDirectories, "create_directories")
def test_create_default_calls_appdirs_and_creates_dirs(
    mock_create_dirs,
    mock_log_dir,
    mock_cache_dir,
    mock_data_dir,
    mock_config_dir,
    fake_settings,
):
    mock_config_dir.return_value = "/tmp/config"
    mock_data_dir.return_value = "/tmp/data"
    mock_cache_dir.return_value = "/tmp/cache"
    mock_log_dir.return_value = "/tmp/logs"

    dirs = AppDirectories.create_default(fake_settings)

    assert dirs.config == Path("/tmp/config")
    assert dirs.data == Path("/tmp/data")
    assert dirs.cache == Path("/tmp/cache")
    assert dirs.logs == Path("/tmp/logs")
    mock_create_dirs.assert_called_once()

def test_create_directories_creates_all_dirs(tmp_path):
    dirs = AppDirectories(
        config=tmp_path / "config",
        data=tmp_path / "data",
        cache=tmp_path / "cache",
        logs=tmp_path / "logs"
    )
    # Ensure directories do not exist before
    for d in [dirs.config, dirs.data, dirs.cache, dirs.logs]:
        assert not d.exists()
    dirs.create_directories()
    # Now they should exist
    for d in [dirs.config, dirs.data, dirs.cache, dirs.logs]:
        assert d.exists()
        assert d.is_dir()