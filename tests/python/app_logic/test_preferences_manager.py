import pytest
import tempfile
from pathlib import Path
import yaml
from ataraxai.app_logic.preferences_manager import PreferencesManager, PREFERENCES_FILENAME
from ataraxai.app_logic.utils.config_schemas.user_preferences_schema import UserPreferences

@pytest.fixture
def temp_config_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

def test_preferences_created_if_not_exist(temp_config_dir):
    pm = PreferencesManager(config_path=temp_config_dir)
    config_file = temp_config_dir / PREFERENCES_FILENAME
    assert config_file.exists()
    assert isinstance(pm.preferences, UserPreferences)

def test_preferences_loaded_if_exist(temp_config_dir):
    prefs_data = {"theme": "dark", "language": "fr"}
    config_file = temp_config_dir / PREFERENCES_FILENAME
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(prefs_data, f)
    pm = PreferencesManager(config_path=temp_config_dir)
    assert pm.preferences.theme == "dark"
    assert pm.preferences.language == "fr"

def test_get_and_set_preferences(temp_config_dir):
    pm = PreferencesManager(config_path=temp_config_dir)
    pm.set("theme", "solarized")
    assert pm.get("theme") == "solarized"
    pm2 = PreferencesManager(config_path=temp_config_dir)
    assert pm2.get("theme") == "solarized"

def test_reload_preferences(temp_config_dir):
    pm = PreferencesManager(config_path=temp_config_dir)
    pm.set("theme", "light")
    config_file = temp_config_dir / PREFERENCES_FILENAME
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.safe_dump({"theme": "dark"}, f)
    pm.reload()
    assert pm.get("theme") == "dark"

def test_get_returns_default_for_missing_key(temp_config_dir):
    pm = PreferencesManager(config_path=temp_config_dir)
    assert pm.get("nonexistent_key", default="default_value") == "default_value"